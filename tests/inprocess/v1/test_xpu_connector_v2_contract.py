# SPDX-License-Identifier: Apache-2.0
"""CPU checks of V2 construction, geometry validation and native arguments."""

# Standard
from collections.abc import Iterator, Sequence
from contextlib import nullcontext
from unittest.mock import Mock, patch

# Third Party
import pytest
import torch

# First Party
from lmcache.inprocess.v1.config import LMCacheEngineConfig
from lmcache.inprocess.v1.gpu_connector import CreateGPUConnector
from lmcache.inprocess.v1.gpu_connector.xpu_connectors import (
    VLLMPagedMemXPUConnectorV2,
)
from lmcache.inprocess.v1.metadata import LMCacheMetadata
from lmcache.utils import EngineType
from lmcache.v1.gpu_connector.kv_format.types import KVLayoutName, LayoutHints
from lmcache.v1.memory_allocators.tensor_memory_allocator import TensorMemoryAllocator
from lmcache.v1.memory_management import MemoryFormat
import lmcache.inprocess.v1.gpu_connector as factory_module
import lmcache.inprocess.v1.gpu_connector.xpu_connectors as xpu_module
import lmcache.lmcache_native as lmcache_native

pytestmark = pytest.mark.no_shared_allocator


class _XPUTensor(torch.Tensor):
    """CPU tensor advertising a device solely for public argument checks."""

    device: torch.device = torch.device("xpu:0")


class _CPUStream:
    """Stream stand-in that does no device work."""

    def synchronize(self) -> None:
        """Complete the mocked stream synchronously."""


def _make_metadata() -> LMCacheMetadata:
    """Build ordinary two-plane model metadata without a group manager."""
    return LMCacheMetadata(
        model_name="xpu-v2-cpu-contract",
        world_size=1,
        local_world_size=1,
        worker_id=0,
        local_worker_id=0,
        kv_dtype=torch.float16,
        kv_shape=(2, 2, 6, 4, 3),
        chunk_size=6,
    )


def _as_mock_xpu(tensor: torch.Tensor) -> torch.Tensor:
    """Expose CPU storage through a public tensor subclass operation."""
    return tensor.as_subclass(_XPUTensor)


@pytest.fixture
def native_call(monkeypatch: pytest.MonkeyPatch) -> Iterator[Mock]:
    """Replace device work while retaining real V2 constructors and discovery."""
    native = Mock()
    monkeypatch.setattr(xpu_module.device_ops, "multi_layer_kv_transfer", native)
    monkeypatch.setattr(factory_module, "torch_device_type", "xpu")
    monkeypatch.setattr(factory_module, "torch_dev", Mock())
    real_empty = torch.empty

    def cpu_empty(
        size: int | Sequence[int],
        *,
        dtype: torch.dtype | None = None,
        device: torch.device | str | None = None,
    ) -> torch.Tensor:
        """Redirect pointer-array allocation to CPU without a device probe."""
        if isinstance(size, int):
            return real_empty(size, dtype=dtype, device="cpu")
        return real_empty(tuple(size), dtype=dtype, device="cpu")

    with (
        patch("torch.xpu.Stream", return_value=_CPUStream()),
        patch("torch.xpu.stream", return_value=nullcontext()),
        patch("torch.empty", side_effect=cpu_empty),
    ):
        yield native


@pytest.mark.parametrize("layout", ["NHD", "HND"])
@pytest.mark.parametrize("factory", [True, False], ids=["factory", "metadata"])
@pytest.mark.parametrize("cached_tokens", [0, 4, 99])
def test_v2_public_calls_forward_hints_geometry_and_prefix(
    native_call: Mock,
    layout: KVLayoutName,
    factory: bool,
    cached_tokens: int,
) -> None:
    """Factory and direct constructors pass component geometry by name."""
    metadata = _make_metadata()
    hints: LayoutHints = {"kv_layout": layout}
    if factory:
        config = LMCacheEngineConfig.from_defaults(
            chunk_size=6,
            enable_pd=True,
            use_layerwise=False,
            use_gpu_connector_v3=False,
        )
        connector = CreateGPUConnector(
            config, metadata, EngineType.VLLM, layout_hints=hints
        )
    else:
        connector = VLLMPagedMemXPUConnectorV2.from_metadata(
            metadata, False, torch.device("xpu:0"), layout_hints=hints
        )
    assert isinstance(connector, VLLMPagedMemXPUConnectorV2)
    assert metadata.kv_layer_groups_manager is None
    assert metadata.get_shapes(4) == [torch.Size((2, 2, 4, 12))]

    shape = (4, 4, 3, 6) if layout == "HND" else (4, 3, 4, 6)
    inner = (18, 6, 1) if layout == "HND" else (24, 6, 1)
    backing = torch.zeros(1 + 2 * 4 * 73, dtype=torch.float16)
    raw_views = [
        _as_mock_xpu(backing.as_strided(shape, (73, *inner), 1 + layer * 4 * 73))
        for layer in range(2)
    ]
    caches = [view.permute(0, 2, 1, 3) for view in raw_views]
    slots = _as_mock_xpu(torch.tensor([0, 1, 10, -1, 4, 8], dtype=torch.int64))
    allocator = TensorMemoryAllocator(torch.empty(4096, dtype=torch.uint8))
    memory_obj = allocator.allocate(metadata.get_shapes(4), metadata.get_dtypes())
    assert memory_obj is not None
    try:
        connector.from_gpu(memory_obj, 2, 6, kvcaches=caches, slot_mapping=slots)
        store_args, store_kwargs = native_call.call_args
        assert len(store_args) == 7
        assert store_kwargs["head_size"] == 3
        assert store_kwargs["block_stride_elems"] == 73
        expected_format = (
            lmcache_native.EngineKVFormat.NL_X_NB_NH_BS_CS
            if layout == "HND"
            else lmcache_native.EngineKVFormat.NL_X_NB_BS_NH_CS
        )
        assert store_args[6] == expected_format
        assert store_kwargs["block_size"] == 3
        assert store_args[4] == 12
        assert store_args[1].tolist() == [view.data_ptr() for view in raw_views]
        assert torch.equal(store_args[2], slots[2:6])
        assert memory_obj.metadata.fmt == MemoryFormat.KV_2LTD
        assert memory_obj.tensor is not None
        assert memory_obj.tensor.shape == (2, 2, 4, 12)

        connector.to_gpu(
            memory_obj,
            2,
            6,
            kvcaches=caches,
            slot_mapping=slots,
            vllm_cached_tokens=cached_tokens,
        )
        load_args, load_kwargs = native_call.call_args
        assert len(load_args) == 7
        assert load_args[5] == lmcache_native.TransferDirection.H2D
        assert load_kwargs["head_size"] == 3
        assert load_kwargs["block_size"] == 3
        assert load_kwargs["block_stride_elems"] == 73
        assert load_kwargs["skip_prefix_n_tokens"] == min(4, max(0, cached_tokens - 2))
        assert metadata.kv_layer_groups_manager is None
    finally:
        allocator.free(memory_obj)
    assert allocator.memcheck()


@pytest.mark.parametrize(
    "invalid_geometry",
    ["odd-content-width", "different-head-width", "different-layer-stride"],
)
def test_v2_rejects_unsupported_geometry_before_native_transfer(
    native_call: Mock, invalid_geometry: str
) -> None:
    """Invalid packed component geometry or heterogeneous strides fail early."""
    metadata = _make_metadata()
    connector = VLLMPagedMemXPUConnectorV2.from_metadata(
        metadata, False, torch.device("xpu:0"), layout_hints={"kv_layout": "NHD"}
    )
    if invalid_geometry == "odd-content-width":
        caches = [
            _as_mock_xpu(torch.zeros(4, 3, 4, 7, dtype=torch.float16)) for _ in range(2)
        ]
    elif invalid_geometry == "different-head-width":
        caches = [
            _as_mock_xpu(torch.zeros(4, 3, 4, 8, dtype=torch.float16)) for _ in range(2)
        ]
    else:
        caches = []
        for stride in (73, 74):
            backing = torch.zeros(1 + 4 * stride, dtype=torch.float16)
            caches.append(
                _as_mock_xpu(backing.as_strided((4, 3, 4, 6), (stride, 24, 6, 1), 1))
            )
    slots = _as_mock_xpu(torch.arange(4, dtype=torch.int64))
    allocator = TensorMemoryAllocator(torch.empty(4096, dtype=torch.uint8))
    memory_obj = allocator.allocate(metadata.get_shapes(4), metadata.get_dtypes())
    assert memory_obj is not None
    try:
        with pytest.raises(
            ValueError, match="(?i)(head|width|stride|geometry|layer|component)"
        ):
            connector.from_gpu(memory_obj, 0, 4, kvcaches=caches, slot_mapping=slots)
        native_call.assert_not_called()
    finally:
        allocator.free(memory_obj)


@pytest.mark.parametrize("direction", ["store", "retrieve"])
@pytest.mark.parametrize("layout", ["NHD", "HND"])
@pytest.mark.parametrize("invalid_layer", ["smaller-block-count", "cpu-device"])
def test_v2_rejects_incompatible_packed_layer_storage(
    native_call: Mock,
    direction: str,
    layout: KVLayoutName,
    invalid_layer: str,
) -> None:
    """Packed layers with different capacity or devices fail before transfer."""
    metadata = _make_metadata()
    connector = VLLMPagedMemXPUConnectorV2.from_metadata(
        metadata, False, torch.device("xpu:0"), layout_hints={"kv_layout": layout}
    )
    shape = (4, 4, 3, 6) if layout == "HND" else (4, 3, 4, 6)
    caches = [_as_mock_xpu(torch.zeros(shape, dtype=torch.float16)) for _ in range(2)]
    if invalid_layer == "smaller-block-count":
        caches[1] = _as_mock_xpu(torch.zeros((2, *shape[1:]), dtype=torch.float16))
    else:
        caches[1] = torch.zeros(shape, dtype=torch.float16)
    slots = _as_mock_xpu(torch.arange(4, dtype=torch.int64))
    allocator = TensorMemoryAllocator(torch.empty(4096, dtype=torch.uint8))
    memory_obj = allocator.allocate(metadata.get_shapes(4), metadata.get_dtypes())
    assert memory_obj is not None
    try:
        transfer = connector.from_gpu if direction == "store" else connector.to_gpu
        with pytest.raises(ValueError, match="block count and XPU device"):
            transfer(memory_obj, 0, 4, kvcaches=caches, slot_mapping=slots)
        native_call.assert_not_called()
    finally:
        allocator.free(memory_obj)
    assert allocator.memcheck()


@pytest.mark.parametrize("direction", ["store", "retrieve"])
@pytest.mark.parametrize(
    "invalid_input",
    [
        "chunk-shape",
        "chunk-dtype",
        "slot-dtype",
        "slot-rank",
        "slot-device",
        "slot-stride",
        "token-range",
        "missing-slots",
    ],
)
def test_v2_rejects_invalid_chunk_or_mapping_before_transfer(
    native_call: Mock,
    direction: str,
    invalid_input: str,
) -> None:
    """Public transfers reject incompatible buffers and slot mappings early."""
    metadata = _make_metadata()
    connector = VLLMPagedMemXPUConnectorV2.from_metadata(
        metadata, False, torch.device("xpu:0"), layout_hints={"kv_layout": "NHD"}
    )
    caches = [
        _as_mock_xpu(torch.zeros(4, 3, 4, 6, dtype=torch.float16)) for _ in range(2)
    ]
    slots = _as_mock_xpu(torch.arange(4, dtype=torch.int64))
    shape = metadata.get_shapes(4)[0]
    dtype = torch.float16
    start, end = 0, 4
    if invalid_input == "chunk-shape":
        shape = torch.Size((2, 2, 3, 12))
    elif invalid_input == "chunk-dtype":
        dtype = torch.float32
    elif invalid_input == "slot-dtype":
        slots = _as_mock_xpu(torch.arange(4, dtype=torch.int32))
    elif invalid_input == "slot-rank":
        slots = _as_mock_xpu(torch.arange(4, dtype=torch.int64).reshape(2, 2))
    elif invalid_input == "slot-device":
        slots = torch.arange(4, dtype=torch.int64)
    elif invalid_input == "slot-stride":
        slots = _as_mock_xpu(torch.arange(8, dtype=torch.int64)[::2])
    elif invalid_input == "token-range":
        start, end = -1, 3
    allocator = TensorMemoryAllocator(torch.empty(4096, dtype=torch.uint8))
    memory_obj = allocator.allocate(shape, dtype)
    assert memory_obj is not None
    kwargs: dict[str, object] = {"kvcaches": caches}
    if invalid_input != "missing-slots":
        kwargs["slot_mapping"] = slots
    transfer = connector.from_gpu if direction == "store" else connector.to_gpu
    try:
        with pytest.raises(ValueError, match="(?i)(chunk|slot)"):
            transfer(memory_obj, start, end, **kwargs)
        native_call.assert_not_called()
    finally:
        allocator.free(memory_obj)


def test_v2_direct_constructor_requires_uniform_component_width(
    native_call: Mock,
) -> None:
    """Equal flattened widths cannot hide differing per-layer component HS."""
    connector = VLLMPagedMemXPUConnectorV2(
        hidden_dim_size=12,
        num_layers=2,
        dtype=torch.float16,
        layout_hints={"kv_layout": "NHD"},
    )
    caches = [
        _as_mock_xpu(torch.zeros(4, 3, 4, 6, dtype=torch.float16)),
        _as_mock_xpu(torch.zeros(4, 3, 3, 8, dtype=torch.float16)),
    ]
    slots = _as_mock_xpu(torch.arange(4, dtype=torch.int64))
    allocator = TensorMemoryAllocator(torch.empty(4096, dtype=torch.uint8))
    memory_obj = allocator.allocate(connector.get_shape(4), torch.float16)
    assert memory_obj is not None
    try:
        with pytest.raises(ValueError, match="uniform geometry"):
            connector.from_gpu(memory_obj, 0, 4, kvcaches=caches, slot_mapping=slots)
        native_call.assert_not_called()
    finally:
        allocator.free(memory_obj)
