# SPDX-License-Identifier: Apache-2.0
"""Numerical V2 transfers against independent CPU indexing of engine storage."""

# Standard
from dataclasses import dataclass

# Third Party
import pytest
import torch

# First Party
from lmcache import device_ops, torch_dev, torch_device_type
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
from lmcache.v1.platform import resolve_device_ops
import lmcache.lmcache_native as lmcache_native

pytestmark = [pytest.mark.xpu, pytest.mark.no_shared_allocator]

_LAYERS = 2
_BLOCKS = 4
_BLOCK_SIZE = 3
_HEADS = 4
_CHUNK_SIZE = 6
_ENGINE_SENTINEL = -11
_MEMORY_SENTINEL = -19
_SLOTS = [0, 1, 10, -1, 4, 8, 2, 6, 3, 9]


@dataclass(frozen=True)
class _Scenario:
    storage: str
    head_size: int
    num_tokens: int
    cached_tokens: int
    factory: bool


_SCENARIOS = [
    pytest.param(_Scenario("tight", 3, 6, 0, True), id="tight-full-factory"),
    pytest.param(_Scenario("padded", 4, 6, 4, False), id="padded-offset-prefix"),
    pytest.param(_Scenario("blocks_first", 3, 4, 4, True), id="blocks-first-partial"),
    pytest.param(_Scenario("permuted", 3, 6, 4, False), id="normalized-full"),
    pytest.param(_Scenario("permuted", 4, 4, 99, True), id="normalized-fully-skipped"),
]


def _make_metadata(
    dtype: torch.dtype, head_size: int, use_mla: bool = False
) -> LMCacheMetadata:
    """Build ordinary model metadata without a layer-group manager."""
    return LMCacheMetadata(
        model_name="xpu-v2-layout-regression",
        world_size=1,
        local_world_size=1,
        worker_id=0,
        local_worker_id=0,
        kv_dtype=dtype,
        kv_shape=(
            _LAYERS,
            1 if use_mla else 2,
            _CHUNK_SIZE,
            1 if use_mla else _HEADS,
            head_size,
        ),
        use_mla=use_mla,
        chunk_size=_CHUNK_SIZE,
    )


def _make_packed_pool(
    dtype: torch.dtype,
    layout: KVLayoutName,
    scenario: _Scenario,
) -> tuple[torch.Tensor, list[torch.Tensor]]:
    """Allocate CPU raw views, including padding and nonzero storage offsets."""
    head_size = scenario.head_size
    shape = (
        (_BLOCKS, _HEADS, _BLOCK_SIZE, 2 * head_size)
        if layout == "HND"
        else (_BLOCKS, _BLOCK_SIZE, _HEADS, 2 * head_size)
    )
    tight = _BLOCK_SIZE * _HEADS * 2 * head_size
    if scenario.storage == "blocks_first":
        block_stride = _LAYERS * tight + 1
        layer_offsets = [1 + layer * tight for layer in range(_LAYERS)]
        pool_size = 1 + _BLOCKS * block_stride
    else:
        padded = scenario.storage == "padded"
        block_stride = tight + int(padded)
        layer_offsets = [1 + layer * _BLOCKS * block_stride for layer in range(_LAYERS)]
        pool_size = 1 + _LAYERS * _BLOCKS * block_stride
    pool = torch.full((pool_size,), _ENGINE_SENTINEL, dtype=dtype)
    inner_stride = (
        (_BLOCK_SIZE * 2 * head_size, 2 * head_size, 1)
        if layout == "HND"
        else (_HEADS * 2 * head_size, 2 * head_size, 1)
    )
    views = [
        pool.as_strided(shape, (block_stride, *inner_stride), offset)
        for offset in layer_offsets
    ]
    return pool, views


def _fill_packed_engine(
    views: list[torch.Tensor], layout: KVLayoutName, head_size: int
) -> None:
    """Populate each CPU K/V component once in the target dtype."""
    for layer, view in enumerate(views):
        for block in range(_BLOCKS):
            for slot in range(_BLOCK_SIZE):
                for head in range(_HEADS):
                    row = (
                        view[block, head, slot]
                        if layout == "HND"
                        else view[block, slot, head]
                    )
                    for component in range(2):
                        for element in range(head_size):
                            row[component * head_size + element] = (
                                layer * 53 + block * 17 + slot * 5 + head * 3 + element
                            ) % 113 + component * 128


def _device_views(
    cpu_pool: torch.Tensor,
    cpu_views: list[torch.Tensor],
    storage: str,
    device: torch.device,
) -> tuple[torch.Tensor, list[torch.Tensor]]:
    """Copy the CPU pool once and expose the corresponding engine views."""
    device_pool = cpu_pool.to(device)
    views = [
        device_pool.as_strided(view.shape, view.stride(), view.storage_offset())
        for view in cpu_views
    ]
    if storage == "permuted":
        views = [view.permute(0, 2, 1, 3) for view in views]
    return device_pool, views


def _expected_packed_store(
    cpu_views: list[torch.Tensor],
    layout: KVLayoutName,
    head_size: int,
    slots: list[int],
    dtype: torch.dtype,
) -> torch.Tensor:
    """Gather raw CPU rows into the public V2 two-plane representation."""
    expected = torch.full(
        (2, _LAYERS, len(slots), _HEADS * head_size),
        _MEMORY_SENTINEL,
        dtype=dtype,
    )
    for token, slot in enumerate(slots):
        if slot < 0:
            continue
        block, block_token = divmod(slot, _BLOCK_SIZE)
        for layer, view in enumerate(cpu_views):
            for head in range(_HEADS):
                row = (
                    view[block, head, block_token]
                    if layout == "HND"
                    else view[block, block_token, head]
                )
                for component in range(2):
                    for element in range(head_size):
                        expected[
                            component, layer, token, head * head_size + element
                        ] = row[component * head_size + element]
    return expected


def _scatter_packed_expected(
    cpu_views: list[torch.Tensor],
    layout: KVLayoutName,
    head_size: int,
    source: torch.Tensor,
    slots: list[int],
    skip: int,
) -> None:
    """Scatter independently populated V2 values by explicit raw indexing."""
    for token, slot in enumerate(slots):
        if token < skip or slot < 0:
            continue
        block, block_token = divmod(slot, _BLOCK_SIZE)
        for layer, view in enumerate(cpu_views):
            for head in range(_HEADS):
                row = (
                    view[block, head, block_token]
                    if layout == "HND"
                    else view[block, block_token, head]
                )
                for component in range(2):
                    for element in range(head_size):
                        row[component * head_size + element] = source[
                            component, layer, token, head * head_size + element
                        ]


@pytest.fixture
def xpu_device() -> torch.device:
    """Require hardware only when a hardware test actually executes."""
    if torch_device_type != "xpu" or not torch_dev.is_available():
        pytest.skip("Requires an available Intel XPU runtime")
    resolved = resolve_device_ops("xpu")
    assert device_ops.multi_layer_kv_transfer is resolved.multi_layer_kv_transfer
    assert resolved.multi_layer_kv_transfer.__module__ == "lmcache.xpu_ops"
    return torch.device("xpu:0")


@pytest.mark.parametrize("layout", ["NHD", "HND"])
@pytest.mark.parametrize("dtype", [torch.float16, torch.bfloat16, torch.float32])
@pytest.mark.parametrize("use_gpu", [False, True], ids=["direct-host", "staging"])
@pytest.mark.parametrize("scenario", _SCENARIOS)
def test_v2_packed_store_and_independent_retrieval(
    xpu_device: torch.device,
    layout: KVLayoutName,
    dtype: torch.dtype,
    use_gpu: bool,
    scenario: _Scenario,
) -> None:
    """Store and retrieve match separate oracles without changing metadata."""
    metadata = _make_metadata(dtype, scenario.head_size)
    assert metadata.kv_layer_groups_manager is None
    config = LMCacheEngineConfig.from_defaults(
        chunk_size=_CHUNK_SIZE,
        use_layerwise=False,
        use_gpu_connector_v3=False,
        enable_pd=not use_gpu,
    )
    hints: LayoutHints = {"kv_layout": layout}
    if scenario.storage == "blocks_first":
        hints = {"kv_layout": "BLHNC" if layout == "HND" else "BLNHC"}
    if scenario.factory:
        connector = CreateGPUConnector(config, metadata, EngineType.VLLM, hints)
    else:
        connector = VLLMPagedMemXPUConnectorV2.from_metadata(
            metadata, use_gpu, xpu_device, layout_hints=hints
        )
    assert isinstance(connector, VLLMPagedMemXPUConnectorV2)
    assert (connector.gpu_buffer is not None) == use_gpu

    start = 2
    end = start + scenario.num_tokens
    cpu_slots = _SLOTS[start:end]
    slot_mapping = torch.tensor(_SLOTS, dtype=torch.int64, device=xpu_device)
    cpu_pool, cpu_views = _make_packed_pool(dtype, layout, scenario)
    _fill_packed_engine(cpu_views, layout, scenario.head_size)
    expected_store = _expected_packed_store(
        cpu_views, layout, scenario.head_size, cpu_slots, dtype
    )
    device_pool, engine_views = _device_views(
        cpu_pool, cpu_views, scenario.storage, xpu_device
    )
    allocator = TensorMemoryAllocator(
        torch.empty(32 * 1024, dtype=torch.uint8, pin_memory=True)
    )
    shape = metadata.get_shapes(scenario.num_tokens)[0]
    memory_obj = allocator.allocate(shape, dtype, fmt=MemoryFormat.KV_2LTD)
    assert memory_obj is not None
    assert memory_obj.tensor is not None
    try:
        memory_obj.tensor.fill_(_MEMORY_SENTINEL)
        connector.from_gpu(
            memory_obj, start, end, kvcaches=engine_views, slot_mapping=slot_mapping
        )
        torch_dev.synchronize()
        assert memory_obj.tensor.shape == (
            2,
            _LAYERS,
            scenario.num_tokens,
            _HEADS * scenario.head_size,
        )
        assert memory_obj.metadata.fmt == MemoryFormat.KV_2LTD
        torch.testing.assert_close(memory_obj.tensor, expected_store, rtol=0, atol=0)
        torch.testing.assert_close(device_pool.cpu(), cpu_pool, rtol=0, atol=0)
        assert metadata.kv_layer_groups_manager is None

        retrieval = (
            (torch.arange(shape.numel(), dtype=torch.int64).remainder(211) - 103)
            .to(dtype)
            .reshape(shape)
        )
        memory_obj.tensor.copy_(retrieval)
        expected_pool, expected_views = _make_packed_pool(dtype, layout, scenario)
        target_pool, target_views = _device_views(
            expected_pool, expected_views, scenario.storage, xpu_device
        )
        skip = min(scenario.num_tokens, max(0, scenario.cached_tokens - start))
        _scatter_packed_expected(
            expected_views,
            layout,
            scenario.head_size,
            retrieval,
            cpu_slots,
            skip,
        )
        target_connector = VLLMPagedMemXPUConnectorV2.from_metadata(
            metadata, use_gpu, xpu_device, layout_hints=hints
        )
        target_connector.to_gpu(
            memory_obj,
            start,
            end,
            kvcaches=target_views,
            slot_mapping=slot_mapping,
            vllm_cached_tokens=scenario.cached_tokens,
        )
        torch_dev.synchronize()
        torch.testing.assert_close(target_pool.cpu(), expected_pool, rtol=0, atol=0)
        assert metadata.kv_layer_groups_manager is None
    finally:
        torch_dev.synchronize()
        allocator.free(memory_obj)
    assert allocator.memcheck()


@pytest.mark.parametrize("use_gpu", [False, True])
@pytest.mark.parametrize(
    "engine_format",
    [
        lmcache_native.EngineKVFormat.NL_X_TWO_NB_BS_NH_HS,
        lmcache_native.EngineKVFormat.NL_X_NB_TWO_BS_NH_HS,
        lmcache_native.EngineKVFormat.NL_X_NB_BS_HS,
    ],
)
def test_v2_legacy_and_mla_independent_directions(
    xpu_device: torch.device,
    use_gpu: bool,
    engine_format: lmcache_native.EngineKVFormat,
) -> None:
    """Existing separate K/V and MLA formats retain their numerical contract."""
    use_mla = engine_format == lmcache_native.EngineKVFormat.NL_X_NB_BS_HS
    components = 1 if use_mla else 2
    heads = 1 if use_mla else _HEADS
    head_size = 4
    dtype = torch.float16
    metadata = _make_metadata(dtype, head_size, use_mla)
    shape = metadata.get_shapes(_CHUNK_SIZE)[0]
    engine_shape: tuple[int, ...]
    if use_mla:
        engine_shape = (_BLOCKS, _BLOCK_SIZE, head_size)
    elif engine_format == lmcache_native.EngineKVFormat.NL_X_TWO_NB_BS_NH_HS:
        engine_shape = (2, _BLOCKS, _BLOCK_SIZE, heads, head_size)
    else:
        engine_shape = (_BLOCKS, 2, _BLOCK_SIZE, heads, head_size)
    cpu_views = [
        (
            torch.arange(torch.Size(engine_shape).numel(), dtype=torch.int64)
            + layer * 200
        )
        .to(dtype)
        .reshape(engine_shape)
        for layer in range(_LAYERS)
    ]
    slots = _SLOTS[2:8]
    slot_mapping = torch.tensor(_SLOTS, dtype=torch.int64, device=xpu_device)
    expected = torch.full(shape, _MEMORY_SENTINEL, dtype=dtype)
    for token, slot in enumerate(slots):
        if slot < 0:
            continue
        block, block_token = divmod(slot, _BLOCK_SIZE)
        for layer, view in enumerate(cpu_views):
            for component in range(components):
                if use_mla:
                    row = view[block, block_token]
                elif (
                    engine_format == lmcache_native.EngineKVFormat.NL_X_TWO_NB_BS_NH_HS
                ):
                    row = view[component, block, block_token].flatten()
                else:
                    row = view[block, component, block_token].flatten()
                expected[component, layer, token] = row
    allocator = TensorMemoryAllocator(
        torch.empty(16 * 1024, dtype=torch.uint8, pin_memory=True)
    )
    fmt = MemoryFormat.KV_MLA_FMT if use_mla else MemoryFormat.KV_2LTD
    memory_obj = allocator.allocate(shape, dtype, fmt=fmt)
    assert memory_obj is not None
    assert memory_obj.tensor is not None
    connector = VLLMPagedMemXPUConnectorV2.from_metadata(
        metadata, use_gpu, xpu_device, layout_hints={"kv_layout": "NHD"}
    )
    try:
        memory_obj.tensor.fill_(_MEMORY_SENTINEL)
        connector.from_gpu(
            memory_obj,
            2,
            8,
            kvcaches=[view.to(xpu_device) for view in cpu_views],
            slot_mapping=slot_mapping,
        )
        torch_dev.synchronize()
        torch.testing.assert_close(memory_obj.tensor, expected, rtol=0, atol=0)
        assert memory_obj.metadata.fmt == fmt

        retrieval = (
            (torch.arange(shape.numel(), dtype=torch.int64).remainder(127) + 17)
            .to(dtype)
            .reshape(shape)
        )
        memory_obj.tensor.copy_(retrieval)
        expected_targets = [
            torch.full(engine_shape, _ENGINE_SENTINEL, dtype=dtype)
            for _ in range(_LAYERS)
        ]
        device_targets = [view.to(xpu_device) for view in expected_targets]
        for token, slot in enumerate(slots):
            if token < 2 or slot < 0:
                continue
            block, block_token = divmod(slot, _BLOCK_SIZE)
            for layer, view in enumerate(expected_targets):
                for component in range(components):
                    row = retrieval[component, layer, token]
                    if use_mla:
                        view[block, block_token] = row
                    elif (
                        engine_format
                        == lmcache_native.EngineKVFormat.NL_X_TWO_NB_BS_NH_HS
                    ):
                        view[component, block, block_token] = row.reshape(
                            heads, head_size
                        )
                    else:
                        view[block, component, block_token] = row.reshape(
                            heads, head_size
                        )
        target_connector = VLLMPagedMemXPUConnectorV2.from_metadata(
            metadata, use_gpu, xpu_device, layout_hints={"kv_layout": "NHD"}
        )
        target_connector.to_gpu(
            memory_obj,
            2,
            8,
            kvcaches=device_targets,
            slot_mapping=slot_mapping,
            vllm_cached_tokens=4,
        )
        torch_dev.synchronize()
        for actual, wanted in zip(device_targets, expected_targets, strict=True):
            torch.testing.assert_close(actual.cpu(), wanted, rtol=0, atol=0)
        assert metadata.kv_layer_groups_manager is None
    finally:
        torch_dev.synchronize()
        allocator.free(memory_obj)
    assert allocator.memcheck()
