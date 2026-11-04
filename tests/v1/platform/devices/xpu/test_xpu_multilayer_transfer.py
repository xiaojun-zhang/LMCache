# SPDX-License-Identifier: Apache-2.0
"""Native XPU transfers against independently indexed engine tensors.

Packed tests use V2's two-plane LMCache representation, rather than the
packed-row representation used by block transfers. Hardware is checked only
when a test executes, so selecting CPU tests does not probe XPU availability.
"""

# Standard
from dataclasses import dataclass
import math

# Third Party
import pytest
import torch

# First Party
from lmcache import torch_dev, torch_device_type
from lmcache.lmcache_native import EngineKVFormat, TransferDirection
from lmcache.v1.platform import resolve_device_ops
from lmcache.v1.platform.base.device_ops import DeviceOps

pytestmark = [pytest.mark.xpu, pytest.mark.no_shared_allocator]

_LAYERS = 2
_BLOCKS = 4
_BLOCK_SIZE = 3
_HEADS = 4
_SENTINEL = -123.0


@dataclass
class _PackedCache:
    pools: list[torch.Tensor]
    layers: list[torch.Tensor]
    pool_indices: list[int]
    block_stride: int


def _make_packed_cache(
    layout: str, dtype: torch.dtype, head_size: int, storage: str
) -> _PackedCache:
    """Build CPU engine inputs while retaining padding and view offsets.

    Args:
        layout: ``NHD`` or ``HND`` physical order.
        dtype: Target dtype, also used by the independent expectations.
        head_size: Width of one K or V component.
        storage: Tight, padded, offset, or blocks-first backing allocation.

    Returns:
        CPU backing pools and raw per-layer views into those pools.
    """
    content_size = 2 * head_size
    if layout == "NHD":
        shape = (_BLOCKS, _BLOCK_SIZE, _HEADS, content_size)
        inner_stride = (_HEADS * content_size, content_size, 1)
    else:
        shape = (_BLOCKS, _HEADS, _BLOCK_SIZE, content_size)
        inner_stride = (_BLOCK_SIZE * content_size, content_size, 1)
    tight_stride = _BLOCK_SIZE * _HEADS * content_size
    block_stride = (
        _LAYERS * tight_stride
        if storage == "blocks-first"
        else tight_stride + (1 if storage == "padded" else 0)
    )
    offset = 1 if storage == "offset" else 0
    pool_count = 1 if storage == "blocks-first" else _LAYERS
    pools = [
        torch.full((_BLOCKS * block_stride + offset + 1,), _SENTINEL, dtype=dtype)
        for _ in range(pool_count)
    ]
    pool_indices = [0] * _LAYERS if pool_count == 1 else list(range(_LAYERS))
    layers = [
        pools[pool_idx].as_strided(
            shape,
            (block_stride, *inner_stride),
            offset + (layer_idx * tight_stride if pool_count == 1 else 0),
        )
        for layer_idx, pool_idx in enumerate(pool_indices)
    ]
    layer_numel = math.prod(shape)
    for layer_idx, layer in enumerate(layers):
        values = (
            torch.arange(
                layer_idx * layer_numel,
                (layer_idx + 1) * layer_numel,
                dtype=torch.float32,
            )
            .div_(16)
            .to(dtype)
        )
        layer.copy_(values.reshape(shape))
    return _PackedCache(pools, layers, pool_indices, block_stride)


def _copy_packed_cache(
    cache: _PackedCache, device: torch.device, *, fill: float | None = None
) -> _PackedCache:
    """Copy entire pools and rebuild views without dropping physical strides.

    Args:
        cache: CPU engine inputs and their views.
        device: Destination device.
        fill: If provided, initialize every destination byte's element with
            this value instead of copying engine contents.

    Returns:
        Pools and views with the original storage offsets and strides.
    """
    pools = [
        (pool if fill is None else torch.full_like(pool, fill)).to(device)
        for pool in cache.pools
    ]
    layers = [
        pools[pool_idx].as_strided(layer.shape, layer.stride(), layer.storage_offset())
        for pool_idx, layer in zip(cache.pool_indices, cache.layers, strict=True)
    ]
    return _PackedCache(pools, layers, cache.pool_indices, cache.block_stride)


def _packed_d2h_expected(
    cache: _PackedCache,
    layout: str,
    head_size: int,
    slots: list[int],
    skip: int,
) -> torch.Tensor:
    """Gather expected two-plane data by raw CPU engine indexing.

    Args:
        cache: CPU engine inputs.
        layout: Physical ``NHD`` or ``HND`` order.
        head_size: One component's head width.
        slots: Chunk-local token-to-engine slot mapping.
        skip: Prefix length whose LMCache destination must stay untouched.

    Returns:
        Expected V2 buffer, with sentinels at negative and skipped slots.
    """
    expected = torch.full(
        (2, _LAYERS, len(slots), _HEADS * head_size),
        _SENTINEL,
        dtype=cache.layers[0].dtype,
    )
    for token, slot in enumerate(slots):
        if token < skip or slot < 0:
            continue
        block, position = divmod(slot, _BLOCK_SIZE)
        for layer_idx, layer in enumerate(cache.layers):
            for component in range(2):
                for head in range(_HEADS):
                    for element in range(head_size):
                        content = component * head_size + element
                        value = (
                            layer[block, position, head, content]
                            if layout == "NHD"
                            else layer[block, head, position, content]
                        )
                        expected[
                            component,
                            layer_idx,
                            token,
                            head * head_size + element,
                        ] = value
    return expected


def _packed_h2d_expected(
    cache: _PackedCache,
    source: torch.Tensor,
    layout: str,
    head_size: int,
    slots: list[int],
    skip: int,
) -> _PackedCache:
    """Scatter an independent CPU two-plane buffer into sentinel engine pools.

    Args:
        cache: Template defining engine view strides and storage offsets.
        source: Independently populated ``[2,L,T,NH*HS]`` buffer.
        layout: Physical ``NHD`` or ``HND`` order.
        head_size: One component's head width.
        slots: Chunk-local token-to-engine slot mapping.
        skip: Prefix length whose engine destinations must stay untouched.

    Returns:
        Expected whole pools, including unrelated slots, offsets and padding.
    """
    expected = _copy_packed_cache(cache, torch.device("cpu"), fill=_SENTINEL)
    for token, slot in enumerate(slots):
        if token < skip or slot < 0:
            continue
        block, position = divmod(slot, _BLOCK_SIZE)
        for layer_idx, layer in enumerate(expected.layers):
            for component in range(2):
                for head in range(_HEADS):
                    for element in range(head_size):
                        content = component * head_size + element
                        value = source[
                            component,
                            layer_idx,
                            token,
                            head * head_size + element,
                        ]
                        if layout == "NHD":
                            layer[block, position, head, content] = value
                        else:
                            layer[block, head, position, content] = value
    return expected


@pytest.fixture(scope="module")
def native_xpu_ops() -> DeviceOps:
    """Require hardware and the resolved native multilayer operation."""
    if torch_device_type != "xpu" or not torch_dev.is_available():
        pytest.skip("Requires an available XPU runtime")
    ops = resolve_device_ops("xpu")
    assert ops.multi_layer_kv_transfer.__module__ == "lmcache.xpu_ops", (
        "This numerical regression must execute the compiled XPU operation"
    )
    return ops


@pytest.mark.parametrize("layout", ["NHD", "HND"])
@pytest.mark.parametrize("dtype", [torch.float16, torch.bfloat16, torch.float32])
@pytest.mark.parametrize("buffer_location", ["host", "xpu"])
@pytest.mark.parametrize(
    ("storage", "head_size", "slots", "skip"),
    [
        pytest.param("tight", 4, [10, 1, 6, 3, 8, 0, 11, 4], 0, id="full-tight"),
        pytest.param("tight", 3, [10, 1, 6, 3, 8, 0, 11, 4], 0, id="odd-component"),
        pytest.param("padded", 4, [10, 1, 6, 3, 8], 0, id="partial-padded"),
        pytest.param("offset", 4, [10, 1, 6, 3, 8, 0, 11, 4], 0, id="unaligned-offset"),
        pytest.param(
            "buffer-offset",
            4,
            [10, 1, 6, 3, 8, 0, 11, 4],
            0,
            id="unaligned-buffer",
        ),
        pytest.param("blocks-first", 8, [10, 1, 6, 3, 8, 0, 11, 4], 0, id="pool-views"),
        pytest.param("padded", 3, [10, 1, 6, -1, 8, 0, 11, -1], 2, id="skip-negative"),
        pytest.param("tight", 3, [10, -1, 6, 3, 8], 5, id="fully-skipped"),
    ],
)
def test_native_packed_multilayer_transfer(
    native_xpu_ops: DeviceOps,
    layout: str,
    dtype: torch.dtype,
    buffer_location: str,
    storage: str,
    head_size: int,
    slots: list[int],
    skip: int,
) -> None:
    """Check D2H and H2D independently, including untouched backing storage."""
    device = torch.device(torch_device_type, 0)
    transfer_device = torch.device(torch_device_type) if storage == "tight" else device
    cache_cpu = _make_packed_cache(layout, dtype, head_size, storage)
    expected_d2h = _packed_d2h_expected(cache_cpu, layout, head_size, slots, skip)
    cache_xpu = _copy_packed_cache(cache_cpu, device)
    pointer_tensor = torch.tensor(
        [layer.data_ptr() for layer in cache_xpu.layers],
        dtype=torch.uint64,
        device=device,
    )
    slot_tensor = torch.tensor(slots, dtype=torch.int64, device=device)
    engine_format = (
        EngineKVFormat.NL_X_NB_BS_NH_CS
        if layout == "NHD"
        else EngineKVFormat.NL_X_NB_NH_BS_CS
    )
    buffer_device = torch.device("cpu") if buffer_location == "host" else device
    buffer_offset = 1 if storage == "buffer-offset" else 0
    output_pool = torch.full(
        (expected_d2h.numel() + buffer_offset + 1,),
        _SENTINEL,
        dtype=dtype,
        device=buffer_device,
        pin_memory=buffer_location == "host",
    )
    output = output_pool[buffer_offset : buffer_offset + expected_d2h.numel()].view(
        expected_d2h.shape
    )
    native_xpu_ops.multi_layer_kv_transfer(
        output,
        pointer_tensor,
        slot_tensor,
        transfer_device,
        _BLOCKS * _BLOCK_SIZE,
        TransferDirection.D2H,
        engine_format,
        block_size=_BLOCK_SIZE,
        head_size=head_size,
        skip_prefix_n_tokens=skip,
        block_stride_elems=0 if storage == "tight" else cache_cpu.block_stride,
    )
    torch_dev.synchronize()
    expected_output_pool = torch.full_like(output_pool, _SENTINEL, device="cpu")
    expected_output_pool[buffer_offset : buffer_offset + expected_d2h.numel()].copy_(
        expected_d2h.flatten()
    )
    assert torch.equal(output_pool.cpu(), expected_output_pool)
    for actual, original in zip(cache_xpu.pools, cache_cpu.pools, strict=True):
        assert torch.equal(actual.cpu(), original)

    source_cpu = (
        torch.arange(expected_d2h.numel(), dtype=torch.float32)
        .div_(8)
        .to(dtype)
        .reshape(expected_d2h.shape)
    )
    expected_h2d = _packed_h2d_expected(
        cache_cpu, source_cpu, layout, head_size, slots, skip
    )
    destination = _copy_packed_cache(cache_cpu, device, fill=_SENTINEL)
    destination_pointers = torch.tensor(
        [layer.data_ptr() for layer in destination.layers],
        dtype=torch.uint64,
        device=device,
    )
    source_pool = torch.full(
        output_pool.shape,
        _SENTINEL,
        dtype=dtype,
        device=buffer_device,
        pin_memory=buffer_location == "host",
    )
    source = source_pool[buffer_offset : buffer_offset + source_cpu.numel()].view(
        source_cpu.shape
    )
    source.copy_(source_cpu)
    native_xpu_ops.multi_layer_kv_transfer(
        source,
        destination_pointers,
        slot_tensor,
        transfer_device,
        _BLOCKS * _BLOCK_SIZE,
        TransferDirection.H2D,
        engine_format,
        block_size=_BLOCK_SIZE,
        head_size=head_size,
        skip_prefix_n_tokens=skip,
        block_stride_elems=0 if storage == "tight" else cache_cpu.block_stride,
    )
    torch_dev.synchronize()
    for actual, expected in zip(destination.pools, expected_h2d.pools, strict=True):
        assert torch.equal(actual.cpu(), expected)
    expected_source_pool = torch.full_like(source_pool, _SENTINEL, device="cpu")
    expected_source_pool[buffer_offset : buffer_offset + source_cpu.numel()].copy_(
        source_cpu.flatten()
    )
    assert torch.equal(source_pool.cpu(), expected_source_pool)


@pytest.mark.parametrize("dtype", [torch.float16, torch.bfloat16, torch.float32])
@pytest.mark.parametrize(
    "engine_format",
    [
        EngineKVFormat.NL_X_TWO_NB_BS_NH_HS,
        EngineKVFormat.NL_X_NB_TWO_BS_NH_HS,
        EngineKVFormat.NL_X_NB_BS_HS,
        EngineKVFormat.NL_X_NBBS_ONE_HS,
    ],
)
def test_native_legacy_multilayer_transfer(
    native_xpu_ops: DeviceOps,
    dtype: torch.dtype,
    engine_format: EngineKVFormat,
) -> None:
    """Preserve separate K/V NHD and MLA in each direction without roundtrips."""
    device = torch.device(torch_device_type, 0)
    is_mla = engine_format in (
        EngineKVFormat.NL_X_NB_BS_HS,
        EngineKVFormat.NL_X_NBBS_ONE_HS,
    )
    head_size = 3
    hidden_size = head_size if is_mla else _HEADS * head_size
    components = 1 if is_mla else 2
    shape: tuple[int, ...]
    if engine_format == EngineKVFormat.NL_X_TWO_NB_BS_NH_HS:
        shape = (2, _BLOCKS, _BLOCK_SIZE, _HEADS, head_size)
    elif engine_format == EngineKVFormat.NL_X_NB_TWO_BS_NH_HS:
        shape = (_BLOCKS, 2, _BLOCK_SIZE, _HEADS, head_size)
    elif engine_format == EngineKVFormat.NL_X_NB_BS_HS:
        shape = (_BLOCKS, _BLOCK_SIZE, head_size)
    else:
        shape = (_BLOCKS * _BLOCK_SIZE, 1, head_size)
    engine_cpu = [
        torch.arange(
            layer * math.prod(shape),
            (layer + 1) * math.prod(shape),
            dtype=torch.float32,
        )
        .div_(16)
        .to(dtype)
        .reshape(shape)
        for layer in range(_LAYERS)
    ]
    slots = [10, 1, 6, -1, 8, 0, 11]
    skip = 2
    buffer_shape = (components, _LAYERS, len(slots), hidden_size)
    expected_d2h = torch.full(buffer_shape, _SENTINEL, dtype=dtype)
    source_cpu = (
        torch.arange(math.prod(buffer_shape), dtype=torch.float32)
        .div_(8)
        .to(dtype)
        .reshape(buffer_shape)
    )
    expected_h2d = [torch.full_like(layer, _SENTINEL) for layer in engine_cpu]
    for token, slot in enumerate(slots):
        if token < skip or slot < 0:
            continue
        block, position = divmod(slot, _BLOCK_SIZE)
        for layer in range(_LAYERS):
            for component in range(components):
                for element in range(hidden_size):
                    head, within_head = divmod(element, head_size)
                    index: tuple[int, ...]
                    if engine_format == EngineKVFormat.NL_X_TWO_NB_BS_NH_HS:
                        index = (component, block, position, head, within_head)
                    elif engine_format == EngineKVFormat.NL_X_NB_TWO_BS_NH_HS:
                        index = (block, component, position, head, within_head)
                    elif engine_format == EngineKVFormat.NL_X_NB_BS_HS:
                        index = (block, position, within_head)
                    else:
                        index = (slot, 0, within_head)
                    expected_d2h[component, layer, token, element] = engine_cpu[layer][
                        index
                    ]
                    expected_h2d[layer][index] = source_cpu[
                        component, layer, token, element
                    ]
    engine_xpu = [layer.to(device) for layer in engine_cpu]
    pointers = torch.tensor(
        [layer.data_ptr() for layer in engine_xpu],
        dtype=torch.uint64,
        device=device,
    )
    slot_tensor = torch.tensor(slots, dtype=torch.int64, device=device)
    output = torch.full(buffer_shape, _SENTINEL, dtype=dtype, device=device)
    native_xpu_ops.multi_layer_kv_transfer(
        output,
        pointers,
        slot_tensor,
        device,
        _BLOCKS * _BLOCK_SIZE,
        TransferDirection.D2H,
        engine_format,
        block_size=_BLOCK_SIZE,
        skip_prefix_n_tokens=skip,
    )
    torch_dev.synchronize()
    assert torch.equal(output.cpu(), expected_d2h)
    for actual, original in zip(engine_xpu, engine_cpu, strict=True):
        assert torch.equal(actual.cpu(), original)

    destination = [
        torch.full_like(layer, _SENTINEL, device=device) for layer in engine_cpu
    ]
    destination_pointers = torch.tensor(
        [layer.data_ptr() for layer in destination],
        dtype=torch.uint64,
        device=device,
    )
    source = source_cpu.to(device)
    native_xpu_ops.multi_layer_kv_transfer(
        source,
        destination_pointers,
        slot_tensor,
        device,
        _BLOCKS * _BLOCK_SIZE,
        TransferDirection.H2D,
        engine_format,
        block_size=_BLOCK_SIZE,
        skip_prefix_n_tokens=skip,
    )
    torch_dev.synchronize()
    for actual, expected in zip(destination, expected_h2d, strict=True):
        assert torch.equal(actual.cpu(), expected)
    assert torch.equal(source.cpu(), source_cpu)


@pytest.mark.parametrize(
    "engine_format",
    [
        EngineKVFormat.NL_X_NB_BS_NH_CS,
        EngineKVFormat.NL_X_NB_NH_BS_CS,
    ],
)
@pytest.mark.parametrize(
    ("components", "head_size", "stride", "skip", "error"),
    [
        pytest.param(1, 4, 0, 0, "contiguous V2", id="packed-row-buffer-rejected"),
        pytest.param(2, 0, 0, 0, "component HS", id="missing-head-size"),
        pytest.param(2, 5, 0, 0, "component HS", id="indivisible-head-size"),
        pytest.param(2, 4, 95, 0, "at least a packed block", id="short-stride"),
        pytest.param(2, 4, -1, 0, "at least a packed block", id="negative-stride"),
        pytest.param(2, 4, 0, -1, "non-negative", id="negative-prefix"),
    ],
)
def test_native_packed_geometry_rejected_before_transfer(
    native_xpu_ops: DeviceOps,
    engine_format: EngineKVFormat,
    components: int,
    head_size: int,
    stride: int,
    skip: int,
    error: str,
) -> None:
    """Reject incompatible V2 operands before modifying either destination."""
    device = torch.device(torch_device_type, 0)
    layout = "NHD" if engine_format == EngineKVFormat.NL_X_NB_BS_NH_CS else "HND"
    cache_cpu = _make_packed_cache(layout, torch.float16, 4, "tight")
    cache_xpu = _copy_packed_cache(cache_cpu, device)
    pointers = torch.tensor(
        [layer.data_ptr() for layer in cache_xpu.layers],
        dtype=torch.uint64,
        device=device,
    )
    slots = torch.tensor([10, 1, 6, 3, 8], dtype=torch.int64, device=device)
    buffer = torch.full(
        (components, _LAYERS, slots.numel(), _HEADS * 4),
        _SENTINEL,
        dtype=torch.float16,
        device=device,
    )
    with pytest.raises(RuntimeError, match=error):
        native_xpu_ops.multi_layer_kv_transfer(
            buffer,
            pointers,
            slots,
            device,
            _BLOCKS * _BLOCK_SIZE,
            TransferDirection.D2H,
            engine_format,
            block_size=_BLOCK_SIZE,
            head_size=head_size,
            skip_prefix_n_tokens=skip,
            block_stride_elems=stride,
        )
    torch_dev.synchronize()
    assert torch.equal(buffer.cpu(), torch.full_like(buffer, _SENTINEL, device="cpu"))
    for actual, original in zip(cache_xpu.pools, cache_cpu.pools, strict=True):
        assert torch.equal(actual.cpu(), original)
