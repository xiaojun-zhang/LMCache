# XPU V2 packed KV transfer

The in-process XPU V2 connector stores non-MLA chunks as
`[2, L, T, NH * HS]` with `MemoryFormat.KV_2LTD`. Engine packed rows contain
K followed by V **within each head**, so copying a whole engine row into a
two-plane chunk would reorder its contents. SYCL unpacks on store and packs on
retrieval; no CPU layout conversion is needed.

## Geometry and addressing

Supported packed formats are `NL_X_NB_BS_NH_CS` (NHD) and
`NL_X_NB_NH_BS_CS` (HND), with `CS = 2 * HS`. The shared format accessors
continue to describe engine geometry: `get_head_size()` returns CS and
`get_hidden_dim_size()` returns `NH * CS`. V2 divides those values by two for
its component geometry using the shared `is_fused_packed` format fact.

For component `q` (K=0, V=1), layer `l`, chunk token `t`, head `h`, and element
`r`, let `D = NH * HS`, `p = slot_mapping[start + t]`, `b = p // BS`, and
`u = p % BS`. Offsets in elements are:

```text
LMCache: ((q * L + l) * T + t) * D + h * HS + r
NHD:     b * block_stride_elems + (u * NH + h) * CS + q * HS + r
HND:     b * block_stride_elems + (h * BS + u) * CS + q * HS + r
```

The factory forwards layout hints to V2 discovery. Discovery precedes pointer
construction; `get_group_data_ptrs()` receives the normalized views, whose
`data_ptr()` includes each storage offset. NHD/HND is selected by the shared
detector, never guessed from middle-dimension sizes.

V2 discovers layout without creating a `KVLayerGroupsManager`. Ordinary
metadata therefore keeps allocating its two-plane fallback before store.
All layers must match the model component width, flattened width and dtype,
and share block count, block size, physical packed block stride and XPU
device. V2 checks every packed layer before publishing pointers so a smaller
layer pool or a pointer from another device cannot enter the transfer kernel.

## Native boundary and copy units

The public SYCL `multi_layer_kv_transfer` appends `block_stride_elems=0`,
matching CUDA's optional argument name and position. Zero means a tight
packed block; positive values must be at least `BS * NH * CS`. Packed views
support padding between blocks, including blocks-first pools and nonzero
storage offsets. Interior padding is rejected by shared normalization.
Non-packed transfers require the new argument to remain zero; V2 rejects
non-tight MLA views before transfer.

For packed transfers, `head_size` means **component HS**, not CS.
The native boundary requires a contiguous `[2,L,T,NH*HS]` buffer, one XPU
pointer per layer, and a contiguous XPU int64 slot mapping matching T.
Legacy separate-K/V and MLA paths retain their former addressing and their
no-op `head_size` semantics. The V3 retrieval call uses named
`head_size=0` and `skip_prefix_n_tokens` for shared-call compatibility; this
does not change V3's initialization or representation.

Copy units are 8, 4, 2 or 1 bytes. The chosen unit divides component head
bytes and physical block-stride bytes and respects the LMCache base address.
Each device-side engine address is checked for unit alignment; misaligned
view pointers use byte copies for that unit. Thus fp16 `NH=4, HS=3` uses
two-byte units even though its 24-byte plane row is divisible by eight.

## Prefixes, staging and ordering

Retrieval skips `min(T, max(0, vllm_cached_tokens - start))` tokens through the
named native skip argument. Negative slots leave the destination unchanged
in both directions; a fully skipped transfer launches no kernel.

Complete stores may use the existing XPU staging buffer; partial chunks
transfer directly. Before a staged store, a nonblocking copy seeds staging
from the destination so skipped rows retain their contents when the complete
staging buffer is copied back. This adds a DMA transfer on staged stores but
no CPU packing or synchronization. Existing store-stream ordering and the
CPU-destination synchronization remain in place.

## Regression and scope

Native tests under `tests/v1/platform/devices/xpu/` and V2 tests under
`tests/inprocess/v1/` compare each direction against independently indexed
CPU engine tensors. They cover packed order, dtypes, odd widths, padding,
storage offsets, prefixes, negative slots, direct/staged stores and legacy
NHD/MLA controls. Hardware fixtures assert that the resolved operation comes
from `lmcache.xpu_ops`; the packed CPU fallback is not a numerical oracle.
Both locations are discovered by the existing Buildkite XPU unit runner.

V3 layout lifecycle, layerwise/blending, MP transfer and end-to-end serving
are separate contracts and are outside this repair.
