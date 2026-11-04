# SPDX-License-Identifier: Apache-2.0
# Standard
from typing import TYPE_CHECKING, List, Optional, Tuple, Union, cast

# Third Party
import torch

# First Party
from lmcache import device_ops
from lmcache.inprocess.v1.compute.blend.utils import LMCBlenderBuilder
from lmcache.inprocess.v1.gpu_connector.gpu_connectors import GPUConnectorInterface
from lmcache.integration.vllm.utils import ENGINE_NAME
from lmcache.logging import init_logger
from lmcache.utils import EngineType, _lmcache_nvtx_annotate
from lmcache.v1.gpu_connector.kv_format.specs.registry import get_spec_class
from lmcache.v1.gpu_connector.utils import (
    DiscoverableKVCache,
    LayoutHints,
    assert_is_vllm_flash_attn_or_flash_infer,
    get_block_size,
    get_device,
    get_dtype,
    get_elements_per_layer,
    get_group_data_ptrs,
    get_head_size,
    get_hidden_dim_size,
    get_num_blocks,
    get_num_layers,
    get_tokens_per_layer,
    normalize_kv_and_discover_format,
    resolve_block_stride,
)
from lmcache.v1.memory_allocators.gpu_memory_allocator import GPUMemoryAllocator
from lmcache.v1.memory_management import MemoryFormat, MemoryObj
import lmcache.lmcache_native as lmcache_native

if TYPE_CHECKING:
    # First Party
    from lmcache.inprocess.v1.metadata import LMCacheMetadata

logger = init_logger(__name__)


class VLLMPagedMemXPUConnectorV2(GPUConnectorInterface):
    """Transfer paged XPU caches using V2's component-major LMCache storage.

    Non-MLA objects remain ``[2, L, T, NH * HS]`` / ``KV_2LTD``, including
    engines whose trailing head row packs K and V into ``CS = 2 * HS``.
    Packed NHD/HND views may have dim-0 block padding, with the same geometry,
    dtype and physical block stride across layers. Legacy NHD and MLA views
    retain their tight-layout transfer contracts.
    """

    def __init__(
        self,
        hidden_dim_size: int,
        num_layers: int,
        use_gpu: bool = False,
        **kwargs: object,
    ) -> None:
        """Construct a V2 connector without initializing a layout group manager.

        Args:
            hidden_dim_size: Elements in one flattened K or V token row.
            num_layers: Number of engine layers.
            use_gpu: Whether to stage complete stores through an XPU buffer.
            **kwargs: ``use_mla`` (bool), ``layout_hints`` (LayoutHints), and
                optional model ``head_size`` (component width). Staging requires
                ``chunk_size`` (int), ``dtype`` (torch.dtype) and ``device``
                (torch.device).

        Raises:
            ValueError: Staging geometry, dtype or device is missing.
        """
        self.hidden_dim_size = hidden_dim_size
        self.num_layers = num_layers
        self.kv_cache_pointers = torch.empty(
            num_layers, dtype=torch.uint64, device="cpu"
        )
        # Not sure we need a dict here. Maybe a single GPU connector always
        # works with a single device?
        self.kv_cache_pointers_on_gpu: dict[int, torch.Tensor] = {}
        self.page_buffer_size = 0

        self.kvcaches: DiscoverableKVCache | None = None

        self.gpu_buffer: Optional[torch.Tensor] = None
        self.use_mla = bool(kwargs.get("use_mla", False))
        self.layout_hints = cast(LayoutHints, kwargs.get("layout_hints") or {})
        self.model_head_size = cast(int | None, kwargs.get("head_size"))
        self.dtype = cast(torch.dtype | None, kwargs.get("dtype"))
        if use_gpu:
            if not all(name in kwargs for name in ("chunk_size", "dtype", "device")):
                raise ValueError("Staging requires chunk_size, dtype and device.")
            if self.dtype is None or kwargs["device"] is None:
                raise ValueError("Staging requires a dtype and an XPU device.")
            shape = self.get_shape(cast(int, kwargs["chunk_size"]))
            self.gpu_buffer = torch.empty(
                shape, dtype=self.dtype, device=cast(torch.device, kwargs["device"])
            )

        self.store_stream = torch.xpu.Stream()
        self.load_stream = torch.xpu.Stream()

    @classmethod
    def from_metadata(
        cls,
        metadata: "LMCacheMetadata",
        use_gpu: bool = False,
        device: Optional[torch.device] = None,
        layout_hints: LayoutHints | None = None,
    ) -> "VLLMPagedMemXPUConnectorV2":
        """Create a connector from LMCacheMetadata.

        Args:
            metadata: The LMCache engine metadata containing model configuration.
            use_gpu: Whether to use GPU intermediate buffer.
            device: The device to use for the connector.
            layout_hints: Physical NHD/HND or blocks-first layout hints.

        Returns:
            A new instance of VLLMPagedMemXPUConnectorV2.
        """
        # Extract parameters from metadata
        # kv_shape: (num_layer, 2 or 1, chunk_size, num_kv_head, head_size)
        num_layers = metadata.kv_shape[0]
        chunk_size = metadata.kv_shape[2]
        num_kv_head = metadata.kv_shape[3]
        head_size = metadata.kv_shape[4]
        hidden_dim_size = num_kv_head * head_size

        return cls(
            hidden_dim_size=hidden_dim_size,
            num_layers=num_layers,
            use_gpu=use_gpu,
            chunk_size=chunk_size,
            dtype=metadata.kv_dtype,
            device=device,
            use_mla=metadata.use_mla,
            head_size=head_size,
            layout_hints=layout_hints,
        )

    @_lmcache_nvtx_annotate
    def to_gpu(
        self, memory_obj: MemoryObj, start: int, end: int, **kwargs: object
    ) -> None:
        """Restore one component-major chunk into the engine cache.

        Args:
            memory_obj: Contiguous V2 chunk, with the engine dtype.
            start: First token in the full sequence's slot mapping.
            end: Exclusive last token.
            **kwargs: ``kvcaches`` (DiscoverableKVCache), ``slot_mapping``
                (1-D XPU int64 Tensor for the full sequence), and optional
                ``vllm_cached_tokens`` (int). Negative slots and the cached
                prefix are left untouched.

        Raises:
            ValueError: Cache geometry, chunk shape, format or mapping is invalid.

        Note:
            Enqueues on the current XPU stream. Batched retrieval owns load
            stream ordering and synchronization.
        """
        self.initialize_kvcaches_ptr(**kwargs)
        if self.kvcaches is None:
            raise ValueError("'kvcaches' must be provided before transfer.")
        kv_cache_pointers = self._initialize_pointers(self.kvcaches)
        tensor, slot_mapping = self._validate_transfer(memory_obj, start, end, kwargs)

        if self.use_mla:
            if memory_obj.metadata.fmt != MemoryFormat.KV_MLA_FMT:
                raise ValueError(
                    "The memory object should be in KV_MLA_FMT format in"
                    " order to be processed by VLLMPagedMemXPUConnector"
                )
        else:
            if memory_obj.metadata.fmt != MemoryFormat.KV_2LTD:
                raise ValueError(
                    "The memory object should be in KV_2LTD format in"
                    " order to be processed by VLLMPagedMemXPUConnector"
                )

        # avoid read/write stream race condition for shared block
        # this will only be potentially non-zero for the first
        # block lmcache is transferring back
        vllm_cached = cast(int, kwargs.get("vllm_cached_tokens", 0))
        skip_prefix_n_tokens = min(end - start, max(0, vllm_cached - start))

        device_ops.multi_layer_kv_transfer(
            tensor,
            kv_cache_pointers,
            slot_mapping[start:end],
            self.device,
            self.page_buffer_size,
            lmcache_native.TransferDirection.H2D,
            self.engine_kv_format,
            block_size=self.block_size,
            head_size=self.head_size,
            skip_prefix_n_tokens=skip_prefix_n_tokens,
            block_stride_elems=self.block_stride_elems,
        )

    @_lmcache_nvtx_annotate
    def from_gpu(
        self, memory_obj: MemoryObj, start: int, end: int, **kwargs: object
    ) -> None:
        """Store engine KV into a V2 chunk, preserving negative-slot rows.

        Args:
            memory_obj: Contiguous destination with shape from ``get_shape``.
            start: First token in the full sequence's slot mapping.
            end: Exclusive last token.
            **kwargs: ``kvcaches`` (DiscoverableKVCache) and ``slot_mapping``
                (1-D XPU int64 Tensor for the full sequence).

        Raises:
            ValueError: Cache geometry, chunk shape or mapping is invalid.

        Note:
            CPU destinations synchronize the store stream before returning.
            Packed KV is unpacked on XPU into the existing two-plane storage;
            metadata's group manager is not initialized by this connector.
        """
        self.initialize_kvcaches_ptr(**kwargs)
        if self.kvcaches is None:
            raise ValueError("'kvcaches' must be provided before transfer.")
        kv_cache_pointers = self._initialize_pointers(self.kvcaches)
        tensor, slot_mapping = self._validate_transfer(memory_obj, start, end, kwargs)

        with torch.xpu.stream(self.store_stream):
            if self.gpu_buffer is None or end - start != self.gpu_buffer.shape[2]:
                device_ops.multi_layer_kv_transfer(
                    tensor,
                    kv_cache_pointers,
                    slot_mapping[start:end],
                    self.device,
                    self.page_buffer_size,
                    lmcache_native.TransferDirection.D2H,
                    self.engine_kv_format,
                    block_size=self.block_size,
                    head_size=self.head_size,
                    block_stride_elems=self.block_stride_elems,
                )
            else:
                # kvcaches -> gpu_buffer -> memobj
                assert self.gpu_buffer.device == self.device
                tmp_gpu_buffer = self.gpu_buffer[:, :, : end - start, :]
                # The kernel leaves negative slots unchanged. Seed staging
                # with the destination so reused buffers preserve those rows.
                tmp_gpu_buffer.copy_(tensor, non_blocking=True)
                device_ops.multi_layer_kv_transfer(
                    tmp_gpu_buffer,
                    kv_cache_pointers,
                    slot_mapping[start:end],
                    self.device,
                    self.page_buffer_size,
                    lmcache_native.TransferDirection.D2H,
                    self.engine_kv_format,
                    block_size=self.block_size,
                    head_size=self.head_size,
                    block_stride_elems=self.block_stride_elems,
                )
                tensor.copy_(tmp_gpu_buffer, non_blocking=True)

        if not tensor.is_xpu:
            # Force a synchronize if the target buffer is NOT XPU device
            # NOTE: for better performance, we may not want to sync for every
            # memory object
            self.store_stream.synchronize()

        if self.use_mla:
            memory_obj.metadata.fmt = MemoryFormat.KV_MLA_FMT
        else:
            memory_obj.metadata.fmt = MemoryFormat.KV_2LTD

    # TODO(Jiayi): need to optimize to enable real batching
    def batched_to_gpu(
        self,
        memory_objs: list[list[MemoryObj]] | list[MemoryObj] | list[int] | None = None,
        starts: list[int] | None = None,
        ends: list[int] | None = None,
        **kwargs: object,
    ) -> None:
        """Restore chunks on the load stream and wait for completion.

        Args:
            memory_objs: Source chunks, one per token range.
            starts: Inclusive sequence offsets.
            ends: Exclusive sequence offsets.
            **kwargs: Cache, slot mapping and cached-prefix arguments of ``to_gpu``.

        Raises:
            ValueError: Any chunk or range list is absent.
        """
        if memory_objs is None or starts is None or ends is None:
            raise ValueError("V2 batched retrieval requires chunks, starts and ends.")
        with torch.xpu.stream(self.load_stream):
            for memory_obj, start, end in zip(
                cast(list[MemoryObj], memory_objs), starts, ends, strict=True
            ):
                self.to_gpu(memory_obj, start, end, **kwargs)
        self.load_stream.synchronize()

    # TODO(Jiayi): need to optimize to enable real batching
    def batched_from_gpu(
        self,
        memory_objs: list[list[MemoryObj]] | list[MemoryObj],
        starts: list[int],
        ends: list[int],
        **kwargs: object,
    ) -> None:
        """Store chunks using the synchronization contract of ``from_gpu``.

        Args:
            memory_objs: Destination chunks, one per token range.
            starts: Inclusive sequence offsets.
            ends: Exclusive sequence offsets.
            **kwargs: Cache and slot mapping arguments of ``from_gpu``.
        """
        for memory_obj, start, end in zip(
            cast(list[MemoryObj], memory_objs), starts, ends, strict=True
        ):
            self.from_gpu(memory_obj, start, end, **kwargs)

    def get_shape(self, num_tokens: int) -> torch.Size:
        """Return V2's ``[components, layers, tokens, NH * HS]`` chunk shape.

        Args:
            num_tokens: Number of tokens in this chunk.

        Returns:
            One component for MLA, otherwise separate K and V planes.
        """
        kv_size = 1 if self.use_mla else 2
        return torch.Size([kv_size, self.num_layers, num_tokens, self.hidden_dim_size])

    def _initialize_pointers(self, kv_caches: DiscoverableKVCache) -> torch.Tensor:
        """Discover and validate engine geometry before publishing layer pointers."""
        self.device = get_device(kv_caches)
        if self.device.type != "xpu":
            raise ValueError("The engine cache must be on XPU.")
        idx = self.device.index
        if idx in self.kv_cache_pointers_on_gpu:
            return self.kv_cache_pointers_on_gpu[idx]

        self.engine_kv_format, self.kvcaches = normalize_kv_and_discover_format(
            kv_caches, EngineType.VLLM, self.layout_hints
        )
        fmt = self.engine_kv_format
        facts = get_spec_class(fmt)
        if facts.is_mla != self.use_mla:
            raise ValueError("Engine MLA layout does not match model metadata.")
        if get_num_layers(self.kvcaches, fmt) != self.num_layers:
            raise ValueError("Engine layer count does not match model metadata.")
        self.num_blocks = get_num_blocks(self.kvcaches, fmt)
        self.block_size = get_block_size(self.kvcaches, fmt)
        self.page_buffer_size = self.num_blocks * self.block_size
        strides = []
        for layer in range(self.num_layers):
            if facts.is_fused_packed:
                layer_caches: DiscoverableKVCache = [
                    cast(list[torch.Tensor], self.kvcaches)[layer]
                ]
                if (
                    get_device(layer_caches) != self.device
                    or get_num_blocks(layer_caches, fmt) != self.num_blocks
                ):
                    raise ValueError(
                        "V2 packed layers must share block count and XPU device."
                    )
            content_size = get_head_size(self.kvcaches, fmt, layer)
            if facts.is_fused_packed and content_size % 2:
                raise ValueError("Packed content size must be twice component HS.")
            # Shared packed specs describe CS, while V2 stores one K/V
            # component per plane and passes component HS to SYCL.
            component_size = (
                content_size // 2 if facts.is_fused_packed else content_size
            )
            if layer == 0:
                self.head_size = component_size
            hidden_dim = get_hidden_dim_size(self.kvcaches, fmt, layer)
            if facts.is_fused_packed:
                hidden_dim //= 2
            if (
                hidden_dim != self.hidden_dim_size
                or component_size != self.head_size
                or (
                    self.model_head_size is not None
                    and component_size != self.model_head_size
                )
                or get_block_size(self.kvcaches, fmt, layer) != self.block_size
            ):
                raise ValueError(
                    "V2 requires uniform geometry matching model metadata."
                )
            dtype = get_dtype(self.kvcaches, fmt, layer)
            if self.dtype is not None and dtype != self.dtype:
                raise ValueError("Engine dtype does not match V2 model metadata.")
            self.dtype = dtype
            stride = resolve_block_stride(self.kvcaches, fmt, layer)
            if facts.is_fused_packed:
                strides.append(stride)
            elif stride is not None and stride != self.block_size * hidden_dim:
                raise ValueError(
                    "V2 supports block padding only for packed KV layouts."
                )
        if facts.is_fused_packed:
            if len(set(strides)) != 1:
                raise ValueError(
                    "V2 requires the same packed block stride in every layer."
                )
            self.block_stride_elems = cast(int, strides[0])
        else:
            self.block_stride_elems = 0
        pointers = get_group_data_ptrs(self.kvcaches, fmt, list(range(self.num_layers)))
        self.kv_cache_pointers = torch.tensor(
            pointers, dtype=torch.uint64, device="cpu"
        )
        self.kv_cache_pointers_on_gpu[idx] = torch.empty(
            len(pointers), dtype=torch.uint64, device=self.device
        )
        self.kv_cache_pointers_on_gpu[idx].copy_(self.kv_cache_pointers)
        return self.kv_cache_pointers_on_gpu[idx]

    def _validate_transfer(
        self, memory_obj: MemoryObj, start: int, end: int, kwargs: dict[str, object]
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Validate host-visible geometry without reading device slot values."""
        tensor = memory_obj.tensor
        if tensor is None:
            raise ValueError("The memory object must contain a tensor.")
        if "slot_mapping" not in kwargs:
            raise ValueError("'slot_mapping' must be provided.")
        slot_mapping = cast(torch.Tensor, kwargs["slot_mapping"])
        if (
            slot_mapping.ndim != 1
            or slot_mapping.dtype != torch.int64
            or slot_mapping.device != self.device
            or not slot_mapping.is_contiguous()
            or not 0 <= start <= end <= slot_mapping.numel()
        ):
            raise ValueError(
                "Expected a contiguous XPU int64 slot mapping and valid range."
            )
        if (
            tensor.shape != self.get_shape(end - start)
            or not tensor.is_contiguous()
            or tensor.dtype != self.dtype
        ):
            raise ValueError(
                "Chunk shape, contiguity and dtype must match V2 geometry."
            )
        return tensor, slot_mapping


class VLLMPagedMemXPUConnectorV3(GPUConnectorInterface):
    def __init__(
        self,
        metadata: "LMCacheMetadata",
        device: torch.device,
        use_gpu: bool = False,
    ):
        assert device.type == "xpu", "The device should be XPU."
        self.metadata = metadata
        self.device = device
        self.use_mla = metadata.use_mla
        self.chunk_size = metadata.chunk_size
        self.use_gpu = use_gpu
        self.kvcaches: Optional[List[torch.Tensor]] = None
        self.page_buffer_size = 0

        self.init = False
        self.group_kv_cache_pointers_on_gpu: Optional[list[torch.Tensor]] = None
        self.group_tmp_buffer: Optional[list[torch.Tensor]] = None

        self.store_stream = torch.xpu.Stream()
        self.load_stream = torch.xpu.Stream()

    @classmethod
    def from_metadata(
        cls,
        metadata: "LMCacheMetadata",
        use_gpu: bool = False,
        device: Optional[torch.device] = None,
    ) -> "VLLMPagedMemXPUConnectorV3":
        assert device is not None
        return cls(metadata, device, use_gpu)

    def _initialize_kv_cache_pointers(self):
        if self.init:
            return
        assert self.metadata.kv_layer_groups_manager.kernel_groups
        if self.use_gpu:
            # init tmp buffer
            tmp_buf_shapes = self.metadata.get_shapes(self.chunk_size)
            tmp_buf_dtypes = self.metadata.get_dtypes()
            assert len(tmp_buf_shapes) == len(tmp_buf_dtypes)
            self.group_tmp_buffer = [
                torch.empty(tmp_buf_shape, dtype=tmp_buf_dtype, device=self.device)
                for tmp_buf_shape, tmp_buf_dtype in zip(
                    tmp_buf_shapes, tmp_buf_dtypes, strict=True
                )
            ]
        self.group_kv_cache_pointers_on_gpu = []
        for group in self.metadata.kv_layer_groups_manager.kernel_groups:
            # init kv cache pointers
            num_layers = group.num_layers
            kv_cache_pointers = torch.empty(
                num_layers, dtype=torch.uint64, device="cpu"
            )
            kv_cache_pointers.numpy()[:] = [
                t.data_ptr()
                for i, t in enumerate(self.kvcaches)
                if i in group.layer_indices
            ]
            kv_cache_pointers_on_gpu = torch.empty(
                num_layers, dtype=torch.uint64, device=self.device
            )
            kv_cache_pointers_on_gpu.copy_(kv_cache_pointers)
            self.group_kv_cache_pointers_on_gpu.append(kv_cache_pointers_on_gpu)

        self.engine_kv_format, self.kvcaches = normalize_kv_and_discover_format(
            self.kvcaches, EngineType.VLLM
        )
        self.num_blocks = get_num_blocks(self.kvcaches, self.engine_kv_format)
        self.block_size = get_block_size(self.kvcaches, self.engine_kv_format)
        self.page_buffer_size = self.num_blocks * self.block_size

        self.init = True
        logger.info("init kv cache pointers success in VLLMPagedMemXPUConnectorV3")

    @_lmcache_nvtx_annotate
    def to_gpu(
        self, memory_obj: MemoryObj, start: int, end: int, **kwargs: object
    ) -> None:
        """Restore initialized V3 groups while preserving the cached prefix.

        Args:
            memory_obj: Source memory object containing the initialized groups.
            start: Inclusive token offset in the full slot mapping.
            end: Exclusive token offset.
            **kwargs: ``kvcaches`` (list[Tensor]), ``slot_mapping`` (Tensor),
                and optional ``vllm_cached_tokens`` (int).

        Note:
            Uses named native skip arguments and leaves legacy head geometry
            unused. Group initialization and representation follow V3's
            existing requirements.
        """
        assert memory_obj.raw_tensor is not None
        assert "slot_mapping" in kwargs
        if self.use_mla:
            assert memory_obj.metadata.fmt == MemoryFormat.KV_MLA_FMT
        else:
            assert memory_obj.metadata.fmt == MemoryFormat.KV_2LTD

        slot_mapping = cast(torch.Tensor, kwargs["slot_mapping"])
        self.initialize_kvcaches_ptr(**kwargs)
        assert self.kvcaches is not None
        assert self.kvcaches[0].device == self.device
        self._initialize_kv_cache_pointers()
        assert self.group_kv_cache_pointers_on_gpu is not None

        # avoid read/write stream race condition for shared block
        # this will only be potentially non-zero for the first
        # block lmcache is transferring back
        vllm_cached = cast(int, kwargs.get("vllm_cached_tokens", 0))
        skip_prefix_n_tokens = min(end - start, max(0, vllm_cached - start))

        for i, kv_cache_pointer in enumerate(self.group_kv_cache_pointers_on_gpu):
            memory_obj_tensor = memory_obj.get_tensor(i)
            assert memory_obj_tensor is not None
            device_ops.multi_layer_kv_transfer(
                memory_obj_tensor,
                kv_cache_pointer,
                slot_mapping[start:end],
                self.device,
                self.page_buffer_size,
                lmcache_native.TransferDirection.H2D,
                self.engine_kv_format,
                self.block_size,
                head_size=0,
                skip_prefix_n_tokens=skip_prefix_n_tokens,
            )

    @_lmcache_nvtx_annotate
    def from_gpu(self, memory_obj: MemoryObj, start: int, end: int, **kwargs):
        assert memory_obj.raw_tensor is not None
        assert "slot_mapping" in kwargs

        slot_mapping: torch.Tensor = kwargs["slot_mapping"]
        self.initialize_kvcaches_ptr(**kwargs)
        assert self.kvcaches is not None
        assert self.kvcaches[0].device == self.device
        self._initialize_kv_cache_pointers()
        assert self.group_kv_cache_pointers_on_gpu is not None
        with torch.xpu.stream(self.store_stream):
            if not self.use_gpu or end - start != self.chunk_size:
                for i, kv_cache_pointer in enumerate(
                    self.group_kv_cache_pointers_on_gpu
                ):
                    memory_obj_tensor = memory_obj.get_tensor(i)
                    assert memory_obj_tensor is not None
                    device_ops.multi_layer_kv_transfer(
                        memory_obj_tensor,
                        kv_cache_pointer,
                        slot_mapping[start:end],
                        self.device,
                        self.page_buffer_size,
                        lmcache_native.TransferDirection.D2H,
                        self.engine_kv_format,
                        self.block_size,
                    )
            else:
                # kvcaches -> gpu_buffer -> memobj
                assert self.group_tmp_buffer is not None
                for i, kv_cache_pointer in enumerate(
                    self.group_kv_cache_pointers_on_gpu
                ):
                    tmp_gpu_buffer = self.group_tmp_buffer[i][:, :, : end - start, :]
                    device_ops.multi_layer_kv_transfer(
                        tmp_gpu_buffer,
                        kv_cache_pointer,
                        slot_mapping[start:end],
                        self.device,
                        self.page_buffer_size,
                        lmcache_native.TransferDirection.D2H,
                        self.engine_kv_format,
                        self.block_size,
                    )
                    memory_obj_tensor = memory_obj.get_tensor(i)
                    assert memory_obj_tensor is not None
                    memory_obj_tensor.copy_(tmp_gpu_buffer, non_blocking=True)

        if not memory_obj.raw_tensor.is_xpu:
            # Force a synchronize if the target buffer is NOT XPU device
            # NOTE: for better performance, we may not want to sync for every
            # memory object
            self.store_stream.synchronize()

        if self.use_mla:
            memory_obj.metadata.fmt = MemoryFormat.KV_MLA_FMT

    def batched_to_gpu(self, memory_objs, starts, ends, **kwargs):
        with torch.xpu.stream(self.load_stream):
            for memory_obj, start, end in zip(memory_objs, starts, ends, strict=False):
                self.to_gpu(memory_obj, start, end, **kwargs)
        self.load_stream.synchronize()

    def batched_from_gpu(self, memory_objs, starts, ends, **kwargs):
        for memory_obj, start, end in zip(memory_objs, starts, ends, strict=False):
            self.from_gpu(memory_obj, start, end, **kwargs)

    def get_shape(self, num_tokens: int) -> torch.Size:
        raise NotImplementedError


class VLLMBufferLayerwiseXPUConnector(GPUConnectorInterface):
    def __init__(
        self,
        hidden_dim_size: int,
        num_layers: int,
        use_gpu: bool = False,
        use_double_buffer: bool = True,
        **kwargs,
    ):
        self.hidden_dim_size = hidden_dim_size
        self.num_layers = num_layers

        self.kvcaches: Optional[List[torch.Tensor]] = None

        # TODO(Jiayi): remove this hardcode
        self.cache_positions = True

        self.fused_rotary_emb = None

        assert use_gpu, "use_gpu must be true in VLLMBufferLayerwiseXPUConnector"
        assert "dtype" in kwargs, "dtype should be provided to create a GPU buffer."
        assert "device" in kwargs, "device should be provided to create a GPU buffer."

        self.dtype = kwargs["dtype"]
        self.device = kwargs["device"]

        self.load_stream = torch.xpu.Stream()
        self.store_stream = torch.xpu.Stream()

        self.buffer_mapping: dict[int, MemoryObj] = {}

        # track gap positions between blended chunks
        self.current_gap_positions = None

        self.use_gpu = use_gpu
        self.gpu_buffer_allocator = None
        self.element_size = torch.tensor([], dtype=self.dtype).element_size()

    @classmethod
    def from_metadata(
        cls,
        metadata: "LMCacheMetadata",
        use_gpu: bool = False,
        device: Optional[torch.device] = None,
    ) -> "VLLMBufferLayerwiseXPUConnector":
        """Create a connector from LMCacheMetadata.

        Args:
            metadata: The LMCache engine metadata containing model configuration.
            use_gpu: Whether to use GPU intermediate buffer.
            device: The device to use for the connector.

        Returns:
            A new instance of VLLMBufferLayerwiseXPUConnector.
        """
        # Extract parameters from metadata
        # kv_shape: (num_layer, 2 or 1, chunk_size, num_kv_head, head_size)
        num_layers = metadata.kv_shape[0]
        num_kv_head = metadata.kv_shape[3]
        head_size = metadata.kv_shape[4]
        hidden_dim_size = num_kv_head * head_size

        return cls(
            hidden_dim_size=hidden_dim_size,
            num_layers=num_layers,
            use_gpu=use_gpu,
            dtype=metadata.kv_dtype,
            device=device,
        )

    def _lazy_initialize_buffer(self, kv_caches):
        """
        Lazily initialize the GPU buffer allocator if it is not initialized yet.
        Currently, we use the `kv_caches` (kv cache pointer) to determine
        the gpu buffer size in gpu connector.
        Also, the first request might be a bit slower due to buffer creation.
        """
        if self.use_gpu and self.gpu_buffer_allocator is None:
            logger.info("Lazily initializing GPU buffer.")
            # NOTE (Jiayi): We use the first layer to determine the gpu buffer size.
            # NOTE (Jiayi): Using the exact number of tokens in the first layer
            # is okay since fragmentation shouldn't exist in the `gpu_buffer_allocator`
            # in layerwise mode.

            self.engine_kv_format, kv_caches = normalize_kv_and_discover_format(
                kv_caches, EngineType.VLLM
            )
            assert_is_vllm_flash_attn_or_flash_infer(self.engine_kv_format)
            self.tokens_per_layer = get_tokens_per_layer(
                kv_caches, self.engine_kv_format
            )
            self.elements_per_layer = get_elements_per_layer(
                kv_caches, self.engine_kv_format
            )
            logger.info(
                "Lazily initializing GPU buffer (max tokens=%s).",
                self.tokens_per_layer,
            )
            gpu_buffer_size = self.elements_per_layer * self.element_size
            self.gpu_buffer_allocator = GPUMemoryAllocator(
                gpu_buffer_size, device=self.device
            )

    def get_kv(self, layer_id: int) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Get the KV cache for the given layer ID.
        This function is used to get the KV cache from the GPU buffer.
        """
        if layer_id not in self.buffer_mapping:
            raise ValueError(f"Layer {layer_id} is not loaded into GPU buffer.")

        gpu_buffer = self.buffer_mapping[layer_id].tensor
        assert gpu_buffer is not None
        return gpu_buffer[0], gpu_buffer[1]

    def to_gpu(self, memory_obj: MemoryObj, start: int, end: int, **kwargs):
        """ """

        raise NotImplementedError

    def from_gpu(self, memory_obj: MemoryObj, start: int, end: int, **kwargs):
        """ """

        raise NotImplementedError

    @_lmcache_nvtx_annotate
    def batched_to_gpu(self, starts: List[int], ends: List[int], **kwargs):
        """
        This function is a generator that moves the KV cache from the memory
        objects to buffer GPU memory. In each iteration i, it (1) loads the KV
        cache of layer i from CPU -> GPU buffer, (2) recovers the positional
        encoding of the layer i-1's KV cache in the GPU buffer, and (3)
        moves the KV cache of layer i-2 from GPU buffer to paged GPU memory.
        In total, this the generator will yield num_layers + 2 times.

        :param starts: The starting indices of the KV cache in the corresponding
            token sequence.

        :param ends: The ending indices of the KV cache in the corresponding
            token sequence.
        """

        self.initialize_kvcaches_ptr(**kwargs)
        assert self.kvcaches is not None, (
            "kvcaches should be provided in kwargs or initialized beforehand."
        )

        if "slot_mapping" not in kwargs:
            raise ValueError("'slot_mapping' should be provided in kwargs.")

        if self.fused_rotary_emb is None and self.cache_positions:
            # TODO(Jiayi): Make this more elegant
            self.lmc_model = LMCBlenderBuilder.get(ENGINE_NAME).layerwise_model
            self.fused_rotary_emb = self.lmc_model.fused_rotary_emb

        slot_mapping: torch.Tensor = kwargs["slot_mapping"]

        self._lazy_initialize_buffer(self.kvcaches)

        num_all_tokens = ends[-1] - starts[0]
        slot_mapping_full = slot_mapping[starts[0] : ends[-1]]

        # compute gap positions
        gap_mask = torch.ones(
            num_all_tokens, dtype=torch.bool, device=slot_mapping_full.device
        )
        buf_offset = starts[0]

        for start, end in zip(starts, ends, strict=False):
            gap_mask[start - buf_offset : end - buf_offset] = False

        self.current_gap_positions = torch.where(gap_mask)[0]

        buf_offset = starts[0]
        if self.cache_positions:
            new_positions_full = torch.arange(
                starts[0], ends[-1], dtype=torch.int64, device=self.kvcaches[0].device
            )

        buffer_shape = self.get_shape(num_all_tokens)
        assert self.gpu_buffer_allocator is not None
        compute_gpu_buffer_obj = self.gpu_buffer_allocator.allocate(
            buffer_shape, self.dtype, MemoryFormat.KV_2TD
        )
        load_gpu_buffer_obj = self.gpu_buffer_allocator.allocate(
            buffer_shape, self.dtype, MemoryFormat.KV_2TD
        )
        assert compute_gpu_buffer_obj is not None, (
            "Failed to allocate GPU buffer in GPUConnector"
        )
        assert load_gpu_buffer_obj is not None, (
            "Failed to allocate GPU buffer in GPUConnector"
        )
        assert compute_gpu_buffer_obj.tensor is not None
        assert load_gpu_buffer_obj.tensor is not None

        # current_stream = torch.xpu.current_stream()

        if self.cache_positions:
            old_positions_full = torch.zeros(
                (num_all_tokens,), dtype=torch.int64, device=self.kvcaches[0].device
            )
        for layer_id in range(self.num_layers + 2):
            if layer_id > 1:
                device_ops.single_layer_kv_transfer(
                    self.buffer_mapping[layer_id - 2].tensor,
                    self.kvcaches[layer_id - 2],
                    slot_mapping_full,
                    lmcache_native.TransferDirection.H2D,
                    self.engine_kv_format,
                    token_major=False,  # shape is [2, num_tokens, hidden_dim]
                )
                del self.buffer_mapping[layer_id - 2]

                logger.debug(
                    "Finished loading layer %d into paged memory", layer_id - 2
                )

            if layer_id > 0 and layer_id <= self.num_layers:
                # NOTE: wait until both compute and load streams are done
                torch.xpu.synchronize()

                # ping-pong the buffers
                compute_gpu_buffer_obj, load_gpu_buffer_obj = (
                    load_gpu_buffer_obj,
                    compute_gpu_buffer_obj,
                )

                if self.cache_positions:
                    assert compute_gpu_buffer_obj.tensor is not None

                    compute_gpu_buffer_obj.tensor[0] = self.fused_rotary_emb(
                        old_positions_full,
                        new_positions_full,
                        compute_gpu_buffer_obj.tensor[0],
                    )

                # gap zeroing after RoPE
                if self.current_gap_positions.numel():
                    compute_gpu_buffer_obj.tensor[:, self.current_gap_positions] = 0.0

                self.buffer_mapping[layer_id - 1] = compute_gpu_buffer_obj

                logger.debug("Finished loading layer %d into buffer", layer_id - 1)

            if layer_id < self.num_layers:
                memory_objs_layer = yield

                # memobj -> gpu_buffer
                with torch.xpu.stream(self.load_stream):
                    for start, end, memory_obj in zip(
                        starts, ends, memory_objs_layer, strict=False
                    ):
                        assert memory_obj.metadata.fmt == MemoryFormat.KV_2TD
                        assert load_gpu_buffer_obj.tensor is not None
                        load_gpu_buffer_obj.tensor[0][
                            start - buf_offset : end - buf_offset
                        ].copy_(memory_obj.tensor[0], non_blocking=True)

                        load_gpu_buffer_obj.tensor[1][
                            start - buf_offset : end - buf_offset
                        ].copy_(memory_obj.tensor[1], non_blocking=True)

                        if self.cache_positions and layer_id == 0:
                            old_positions_full[
                                start - buf_offset : end - buf_offset
                            ] = memory_obj.metadata.cached_positions

            elif layer_id == self.num_layers:
                yield

        # free the buffer memory
        load_gpu_buffer_obj.ref_count_down()
        compute_gpu_buffer_obj.ref_count_down()

        assert len(self.buffer_mapping) == 0, (
            "There are still layers in the buffer mapping after "
            "releasing the GPU buffers."
        )

        yield

    # TODO(Jiayi): Reduce repetitive operations in `batched_to_gpu`
    # and `batched_from_gpu`.
    @_lmcache_nvtx_annotate
    def batched_from_gpu(
        self,
        memory_objs: Union[List[List[MemoryObj]], List[MemoryObj]],
        starts: List[int],
        ends: List[int],
        **kwargs,
    ):
        """
        This function is a generator that moves the KV cache from the paged GPU
        memory to the memory objects. The first iteration will prepare some
        related metadata and initiate the transfer in the first layer. In each
        of the following iterations, it will first wait until the storing of
        previous layer finishes, and then initiate string the KV cache of the
        current layer one. The storing process of the KV cache is paged GPU
        memory -> GPU buffer -> memory objects. The last iteration simply waits
        for the last layer to finish.
        In total, this the generator will yield num_layers + 1 times.

        :param memory_objs: The memory objects to store the KV cache. The first
            dimension is the number of layers, and the second dimension is the
            number of memory objects (i.e., number of chunks) for each layer.

        :param starts: The starting indices of the KV cache in the corresponding
            token sequence.

        :param ends: The ending indices of the KV cache in the corresponding
            token sequence.

        :raises ValueError: If 'kvcaches' is not provided in kwargs.

        :raises ValueError: If 'slot_mapping' is not provided in kwargs.
        """

        self.initialize_kvcaches_ptr(**kwargs)
        assert self.kvcaches is not None, (
            "kvcaches should be provided in kwargs or initialized beforehand."
        )

        if "slot_mapping" not in kwargs:
            raise ValueError("'slot_mapping' should be provided in kwargs.")

        slot_mapping: torch.Tensor = kwargs["slot_mapping"]

        self._lazy_initialize_buffer(self.kvcaches)

        buf_start = 0
        slot_mapping_chunks = []
        buf_starts_ends = []
        old_positions_chunks = []
        for start, end in zip(starts, ends, strict=False):
            buf_end = buf_start + end - start
            buf_starts_ends.append((buf_start, buf_end))
            slot_mapping_chunks.append(slot_mapping[start:end])
            buf_start = buf_end
            if self.cache_positions:
                old_positions_chunks.append(
                    torch.arange(
                        start, end, device=self.kvcaches[0].device, dtype=torch.int64
                    )
                )

        slot_mapping_full = torch.cat(slot_mapping_chunks, dim=0)

        num_tokens = len(slot_mapping_full)
        buffer_shape = self.get_shape(num_tokens)
        assert self.gpu_buffer_allocator is not None
        tmp_gpu_buffer_obj = self.gpu_buffer_allocator.allocate(
            buffer_shape, self.dtype, MemoryFormat.KV_2TD
        )
        assert tmp_gpu_buffer_obj is not None, (
            "Failed to allocate GPU buffer in GPUConnector"
        )
        assert tmp_gpu_buffer_obj.tensor is not None

        current_stream = torch.xpu.current_stream()

        for layer_id in range(self.num_layers):
            memory_objs_layer = memory_objs[layer_id]
            # kvcaches -> gpu_buffer -> memobj
            with torch.xpu.stream(self.store_stream):
                self.store_stream.wait_stream(current_stream)
                device_ops.single_layer_kv_transfer(
                    tmp_gpu_buffer_obj.tensor,
                    self.kvcaches[layer_id],
                    slot_mapping_full,
                    lmcache_native.TransferDirection.D2H,
                    self.engine_kv_format,
                    token_major=False,  # shape is [2, num_tokens, hidden_dim]
                )
                for (buf_start, buf_end), memory_obj, old_positions in zip(
                    buf_starts_ends,
                    memory_objs_layer,
                    old_positions_chunks,
                    strict=False,
                ):
                    assert memory_obj.tensor is not None
                    memory_obj.tensor[0].copy_(
                        tmp_gpu_buffer_obj.tensor[0][buf_start:buf_end],
                        non_blocking=True,
                    )
                    memory_obj.tensor[1].copy_(
                        tmp_gpu_buffer_obj.tensor[1][buf_start:buf_end],
                        non_blocking=True,
                    )
                    if self.cache_positions:
                        memory_obj.metadata.cached_positions = old_positions

            yield
            self.store_stream.synchronize()
            logger.debug("Finished offloading layer %d", layer_id)

        # free the buffer memory
        tmp_gpu_buffer_obj.ref_count_down()
        yield

    def get_shape(self, num_tokens: int) -> torch.Size:
        return torch.Size([2, num_tokens, self.hidden_dim_size])


class VLLMPagedMemLayerwiseXPUConnector(GPUConnectorInterface):
    """ """

    def __init__(
        self,
        hidden_dim_size: int,
        num_layers: int,
        use_gpu: bool = False,
        **kwargs,
    ):
        self.hidden_dim_size = hidden_dim_size
        self.num_layers = num_layers
        self.use_gpu = use_gpu

        self.gpu_buffer_allocator = None

        assert "chunk_size" in kwargs, (
            "chunk_size should be provided to create a GPU buffer."
        )
        assert "dtype" in kwargs, "dtype should be provided to create a GPU buffer."
        assert "device" in kwargs, "device should be provided to create a GPU buffer."

        self.dtype = kwargs["dtype"]
        self.device = kwargs["device"]

        self.kvcaches: Optional[List[torch.Tensor]] = None

        # All sizes are in bytes
        self.element_size = torch.tensor([], dtype=self.dtype).element_size()

        self.load_stream = torch.xpu.Stream()
        self.store_stream = torch.xpu.Stream()

        self.use_mla = "use_mla" in kwargs and kwargs["use_mla"]

    @classmethod
    def from_metadata(
        cls,
        metadata: "LMCacheMetadata",
        use_gpu: bool = False,
        device: Optional[torch.device] = None,
    ) -> "VLLMPagedMemLayerwiseXPUConnector":
        """Create a connector from LMCacheMetadata.

        Args:
            metadata: The LMCache engine metadata containing model configuration.
            use_gpu: Whether to use GPU intermediate buffer.
            device: The device to use for the connector.

        Returns:
            A new instance of VLLMPagedMemLayerwiseXPUConnector.
        """
        # Extract parameters from metadata
        # kv_shape: (num_layer, 2 or 1, chunk_size, num_kv_head, head_size)
        num_layers = metadata.kv_shape[0]
        chunk_size = metadata.kv_shape[2]
        num_kv_head = metadata.kv_shape[3]
        head_size = metadata.kv_shape[4]
        hidden_dim_size = num_kv_head * head_size

        return cls(
            hidden_dim_size=hidden_dim_size,
            num_layers=num_layers,
            use_gpu=use_gpu,
            chunk_size=chunk_size,
            dtype=metadata.kv_dtype,
            device=device,
            use_mla=metadata.use_mla,
        )

    def _lazy_initialize_buffer(self, kv_caches):
        """
        Lazily initialize the GPU buffer allocator if it is not initialized yet.
        Currently, we use the `kv_caches` (kv cache pointer) to determine
        the gpu buffer size in gpu connector.
        Also, the first request might be a bit slower due to buffer creation.
        """
        if self.use_gpu and self.gpu_buffer_allocator is None:
            logger.info("Lazily initializing GPU buffer.")
            # NOTE (Jiayi): We use the first layer to determine the gpu buffer size.
            # NOTE (Jiayi): Using the exact number of tokens in the first layer
            # is okay since fragmentation shouldn't exist in the `gpu_buffer_allocator`
            # in layerwise mode.

            self.engine_kv_format, kv_caches = normalize_kv_and_discover_format(
                kv_caches, EngineType.VLLM
            )
            assert_is_vllm_flash_attn_or_flash_infer(self.engine_kv_format)
            self.tokens_per_layer = get_tokens_per_layer(
                kv_caches, self.engine_kv_format
            )
            self.elements_per_layer = get_elements_per_layer(
                kv_caches, self.engine_kv_format
            )
            logger.info(
                "Lazily initializing GPU buffer (max tokens=%s).",
                self.tokens_per_layer,
            )
            gpu_buffer_size = self.elements_per_layer * self.element_size
            self.gpu_buffer_allocator = GPUMemoryAllocator(
                gpu_buffer_size, device=self.device
            )

    def to_gpu(self, memory_obj: MemoryObj, start: int, end: int, **kwargs):
        """ """

        raise NotImplementedError

    def from_gpu(self, memory_obj: MemoryObj, start: int, end: int, **kwargs):
        """ """

        raise NotImplementedError

    @_lmcache_nvtx_annotate
    def batched_to_gpu(self, starts: List[int], ends: List[int], **kwargs):
        """
        This function is a generator that moves the KV cache from the memory
        objects to paged GPU memory. The first iteration will prepare some
        related metadata. In each of the following iterations, it will first
        wait until the loading of the previous layer finish, and then load
        one layer of KV cache from the memory objects -> GPU buffer ->
        paged GPU memory. The last iteration simply waits for the last layer
        to finish.
        In total, this the generator will yield num_layers + 2 times.

        :param starts: The starting indices of the KV cache in the corresponding
            token sequence.

        :param ends: The ending indices of the KV cache in the corresponding
            token sequence.

        :raises ValueError: If 'slot_mapping' is not provided in kwargs.
        """

        self.initialize_kvcaches_ptr(**kwargs)
        assert self.kvcaches is not None, (
            "kvcaches should be provided in kwargs or initialized beforehand."
        )

        if "slot_mapping" not in kwargs:
            raise ValueError("'slot_mapping' should be provided in kwargs.")

        if "sync" not in kwargs:
            raise ValueError("'sync' should be provided in kwargs.")

        slot_mapping: torch.Tensor = kwargs["slot_mapping"]
        sync: bool = kwargs["sync"]

        self._lazy_initialize_buffer(self.kvcaches)

        slot_mapping_chunks = []
        for start, end in zip(starts, ends, strict=False):
            slot_mapping_chunks.append(slot_mapping[start:end])

        # TODO(Jiayi): Optimize away this `cat`
        slot_mapping_full = torch.cat(slot_mapping_chunks, dim=0)

        num_tokens = len(slot_mapping_full)

        tmp_gpu_buffer_obj: Optional[MemoryObj] = None
        if self.use_gpu:
            buffer_shape = self.get_shape(num_tokens)
            assert self.gpu_buffer_allocator is not None
            tmp_gpu_buffer_obj = self.gpu_buffer_allocator.allocate(
                buffer_shape, self.dtype, MemoryFormat.KV_T2D
            )
            assert tmp_gpu_buffer_obj is not None, (
                "Failed to allocate GPU buffer in GPUConnector"
            )
            assert tmp_gpu_buffer_obj.tensor is not None

        offset = starts[0]
        current_stream = torch.xpu.current_stream()

        for layer_id in range(self.num_layers):
            memory_objs_layer = yield
            if sync:
                current_stream.wait_stream(self.load_stream)
            if layer_id > 0:
                logger.debug("Finished loading layer %d", layer_id - 1)

            # memobj -> gpu_buffer -> kvcaches
            with torch.xpu.stream(self.load_stream):
                for start, end, memory_obj in zip(
                    starts, ends, memory_objs_layer, strict=False
                ):
                    # Validate memory format
                    if self.use_mla:
                        assert memory_obj.metadata.fmt == MemoryFormat.KV_MLA_FMT, (
                            f"Expected memory format {MemoryFormat.KV_MLA_FMT}, "
                            f"got {memory_obj.metadata.fmt}"
                        )
                    else:
                        assert memory_obj.metadata.fmt == MemoryFormat.KV_T2D, (
                            f"Expected memory format {MemoryFormat.KV_T2D}, "
                            f"got {memory_obj.metadata.fmt}"
                        )
                    if self.use_gpu:
                        tmp_gpu_buffer_obj.tensor[start - offset : end - offset].copy_(
                            memory_obj.tensor, non_blocking=True
                        )
                    else:
                        device_ops.single_layer_kv_transfer(
                            memory_obj.tensor,
                            self.kvcaches[layer_id],
                            slot_mapping[start:end],
                            lmcache_native.TransferDirection.H2D,
                            self.engine_kv_format,
                            token_major=True,
                        )

                if self.use_gpu:
                    device_ops.single_layer_kv_transfer(
                        tmp_gpu_buffer_obj.tensor,
                        self.kvcaches[layer_id],
                        slot_mapping_full,
                        lmcache_native.TransferDirection.H2D,
                        self.engine_kv_format,
                        token_major=True,
                    )
        yield

        # synchronize the last layer
        if sync:
            current_stream.wait_stream(self.load_stream)

        # free the buffer memory
        if tmp_gpu_buffer_obj is not None:
            tmp_gpu_buffer_obj.ref_count_down()

        logger.debug("Finished loading all %d layers.", self.num_layers)
        yield

    @_lmcache_nvtx_annotate
    def batched_from_gpu(
        self,
        memory_objs: Union[List[List[MemoryObj]]],
        starts: List[int],
        ends: List[int],
        **kwargs,
    ):
        """
        This function is a generator that moves the KV cache from the paged GPU
        memory to the memory objects. The first iteration will prepare some
        related metadata and initiate the transfer in the first layer. In each
        of the following iterations, it will first wait until the storing of
        previous layer finishes, and then initiate string the KV cache of the
        current layer one. The storing process of the KV cache is paged GPU
        memory -> GPU buffer -> memory objects. The last iteration simply waits
        for the last layer to finish.
        In total, this the generator will yield num_layers + 1 times.

        :param memory_objs: The memory objects to store the KV cache. The first
            dimension is the number of layers, and the second dimension is the
            number of memory objects (i.e., number of chunks) for each layer.

        :param starts: The starting indices of the KV cache in the corresponding
            token sequence.

        :param ends: The ending indices of the KV cache in the corresponding
            token sequence.

        :raises ValueError: If 'slot_mapping' is not provided in kwargs.
        """

        self.initialize_kvcaches_ptr(**kwargs)
        assert self.kvcaches is not None, (
            "kvcaches should be provided in kwargs or initialized beforehand."
        )

        if "slot_mapping" not in kwargs:
            raise ValueError("'slot_mapping' should be provided in kwargs.")

        if "sync" not in kwargs:
            raise ValueError("'sync' should be provided in kwargs.")

        slot_mapping: torch.Tensor = kwargs["slot_mapping"]
        sync: bool = kwargs["sync"]

        self._lazy_initialize_buffer(self.kvcaches)

        slot_mapping_chunks = []
        for start, end in zip(starts, ends, strict=False):
            slot_mapping_chunks.append(slot_mapping[start:end])

        slot_mapping_full = torch.cat(slot_mapping_chunks, dim=0)

        num_tokens = len(slot_mapping_full)

        tmp_gpu_buffer_obj: Optional[MemoryObj] = None
        if self.use_gpu:
            buffer_shape = self.get_shape(num_tokens)
            assert self.gpu_buffer_allocator is not None
            tmp_gpu_buffer_obj = self.gpu_buffer_allocator.allocate(
                buffer_shape, self.dtype, MemoryFormat.KV_T2D
            )
            assert tmp_gpu_buffer_obj is not None, (
                "Failed to allocate GPU buffer in GPUConnector"
            )
            assert tmp_gpu_buffer_obj.tensor is not None

        offset = starts[0]
        current_stream = torch.xpu.current_stream()

        for layer_id in range(self.num_layers):
            memory_objs_layer = memory_objs[layer_id]
            # kvcaches -> gpu_buffer -> memobj
            with torch.xpu.stream(self.store_stream):
                self.store_stream.wait_stream(current_stream)
                if self.use_gpu:
                    device_ops.single_layer_kv_transfer(
                        tmp_gpu_buffer_obj.tensor,
                        self.kvcaches[layer_id],
                        slot_mapping_full,
                        lmcache_native.TransferDirection.D2H,
                        self.engine_kv_format,
                        token_major=True,
                    )
                for start, end, memory_obj in zip(
                    starts, ends, memory_objs_layer, strict=False
                ):
                    assert memory_obj.tensor is not None
                    if self.use_gpu:
                        memory_obj.tensor.copy_(
                            tmp_gpu_buffer_obj.tensor[start - offset : end - offset],
                            non_blocking=True,
                        )
                    else:
                        device_ops.single_layer_kv_transfer(
                            memory_obj.tensor,
                            self.kvcaches[layer_id],
                            slot_mapping[start:end],
                            lmcache_native.TransferDirection.D2H,
                            self.engine_kv_format,
                            token_major=True,
                        )
                    # Set metadata format
                    if self.use_mla:
                        memory_obj.metadata.fmt = MemoryFormat.KV_MLA_FMT

            yield
            if sync:
                self.store_stream.synchronize()
            logger.debug("Finished offloading layer %d", layer_id)

        # free the buffer memory
        if tmp_gpu_buffer_obj is not None:
            tmp_gpu_buffer_obj.ref_count_down()

        yield

    def get_shape(self, num_tokens: int) -> torch.Size:
        if self.use_mla:
            # MLA format: [num_tokens, hidden_dim_size]
            return torch.Size([num_tokens, self.hidden_dim_size])
        else:
            # Standard format: [num_tokens, 2, hidden_dim_size]
            return torch.Size([num_tokens, 2, self.hidden_dim_size])
