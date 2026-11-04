XPU
===

In-process V2 packed KV caches
----------------------------------------

The in-process XPU V2 connector supports per-layer packed NHD
``[num_blocks, block_size, num_heads, 2 * head_size]`` and HND
``[num_blocks, num_heads, block_size, 2 * head_size]`` caches. Each head row
contains K followed by V. Stores unpack those rows into LMCache's existing
two-plane ``KV_2LTD`` chunks; retrieval packs them back on XPU.

Select V2 with the existing configuration:

.. code-block:: yaml

   use_layerwise: false
   use_gpu_connector_v3: false

Engine integrations must forward the physical layout hint, for example
``{"kv_layout": "HND"}`` or ``{"kv_layout": "BLHNC"}``. Dimension sizes alone
cannot distinguish NHD from HND. Blocks-first ``BLNHC`` and ``BLHNC`` views
and padding between packed blocks are supported; padding inside a block is
rejected. V2 requires uniform layer geometry, dtype and packed block stride.
Legacy NHD separate-K/V layouts and tightly stored MLA caches retain their
existing representation.

Retrieval preserves slots already covered by ``vllm_cached_tokens``.
Negative slot mappings preserve the destination in both directions, including
stores through the XPU staging buffer.

Packed V3 initialization and representation, layerwise transfer and blending
have separate requirements and are outside V2 packed-layout support.
