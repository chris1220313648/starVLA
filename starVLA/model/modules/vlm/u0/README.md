Native UNIS model/config/tokenizer copied from Xiaomi-Robotics-U0's
`training/src/wm_fsdp/{models,tokenizers}` at commit
`1ae7fe3c32bb0fd251cb2f4e0ab55944daf92285` on 2026-09-10. Original Apache-2.0
notices are retained in the source files. The IBQ implementation and weights
are shared with starVLA's existing Emu3.5 integration.

U0Fast explicitly supports eager and FlashAttention 2. The upstream SDPA
implementation omits Q/K normalization and is not enabled by this adapter.
