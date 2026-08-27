# denoize C ABI v1

`denoize-c` is the stable native boundary for finite and incremental scalar
processing. Include `include/denoize.h`, link the platform static or shared
library, initialize every versioned structure, create one processor per stream,
then call `process` zero or more times followed by `finish`.

The processor is owned by its creating thread. The separately allocated cancel
token may be used from another thread and performs one atomic store without
waiting. Cancellation is observed between calls, so callers must keep
`max_frames_per_call` small enough for their responsiveness target. This ABI is
incremental but does not promise hard-real-time allocation behavior; live hosts
must run it on a worker and communicate with the audio callback through bounded
preallocated queues.

Token destruction is not concurrent with token use: the owner must first join
or otherwise synchronize every cancellation/reset caller. Destroying a live
opaque handle twice, or using it after destruction, is outside the C contract.

No function returns borrowed Rust memory, an enum layout, or a Rust error. Audio
and diagnostic buffers are caller-owned. Diagnostic strings are copied and
always report the required NUL-inclusive length. Unknown versions, flags,
nonzero reserved fields, invalid resource bounds, and use from the wrong thread
fail closed.

Only the classical scalar backend is available through ABI v1. An unavailable
backend is never silently replaced with a different recipe.
