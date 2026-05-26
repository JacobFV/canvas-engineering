# canvas_engineering.dispatch

Per-connection attention dispatch. Routes each topology connection to
its resolved attention function and combines per-edge contributions
according to `ConnectionProgram.write_mode` (`add` / `replace` /
`gate`). When a `CanvasProgram` is attached, the dispatcher also
evaluates per-edge `trigger` expressions, honors per-operator defaults
from `OPERATOR_DEFAULTS`, applies `MaskSpec` sparsity patterns, runs
`SlotBindingModule` for regions with an `IdentitySpec`, and lets a
`CortexRegistry` override the attention fn for intra-cortex edges with
the cortex's declared `local_backend`.

::: canvas_engineering.dispatch.AttentionDispatcher
