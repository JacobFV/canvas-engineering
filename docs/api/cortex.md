# canvas_engineering.cortex

Named spatial zones (cortices) for grouping regions that share a
preferred local attention backend. When the `AttentionDispatcher` is
constructed with a `CortexRegistry`, intra-cortex edges substitute the
cortex's declared `local_backend` for the resolved attention fn.

::: canvas_engineering.cortex.CortexSpec

::: canvas_engineering.cortex.CortexRegistry
