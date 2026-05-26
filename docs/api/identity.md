# canvas_engineering.identity

Persistent object identity for canvas regions. Attach an `IdentitySpec`
to a `RegionProgram` to have the dispatcher auto-instantiate a
`SlotBindingModule` for that region: cross-attention from a fixed bank
of identity slots into the region's observations, with optional
birth/death gating for variable-count tracklets.

::: canvas_engineering.identity.IdentitySpec

::: canvas_engineering.identity.SlotBindingModule
