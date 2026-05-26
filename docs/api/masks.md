# canvas_engineering.masks

Attention mask specifications and utilities. Attach a `MaskSpec` to a
`ConnectionProgram.mask_spec` to control the sparsity pattern of
attention between two regions. The dispatcher pre-computes
`mask_to_index_pairs(...)` at construction time and iterates the
resulting (`src_idx`, `dst_idx`) pairs in its dense branch.

::: canvas_engineering.masks.MaskSpec

::: canvas_engineering.masks.Rect

::: canvas_engineering.masks.rect_cover

::: canvas_engineering.masks.mask_to_index_pairs
