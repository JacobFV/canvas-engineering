# canvas_engineering.compiler

Program compilation: lower a CanvasProgram to a deploy-ready execution
plan. `ProgramCompiler.compile()` accepts an optional runtime
`torch.nn.Module` and actually materializes the compile modes —
freezing parameters, replacing them with same-valued buffers for
`constant` regions, and serializing `state_dict` to disk for `export`
regions.

::: canvas_engineering.compiler.CompiledProgram

::: canvas_engineering.compiler.ProgramCompiler
