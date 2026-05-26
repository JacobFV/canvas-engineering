# canvas_engineering.clock_ir

Composable AST for region firing rules. `ClockExpr` nodes can be
attached to a `ClockSpec` via the `expr=` field; when set, the
expression supersedes the flat `mode` / `period` / `event_source`
fields. The full IR serializes to a plain `dict` for JSON persistence.

## Base

::: canvas_engineering.clock_ir.ClockExpr

::: canvas_engineering.clock_ir.ClockContext

## Leaves

::: canvas_engineering.clock_ir.Periodic

::: canvas_engineering.clock_ir.OnEvent

::: canvas_engineering.clock_ir.BoundaryExpr

## Combinators and decorators

::: canvas_engineering.clock_ir.And

::: canvas_engineering.clock_ir.Or

::: canvas_engineering.clock_ir.Not

::: canvas_engineering.clock_ir.Cooldown

::: canvas_engineering.clock_ir.MaxSilence

## Sugar

::: canvas_engineering.clock_ir.periodic

::: canvas_engineering.clock_ir.on

::: canvas_engineering.clock_ir.boundary
