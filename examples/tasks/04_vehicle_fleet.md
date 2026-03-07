# 04: Autonomous Vehicle Fleet — Cooperative Trajectory Prediction

## Purpose
Multi-agent coordination. Show that ring topology between vehicles and
interleaved layout create measurable advantages for trajectory prediction.

## Data
Synthetic but physically grounded:
- N vehicles on a 2D road with lanes
- Each vehicle has position, velocity, heading
- Vehicles must avoid collisions and maintain lane discipline
- Generate trajectories with a social force model (repulsion + lane attraction)

## Type hierarchy
```
Vehicle:
  position: Field(1, 4)              — x, y, vx, vy
  heading: Field(1, 2)               — sin(theta), cos(theta)
  lane_context: Field(1, 4, is_output=False) — nearest lane info
  intent: Field(2, 4)                — latent driving intent
  trajectory: Field(4, 4, loss_weight=4.0)   — predicted future 2s

FleetScene:
  traffic_state: Field(4, 4)         — global scene summary
  vehicles: list[Vehicle]            — 4-8 vehicles per scene
```

## Training
- Behavioral cloning on social-force trajectories
- But also: **contrastive loss on intent field** — vehicles with similar
  future trajectories should have similar intents, vehicles about to
  diverge should have different intents. This is learned structure in
  the intent field, not just regression.
- Compare: isolated vehicles vs ring connectivity vs dense connectivity

## Visualization
`assets/examples/04_fleet.png`:
(a) Bird's-eye view of a scene with predicted vs true trajectories
(b) Intent field similarity matrix (vehicles x vehicles) — does ring
    topology create meaningful intent clustering?
(c) Training curves: isolated vs ring vs dense
(d) Collision rate and trajectory error comparison
(e) Attention weights between vehicles — who looks at whom?

## Key message
"Ring connectivity between vehicles isn't just a topology choice — it
creates structured intent representations where nearby vehicles learn
correlated plans. The contrastive loss on the intent field turns an
SFT problem into a representation learning problem."
