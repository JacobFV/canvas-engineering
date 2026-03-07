# 05: Protein Binding Affinity from Sequence

## Purpose
Show canvas types on a molecular biology problem. Demonstrate that the
type hierarchy (chain -> interface -> complex) with matched-field
cross-chain connectivity learns binding affinity better than a flat model.

## Data
Synthetic protein-like sequences:
- Generate random "amino acid" sequences (20-letter alphabet, length 50-100)
- Binding affinity is a function of:
  - Sequence similarity at specific "binding site" positions (positions 20-30)
  - Complementarity (charge matching: + binds -, hydrophobic binds hydrophobic)
  - A nonlinear interaction term
- This is synthetic but captures the essential structure of real binding

## Type hierarchy
```
Chain:
  sequence: Field(8, 8)              — per-residue embeddings
  structure: Field(4, 4)             — secondary structure prediction
  binding_site: Field(4, 4, loss_weight=3.0) — interface residues

Complex:
  interaction: Field(4, 4, loss_weight=4.0) — inter-chain interaction
  affinity: Field(1, 1, loss_weight=5.0)    — binding affinity scalar
  chains: list[Chain]                        — 2 chains
```

## Training
- Multi-task: predict per-residue structure + binding site location + affinity
- **Mutual information maximization** between chain binding_site fields:
  the two chains' binding sites should have high MI (they co-evolve)
- Compare: flat (all positions in one region) vs structured canvas

## Visualization
`assets/examples/05_protein.png`:
(a) Canvas layout with chains colored differently
(b) Cross-chain attention heatmap (which residues attend across chains?)
(c) Binding affinity: predicted vs true (scatter plot, both models)
(d) Binding site attention: does the model learn to focus on positions 20-30?
(e) MI between chain binding sites over training

## Key message
"Matched-field connectivity between chains isn't hand-engineering —
it's declaring the structural prior that chains interact through
corresponding interface residues. The model fills in the details."
