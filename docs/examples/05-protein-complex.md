# Example 05: Protein Binding Affinity

Binding affinity prediction from sequence. The type hierarchy (chain → interface → complex) with matched-field cross-chain connectivity learns binding affinity better than a flat model.

**Source**: [`examples/05_protein_complex.py`](https://github.com/JacobFV/canvas-engineering/blob/main/examples/05_protein_complex.py) *(coming soon)*

## What it demonstrates

- **Matched-field cross-chain connectivity** — receptor chain fields attend to ligand chain fields of the same semantic type
- **Interface region** — explicit latent region for the binding pocket, wired to both chains
- **Hierarchical loss** — per-residue contact prediction + complex-level affinity
- **Transfer distance** — semantic type embeddings predict adapter cost between protein families

## Type hierarchy

```python
@dataclass
class ProteinChain:
    sequence: Field = Field(4, 8, is_output=False)   # residue embeddings
    structure: Field = Field(4, 8)                    # secondary structure
    surface: Field = Field(2, 4)                      # solvent-accessible surface

@dataclass
class Complex:
    receptor: ProteinChain = field(default_factory=ProteinChain)
    ligand: ProteinChain = field(default_factory=ProteinChain)
    interface: Field = Field(4, 8)                    # binding pocket
    affinity: Field = Field(1, 1, loss_weight=5.0)   # pKd prediction
```

## Key insight

The `interface` region acts as a learned aggregation of cross-chain information — analogous to a pointer in the type system. Its latent content can be inspected to understand what structural features drive affinity.

!!! note "Task spec"
    Full implementation details in [`examples/tasks/05_protein_complex.md`](https://github.com/JacobFV/canvas-engineering/blob/main/examples/tasks/05_protein_complex.md).
