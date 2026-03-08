# Example 05: Protein Folding Complex

4-chain protein complex binding affinity prediction. Structured canvas with matched-field cross-chain connectivity vs a flat baseline — the type hierarchy (chain → interface → complex) with `array_element="matched_fields"` learns binding affinity and stability prediction.

**Source**: [`examples/05_protein_folding_complex.py`](https://github.com/JacobFV/canvas-engineering/blob/main/examples/05_protein_folding_complex.py)

## Results

<p align="center">
  <img src="https://raw.githubusercontent.com/JacobFV/canvas-engineering/main/assets/examples/05_protein.png" alt="Example 05 results" width="100%">
</p>

<p align="center">
  <img src="https://raw.githubusercontent.com/JacobFV/canvas-engineering/main/assets/examples/05_protein.gif" alt="Example 05 protein binding animation" width="100%">
</p>

<video width="100%" controls>
  <source src="https://raw.githubusercontent.com/JacobFV/canvas-engineering/main/assets/examples/05_protein.mp4" type="video/mp4">
</video>

**4×4 panel figure**: Chain conformations, binding site contact maps, affinity predictions, stability analysis, secondary structure, topology diagrams, and training curves.

**Animation**: 4-chain protein complex docking simulation — free chains approaching, initial docking, conformational tightening, and stable complex formation with charge interactions, hydrogen bonds, and binding energy readout.

## Type hierarchy

```python
@dataclass
class Chain:
    sequence: Field = Field(4, 4)
    structure: Field = Field(2, 2)
    binding_site: Field = Field(2, 4, loss_weight=3.0,
                                semantic_type="protein binding interface")
    contacts: Field = Field(2, 4, loss_weight=2.0)

@dataclass
class Complex:
    interaction: Field = Field(2, 4, loss_weight=4.0)
    affinity: Field = Field(1, 1, loss_weight=5.0)
    stability: Field = Field(1, 2, loss_weight=3.0)
    chains: list  # 4 chains: heavy, light, antigen, cofactor
```

Flat baseline for comparison:

```python
@dataclass
class FlatComplex:
    all_residues: Field = Field(8, 8)
    structure: Field = Field(4, 2)
    affinity: Field = Field(1, 1, loss_weight=5.0)
```

## Connectivity

```python
# Structured — matched-field cross-chain attention
bound_structured = compile_schema(
    Complex(chains=[Chain(), Chain(), Chain(), Chain()]),
    T=1, H=16, W=16, d_model=48,
    connectivity=ConnectivityPolicy(
        intra="dense",
        parent_child="hub_spoke",
        array_element="matched_fields",
    ),
    layout_strategy=LayoutStrategy.INTERLEAVED,
)
# 19 fields, 256 positions, 217 connections

# Flat baseline
bound_flat = compile_schema(
    FlatComplex(), T=1, H=16, W=16, d_model=48,
    connectivity=ConnectivityPolicy(intra="dense"),
)
# 3 fields, 256 positions, 9 connections
```

## Key insight

`array_element="matched_fields"` connects binding site fields across chains — receptor chain binding sites attend to ligand chain binding sites of the same semantic type. The `interaction` region acts as a learned aggregation of cross-chain information.

## Run it

```bash
python examples/05_protein_folding_complex.py
# Generates: assets/examples/05_protein.{png,gif,mp4}
```
