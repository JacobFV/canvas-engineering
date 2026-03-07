"""Protein complex: multi-chain structure prediction with binding site focus.

Model a protein-protein interaction complex as a canvas type hierarchy.
Each protein chain has:
- Residue-level features (per-residue embeddings on the canvas)
- Secondary structure state (helix/sheet/coil as latent)
- A binding interface region with elevated loss weight

The complex-level type adds:
- An interaction field representing the interface between chains
- Solvent accessibility context (input-only, from DSSP or similar)

This mirrors how AlphaFold-Multimer and RoseTTAFold-AllAtom work internally,
but expressed declaratively. The key insight: the binding interface gets
higher loss weight because that's where drug design cares about accuracy.

Layout strategy: INTERLEAVED — residue features from both chains are
spatially adjacent, so the model naturally learns inter-chain correlations
through spatial attention priors.
"""

from dataclasses import dataclass, field as dc_field
from canvas_engineering import Field, compile_schema, ConnectivityPolicy, LayoutStrategy


@dataclass
class ProteinChain:
    """One protein chain in a complex."""
    residues: Field = Field(16, 16)       # 256 positions: per-residue features
    secondary_structure: Field = Field(4, 4)  # 16 positions: SS state
    binding_interface: Field = Field(4, 8,    # 32 positions: interface residues
                                    loss_weight=3.0,  # drug design cares here
                                    semantic_type="protein binding interface residues")


@dataclass
class ProteinComplex:
    """Multi-chain protein complex with interaction modeling."""
    # Complex-level fields
    interaction: Field = Field(8, 8,     # 64 positions: inter-chain interface
                               loss_weight=4.0,
                               attn="cross_attention",
                               semantic_type="protein-protein interaction interface")

    solvent: Field = Field(4, 4,         # 16 positions: solvent accessibility
                           is_output=False,  # input-only context
                           semantic_type="DSSP solvent accessibility")

    # The chains
    chains: list = dc_field(default_factory=list)


# --- Antibody-antigen complex (2 chains) ---
antibody_antigen = ProteinComplex(
    chains=[
        ProteinChain(
            residues=Field(16, 16),       # antibody: standard size
            binding_interface=Field(8, 8, # larger interface for CDR loops
                                   loss_weight=5.0,
                                   semantic_type="antibody CDR loop binding residues"),
        ),
        ProteinChain(
            residues=Field(12, 12),       # antigen: different size
            binding_interface=Field(4, 8,
                                   loss_weight=3.0,
                                   semantic_type="antigen epitope residues"),
        ),
    ],
)

bound = compile_schema(
    antibody_antigen, T=1, H=64, W=64, d_model=384,
    connectivity=ConnectivityPolicy(
        intra="dense",                    # within each chain: full attention
        parent_child="hub_spoke",         # complex interaction <-> all chain fields
        array_element="matched_fields",   # chain[0].binding <-> chain[1].binding
        temporal="dense",                 # no temporal (single structure)
    ),
    layout_strategy=LayoutStrategy.INTERLEAVED,
)

print("=== Antibody-Antigen Complex ===")
print(bound.summary())

# The binding interface gets most of the loss attention
weights = bound.layout.loss_weight_mask("cpu")
total = weights.sum().item()
print("\nLoss weight distribution:")
for name, bf in sorted(bound.fields.items()):
    indices = bf.indices()
    w = sum(weights[i].item() for i in indices)
    if w > 0:
        print(f"  {name}: {w/total*100:.1f}%"
              f"  (loss_weight={bf.spec.loss_weight})")

# Cross-chain connections via matched_fields
print("\nCross-chain connections (binding interface interactions):")
for c in bound.topology.connections:
    if ("chains[0]" in c.src and "chains[1]" in c.dst):
        print(f"  {c.src} -> {c.dst}")


# --- Trimer (3-chain complex, e.g., viral spike protein) ---
print("\n\n=== Viral Spike Protein Trimer ===")
spike = ProteinComplex(
    interaction=Field(12, 12,          # larger interface for 3-way
                      loss_weight=4.0,
                      semantic_type="trimeric spike protein interface"),
    chains=[
        ProteinChain(residues=Field(12, 12),
                     semantic_type="spike protomer") for _ in range(3)
    ],
)

bound3 = compile_schema(
    spike, T=1, H=64, W=64, d_model=384,
    connectivity=ConnectivityPolicy(
        intra="dense",
        parent_child="hub_spoke",
        array_element="ring",            # protomers form a ring (3-fold symmetry)
    ),
)

print(f"Protomers: 3")
print(f"Total fields: {len(bound3.field_names)}")
ring_conns = [c for c in bound3.topology.connections
              if "chains[" in c.src and "chains[" in c.dst
              and c.src.split("]")[0] != c.dst.split("]")[0]]
print(f"Inter-protomer connections: {len(ring_conns)}")
print("Ring topology (3-fold symmetry):")
pairs = set()
for c in ring_conns:
    a, b = c.src.split("]")[0]+"]", c.dst.split("]")[0]+"]"
    pairs.add(tuple(sorted([a, b])))
for a, b in sorted(pairs):
    print(f"  {a} <-> {b}")
