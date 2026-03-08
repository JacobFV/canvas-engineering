"""Protein Complex: binding affinity from synthetic sequences with canvas types.

Synthetic protein-like sequences where binding affinity depends on
complementarity at specific binding-site positions. Two models:
  1. Flat baseline: all residues in one big region
  2. Canvas structured: chain -> binding_site -> complex interaction
     with matched-field cross-chain connectivity

Multi-task: predict per-residue structure + binding site + affinity.
Mutual information maximization between chain binding sites.

Run:  python examples/05_protein_folding_complex.py
Out:  assets/examples/05_protein.png
"""

import os, math
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from dataclasses import dataclass, field as dc_field

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

from canvas_engineering import Field, compile_schema, ConnectivityPolicy, LayoutStrategy

ASSETS = os.path.join(os.path.dirname(os.path.dirname(__file__)), "assets", "examples")
os.makedirs(ASSETS, exist_ok=True)

torch.manual_seed(42)
np.random.seed(42)

SEQ_LEN = 50        # residues per chain
ALPHABET = 20       # amino acid types
BINDING_START = 20   # binding site starts at residue 20
BINDING_END = 30     # binding site ends at residue 30


# ── 1. Type declarations ─────────────────────────────────────────────

@dataclass
class Chain:
    sequence: Field = Field(4, 4)                          # 16 pos: per-residue embeddings
    structure: Field = Field(2, 2)                         # 4 pos: secondary structure
    binding_site: Field = Field(2, 4, loss_weight=3.0,
                                semantic_type="protein binding interface residues")

@dataclass
class Complex:
    interaction: Field = Field(2, 4, loss_weight=4.0,
                               semantic_type="protein-protein interaction interface")
    affinity: Field = Field(1, 1, loss_weight=5.0)         # binding affinity scalar
    chains: list = dc_field(default_factory=list)


bound_structured = compile_schema(
    Complex(chains=[Chain(), Chain()]),
    T=1, H=16, W=16, d_model=48,
    connectivity=ConnectivityPolicy(
        intra="dense",
        parent_child="hub_spoke",
        array_element="matched_fields",  # chain[0].binding <-> chain[1].binding
    ),
    layout_strategy=LayoutStrategy.INTERLEAVED,
)

# Flat: everything in one big region (no chain separation)
@dataclass
class FlatComplex:
    all_residues: Field = Field(8, 8)                  # 64 pos for both chains
    structure: Field = Field(4, 2)                     # 8 pos
    affinity: Field = Field(1, 1, loss_weight=5.0)

bound_flat = compile_schema(
    FlatComplex(), T=1, H=16, W=16, d_model=48,
    connectivity=ConnectivityPolicy(intra="dense"),
)

print("Structured:", bound_structured.summary())
print("Flat:", bound_flat.summary())


# ── 2. Synthetic data: protein binding ───────────────────────────────

# Define amino acid properties (charge, hydrophobicity, size)
AA_PROPERTIES = torch.randn(ALPHABET, 3)
AA_PROPERTIES[:, 0] = torch.linspace(-1, 1, ALPHABET)   # charge
AA_PROPERTIES[:, 1] = torch.sin(torch.arange(ALPHABET).float() * 0.5)  # hydrophobicity
AA_PROPERTIES[:, 2] = torch.linspace(0.5, 1.5, ALPHABET)  # size


def generate_protein_data(n_samples=4096):
    """Generate synthetic protein pairs with binding affinity.

    Binding rules:
    1. Charge complementarity at binding site (+ binds -)
    2. Hydrophobic matching at binding site (similar hydrophobicity binds)
    3. Nonlinear interaction term (size compatibility)
    """
    # Random sequences
    seq_a = torch.randint(0, ALPHABET, (n_samples, SEQ_LEN))
    seq_b = torch.randint(0, ALPHABET, (n_samples, SEQ_LEN))

    # One-hot encode
    onehot_a = F.one_hot(seq_a, ALPHABET).float()  # (N, L, 20)
    onehot_b = F.one_hot(seq_b, ALPHABET).float()

    # Properties at binding site
    props_a = AA_PROPERTIES[seq_a[:, BINDING_START:BINDING_END]]  # (N, 10, 3)
    props_b = AA_PROPERTIES[seq_b[:, BINDING_START:BINDING_END]]

    # Binding affinity components
    # 1. Charge complementarity: opposite charges attract
    charge_comp = -(props_a[:, :, 0] * props_b[:, :, 0]).mean(dim=1)  # (N,)

    # 2. Hydrophobic matching: similar hydrophobicity binds
    hydro_match = -((props_a[:, :, 1] - props_b[:, :, 1]) ** 2).mean(dim=1)

    # 3. Size compatibility: similar sizes fit better
    size_compat = -((props_a[:, :, 2] - props_b[:, :, 2]) ** 2).mean(dim=1)

    # Nonlinear interaction: tanh of combined effect
    affinity = torch.tanh(charge_comp + 0.5 * hydro_match + 0.3 * size_compat)
    affinity = affinity + torch.randn_like(affinity) * 0.05  # noise

    # Secondary structure: periodic pattern based on sequence
    # (helix=0, sheet=1, coil=2) — simplified
    structure_a = (seq_a % 3).float()  # (N, L)
    structure_b = (seq_b % 3).float()

    # Binding site labels: 1 if in binding region, 0 otherwise
    binding_mask = torch.zeros(SEQ_LEN)
    binding_mask[BINDING_START:BINDING_END] = 1.0

    # Pool sequences into fixed-size features for canvas
    # Average pool to match canvas field sizes
    def pool_seq(onehot, n_out=16):
        """Pool (N, L, 20) -> (N, n_out) via learned-like projection."""
        # Use mean over chunks
        chunk = SEQ_LEN // n_out
        pooled = []
        for i in range(n_out):
            start = i * chunk
            end = min(start + chunk, SEQ_LEN)
            pooled.append(onehot[:, start:end].mean(dim=1).mean(dim=-1))
        return torch.stack(pooled, dim=1)

    seq_feat_a = pool_seq(onehot_a, 16)
    seq_feat_b = pool_seq(onehot_b, 16)

    # Binding site features (properties at binding positions)
    bind_feat_a = props_a.reshape(n_samples, -1)[:, :8]  # (N, 8) — first 8 dims
    bind_feat_b = props_b.reshape(n_samples, -1)[:, :8]

    # Structure features
    struct_feat_a = pool_seq(structure_a.unsqueeze(-1).expand(-1, -1, 1), 4).squeeze(-1) if structure_a.dim() == 2 else structure_a[:, :4]
    struct_feat_b = pool_seq(structure_b.unsqueeze(-1).expand(-1, -1, 1), 4).squeeze(-1) if structure_b.dim() == 2 else structure_b[:, :4]

    # Simple structure pooling
    struct_a = torch.zeros(n_samples, 4)
    struct_b = torch.zeros(n_samples, 4)
    chunk = SEQ_LEN // 4
    for i in range(4):
        struct_a[:, i] = structure_a[:, i*chunk:(i+1)*chunk].float().mean(dim=1)
        struct_b[:, i] = structure_b[:, i*chunk:(i+1)*chunk].float().mean(dim=1)

    return {
        'seq_a': seq_feat_a, 'seq_b': seq_feat_b,       # (N, 16) each
        'struct_a': struct_a, 'struct_b': struct_b,       # (N, 4) each
        'bind_a': bind_feat_a, 'bind_b': bind_feat_b,     # (N, 8) each
        'affinity': affinity,                              # (N,)
        'raw_seq_a': seq_a, 'raw_seq_b': seq_b,           # for analysis
        'props_a': props_a, 'props_b': props_b,           # binding site props
    }


print("\nGenerating synthetic protein data...")
train_data = generate_protein_data(4096)
val_data = generate_protein_data(1024)
print(f"  Affinity range: [{train_data['affinity'].min():.2f}, {train_data['affinity'].max():.2f}]")


# ── 3. Models ────────────────────────────────────────────────────────

class StructuredProteinModel(nn.Module):
    """Canvas model with per-chain structure and cross-chain interaction."""

    def __init__(self, bound, d=48, nhead=4):
        super().__init__()
        self.bound = bound
        self.d = d
        N = bound.layout.num_positions

        self.pos_emb = nn.Parameter(torch.randn(1, N, d) * 0.02)
        layer = nn.TransformerEncoderLayer(
            d_model=d, nhead=nhead, dim_feedforward=128,
            dropout=0.0, batch_first=True)
        self.encoder = nn.TransformerEncoder(layer, num_layers=2)
        mask = bound.topology.to_additive_mask(bound.layout)
        self.register_buffer('mask', mask)

        # Input projections per field
        def field_size(name):
            return len(bound.layout.region_indices(name))

        self.seq_proj_a = nn.Linear(16, field_size('chains[0].sequence') * d)
        self.seq_proj_b = nn.Linear(16, field_size('chains[1].sequence') * d)
        self.bind_proj_a = nn.Linear(8, field_size('chains[0].binding_site') * d)
        self.bind_proj_b = nn.Linear(8, field_size('chains[1].binding_site') * d)

        # Output heads
        aff_n = field_size('affinity')
        struct_n_a = field_size('chains[0].structure')
        struct_n_b = field_size('chains[1].structure')
        interact_n = field_size('interaction')

        self.affinity_head = nn.Linear(aff_n * d, 1)
        self.struct_head_a = nn.Linear(struct_n_a * d, 4)
        self.struct_head_b = nn.Linear(struct_n_b * d, 4)

        # For MI analysis
        self.bind_read_a = nn.Linear(field_size('chains[0].binding_site') * d, 16)
        self.bind_read_b = nn.Linear(field_size('chains[1].binding_site') * d, 16)

        self._field_sizes = {
            'seq_a': field_size('chains[0].sequence'),
            'seq_b': field_size('chains[1].sequence'),
            'bind_a': field_size('chains[0].binding_site'),
            'bind_b': field_size('chains[1].binding_site'),
            'struct_a': struct_n_a,
            'struct_b': struct_n_b,
            'aff': aff_n,
            'interact': interact_n,
        }

    def forward(self, data):
        B = data['seq_a'].shape[0]
        canvas = self.pos_emb.expand(B, -1, -1).clone()

        # Place chain features
        for chain_idx, suffix in [(0, 'a'), (1, 'b')]:
            seq_idx = self.bound.layout.region_indices(f'chains[{chain_idx}].sequence')
            bind_idx = self.bound.layout.region_indices(f'chains[{chain_idx}].binding_site')

            seq_proj = getattr(self, f'seq_proj_{suffix}')
            bind_proj = getattr(self, f'bind_proj_{suffix}')

            seq_emb = seq_proj(data[f'seq_{suffix}']).reshape(B, self._field_sizes[f'seq_{suffix}'], self.d)
            bind_emb = bind_proj(data[f'bind_{suffix}']).reshape(B, self._field_sizes[f'bind_{suffix}'], self.d)

            canvas[:, seq_idx] = canvas[:, seq_idx] + seq_emb
            canvas[:, bind_idx] = canvas[:, bind_idx] + bind_emb

        canvas = self.encoder(canvas, mask=self.mask)

        # Read outputs
        aff_idx = self.bound.layout.region_indices('affinity')
        struct_a_idx = self.bound.layout.region_indices('chains[0].structure')
        struct_b_idx = self.bound.layout.region_indices('chains[1].structure')
        bind_a_idx = self.bound.layout.region_indices('chains[0].binding_site')
        bind_b_idx = self.bound.layout.region_indices('chains[1].binding_site')

        affinity = self.affinity_head(canvas[:, aff_idx].reshape(B, -1)).squeeze(-1)
        struct_a = self.struct_head_a(canvas[:, struct_a_idx].reshape(B, -1))
        struct_b = self.struct_head_b(canvas[:, struct_b_idx].reshape(B, -1))

        # Binding site embeddings for MI
        bind_emb_a = self.bind_read_a(canvas[:, bind_a_idx].reshape(B, -1))
        bind_emb_b = self.bind_read_b(canvas[:, bind_b_idx].reshape(B, -1))

        return {
            'affinity': affinity,
            'struct_a': struct_a, 'struct_b': struct_b,
            'bind_emb_a': bind_emb_a, 'bind_emb_b': bind_emb_b,
        }


class FlatProteinModel(nn.Module):
    """Flat baseline: everything in one region."""

    def __init__(self, bound, d=48, nhead=4):
        super().__init__()
        self.bound = bound
        self.d = d
        N = bound.layout.num_positions

        self.pos_emb = nn.Parameter(torch.randn(1, N, d) * 0.02)
        layer = nn.TransformerEncoderLayer(
            d_model=d, nhead=nhead, dim_feedforward=128,
            dropout=0.0, batch_first=True)
        self.encoder = nn.TransformerEncoder(layer, num_layers=2)
        mask = bound.topology.to_additive_mask(bound.layout)
        self.register_buffer('mask', mask)

        res_n = len(bound.layout.region_indices('all_residues'))
        struct_n = len(bound.layout.region_indices('structure'))
        aff_n = len(bound.layout.region_indices('affinity'))

        # Input: concat both chains' features
        self.input_proj = nn.Linear(16 + 16 + 8 + 8, res_n * d)
        self.affinity_head = nn.Linear(aff_n * d, 1)
        self.struct_head = nn.Linear(struct_n * d, 8)  # both chains' structure

        self.res_n = res_n
        self.struct_n = struct_n
        self.aff_n = aff_n

    def forward(self, data):
        B = data['seq_a'].shape[0]
        canvas = self.pos_emb.expand(B, -1, -1).clone()

        # Concatenate everything
        all_in = torch.cat([data['seq_a'], data['seq_b'],
                            data['bind_a'], data['bind_b']], dim=-1)
        res_idx = self.bound.layout.region_indices('all_residues')
        res_emb = self.input_proj(all_in).reshape(B, self.res_n, self.d)
        canvas[:, res_idx] = canvas[:, res_idx] + res_emb

        canvas = self.encoder(canvas, mask=self.mask)

        aff_idx = self.bound.layout.region_indices('affinity')
        struct_idx = self.bound.layout.region_indices('structure')

        affinity = self.affinity_head(canvas[:, aff_idx].reshape(B, -1)).squeeze(-1)
        struct_all = self.struct_head(canvas[:, struct_idx].reshape(B, -1))

        return {
            'affinity': affinity,
            'struct_a': struct_all[:, :4], 'struct_b': struct_all[:, 4:],
            'bind_emb_a': None, 'bind_emb_b': None,
        }


# ── 4. Training ──────────────────────────────────────────────────────

def mi_loss(emb_a, emb_b):
    """Encourage high mutual information between binding site embeddings.

    Uses a simple contrastive approach: matched pairs should be more similar
    than random pairs.
    """
    B = emb_a.shape[0]
    a_norm = F.normalize(emb_a, dim=-1)
    b_norm = F.normalize(emb_b, dim=-1)

    # Positive: matched pairs (diagonal)
    pos = (a_norm * b_norm).sum(dim=-1)  # (B,)

    # Negative: random pairs
    perm = torch.randperm(B)
    neg = (a_norm * b_norm[perm]).sum(dim=-1)

    # InfoNCE-style
    loss = -torch.log(torch.exp(pos / 0.1) / (torch.exp(pos / 0.1) + torch.exp(neg / 0.1) + 1e-8))
    return loss.mean()


def train_protein(model, label, use_mi=False, n_epochs=400, bs=128):
    opt = torch.optim.AdamW(model.parameters(), lr=2e-3, weight_decay=1e-4)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, n_epochs)
    losses = []
    mi_losses_hist = []

    for ep in range(n_epochs):
        idx = torch.randint(0, len(train_data['affinity']), (bs,))
        batch = {k: v[idx] if isinstance(v, torch.Tensor) else v for k, v in train_data.items()}

        out = model(batch)

        # Multi-task loss
        aff_loss = F.mse_loss(out['affinity'], batch['affinity'])
        struct_loss = F.mse_loss(out['struct_a'], batch['struct_a']) + \
                      F.mse_loss(out['struct_b'], batch['struct_b'])

        loss = 5.0 * aff_loss + struct_loss

        mi_val = 0.0
        if use_mi and out['bind_emb_a'] is not None:
            mi_val = mi_loss(out['bind_emb_a'], out['bind_emb_b'])
            loss = loss + 0.5 * mi_val

        opt.zero_grad()
        loss.backward()
        opt.step()
        sched.step()
        losses.append(loss.item())
        mi_losses_hist.append(mi_val.item() if isinstance(mi_val, torch.Tensor) else mi_val)

        if ep % 200 == 0:
            print(f"  [{label}] ep {ep:3d}: loss={loss.item():.4f}, aff={aff_loss.item():.4f}")

    model.eval()
    with torch.no_grad():
        out_val = model(val_data)
        val_aff = F.mse_loss(out_val['affinity'], val_data['affinity']).item()
    print(f"  [{label}] val_affinity_mse={val_aff:.4f}")
    return model, losses, mi_losses_hist, val_aff


print("\nTraining flat model...")
flat_model = FlatProteinModel(bound_flat)
flat_model, flat_losses, flat_mi, flat_val = train_protein(flat_model, "flat", use_mi=False)

print("Training structured model...")
struct_model = StructuredProteinModel(bound_structured)
struct_model, struct_losses, struct_mi, struct_val = train_protein(struct_model, "structured", use_mi=True)


# ── 5. Visualization ────────────────────────────────────────────────

fig, axes = plt.subplots(2, 3, figsize=(16, 10), dpi=150)
fig.patch.set_facecolor('white')
fig.suptitle('Protein Complex: Structured Canvas vs Flat Baseline',
             fontsize=16, fontweight='bold', y=0.99)

CS, CF = '#4A90D9', '#E8734A'  # structured, flat

# (a) Canvas layout
ax = axes[0, 0]
ax.set_title('Structured Canvas Layout', fontsize=11, fontweight='bold')
H, W = bound_structured.layout.H, bound_structured.layout.W
grid = np.ones((H, W, 3)) * 0.95
field_colors = {
    'interaction': '#E74C3C',
    'affinity': '#F39C12',
    'sequence': '#3498DB',
    'structure': '#9B59B6',
    'binding_site': '#2ECC71',
}
for name, bf in bound_structured.fields.items():
    color_key = name.split('.')[-1] if '.' in name else name
    color = field_colors.get(color_key, '#BDC3C7')
    r, g, b = int(color[1:3], 16)/255, int(color[3:5], 16)/255, int(color[5:7], 16)/255
    h0, h1 = bf.spec.bounds[2], bf.spec.bounds[3]
    w0, w1 = bf.spec.bounds[4], bf.spec.bounds[5]
    grid[h0:h1, w0:w1] = [r, g, b]

ax.imshow(grid, aspect='equal', interpolation='nearest')
from matplotlib.patches import Patch
legend_elements = [Patch(facecolor=c, label=n) for n, c in field_colors.items()]
ax.legend(handles=legend_elements, fontsize=7, loc='lower right')
ax.set_xlabel('W'); ax.set_ylabel('H')

# (b) Cross-chain attention analysis
ax = axes[0, 1]
ax.set_title('Cross-Chain Connections', fontsize=11, fontweight='bold')
# Count connections by field type
from collections import defaultdict
conn_counts = defaultdict(int)
for c in bound_structured.topology.connections:
    if "chains[0]" in c.src and "chains[1]" in c.dst:
        src_field = c.src.split('.')[-1]
        dst_field = c.dst.split('.')[-1]
        conn_counts[f'{src_field}\n->\n{dst_field}'] += 1

if conn_counts:
    labels = list(conn_counts.keys())
    values = list(conn_counts.values())
    bars = ax.barh(range(len(labels)), values, color=CS, alpha=0.7)
    ax.set_yticks(range(len(labels)))
    ax.set_yticklabels(labels, fontsize=8)
    ax.set_xlabel('Number of Connections')
    for bar, val in zip(bars, values):
        ax.text(bar.get_width() + 0.3, bar.get_y() + bar.get_height()/2,
                str(val), va='center', fontsize=9, fontweight='bold')
else:
    ax.text(0.5, 0.5, 'matched_fields\ncross-chain', transform=ax.transAxes,
            ha='center', va='center', fontsize=12)
ax.grid(True, alpha=0.2, axis='x')

# (c) Binding affinity: predicted vs true
ax = axes[0, 2]
ax.set_title('Binding Affinity: Predicted vs True', fontsize=11, fontweight='bold')
struct_model.eval(); flat_model.eval()
with torch.no_grad():
    pred_struct = struct_model(val_data)['affinity'].numpy()
    pred_flat = flat_model(val_data)['affinity'].numpy()
true_aff = val_data['affinity'].numpy()
ax.scatter(true_aff, pred_struct, s=5, alpha=0.3, color=CS, label=f'structured (MSE={struct_val:.3f})')
ax.scatter(true_aff, pred_flat, s=5, alpha=0.3, color=CF, label=f'flat (MSE={flat_val:.3f})')
lims = [-1.5, 1.5]
ax.plot(lims, lims, 'k--', lw=1, alpha=0.4, label='perfect')
ax.set_xlim(lims); ax.set_ylim(lims)
ax.set_xlabel('True Affinity')
ax.set_ylabel('Predicted Affinity')
ax.legend(fontsize=8, markerscale=3)
ax.set_aspect('equal')
ax.grid(True, alpha=0.2)

# (d) Training curves
ax = axes[1, 0]
ax.set_title('Training Loss', fontsize=11, fontweight='bold')
w = 30
def smooth(a, w=w): return np.convolve(a, np.ones(w)/w, mode='valid')
ax.plot(smooth(struct_losses), color=CS, lw=2, label='structured')
ax.plot(smooth(flat_losses), color=CF, lw=2, label='flat')
ax.legend(fontsize=9)
ax.set_xlabel('Epoch'); ax.set_ylabel('Loss')
ax.grid(True, alpha=0.2)

# Winner
better = 'structured' if struct_val < flat_val else 'flat'
ratio = max(struct_val, flat_val) / max(min(struct_val, flat_val), 1e-8)
color = CS if struct_val < flat_val else CF
ax.text(0.98, 0.95, f'{better}: {ratio:.2f}x better',
        transform=ax.transAxes, ha='right', va='top', fontsize=10,
        fontweight='bold', color=color,
        bbox=dict(boxstyle='round,pad=0.3', facecolor=color, alpha=0.15))

# (e) MI loss over training
ax = axes[1, 1]
ax.set_title('Binding Site MI Loss (structured only)', fontsize=11, fontweight='bold')
if any(v > 0 for v in struct_mi):
    ax.plot(smooth(struct_mi), color='#2ECC71', lw=2)
    ax.set_xlabel('Epoch')
    ax.set_ylabel('MI Loss (lower = higher mutual info)')
    ax.grid(True, alpha=0.2)
else:
    ax.text(0.5, 0.5, 'No MI loss recorded', transform=ax.transAxes,
            ha='center', va='center')

# (f) Affinity error by true affinity (binned)
ax = axes[1, 2]
ax.set_title('Error by Affinity Strength', fontsize=11, fontweight='bold')
bins = np.linspace(-1, 1, 8)
for pred, color, label in [(pred_struct, CS, 'structured'), (pred_flat, CF, 'flat')]:
    bin_errors = []
    bin_centers = []
    for i in range(len(bins) - 1):
        mask = (true_aff >= bins[i]) & (true_aff < bins[i + 1])
        if mask.sum() > 5:
            err = np.abs(pred[mask] - true_aff[mask]).mean()
            bin_errors.append(err)
            bin_centers.append((bins[i] + bins[i + 1]) / 2)
    ax.plot(bin_centers, bin_errors, 'o-', color=color, lw=2, label=label, markersize=5)

ax.set_xlabel('True Affinity')
ax.set_ylabel('Mean Absolute Error')
ax.legend(fontsize=9)
ax.grid(True, alpha=0.2)

plt.tight_layout(rect=[0, 0, 1, 0.97])
path = os.path.join(ASSETS, "05_protein.png")
fig.savefig(path, bbox_inches='tight', facecolor='white', dpi=150)
plt.close()
print(f"\nSaved {path}")


# ── 6. Animation: high-fidelity protein binding ─────────────────────

import matplotlib.animation as animation
from matplotlib.collections import LineCollection

print("Generating protein binding animation...")

np.random.seed(42)
N_FRAMES = 120  # longer rollout

# Generate helical backbone geometry for each chain
def make_helix(n_residues, radius=2.5, pitch=0.35, center=np.array([0, 0])):
    """Generate a helical backbone (2D projection of 3D alpha-helix)."""
    t = np.linspace(0, n_residues * pitch, n_residues)
    x = radius * np.cos(t * 2.5) + center[0] + np.arange(n_residues) * 0.15
    y = radius * np.sin(t * 2.5) + center[1]
    return np.stack([x, y], axis=-1)

chain_a_base = make_helix(SEQ_LEN, radius=2.0, pitch=0.3, center=np.array([-4, 0]))
chain_b_base = make_helix(SEQ_LEN, radius=2.0, pitch=0.35, center=np.array([4, 0]))
# Mirror chain B so binding sites face each other
chain_b_base[:, 0] = -chain_b_base[:, 0] + 8

# Properties for coloring — hydrophobicity for surface, charge for interactions
charge_a = AA_PROPERTIES[train_data['raw_seq_a'][0].numpy(), 0].numpy()
charge_b = AA_PROPERTIES[train_data['raw_seq_b'][0].numpy(), 0].numpy()
hydro_a = AA_PROPERTIES[train_data['raw_seq_a'][0].numpy(), 1].numpy()
hydro_b = AA_PROPERTIES[train_data['raw_seq_b'][0].numpy(), 1].numpy()
size_a = AA_PROPERTIES[train_data['raw_seq_a'][0].numpy(), 2].numpy()
size_b = AA_PROPERTIES[train_data['raw_seq_b'][0].numpy(), 2].numpy()

# Sidechain directions (perpendicular to backbone, with some variation)
def sidechain_dirs(backbone, length_scale):
    dirs = np.zeros_like(backbone)
    for i in range(len(backbone)):
        if i < len(backbone) - 1:
            tang = backbone[i+1] - backbone[i]
        else:
            tang = backbone[i] - backbone[i-1]
        norm = np.array([-tang[1], tang[0]])
        norm = norm / (np.linalg.norm(norm) + 1e-8)
        # Alternate sides + noise
        side = 1 if i % 2 == 0 else -1
        dirs[i] = norm * side * length_scale[i] * 0.8
    return dirs

sc_dirs_a = sidechain_dirs(chain_a_base, size_a)
sc_dirs_b = sidechain_dirs(chain_b_base, size_b)

aff_val = train_data['affinity'][0].item()

fig_anim, ax_anim = plt.subplots(1, 1, figsize=(10, 7), dpi=120)
fig_anim.patch.set_facecolor('#080818')

# Precompute thermal noise for consistency across frames
rng = np.random.RandomState(42)
thermal_a = rng.randn(N_FRAMES, SEQ_LEN, 2) * 0.08
thermal_b = rng.randn(N_FRAMES, SEQ_LEN, 2) * 0.08

def animate_protein(frame):
    ax_anim.clear()
    ax_anim.set_facecolor('#080818')

    # Phase 1 (0-40): approach
    # Phase 2 (40-70): docking — binding sites lock
    # Phase 3 (70-100): conformational change — tighter binding
    # Phase 4 (100-120): stable complex with readouts

    if frame < 40:
        t = frame / 40.0
        # Smooth ease-in-out
        t = t * t * (3 - 2 * t)
        offset_a = np.array([-8 + 5 * t, 0])
        offset_b = np.array([8 - 5 * t, 0])
        bind_strength = 0.0
    elif frame < 70:
        t = (frame - 40) / 30.0
        offset_a = np.array([-3 - 0.5 * t, 0])
        offset_b = np.array([3 + 0.5 * t, 0])
        bind_strength = t
    elif frame < 100:
        t = (frame - 70) / 30.0
        offset_a = np.array([-3.5 + 0.3 * t, 0.2 * np.sin(t * np.pi)])
        offset_b = np.array([3.5 - 0.3 * t, -0.2 * np.sin(t * np.pi)])
        bind_strength = 1.0
    else:
        offset_a = np.array([-3.2, 0])
        offset_b = np.array([3.2, 0])
        bind_strength = 1.0

    # Add thermal motion (seeded for smoothness)
    pos_a = chain_a_base + offset_a + thermal_a[frame]
    pos_b = chain_b_base + offset_b + thermal_b[frame]

    # Sidechain endpoints
    sc_a = pos_a + sc_dirs_a + thermal_a[frame] * 0.3
    sc_b = pos_b + sc_dirs_b + thermal_b[frame] * 0.3

    # Draw water/solvent as faint dots
    if frame < 100:
        n_water = 80
        wx = rng.uniform(-14, 14, n_water)
        wy = rng.uniform(-9, 9, n_water)
        ax_anim.scatter(wx, wy, s=2, color='#1a2a4a', alpha=0.15, zorder=0)

    # Draw backbone with smooth spline-like segments
    for chain_pos, chain_color, alpha in [
        (pos_a, '#3498DB', 0.8), (pos_b, '#E74C3C', 0.8)
    ]:
        # Backbone as thick tube
        segments = np.array([[chain_pos[i], chain_pos[i+1]]
                             for i in range(len(chain_pos)-1)])
        lc = LineCollection(segments, colors=chain_color, linewidths=2.5,
                            alpha=alpha, zorder=2)
        ax_anim.add_collection(lc)

    # Draw sidechains as thin sticks
    for i in range(SEQ_LEN):
        ax_anim.plot([pos_a[i, 0], sc_a[i, 0]], [pos_a[i, 1], sc_a[i, 1]],
                     '-', color='#5DADE2', lw=0.6, alpha=0.4, zorder=2)
        ax_anim.plot([pos_b[i, 0], sc_b[i, 0]], [pos_b[i, 1], sc_b[i, 1]],
                     '-', color='#EC7063', lw=0.6, alpha=0.4, zorder=2)

    # Draw residues — size by sidechain size, color by charge
    for pos, charge, hydro, sizes, base_color in [
        (pos_a, charge_a, hydro_a, size_a, '#3498DB'),
        (pos_b, charge_b, hydro_b, size_b, '#E74C3C'),
    ]:
        # C-alpha atoms (backbone)
        ax_anim.scatter(pos[:, 0], pos[:, 1], s=sizes * 25,
                        c=charge, cmap='coolwarm', vmin=-1, vmax=1,
                        zorder=3, edgecolors=base_color, linewidths=0.3, alpha=0.85)

    # Highlight binding site with glow
    bind_a = pos_a[BINDING_START:BINDING_END]
    bind_b = pos_b[BINDING_START:BINDING_END]

    glow_alpha = 0.15 + 0.15 * np.sin(frame * 0.15)
    ax_anim.scatter(bind_a[:, 0], bind_a[:, 1], s=120, facecolors='none',
                     edgecolors='#2ECC71', lw=2, zorder=4, alpha=0.6 + glow_alpha)
    ax_anim.scatter(bind_b[:, 0], bind_b[:, 1], s=120, facecolors='none',
                     edgecolors='#2ECC71', lw=2, zorder=4, alpha=0.6 + glow_alpha)

    # Binding interactions — charge complementarity lines
    if bind_strength > 0:
        for i in range(BINDING_START, BINDING_END):
            for j in range(BINDING_START, BINDING_END):
                dist = np.sqrt(((pos_a[i] - pos_b[j])**2).sum())
                charge_interact = -charge_a[i] * charge_b[j]  # opposite charges attract
                if dist < 5.0 and charge_interact > 0.1:
                    strength = charge_interact * bind_strength
                    color = '#2ECC71' if charge_interact > 0.3 else '#F1C40F'
                    ax_anim.plot([pos_a[i, 0], pos_b[j, 0]],
                                 [pos_a[i, 1], pos_b[j, 1]],
                                 '-', color=color, lw=1.2 * strength,
                                 alpha=0.5 * strength, zorder=3)

    # Hydrogen bonds (backbone-backbone, when close)
    if bind_strength > 0.5:
        for i in range(BINDING_START, BINDING_END, 2):
            for j in range(BINDING_START, BINDING_END, 2):
                dist = np.sqrt(((pos_a[i] - pos_b[j])**2).sum())
                if dist < 3.5:
                    ax_anim.plot([pos_a[i, 0], pos_b[j, 0]],
                                 [pos_a[i, 1], pos_b[j, 1]],
                                 ':', color='#AED6F1', lw=0.8,
                                 alpha=0.3 * bind_strength, zorder=2)

    # UI elements
    ax_anim.set_xlim(-14, 14)
    ax_anim.set_ylim(-9, 9)
    ax_anim.set_aspect('equal')
    ax_anim.axis('off')

    # Labels
    ax_anim.text(0.02, 0.97, 'Chain A (antibody)', color='#3498DB',
                 fontsize=10, fontweight='bold', transform=ax_anim.transAxes, va='top')
    ax_anim.text(0.98, 0.97, 'Chain B (antigen)', color='#E74C3C',
                 fontsize=10, fontweight='bold', transform=ax_anim.transAxes, va='top', ha='right')

    # Phase indicator
    if frame < 40:
        phase = 'Approach'
    elif frame < 70:
        phase = 'Docking'
    elif frame < 100:
        phase = 'Conformational change'
    else:
        phase = 'Stable complex'
    ax_anim.text(0.5, 0.02, phase, transform=ax_anim.transAxes,
                 color='white', fontsize=11, ha='center', fontfamily='monospace',
                 alpha=0.7)

    # Binding energy readout
    if frame >= 40:
        energy = -aff_val * bind_strength
        ax_anim.text(0.5, 0.97, f'Binding energy: {energy:.3f}',
                     transform=ax_anim.transAxes, color='#2ECC71', fontsize=10,
                     ha='center', va='top', fontweight='bold', fontfamily='monospace')

    # Legend
    if frame > 5:
        ax_anim.text(0.02, 0.08, 'Red/blue = charge', color='#888888', fontsize=7,
                     transform=ax_anim.transAxes)
        ax_anim.text(0.02, 0.04, 'Green rings = binding site', color='#2ECC71', fontsize=7,
                     transform=ax_anim.transAxes)

anim = animation.FuncAnimation(fig_anim, animate_protein,
                                frames=N_FRAMES, interval=80)
gif_path = os.path.join(ASSETS, "05_protein.gif")
anim.save(gif_path, writer='pillow', fps=12)
plt.close()
print(f"Saved {gif_path}")
