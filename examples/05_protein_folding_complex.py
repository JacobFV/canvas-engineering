"""Protein Complex: 4-chain binding affinity with canvas types.

Synthetic protein-like sequences where binding affinity depends on
complementarity at specific binding-site positions. Four chains:
  - Heavy chain (antibody)
  - Light chain (antibody)
  - Antigen
  - Cofactor

Multi-task: per-residue structure (H/E/C), contact prediction,
binding site identification, and scalar affinity.

Two models compared:
  1. Flat baseline: all residues in one big region
  2. Canvas structured: chain -> binding_site -> complex interaction
     with matched-field cross-chain connectivity + MI loss

Run:  python examples/05_protein_folding_complex.py
Out:  assets/examples/05_protein.png
      assets/examples/05_protein.gif
"""

import os, math
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from dataclasses import dataclass, field as dc_field
from collections import defaultdict

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.animation as animation
from matplotlib.collections import LineCollection
from matplotlib.patches import Patch, FancyArrowPatch
from matplotlib import patheffects
import matplotlib.gridspec as gridspec

from canvas_engineering import Field, compile_schema, ConnectivityPolicy, LayoutStrategy

ASSETS = os.path.join(os.path.dirname(os.path.dirname(__file__)), "assets", "examples")
os.makedirs(ASSETS, exist_ok=True)

torch.manual_seed(42)
np.random.seed(42)

SEQ_LEN = 50        # residues per chain
ALPHABET = 20       # amino acid types
BINDING_START = 20   # binding site starts at residue 20
BINDING_END = 30     # binding site ends at residue 30
N_CHAINS = 4
CHAIN_NAMES = ['heavy', 'light', 'antigen', 'cofactor']
CHAIN_COLORS_HEX = ['#4A90FF', '#FF4A6A', '#3DDC84', '#FFD700']
CHAIN_COLORS_DIM = ['#2A5090', '#902A3A', '#1D7C44', '#907700']

# Dark theme palette
BG_DARK = '#080818'
BG_PANEL = '#0C0C24'
TEXT_DIM = '#556677'
TEXT_BRIGHT = '#CCDDEE'
GRID_COLOR = '#1A1A3A'
ACCENT_GREEN = '#00FF88'
ACCENT_CYAN = '#00CCFF'
ACCENT_MAGENTA = '#FF44CC'
ACCENT_GOLD = '#FFD700'


# ── 1. Type declarations ─────────────────────────────────────────────

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
    chains: list = dc_field(default_factory=list)


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

# Flat: everything in one big region (no chain separation)
@dataclass
class FlatComplex:
    all_residues: Field = Field(8, 8)
    structure: Field = Field(4, 2)
    affinity: Field = Field(1, 1, loss_weight=5.0)

bound_flat = compile_schema(
    FlatComplex(), T=1, H=16, W=16, d_model=48,
    connectivity=ConnectivityPolicy(intra="dense"),
)

print("Structured:", bound_structured.summary())
print("Flat:", bound_flat.summary())


# ── 2. Synthetic data: protein binding ───────────────────────────────

AA_PROPERTIES = torch.randn(ALPHABET, 3)
AA_PROPERTIES[:, 0] = torch.linspace(-1, 1, ALPHABET)       # charge
AA_PROPERTIES[:, 1] = torch.sin(torch.arange(ALPHABET).float() * 0.5)  # hydrophobicity
AA_PROPERTIES[:, 2] = torch.linspace(0.5, 1.5, ALPHABET)    # size

# Secondary structure assignment: deterministic from sequence
# 0=helix, 1=sheet, 2=coil
def assign_secondary_structure(seq):
    """Assign secondary structure based on sequence patterns."""
    N, L = seq.shape
    ss = torch.zeros(N, L, dtype=torch.long)
    for i in range(L):
        # Helix-forming: residues 0-6
        # Sheet-forming: residues 7-13
        # Coil: residues 14-19
        ss[:, i] = torch.where(seq[:, i] < 7, torch.zeros_like(seq[:, i]),
                   torch.where(seq[:, i] < 14, torch.ones_like(seq[:, i]),
                   2 * torch.ones_like(seq[:, i])))
    return ss


def generate_protein_data(n_samples=4096):
    """Generate synthetic 4-chain protein complex data with binding affinity.

    Binding rules:
    1. Charge complementarity at binding site (+ binds -)
    2. Hydrophobic matching at binding site
    3. Size compatibility (nonlinear)
    4. Heavy-light chain cooperativity bonus
    """
    seqs = []
    for _ in range(N_CHAINS):
        seqs.append(torch.randint(0, ALPHABET, (n_samples, SEQ_LEN)))

    onehots = [F.one_hot(s, ALPHABET).float() for s in seqs]

    # Properties at binding sites for each chain
    props = [AA_PROPERTIES[s[:, BINDING_START:BINDING_END]] for s in seqs]

    # Pairwise binding contributions
    total_affinity = torch.zeros(n_samples)
    pair_weights = {
        (0, 2): 1.0,   # heavy-antigen (primary)
        (1, 2): 0.7,   # light-antigen
        (0, 1): 0.3,   # heavy-light cooperativity
        (3, 2): 0.5,   # cofactor-antigen
        (3, 0): 0.2,   # cofactor-heavy
    }

    for (i, j), w in pair_weights.items():
        charge_comp = -(props[i][:, :, 0] * props[j][:, :, 0]).mean(dim=1)
        hydro_match = -((props[i][:, :, 1] - props[j][:, :, 1]) ** 2).mean(dim=1)
        size_compat = -((props[i][:, :, 2] - props[j][:, :, 2]) ** 2).mean(dim=1)
        total_affinity += w * (charge_comp + 0.5 * hydro_match + 0.3 * size_compat)

    affinity = torch.tanh(total_affinity)
    affinity = affinity + torch.randn_like(affinity) * 0.05

    # Stability: depends on internal chain cohesion
    stability = torch.zeros(n_samples, 2)
    for ci in range(N_CHAINS):
        internal = AA_PROPERTIES[seqs[ci]].std(dim=1).mean(dim=-1)
        stability[:, 0] += internal
    stability[:, 0] = torch.tanh(stability[:, 0] / N_CHAINS)
    stability[:, 1] = affinity * 0.8 + torch.randn(n_samples) * 0.1  # correlated
    stability = stability + torch.randn_like(stability) * 0.05

    # Secondary structure per chain
    structures = [assign_secondary_structure(s) for s in seqs]

    # Contact prediction: pairwise residue contacts (binary, simplified)
    # For each chain pair in binding region, contacts based on distance in property space
    contact_maps = []
    bind_len = BINDING_END - BINDING_START
    for ci in range(N_CHAINS):
        # Contacts with next chain in the complex ring
        cj = (ci + 1) % N_CHAINS
        dist = torch.cdist(props[ci], props[cj])  # (N, bind_len, bind_len)
        contacts = (dist < 1.0).float()  # binary contacts
        # Pool to fixed size (8 features)
        contact_feat = contacts.reshape(n_samples, -1)
        # Take first 8 dims
        if contact_feat.shape[1] >= 8:
            contact_feat = contact_feat[:, :8]
        else:
            contact_feat = F.pad(contact_feat, (0, 8 - contact_feat.shape[1]))
        contact_maps.append(contact_feat)

    # Pool sequences into fixed-size features for canvas
    def pool_seq(onehot, n_out=16):
        chunk = SEQ_LEN // n_out
        pooled = []
        for i in range(n_out):
            start = i * chunk
            end = min(start + chunk, SEQ_LEN)
            pooled.append(onehot[:, start:end].mean(dim=1).mean(dim=-1))
        return torch.stack(pooled, dim=1)

    seq_feats = [pool_seq(oh, 16) for oh in onehots]
    bind_feats = [p.reshape(n_samples, -1)[:, :8] for p in props]

    # Structure features (pooled)
    struct_feats = []
    for ss in structures:
        sf = torch.zeros(n_samples, 4)
        chunk = SEQ_LEN // 4
        for i in range(4):
            sf[:, i] = ss[:, i*chunk:(i+1)*chunk].float().mean(dim=1)
        struct_feats.append(sf)

    result = {
        'affinity': affinity,
        'stability': stability,
    }
    for ci in range(N_CHAINS):
        result[f'seq_{ci}'] = seq_feats[ci]
        result[f'bind_{ci}'] = bind_feats[ci]
        result[f'struct_{ci}'] = struct_feats[ci]
        result[f'contacts_{ci}'] = contact_maps[ci]
        result[f'raw_seq_{ci}'] = seqs[ci]
        result[f'props_{ci}'] = props[ci]
        result[f'ss_{ci}'] = structures[ci]

    return result


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

        def field_size(name):
            return len(bound.layout.region_indices(name))

        # Per-chain projections
        self.seq_projs = nn.ModuleList([
            nn.Linear(16, field_size(f'chains[{i}].sequence') * d)
            for i in range(N_CHAINS)
        ])
        self.bind_projs = nn.ModuleList([
            nn.Linear(8, field_size(f'chains[{i}].binding_site') * d)
            for i in range(N_CHAINS)
        ])

        # Output heads
        aff_n = field_size('affinity')
        stab_n = field_size('stability')
        self.affinity_head = nn.Linear(aff_n * d, 1)
        self.stability_head = nn.Linear(stab_n * d, 2)

        self.struct_heads = nn.ModuleList([
            nn.Linear(field_size(f'chains[{i}].structure') * d, 4)
            for i in range(N_CHAINS)
        ])
        self.contact_heads = nn.ModuleList([
            nn.Linear(field_size(f'chains[{i}].contacts') * d, 8)
            for i in range(N_CHAINS)
        ])

        # For MI analysis
        self.bind_reads = nn.ModuleList([
            nn.Linear(field_size(f'chains[{i}].binding_site') * d, 16)
            for i in range(N_CHAINS)
        ])

        self._field_sizes = {}
        for i in range(N_CHAINS):
            self._field_sizes[f'seq_{i}'] = field_size(f'chains[{i}].sequence')
            self._field_sizes[f'bind_{i}'] = field_size(f'chains[{i}].binding_site')
            self._field_sizes[f'struct_{i}'] = field_size(f'chains[{i}].structure')
            self._field_sizes[f'contacts_{i}'] = field_size(f'chains[{i}].contacts')
        self._field_sizes['aff'] = aff_n
        self._field_sizes['stab'] = stab_n
        self._field_sizes['interact'] = field_size('interaction')

    def forward(self, data):
        B = data['seq_0'].shape[0]
        canvas = self.pos_emb.expand(B, -1, -1).clone()

        for ci in range(N_CHAINS):
            seq_idx = self.bound.layout.region_indices(f'chains[{ci}].sequence')
            bind_idx = self.bound.layout.region_indices(f'chains[{ci}].binding_site')

            seq_emb = self.seq_projs[ci](data[f'seq_{ci}']).reshape(
                B, self._field_sizes[f'seq_{ci}'], self.d)
            bind_emb = self.bind_projs[ci](data[f'bind_{ci}']).reshape(
                B, self._field_sizes[f'bind_{ci}'], self.d)

            canvas[:, seq_idx] = canvas[:, seq_idx] + seq_emb
            canvas[:, bind_idx] = canvas[:, bind_idx] + bind_emb

        canvas = self.encoder(canvas, mask=self.mask)

        # Read outputs
        aff_idx = self.bound.layout.region_indices('affinity')
        stab_idx = self.bound.layout.region_indices('stability')

        affinity = self.affinity_head(canvas[:, aff_idx].reshape(B, -1)).squeeze(-1)
        stability = self.stability_head(canvas[:, stab_idx].reshape(B, -1))

        structs = []
        contacts = []
        bind_embs = []
        for ci in range(N_CHAINS):
            s_idx = self.bound.layout.region_indices(f'chains[{ci}].structure')
            c_idx = self.bound.layout.region_indices(f'chains[{ci}].contacts')
            b_idx = self.bound.layout.region_indices(f'chains[{ci}].binding_site')

            structs.append(self.struct_heads[ci](canvas[:, s_idx].reshape(B, -1)))
            contacts.append(self.contact_heads[ci](canvas[:, c_idx].reshape(B, -1)))
            bind_embs.append(self.bind_reads[ci](canvas[:, b_idx].reshape(B, -1)))

        return {
            'affinity': affinity,
            'stability': stability,
            'structs': structs,
            'contacts': contacts,
            'bind_embs': bind_embs,
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

        # Input: concat all chains' features
        self.input_proj = nn.Linear(16 * N_CHAINS + 8 * N_CHAINS, res_n * d)
        self.affinity_head = nn.Linear(aff_n * d, 1)
        self.struct_head = nn.Linear(struct_n * d, 4 * N_CHAINS)
        self.stability_head = nn.Linear(aff_n * d, 2)

        self.res_n = res_n
        self.struct_n = struct_n
        self.aff_n = aff_n

    def forward(self, data):
        B = data['seq_0'].shape[0]
        canvas = self.pos_emb.expand(B, -1, -1).clone()

        all_in = torch.cat(
            [data[f'seq_{ci}'] for ci in range(N_CHAINS)] +
            [data[f'bind_{ci}'] for ci in range(N_CHAINS)],
            dim=-1
        )
        res_idx = self.bound.layout.region_indices('all_residues')
        res_emb = self.input_proj(all_in).reshape(B, self.res_n, self.d)
        canvas[:, res_idx] = canvas[:, res_idx] + res_emb

        canvas = self.encoder(canvas, mask=self.mask)

        aff_idx = self.bound.layout.region_indices('affinity')
        struct_idx = self.bound.layout.region_indices('structure')

        affinity = self.affinity_head(canvas[:, aff_idx].reshape(B, -1)).squeeze(-1)
        struct_all = self.struct_head(canvas[:, struct_idx].reshape(B, -1))
        stability = self.stability_head(canvas[:, aff_idx].reshape(B, -1))

        structs = [struct_all[:, ci*4:(ci+1)*4] for ci in range(N_CHAINS)]

        return {
            'affinity': affinity,
            'stability': stability,
            'structs': structs,
            'contacts': [torch.zeros(B, 8) for _ in range(N_CHAINS)],
            'bind_embs': [None] * N_CHAINS,
        }


# ── 4. Training ──────────────────────────────────────────────────────

def mi_loss(emb_a, emb_b):
    """Encourage high mutual information between binding site embeddings."""
    B = emb_a.shape[0]
    a_norm = F.normalize(emb_a, dim=-1)
    b_norm = F.normalize(emb_b, dim=-1)
    pos = (a_norm * b_norm).sum(dim=-1)
    perm = torch.randperm(B)
    neg = (a_norm * b_norm[perm]).sum(dim=-1)
    loss = -torch.log(torch.exp(pos / 0.1) / (torch.exp(pos / 0.1) + torch.exp(neg / 0.1) + 1e-8))
    return loss.mean()


def train_protein(model, label, use_mi=False, n_epochs=400, bs=128):
    opt = torch.optim.AdamW(model.parameters(), lr=2e-3, weight_decay=1e-4)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, n_epochs)
    losses = []
    mi_losses_hist = []

    for ep in range(n_epochs):
        idx = torch.randint(0, len(train_data['affinity']), (bs,))
        batch = {k: v[idx] if isinstance(v, torch.Tensor) else v
                 for k, v in train_data.items()}

        out = model(batch)

        # Multi-task loss
        aff_loss = F.mse_loss(out['affinity'], batch['affinity'])
        stab_loss = F.mse_loss(out['stability'], batch['stability'])

        struct_loss = sum(
            F.mse_loss(out['structs'][ci], batch[f'struct_{ci}'])
            for ci in range(N_CHAINS)
        )

        contact_loss = torch.tensor(0.0)
        if use_mi:
            contact_loss = sum(
                F.mse_loss(out['contacts'][ci], batch[f'contacts_{ci}'])
                for ci in range(N_CHAINS)
            )

        loss = 5.0 * aff_loss + 3.0 * stab_loss + struct_loss + 2.0 * contact_loss

        mi_val = 0.0
        if use_mi and out['bind_embs'][0] is not None:
            # MI between all pairs of binding site embeddings
            for ci in range(N_CHAINS):
                for cj in range(ci + 1, N_CHAINS):
                    mi_val = mi_val + mi_loss(out['bind_embs'][ci], out['bind_embs'][cj])
            loss = loss + 0.3 * mi_val

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
        val_stab = F.mse_loss(out_val['stability'], val_data['stability']).item()
    print(f"  [{label}] val_affinity_mse={val_aff:.4f}, val_stability_mse={val_stab:.4f}")
    return model, losses, mi_losses_hist, val_aff, val_stab


print("\nTraining flat model...")
flat_model = FlatProteinModel(bound_flat)
flat_model, flat_losses, flat_mi, flat_val_aff, flat_val_stab = train_protein(
    flat_model, "flat", use_mi=False)

print("Training structured model...")
struct_model = StructuredProteinModel(bound_structured)
struct_model, struct_losses, struct_mi, struct_val_aff, struct_val_stab = train_protein(
    struct_model, "structured", use_mi=True)


# ── 5. Visualization: 4x4 multi-panel dark figure ───────────────────

print("\nGenerating 4x4 visualization...")

fig = plt.figure(figsize=(24, 24), dpi=150)
fig.patch.set_facecolor(BG_DARK)

gs = gridspec.GridSpec(4, 4, figure=fig, hspace=0.35, wspace=0.30,
                       left=0.04, right=0.97, top=0.95, bottom=0.03)

fig.suptitle('PROTEIN COMPLEX ANALYSIS — 4-CHAIN STRUCTURED CANVAS',
             fontsize=20, fontweight='bold', color=TEXT_BRIGHT, y=0.98,
             fontfamily='monospace')

def style_ax(ax, title='', xlabel='', ylabel=''):
    ax.set_facecolor(BG_PANEL)
    ax.set_title(title, fontsize=10, fontweight='bold', color=TEXT_BRIGHT,
                 fontfamily='monospace', pad=8)
    ax.set_xlabel(xlabel, fontsize=8, color=TEXT_DIM, fontfamily='monospace')
    ax.set_ylabel(ylabel, fontsize=8, color=TEXT_DIM, fontfamily='monospace')
    ax.tick_params(colors=TEXT_DIM, labelsize=7)
    for spine in ax.spines.values():
        spine.set_color('#1A1A3A')
        spine.set_linewidth(0.5)
    ax.grid(True, alpha=0.08, color='#334455')


# ── (0,0) 3D-like protein structure rendering ────────────────────────
ax = fig.add_subplot(gs[0, 0])
style_ax(ax, '3D PROTEIN STRUCTURE (2D PROJECTION)')

np.random.seed(42)

def make_helix_3d(n_residues, radius=2.0, pitch=0.35, center=np.array([0, 0, 0])):
    t = np.linspace(0, n_residues * pitch, n_residues)
    x = radius * np.cos(t * 2.5) + center[0] + np.arange(n_residues) * 0.12
    y = radius * np.sin(t * 2.5) + center[1]
    z = center[2] + np.sin(t * 1.5) * 0.8
    return np.stack([x, y, z], axis=-1)

# 4 chains approaching a common center
chain_centers = [
    np.array([-5, -3, 0]),
    np.array([-5, 3, 0]),
    np.array([5, 0, -1]),
    np.array([3, -4, 1]),
]

chain_positions_3d = [make_helix_3d(SEQ_LEN, radius=1.5, pitch=0.3, center=c)
                      for c in chain_centers]

# Simple 3D -> 2D projection with depth shading
view_angle = 0.4
for ci, (pos3d, color, name) in enumerate(zip(
        chain_positions_3d, CHAIN_COLORS_HEX, CHAIN_NAMES)):
    # Rotate for viewing angle
    cos_a, sin_a = np.cos(view_angle), np.sin(view_angle)
    x2d = pos3d[:, 0] * cos_a - pos3d[:, 2] * sin_a
    y2d = pos3d[:, 1]
    depth = pos3d[:, 0] * sin_a + pos3d[:, 2] * cos_a

    # Depth-based alpha and size
    depth_norm = (depth - depth.min()) / (depth.max() - depth.min() + 1e-8)
    alphas = 0.4 + 0.5 * (1 - depth_norm)
    sizes = 8 + 20 * (1 - depth_norm)

    # Backbone
    segments = np.array([[x2d[i], y2d[i], x2d[i+1], y2d[i+1]]
                         for i in range(len(x2d)-1)])
    for i in range(len(segments)):
        ax.plot([segments[i, 0], segments[i, 2]],
                [segments[i, 1], segments[i, 3]],
                '-', color=color, lw=1.5, alpha=float(alphas[i]) * 0.7)

    # Residues with depth shading
    ax.scatter(x2d, y2d, s=sizes, c=[color]*len(x2d),
               alpha=alphas, zorder=3, edgecolors='none')

    # Binding site glow
    bs = slice(BINDING_START, BINDING_END)
    ax.scatter(x2d[bs], y2d[bs], s=sizes[bs]*2, facecolors='none',
               edgecolors=ACCENT_GREEN, lw=1.2, alpha=0.6, zorder=4)

    # Label
    ax.text(x2d[0], y2d[0] - 1.0, name.upper(), fontsize=6, color=color,
            fontfamily='monospace', fontweight='bold', ha='center', alpha=0.9)

ax.set_xlim(-10, 12)
ax.set_ylim(-8, 8)
ax.set_aspect('equal')
ax.axis('off')


# ── (0,1) Contact map heatmap ────────────────────────────────────────
ax = fig.add_subplot(gs[0, 1])
style_ax(ax, 'INTER-CHAIN CONTACT DISTANCE MAP', 'residue (chain j)', 'residue (chain i)')

# Build full 4-chain pairwise contact distance matrix at binding site
bind_len = BINDING_END - BINDING_START
full_size = bind_len * N_CHAINS
contact_matrix = np.ones((full_size, full_size)) * np.nan

sample_idx = 0
for ci in range(N_CHAINS):
    for cj in range(N_CHAINS):
        props_i = train_data[f'props_{ci}'][sample_idx].numpy()
        props_j = train_data[f'props_{cj}'][sample_idx].numpy()
        dist = np.sqrt(((props_i[:, None] - props_j[None, :]) ** 2).sum(axis=-1))
        r0, r1 = ci * bind_len, (ci + 1) * bind_len
        c0, c1 = cj * bind_len, (cj + 1) * bind_len
        contact_matrix[r0:r1, c0:c1] = dist

im = ax.imshow(contact_matrix, cmap='inferno', aspect='equal', interpolation='bilinear')
cb = plt.colorbar(im, ax=ax, shrink=0.7, pad=0.02)
cb.set_label('property distance', fontsize=6, color=TEXT_DIM)
cb.ax.tick_params(labelsize=5, colors=TEXT_DIM)

# Chain boundary lines and labels
for i in range(1, N_CHAINS):
    ax.axhline(i * bind_len - 0.5, color=ACCENT_CYAN, lw=0.5, alpha=0.5)
    ax.axvline(i * bind_len - 0.5, color=ACCENT_CYAN, lw=0.5, alpha=0.5)

for ci, name in enumerate(CHAIN_NAMES):
    mid = ci * bind_len + bind_len // 2
    ax.text(mid, -2.5, name[:3].upper(), fontsize=5, color=CHAIN_COLORS_HEX[ci],
            ha='center', fontfamily='monospace', fontweight='bold')
    ax.text(-2.5, mid, name[:3].upper(), fontsize=5, color=CHAIN_COLORS_HEX[ci],
            ha='right', va='center', fontfamily='monospace', fontweight='bold')


# ── (0,2) Binding energy landscape (PCA of binding embeddings) ───────
ax = fig.add_subplot(gs[0, 2])
style_ax(ax, 'BINDING ENERGY LANDSCAPE', 'PC1', 'PC2')

struct_model.eval()
with torch.no_grad():
    out_struct = struct_model(val_data)
    bind_embs_all = torch.cat(out_struct['bind_embs'], dim=-1)  # (N, 16*4)

# PCA
centered = bind_embs_all - bind_embs_all.mean(dim=0)
U, S, V = torch.svd_lowrank(centered, q=2)
pca = (centered @ V).numpy()
aff_vals = val_data['affinity'].numpy()

sc = ax.scatter(pca[:, 0], pca[:, 1], c=aff_vals, s=4, alpha=0.5,
                cmap='coolwarm', vmin=-1, vmax=1)
plt.colorbar(sc, ax=ax, shrink=0.7, label='affinity', pad=0.02)

# Contour lines
from scipy.ndimage import gaussian_filter
xg = np.linspace(pca[:, 0].min(), pca[:, 0].max(), 50)
yg = np.linspace(pca[:, 1].min(), pca[:, 1].max(), 50)
xx, yy = np.meshgrid(xg, yg)
# KDE-like density
from scipy.stats import gaussian_kde
try:
    kde = gaussian_kde(pca.T)
    zz = kde(np.vstack([xx.ravel(), yy.ravel()])).reshape(xx.shape)
    ax.contour(xx, yy, zz, levels=6, colors=ACCENT_CYAN, linewidths=0.4, alpha=0.5)
except Exception:
    pass


# ── (0,3) Ramachandran-like scatter ──────────────────────────────────
ax = fig.add_subplot(gs[0, 3])
style_ax(ax, 'RAMACHANDRAN (phi/psi PROXY)', 'phi proxy', 'psi proxy')

# Generate pseudo phi/psi from sequence properties
sample_seqs = train_data['raw_seq_0'][:200].numpy()
sample_ss = train_data['ss_0'][:200].numpy()
phi_proxy = np.sin(sample_seqs * 0.3) * 180
psi_proxy = np.cos(sample_seqs * 0.5 + 1) * 180

ss_colors_map = {0: '#FF4444', 1: '#44AAFF', 2: '#888888'}  # H, E, C
ss_labels = {0: 'Helix', 1: 'Sheet', 2: 'Coil'}

for ss_type, color in ss_colors_map.items():
    mask = sample_ss.ravel() == ss_type
    phi_flat = phi_proxy.ravel()[mask]
    psi_flat = psi_proxy.ravel()[mask]
    ax.scatter(phi_flat[:800], psi_flat[:800], s=3, c=color, alpha=0.4,
               label=ss_labels[ss_type], edgecolors='none')

ax.set_xlim(-200, 200)
ax.set_ylim(-200, 200)
ax.legend(fontsize=6, loc='upper right', framealpha=0.3,
          edgecolor='none', facecolor=BG_PANEL, labelcolor=TEXT_DIM)
ax.axhline(0, color=TEXT_DIM, lw=0.3, alpha=0.3)
ax.axvline(0, color=TEXT_DIM, lw=0.3, alpha=0.3)


# ── (1,0) Secondary structure strip ─────────────────────────────────
ax = fig.add_subplot(gs[1, 0])
style_ax(ax, 'SECONDARY STRUCTURE STRIPS', 'residue position', 'chain')

ss_cmap = {0: '#FF3333', 1: '#3388FF', 2: '#555555'}  # H=red, E=blue, C=gray
for ci in range(N_CHAINS):
    ss = train_data[f'ss_{ci}'][0].numpy()  # first sample
    for ri in range(SEQ_LEN):
        color = ss_cmap[int(ss[ri])]
        ax.barh(ci, 1, left=ri, height=0.7, color=color, edgecolor='none')

ax.set_yticks(range(N_CHAINS))
ax.set_yticklabels([n.upper() for n in CHAIN_NAMES], fontsize=7,
                    fontfamily='monospace', color=TEXT_DIM)
ax.set_xlim(0, SEQ_LEN)
ax.set_ylim(-0.5, N_CHAINS - 0.5)

# Binding site marker
ax.axvspan(BINDING_START, BINDING_END, alpha=0.15, color=ACCENT_GREEN, zorder=0)
ax.text(BINDING_START + (BINDING_END - BINDING_START)/2, N_CHAINS - 0.1,
        'BINDING\nSITE', fontsize=5, color=ACCENT_GREEN, ha='center', va='bottom',
        fontfamily='monospace', fontweight='bold')

legend_elements = [Patch(facecolor='#FF3333', label='H (helix)'),
                   Patch(facecolor='#3388FF', label='E (sheet)'),
                   Patch(facecolor='#555555', label='C (coil)')]
ax.legend(handles=legend_elements, fontsize=5, loc='lower right',
          framealpha=0.3, edgecolor='none', facecolor=BG_PANEL,
          labelcolor=TEXT_DIM)


# ── (1,1) Affinity prediction scatter ────────────────────────────────
ax = fig.add_subplot(gs[1, 1])
style_ax(ax, 'AFFINITY: PREDICTED vs TRUE', 'true affinity', 'predicted affinity')

struct_model.eval()
flat_model.eval()
with torch.no_grad():
    pred_struct = struct_model(val_data)['affinity'].numpy()
    pred_flat = flat_model(val_data)['affinity'].numpy()
true_aff = val_data['affinity'].numpy()

ax.scatter(true_aff, pred_flat, s=6, alpha=0.3, color=CHAIN_COLORS_HEX[1],
           label=f'flat (MSE={flat_val_aff:.3f})', zorder=2, edgecolors='none')
ax.scatter(true_aff, pred_struct, s=6, alpha=0.4, color=ACCENT_CYAN,
           label=f'structured (MSE={struct_val_aff:.3f})', zorder=3, edgecolors='none')
lims = [-1.5, 1.5]
ax.plot(lims, lims, '--', color=TEXT_DIM, lw=1, alpha=0.5)
ax.set_xlim(lims)
ax.set_ylim(lims)
ax.set_aspect('equal')
ax.legend(fontsize=6, loc='upper left', framealpha=0.3,
          edgecolor='none', facecolor=BG_PANEL, labelcolor=TEXT_DIM)


# ── (1,2) Per-residue error heatmap ──────────────────────────────────
ax = fig.add_subplot(gs[1, 2])
style_ax(ax, 'PER-RESIDUE STRUCTURE ERROR', 'structure feature idx', 'chain')

with torch.no_grad():
    out_s = struct_model(val_data)

error_matrix = np.zeros((N_CHAINS, 4))
for ci in range(N_CHAINS):
    pred = out_s['structs'][ci].numpy()
    true = val_data[f'struct_{ci}'].numpy()
    error_matrix[ci] = np.abs(pred - true).mean(axis=0)

im = ax.imshow(error_matrix, cmap='magma', aspect='auto', interpolation='nearest')
plt.colorbar(im, ax=ax, shrink=0.7, label='MAE', pad=0.02)
ax.set_yticks(range(N_CHAINS))
ax.set_yticklabels([n.upper() for n in CHAIN_NAMES], fontsize=7,
                    fontfamily='monospace')
ax.set_xticks(range(4))
ax.set_xticklabels(['q1', 'q2', 'q3', 'q4'], fontsize=7, fontfamily='monospace')

# Annotate values
for ci in range(N_CHAINS):
    for fi in range(4):
        ax.text(fi, ci, f'{error_matrix[ci, fi]:.3f}', ha='center', va='center',
                fontsize=6, color='white' if error_matrix[ci, fi] > error_matrix.mean() else TEXT_DIM,
                fontfamily='monospace')


# ── (1,3) Training curves (dark theme, log scale) ────────────────────
ax = fig.add_subplot(gs[1, 3])
style_ax(ax, 'TRAINING LOSS (LOG SCALE)', 'epoch', 'loss')

w = 20
def smooth(a, w=w):
    return np.convolve(a, np.ones(w)/w, mode='valid')

ax.plot(smooth(struct_losses), color=ACCENT_CYAN, lw=1.5, label='structured', alpha=0.9)
ax.plot(smooth(flat_losses), color=CHAIN_COLORS_HEX[1], lw=1.5, label='flat', alpha=0.9)
ax.set_yscale('log')
ax.legend(fontsize=7, loc='upper right', framealpha=0.3,
          edgecolor='none', facecolor=BG_PANEL, labelcolor=TEXT_DIM)

# Winner annotation
better = 'structured' if struct_val_aff < flat_val_aff else 'flat'
ratio = max(struct_val_aff, flat_val_aff) / max(min(struct_val_aff, flat_val_aff), 1e-8)
winner_color = ACCENT_CYAN if struct_val_aff < flat_val_aff else CHAIN_COLORS_HEX[1]
ax.text(0.98, 0.05, f'{better}: {ratio:.2f}x better',
        transform=ax.transAxes, ha='right', va='bottom', fontsize=8,
        fontweight='bold', color=winner_color, fontfamily='monospace',
        bbox=dict(boxstyle='round,pad=0.3', facecolor=winner_color, alpha=0.15,
                  edgecolor='none'))


# ── (2,0) Cross-chain attention weights ──────────────────────────────
ax = fig.add_subplot(gs[2, 0])
style_ax(ax, 'CROSS-CHAIN CONNECTIVITY', '', '')

# Build cross-chain connection count matrix
cross_matrix = np.zeros((N_CHAINS, N_CHAINS))
for c in bound_structured.topology.connections:
    for ci in range(N_CHAINS):
        for cj in range(N_CHAINS):
            if ci == cj:
                continue
            if f'chains[{ci}]' in c.src and f'chains[{cj}]' in c.dst:
                cross_matrix[ci, cj] += 1

im = ax.imshow(cross_matrix, cmap='viridis', aspect='equal', interpolation='nearest')
plt.colorbar(im, ax=ax, shrink=0.7, label='connections', pad=0.02)
ax.set_xticks(range(N_CHAINS))
ax.set_yticks(range(N_CHAINS))
ax.set_xticklabels([n[:3].upper() for n in CHAIN_NAMES], fontsize=7,
                    fontfamily='monospace')
ax.set_yticklabels([n[:3].upper() for n in CHAIN_NAMES], fontsize=7,
                    fontfamily='monospace')

for ci in range(N_CHAINS):
    for cj in range(N_CHAINS):
        val = int(cross_matrix[ci, cj])
        if val > 0:
            ax.text(cj, ci, str(val), ha='center', va='center', fontsize=7,
                    color='white', fontweight='bold', fontfamily='monospace')


# ── (2,1) Electrostatic surface ──────────────────────────────────────
ax = fig.add_subplot(gs[2, 1])
style_ax(ax, 'ELECTROSTATIC SURFACE (BINDING INTERFACE)')

# Charge distribution at binding interface between heavy and antigen
charges_h = AA_PROPERTIES[train_data['raw_seq_0'][0, BINDING_START:BINDING_END].numpy(), 0].numpy()
charges_a = AA_PROPERTIES[train_data['raw_seq_2'][0, BINDING_START:BINDING_END].numpy(), 0].numpy()

# Create a 2D surface-like plot
bind_len = BINDING_END - BINDING_START
x = np.arange(bind_len)
y_h = np.zeros(bind_len) + 1
y_a = np.zeros(bind_len) - 1

# Draw as filled bars
for i in range(bind_len):
    color_h = plt.cm.coolwarm((charges_h[i] + 1) / 2)
    color_a = plt.cm.coolwarm((charges_a[i] + 1) / 2)
    ax.bar(i, 0.8, bottom=0.6, color=color_h, edgecolor='none', width=0.9)
    ax.bar(i, 0.8, bottom=-1.4, color=color_a, edgecolor='none', width=0.9)

    # Interaction lines for complementary charges
    if charges_h[i] * charges_a[i] < -0.2:
        strength = abs(charges_h[i] * charges_a[i])
        ax.plot([i, i], [0.6, -0.6], '-', color=ACCENT_GREEN,
                lw=strength * 3, alpha=0.4)

ax.text(bind_len/2, 1.8, 'HEAVY CHAIN', fontsize=7, color=CHAIN_COLORS_HEX[0],
        ha='center', fontfamily='monospace', fontweight='bold')
ax.text(bind_len/2, -2.0, 'ANTIGEN', fontsize=7, color=CHAIN_COLORS_HEX[2],
        ha='center', fontfamily='monospace', fontweight='bold')
ax.set_ylim(-2.5, 2.5)
ax.set_xlim(-0.5, bind_len - 0.5)
ax.axhline(0, color=TEXT_DIM, lw=0.3, ls='--')
ax.text(bind_len + 0.5, 1.0, '+', fontsize=10, color='#FF4444', fontfamily='monospace')
ax.text(bind_len + 0.5, -1.0, '-', fontsize=10, color='#4444FF', fontfamily='monospace')


# ── (2,2) Canvas layout grid ─────────────────────────────────────────
ax = fig.add_subplot(gs[2, 2])
style_ax(ax, 'STRUCTURED CANVAS LAYOUT')

H, W = bound_structured.layout.H, bound_structured.layout.W
grid = np.ones((H, W, 3)) * 0.05  # near-black

field_color_map = {
    'interaction': '#E74C3C',
    'affinity': '#FFD700',
    'stability': '#FF8C00',
    'sequence': '#4A90FF',
    'structure': '#9B59B6',
    'binding_site': '#00FF88',
    'contacts': '#FF44CC',
}

for name, bf in bound_structured.fields.items():
    color_key = name.split('.')[-1] if '.' in name else name
    color = field_color_map.get(color_key, '#333333')
    r, g, b = int(color[1:3], 16)/255, int(color[3:5], 16)/255, int(color[5:7], 16)/255
    h0, h1 = bf.spec.bounds[2], bf.spec.bounds[3]
    w0, w1 = bf.spec.bounds[4], bf.spec.bounds[5]
    grid[h0:h1, w0:w1] = [r, g, b]

ax.imshow(grid, aspect='equal', interpolation='nearest')
legend_elements = [Patch(facecolor=c, label=n, edgecolor='none')
                   for n, c in field_color_map.items()]
ax.legend(handles=legend_elements, fontsize=5, loc='lower right', ncol=2,
          framealpha=0.3, edgecolor='none', facecolor=BG_PANEL, labelcolor=TEXT_DIM)


# ── (2,3) MI loss curve ─────────────────────────────────────────────
ax = fig.add_subplot(gs[2, 3])
style_ax(ax, 'MUTUAL INFORMATION LOSS', 'epoch', 'MI loss')

if any(v > 0 for v in struct_mi):
    ax.plot(smooth(struct_mi), color=ACCENT_GREEN, lw=1.5, alpha=0.9)
    ax.fill_between(range(len(smooth(struct_mi))), smooth(struct_mi),
                     alpha=0.1, color=ACCENT_GREEN)
    ax.text(0.98, 0.95, 'binding site\ncross-chain MI',
            transform=ax.transAxes, ha='right', va='top', fontsize=7,
            color=ACCENT_GREEN, fontfamily='monospace',
            bbox=dict(boxstyle='round,pad=0.3', facecolor=ACCENT_GREEN,
                      alpha=0.1, edgecolor='none'))


# ── (3,0) Binding site close-up ─────────────────────────────────────
ax = fig.add_subplot(gs[3, 0])
style_ax(ax, 'BINDING SITE CLOSE-UP')

# Show binding residues of heavy (top) and antigen (bottom) with interaction lines
np.random.seed(7)
bind_x = np.arange(bind_len) * 1.2
heavy_y = np.ones(bind_len) * 2 + np.random.randn(bind_len) * 0.15
antigen_y = np.ones(bind_len) * -2 + np.random.randn(bind_len) * 0.15
light_y = np.ones(bind_len) * 3.5 + np.random.randn(bind_len) * 0.15
cofactor_y = np.ones(bind_len) * -3.5 + np.random.randn(bind_len) * 0.15

for positions, y_pos, color, name in [
    (bind_x, heavy_y, CHAIN_COLORS_HEX[0], 'HVY'),
    (bind_x, light_y, CHAIN_COLORS_HEX[1], 'LGT'),
    (bind_x, antigen_y, CHAIN_COLORS_HEX[2], 'AGN'),
    (bind_x, cofactor_y, CHAIN_COLORS_HEX[3], 'COF'),
]:
    ax.scatter(positions, y_pos, s=40, c=color, zorder=4, edgecolors='white',
               linewidths=0.3, alpha=0.9)
    # Backbone
    ax.plot(positions, y_pos, '-', color=color, lw=1.5, alpha=0.5, zorder=3)
    ax.text(-1.5, y_pos.mean(), name, fontsize=6, color=color,
            fontfamily='monospace', fontweight='bold', va='center')

# Interaction lines (heavy-antigen)
for i in range(bind_len):
    charge_h = charges_h[i]
    charge_a = charges_a[i]
    interact = -charge_h * charge_a
    if interact > 0.15:
        ax.plot([bind_x[i], bind_x[i]], [heavy_y[i], antigen_y[i]],
                ':', color=ACCENT_GREEN, lw=interact * 2.5, alpha=0.4, zorder=2)
    # H-bonds as cyan dashes
    if i % 3 == 0:
        ax.plot([bind_x[i], bind_x[i]], [heavy_y[i], antigen_y[i]],
                '--', color=ACCENT_CYAN, lw=0.5, alpha=0.3, zorder=1)

ax.set_xlim(-3, bind_len * 1.2 + 1)
ax.set_ylim(-5, 5)
ax.axis('off')


# ── (3,1) Model comparison summary bars ──────────────────────────────
ax = fig.add_subplot(gs[3, 1])
style_ax(ax, 'MODEL COMPARISON', '', 'MSE')

metrics = {
    'Affinity': (struct_val_aff, flat_val_aff),
    'Stability': (struct_val_stab, flat_val_stab),
}

# Compute structure MSE
with torch.no_grad():
    out_s2 = struct_model(val_data)
    out_f2 = flat_model(val_data)
    struct_struct_mse = np.mean([
        F.mse_loss(out_s2['structs'][ci], val_data[f'struct_{ci}']).item()
        for ci in range(N_CHAINS)
    ])
    flat_struct_mse = np.mean([
        F.mse_loss(out_f2['structs'][ci], val_data[f'struct_{ci}']).item()
        for ci in range(N_CHAINS)
    ])
    metrics['Structure'] = (struct_struct_mse, flat_struct_mse)

x_pos = np.arange(len(metrics))
width = 0.35
s_vals = [v[0] for v in metrics.values()]
f_vals = [v[1] for v in metrics.values()]

bars1 = ax.bar(x_pos - width/2, s_vals, width, label='structured',
               color=ACCENT_CYAN, alpha=0.8, edgecolor='none')
bars2 = ax.bar(x_pos + width/2, f_vals, width, label='flat',
               color=CHAIN_COLORS_HEX[1], alpha=0.8, edgecolor='none')
ax.set_xticks(x_pos)
ax.set_xticklabels(list(metrics.keys()), fontsize=7, fontfamily='monospace',
                    color=TEXT_DIM)
ax.legend(fontsize=7, framealpha=0.3, edgecolor='none', facecolor=BG_PANEL,
          labelcolor=TEXT_DIM)

# Annotate values
for bar, val in zip(bars1, s_vals):
    ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.002,
            f'{val:.4f}', ha='center', fontsize=5, color=ACCENT_CYAN,
            fontfamily='monospace')
for bar, val in zip(bars2, f_vals):
    ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.002,
            f'{val:.4f}', ha='center', fontsize=5, color=CHAIN_COLORS_HEX[1],
            fontfamily='monospace')


# ── (3,2) Stability prediction over perturbation ─────────────────────
ax = fig.add_subplot(gs[3, 2])
style_ax(ax, 'STABILITY vs MUTATION RATE', 'mutation fraction', 'predicted stability')

# Simulate mutations: progressively mutate the binding site and re-predict
mutation_fracs = np.linspace(0, 1.0, 12)
stab_means_struct = []
stab_means_flat = []

np.random.seed(42)
for frac in mutation_fracs:
    perturbed = {k: v.clone() if isinstance(v, torch.Tensor) else v
                 for k, v in val_data.items()}
    # Mutate binding features
    for ci in range(N_CHAINS):
        noise = torch.randn_like(perturbed[f'bind_{ci}']) * frac * 2
        perturbed[f'bind_{ci}'] = perturbed[f'bind_{ci}'] + noise

    with torch.no_grad():
        out_s = struct_model(perturbed)
        out_f = flat_model(perturbed)
        stab_means_struct.append(out_s['stability'][:, 0].mean().item())
        stab_means_flat.append(out_f['stability'][:, 0].mean().item())

ax.plot(mutation_fracs, stab_means_struct, 'o-', color=ACCENT_CYAN, lw=2,
        markersize=4, label='structured', alpha=0.9)
ax.plot(mutation_fracs, stab_means_flat, 's--', color=CHAIN_COLORS_HEX[1], lw=2,
        markersize=4, label='flat', alpha=0.9)
ax.axvspan(0, 0.15, alpha=0.08, color=ACCENT_GREEN)
ax.text(0.07, ax.get_ylim()[0] if stab_means_struct else 0, 'wild\ntype',
        fontsize=5, color=ACCENT_GREEN, ha='center', va='bottom', fontfamily='monospace')
ax.legend(fontsize=7, framealpha=0.3, edgecolor='none', facecolor=BG_PANEL,
          labelcolor=TEXT_DIM)


# ── (3,3) Loss weight pie chart ──────────────────────────────────────
ax = fig.add_subplot(gs[3, 3])
style_ax(ax, 'LOSS WEIGHT ALLOCATION')

weight_data = {
    'affinity': 5.0,
    'interaction': 4.0,
    'binding_site': 3.0,
    'stability': 3.0,
    'contacts': 2.0,
    'sequence': 1.0,
    'structure': 1.0,
}

colors_pie = ['#FFD700', '#E74C3C', '#00FF88', '#FF8C00',
              '#FF44CC', '#4A90FF', '#9B59B6']
labels_pie = list(weight_data.keys())
values_pie = list(weight_data.values())

wedges, texts, autotexts = ax.pie(
    values_pie, labels=None, autopct='%1.0f%%',
    colors=colors_pie, startangle=90,
    pctdistance=0.8,
    wedgeprops=dict(edgecolor=BG_DARK, linewidth=1.5)
)

for text in autotexts:
    text.set_fontsize(6)
    text.set_fontfamily('monospace')
    text.set_color('white')
    text.set_fontweight('bold')

# Legend instead of labels
ax.legend(wedges, labels_pie, fontsize=6, loc='center left',
          bbox_to_anchor=(-0.3, 0.5),
          framealpha=0.3, edgecolor='none', facecolor=BG_PANEL,
          labelcolor=TEXT_DIM)


# ── Save figure ──────────────────────────────────────────────────────
path = os.path.join(ASSETS, "05_protein.png")
fig.savefig(path, bbox_inches='tight', facecolor=BG_DARK, dpi=150)
plt.close()
print(f"\nSaved {path}")


# ── 6. Animation: high-fidelity protein complex assembly ─────────────

print("Generating protein binding animation...")

np.random.seed(42)
N_FRAMES = 160

# Generate helical backbone geometry for each chain
def make_helix(n_residues, radius=2.5, pitch=0.35, center=np.array([0, 0])):
    t = np.linspace(0, n_residues * pitch, n_residues)
    x = radius * np.cos(t * 2.5) + center[0] + np.arange(n_residues) * 0.15
    y = radius * np.sin(t * 2.5) + center[1]
    return np.stack([x, y], axis=-1)

# 4 chain base positions (far apart initially)
chain_bases = [
    make_helix(SEQ_LEN, radius=2.0, pitch=0.3, center=np.array([-4, -3])),
    make_helix(SEQ_LEN, radius=1.8, pitch=0.28, center=np.array([-4, 3])),
    make_helix(SEQ_LEN, radius=2.2, pitch=0.32, center=np.array([4, 0])),
    make_helix(SEQ_LEN, radius=1.5, pitch=0.25, center=np.array([2, -5])),
]
# Mirror chains 2,3 so binding sites face inward
chain_bases[2][:, 0] = -chain_bases[2][:, 0] + 8
chain_bases[3][:, 0] = -chain_bases[3][:, 0] + 4
chain_bases[3][:, 1] = -chain_bases[3][:, 1] - 2

# Properties for coloring
chain_charges = [
    AA_PROPERTIES[train_data[f'raw_seq_{ci}'][0].numpy(), 0].numpy()
    for ci in range(N_CHAINS)
]
chain_sizes = [
    AA_PROPERTIES[train_data[f'raw_seq_{ci}'][0].numpy(), 2].numpy()
    for ci in range(N_CHAINS)
]

# Sidechain directions
def sidechain_dirs(backbone, length_scale):
    dirs = np.zeros_like(backbone)
    for i in range(len(backbone)):
        if i < len(backbone) - 1:
            tang = backbone[i+1] - backbone[i]
        else:
            tang = backbone[i] - backbone[i-1]
        norm = np.array([-tang[1], tang[0]])
        norm = norm / (np.linalg.norm(norm) + 1e-8)
        side = 1 if i % 2 == 0 else -1
        dirs[i] = norm * side * length_scale[i] * 0.8
    return dirs

sc_dirs = [sidechain_dirs(chain_bases[ci], chain_sizes[ci]) for ci in range(N_CHAINS)]

aff_val = train_data['affinity'][0].item()

fig_anim, ax_anim = plt.subplots(1, 1, figsize=(12, 8), dpi=120)
fig_anim.patch.set_facecolor(BG_DARK)

# Precompute thermal noise
rng = np.random.RandomState(42)
thermal = [rng.randn(N_FRAMES, SEQ_LEN, 2) * 0.06 for _ in range(N_CHAINS)]

# Starting offsets (far from center)
start_offsets = [
    np.array([-12, -6]),
    np.array([-12, 6]),
    np.array([12, 2]),
    np.array([8, -8]),
]
# Final docked offsets (close together)
dock_offsets = [
    np.array([-2.5, -1.5]),
    np.array([-2.5, 1.5]),
    np.array([2.5, 0]),
    np.array([1.0, -2.5]),
]

def ease_in_out(t):
    return t * t * (3 - 2 * t)

def animate_protein(frame):
    ax_anim.clear()
    ax_anim.set_facecolor(BG_DARK)

    # Phase 1 (0-50): Free chains approaching with thermal motion
    # Phase 2 (50-85): Initial contact / docking with binding site glow
    # Phase 3 (85-120): Conformational change / tightening
    # Phase 4 (120-160): Stable complex with readouts + rotating view

    # Compute view angle for rotation effect
    if frame >= 120:
        view_t = (frame - 120) / 40.0
        view_rot = view_t * 0.3  # subtle rotation
    else:
        view_rot = 0.0

    # Compute per-chain offsets based on phase
    offsets = []
    bind_strength = 0.0
    thermal_scale = 1.0

    for ci in range(N_CHAINS):
        if frame < 50:
            t = ease_in_out(frame / 50.0)
            off = start_offsets[ci] * (1 - t) + dock_offsets[ci] * t * 0.4
            bind_strength = 0.0
            thermal_scale = 1.5 - 0.5 * t
        elif frame < 85:
            t = ease_in_out((frame - 50) / 35.0)
            off = dock_offsets[ci] * (0.4 + 0.4 * t)
            bind_strength = t
            thermal_scale = 1.0 - 0.3 * t
        elif frame < 120:
            t = ease_in_out((frame - 85) / 35.0)
            off = dock_offsets[ci] * (0.8 + 0.2 * t)
            # Slight conformational wiggle
            wiggle = 0.3 * np.sin(t * np.pi * 2) * np.array([
                np.cos(ci * np.pi / 2), np.sin(ci * np.pi / 2)])
            off = off + wiggle
            bind_strength = 1.0
            thermal_scale = 0.7 - 0.3 * t
        else:
            off = dock_offsets[ci]
            bind_strength = 1.0
            thermal_scale = 0.4

        offsets.append(off)

    # Apply rotation for later frames
    cos_r, sin_r = np.cos(view_rot), np.sin(view_rot)

    # Draw each chain
    all_pos = []
    for ci in range(N_CHAINS):
        pos = chain_bases[ci] + offsets[ci] + thermal[ci][frame] * thermal_scale

        # Apply rotation
        if view_rot != 0:
            cx, cy = pos.mean(axis=0)
            pos_c = pos - np.array([cx, cy])
            pos_r = np.zeros_like(pos_c)
            pos_r[:, 0] = pos_c[:, 0] * cos_r - pos_c[:, 1] * sin_r
            pos_r[:, 1] = pos_c[:, 0] * sin_r + pos_c[:, 1] * cos_r
            pos = pos_r + np.array([cx, cy])

        all_pos.append(pos)

        color = CHAIN_COLORS_HEX[ci]
        dim_color = CHAIN_COLORS_DIM[ci]

        # Backbone
        segments = np.array([[pos[i], pos[i+1]] for i in range(len(pos)-1)])
        lc = LineCollection(segments, colors=color, linewidths=2.0,
                            alpha=0.7, zorder=2)
        ax_anim.add_collection(lc)

        # Sidechains
        sc = pos + sc_dirs[ci] + thermal[ci][frame] * 0.2
        for i in range(0, SEQ_LEN, 2):
            ax_anim.plot([pos[i, 0], sc[i, 0]], [pos[i, 1], sc[i, 1]],
                         '-', color=dim_color, lw=0.4, alpha=0.3, zorder=2)

        # Residues colored by charge
        ax_anim.scatter(pos[:, 0], pos[:, 1], s=chain_sizes[ci] * 18,
                        c=chain_charges[ci], cmap='coolwarm', vmin=-1, vmax=1,
                        zorder=3, edgecolors=color, linewidths=0.2, alpha=0.8)

        # Binding site glow
        bind_pos = pos[BINDING_START:BINDING_END]
        glow_alpha = 0.15 + 0.15 * np.sin(frame * 0.12 + ci)
        ax_anim.scatter(bind_pos[:, 0], bind_pos[:, 1], s=100,
                        facecolors='none', edgecolors=ACCENT_GREEN,
                        lw=1.5, zorder=4, alpha=0.5 + glow_alpha)

    # Binding interactions between chain pairs
    if bind_strength > 0:
        pair_list = [(0, 2), (1, 2), (3, 2), (0, 1)]
        for ci, cj in pair_list:
            for i in range(BINDING_START, BINDING_END, 1):
                for j in range(BINDING_START, BINDING_END, 2):
                    dist = np.sqrt(((all_pos[ci][i] - all_pos[cj][j])**2).sum())
                    charge_interact = -chain_charges[ci][i] * chain_charges[cj][j]
                    if dist < 6.0 and charge_interact > 0.15:
                        strength = charge_interact * bind_strength
                        color = ACCENT_GREEN if charge_interact > 0.3 else ACCENT_GOLD
                        ax_anim.plot(
                            [all_pos[ci][i, 0], all_pos[cj][j, 0]],
                            [all_pos[ci][i, 1], all_pos[cj][j, 1]],
                            '-', color=color, lw=0.8 * strength,
                            alpha=0.3 * strength, zorder=2)

        # Hydrogen bonds (dotted lines)
        if bind_strength > 0.5:
            for ci, cj in [(0, 2), (1, 2)]:
                for i in range(BINDING_START, BINDING_END, 3):
                    for j in range(BINDING_START, BINDING_END, 3):
                        dist = np.sqrt(((all_pos[ci][i] - all_pos[cj][j])**2).sum())
                        if dist < 4.0:
                            ax_anim.plot(
                                [all_pos[ci][i, 0], all_pos[cj][j, 0]],
                                [all_pos[ci][i, 1], all_pos[cj][j, 1]],
                                ':', color='#88CCFF', lw=0.6,
                                alpha=0.25 * bind_strength, zorder=2)

        # Charge arcs for strong interactions
        if bind_strength > 0.7:
            for ci, cj in [(0, 2)]:
                for i in range(BINDING_START, BINDING_END, 4):
                    j = BINDING_START + (i - BINDING_START + 2) % (BINDING_END - BINDING_START)
                    charge_interact = -chain_charges[ci][i] * chain_charges[cj][j]
                    if charge_interact > 0.4:
                        mid_x = (all_pos[ci][i, 0] + all_pos[cj][j, 0]) / 2
                        mid_y = (all_pos[ci][i, 1] + all_pos[cj][j, 1]) / 2
                        # Draw a small arc via bezier approximation
                        t_arc = np.linspace(0, 1, 20)
                        arc_x = (1-t_arc)**2 * all_pos[ci][i, 0] + \
                                2*(1-t_arc)*t_arc * (mid_x + 0.5) + \
                                t_arc**2 * all_pos[cj][j, 0]
                        arc_y = (1-t_arc)**2 * all_pos[ci][i, 1] + \
                                2*(1-t_arc)*t_arc * (mid_y + 1.0) + \
                                t_arc**2 * all_pos[cj][j, 1]
                        color_arc = '#FF6644' if chain_charges[ci][i] > 0 else '#4466FF'
                        ax_anim.plot(arc_x, arc_y, '-', color=color_arc,
                                     lw=0.8, alpha=0.3 * bind_strength)

    # UI elements
    ax_anim.set_xlim(-18, 18)
    ax_anim.set_ylim(-12, 12)
    ax_anim.set_aspect('equal')
    ax_anim.axis('off')

    # Chain labels
    for ci, name in enumerate(CHAIN_NAMES):
        x_lab = 0.02 + ci * 0.24
        ax_anim.text(x_lab, 0.97, name.upper(),
                     color=CHAIN_COLORS_HEX[ci], fontsize=9, fontweight='bold',
                     transform=ax_anim.transAxes, va='top', fontfamily='monospace')

    # Phase indicator
    if frame < 50:
        phase = 'PHASE 1: FREE CHAINS APPROACHING'
    elif frame < 85:
        phase = 'PHASE 2: INITIAL DOCKING'
    elif frame < 120:
        phase = 'PHASE 3: CONFORMATIONAL TIGHTENING'
    else:
        phase = 'PHASE 4: STABLE COMPLEX'
    ax_anim.text(0.5, 0.02, phase, transform=ax_anim.transAxes,
                 color=TEXT_BRIGHT, fontsize=10, ha='center', fontfamily='monospace',
                 alpha=0.7)

    # Binding energy readout
    if frame >= 50:
        energy = -aff_val * bind_strength
        ax_anim.text(0.5, 0.97, f'BINDING ENERGY: {energy:.3f} kcal/mol',
                     transform=ax_anim.transAxes, color=ACCENT_GREEN, fontsize=10,
                     ha='center', va='top', fontweight='bold', fontfamily='monospace',
                     path_effects=[patheffects.withStroke(linewidth=2, foreground=BG_DARK)])

    # Progress bar at bottom
    progress = frame / N_FRAMES
    ax_anim.plot([-17, -17 + 34 * progress], [-11.5, -11.5],
                 '-', color=ACCENT_CYAN, lw=3, alpha=0.6, solid_capstyle='round')
    ax_anim.plot([-17, 17], [-11.5, -11.5],
                 '-', color=TEXT_DIM, lw=1, alpha=0.2)

    # Legend
    if frame > 10:
        legend_items = [
            ('charge (red/blue)', '#AAAAAA'),
            ('binding site', ACCENT_GREEN),
            ('H-bonds', '#88CCFF'),
            ('charge arcs', '#FF6644'),
        ]
        for li, (text, color) in enumerate(legend_items):
            ax_anim.text(0.98, 0.12 - li * 0.035, text,
                         color=color, fontsize=6, fontfamily='monospace',
                         transform=ax_anim.transAxes, ha='right', alpha=0.6)

anim = animation.FuncAnimation(fig_anim, animate_protein,
                                frames=N_FRAMES, interval=70)
gif_path = os.path.join(ASSETS, "05_protein.gif")
anim.save(gif_path, writer='pillow', fps=14)
plt.close()
print(f"Saved {gif_path}")
