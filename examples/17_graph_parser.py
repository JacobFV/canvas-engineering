"""Graph parser: state with parser tags, operator types, fixpoint iteration.

Demonstrates v2 operator types (observe, bind), ClockExpr IR for
fixpoint-driven iteration, and the residual family for parse errors.

Architecture:
  tokens (observation) - input token sequence
  spans (state, tags=parser) - detected spans/constituents
  nodes (state, tags=parser,object) - parse tree nodes
  parse_error (residual) - parse quality error signal

Operators:
  tokens -> spans = observe (write evidence)
  spans -> nodes = bind (structural binding)

The parser iterates until parse_error drops below a threshold (fixpoint).
It learns to parse bracket sequences like "(()(()))" into a tree.

Run:  python examples/17_graph_parser.py
Out:  assets/examples/17_graph_parser.png
"""

import os
import math
import numpy as np
import torch
import torch.nn as nn
from dataclasses import dataclass

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

from canvas_engineering import (
    Field, compile_program, ConnectivityPolicy,
    CanvasProgram, RegionProgram, ConnectionProgram, ClockSpec,
    RegionScheduler, ResidualSpec, ResidualAccumulator,
    ClockExpr, Periodic, OnEvent, Or,
    clock_periodic, clock_on,
)

ASSETS = os.path.join(os.path.dirname(os.path.dirname(__file__)), "assets", "examples")
os.makedirs(ASSETS, exist_ok=True)

torch.manual_seed(42)

# ── 1. Declare types ──────────────────────────────────────────────────

@dataclass
class GraphParser:
    tokens: Field = Field(2, 4, family="observation",
                          semantic_type="input token embeddings")
    spans: Field = Field(2, 4, family="state", tags=("parser",),
                         loss_weight=2.0,
                         semantic_type="detected span boundaries")
    nodes: Field = Field(2, 4, family="state", tags=("parser", "object"),
                         loss_weight=3.0,
                         semantic_type="parse tree node types")
    parse_error: Field = Field(1, 2, family="residual",
                               semantic_type="parse quality error signal")


# ── 2. Compile with operator semantics ────────────────────────────────

parser = GraphParser()
bound, program = compile_program(
    parser, T=1, H=8, W=8, d_model=48,
    connectivity=ConnectivityPolicy(intra="dense", temporal="same_frame"),
)

# Override connection operators
program.connections[("tokens", "spans")] = ConnectionProgram(operator="observe")
program.connections[("spans", "nodes")] = ConnectionProgram(operator="bind")
program.connections[("nodes", "parse_error")] = ConnectionProgram(operator="emit_residual")

# Add fixpoint clock: iterate while parse_error > threshold
program.regions["spans"] = RegionProgram(
    family="state", tags=("parser",),
    clock=ClockSpec(
        mode="on_event",
        event_source="parse_error.prediction",
        event_threshold=0.2,
        max_silence=3,
        max_inner_steps=5,
        domain="internal",
    ),
)
program.regions["parse_error"] = RegionProgram(
    family="residual",
    carrier="residual",
)

print("=== Graph Parser ===")
print(bound.summary())
print()
print(program.summary())

# Show operators
print("\nConnection operators:")
for (src, dst), cp in program.connections.items():
    print(f"  {src} -> {dst}: operator={cp.operator}")

# Show ClockExpr IR
fixpoint_expr = Or(
    clock_periodic(1),
    clock_on("parse_error.prediction", gt=0.2),
)
print(f"\nFixpoint clock expr: {fixpoint_expr}")
print(f"  Serialized: {fixpoint_expr.to_dict()}")
print(f"  Round-trip: {ClockExpr.from_dict(fixpoint_expr.to_dict())}")


# ── 3. Generate synthetic data ────────────────────────────────────────
# Bracket sequences: learn to identify matching brackets and nesting depth.
# Input: sequence of tokens ( = 1, ) = -1, pad = 0
# Target spans: [start, end, depth] for each matching pair
# Target nodes: nesting depth at each position

SEQ_LEN = 8
TOKEN_DIM = 8   # one-hot-ish embedding
SPAN_DIM = 8    # span boundary encodings
NODE_DIM = 8    # parse tree node features
ERROR_DIM = 2   # parse quality

def generate_bracket_data(n_samples=2048):
    """Generate bracket sequences with parse targets."""
    tokens_list = []
    spans_list = []
    nodes_list = []
    valid_list = []

    for _ in range(n_samples):
        # Generate valid bracket sequence
        seq = []
        depth = 0
        max_depth = 0
        for pos in range(SEQ_LEN):
            if pos >= SEQ_LEN - 1:
                # Must close remaining brackets
                if depth > 0:
                    seq.append(-1)
                    depth -= 1
                else:
                    seq.append(0)
            elif depth == 0 or (torch.rand(1).item() < 0.6 and depth < 3):
                seq.append(1)
                depth += 1
                max_depth = max(max_depth, depth)
            else:
                seq.append(-1)
                depth -= 1

        # Pad to SEQ_LEN
        while len(seq) < SEQ_LEN:
            seq.append(0)
        seq = seq[:SEQ_LEN]

        # Token encoding: embed as 8-dim
        token_emb = torch.zeros(TOKEN_DIM)
        for i, s in enumerate(seq[:TOKEN_DIM]):
            token_emb[i] = float(s)

        # Span targets: encode depth at each position
        span_emb = torch.zeros(SPAN_DIM)
        d = 0
        for i, s in enumerate(seq[:SPAN_DIM]):
            if s == 1:
                d += 1
            span_emb[i] = d / max(max_depth, 1)  # normalized depth
            if s == -1:
                d = max(0, d - 1)

        # Node targets: binary features about parse structure
        node_emb = torch.zeros(NODE_DIM)
        node_emb[0] = float(max_depth) / 4.0  # max nesting
        node_emb[1] = float(sum(1 for s in seq if s == 1)) / SEQ_LEN  # open bracket fraction
        node_emb[2] = float(depth == 0)  # is balanced
        node_emb[3] = float(max_depth > 1)  # has nesting
        # Fill remaining with derived features
        for i in range(4, NODE_DIM):
            node_emb[i] = span_emb[i] * node_emb[i % 4]

        is_valid = (depth == 0)

        tokens_list.append(token_emb)
        spans_list.append(span_emb)
        nodes_list.append(node_emb)
        valid_list.append(float(is_valid))

    return {
        'tokens': torch.stack(tokens_list),
        'spans': torch.stack(spans_list),
        'nodes': torch.stack(nodes_list),
        'valid': torch.tensor(valid_list),
    }

data_tr = generate_bracket_data()
data_val = generate_bracket_data(512)


# ── 4. Build parser model with fixpoint iteration ────────────────────

class ParserModel(nn.Module):
    def __init__(self, bound_schema, d_model=48, nhead=4, max_iterations=5):
        super().__init__()
        self.bound = bound_schema
        self.d = d_model
        self.max_iterations = max_iterations
        N = bound_schema.layout.num_positions

        self.pos_emb = nn.Parameter(torch.randn(1, N, d_model) * 0.02)
        layer = nn.TransformerEncoderLayer(
            d_model=d_model, nhead=nhead, dim_feedforward=192,
            dropout=0.0, batch_first=True,
        )
        self.encoder = nn.TransformerEncoder(layer, num_layers=2)
        mask = bound_schema.topology.to_additive_mask(bound_schema.layout)
        self.register_buffer('attn_mask', mask)

        tok_n = len(bound_schema.layout.region_indices("tokens"))
        span_n = len(bound_schema.layout.region_indices("spans"))
        node_n = len(bound_schema.layout.region_indices("nodes"))
        err_n = len(bound_schema.layout.region_indices("parse_error"))

        self.tok_proj = nn.Linear(TOKEN_DIM, tok_n * d_model)
        self.span_out = nn.Linear(span_n * d_model, SPAN_DIM)
        self.node_out = nn.Linear(node_n * d_model, NODE_DIM)
        self.err_out = nn.Linear(err_n * d_model, ERROR_DIM)

        # Iterative refinement: residual connection for spans
        self.span_refine = nn.Linear(SPAN_DIM, span_n * d_model)

        self.tok_n = tok_n
        self.span_n = span_n
        self.node_n = node_n
        self.err_n = err_n

    def forward(self, tokens, n_iterations=None):
        if n_iterations is None:
            n_iterations = self.max_iterations

        B = tokens.shape[0]
        tok_idx = self.bound.layout.region_indices("tokens")
        span_idx = self.bound.layout.region_indices("spans")
        node_idx = self.bound.layout.region_indices("nodes")
        err_idx = self.bound.layout.region_indices("parse_error")

        # Initial pass
        canvas = self.pos_emb.expand(B, -1, -1).clone()
        canvas[:, tok_idx] = canvas[:, tok_idx] + \
            self.tok_proj(tokens).reshape(B, self.tok_n, self.d)

        all_spans = []
        all_errors = []
        span_state = torch.zeros(B, SPAN_DIM, device=tokens.device)

        for iteration in range(n_iterations):
            # Inject span refinement from previous iteration
            if iteration > 0:
                canvas[:, span_idx] = canvas[:, span_idx] + \
                    self.span_refine(span_state).reshape(B, self.span_n, self.d)

            canvas_out = self.encoder(canvas, mask=self.attn_mask)

            span_state = self.span_out(canvas_out[:, span_idx].reshape(B, -1))
            error = self.err_out(canvas_out[:, err_idx].reshape(B, -1))

            all_spans.append(span_state)
            all_errors.append(error)

        # Final node output
        nodes = self.node_out(canvas_out[:, node_idx].reshape(B, -1))

        return span_state, nodes, all_spans, all_errors


model = ParserModel(bound)
optimizer = torch.optim.AdamW(model.parameters(), lr=2e-3, weight_decay=1e-4)
sched = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, 500)


# ── 5. Train ──────────────────────────────────────────────────────────

losses_total = []
convergence_history = []
n_epochs = 500
batch_size = 64

print("\nTraining graph parser with fixpoint iteration...")
for epoch in range(n_epochs):
    idx = torch.randint(0, len(data_tr['tokens']), (batch_size,))
    spans, nodes, all_spans, all_errors = model(data_tr['tokens'][idx])

    # Loss on final output
    span_loss = ((spans - data_tr['spans'][idx]) ** 2).mean() * 2.0
    node_loss = ((nodes - data_tr['nodes'][idx]) ** 2).mean() * 3.0

    # Fixpoint convergence loss: each iteration should improve
    iter_losses = []
    for s in all_spans:
        iter_losses.append(((s - data_tr['spans'][idx]) ** 2).mean())
    convergence_loss = sum(iter_losses) * 0.5

    loss = span_loss + node_loss + convergence_loss

    optimizer.zero_grad()
    loss.backward()
    optimizer.step()
    sched.step()

    losses_total.append(loss.item())
    convergence_history.append([il.item() for il in iter_losses])

    if epoch % 100 == 0:
        iter_mses = [il.item() for il in iter_losses]
        print(f"  epoch {epoch:3d}: loss={loss.item():.4f} iter_errors={[f'{x:.3f}' for x in iter_mses]}")


# ── 6. Evaluate ───────────────────────────────────────────────────────

model.eval()
with torch.no_grad():
    spans_val, nodes_val, all_spans_val, all_errors_val = model(data_val['tokens'])

    span_mse = ((spans_val - data_val['spans']) ** 2).mean().item()
    node_mse = ((nodes_val - data_val['nodes']) ** 2).mean().item()

    # Per-iteration quality
    iter_qualities = []
    for s in all_spans_val:
        iter_qualities.append(((s - data_val['spans']) ** 2).mean().item())

    # Quality vs number of iterations
    qualities_by_n = []
    for n in range(1, 6):
        s, nd, _, _ = model(data_val['tokens'], n_iterations=n)
        q = ((s - data_val['spans']) ** 2).mean().item()
        qualities_by_n.append(q)

    print(f"\n  Span MSE: {span_mse:.4f}")
    print(f"  Node MSE: {node_mse:.4f}")
    print(f"  Per-iteration quality: {[f'{q:.4f}' for q in iter_qualities]}")


# ── 7. Visualize ──────────────────────────────────────────────────────

PARSER_COLORS = {
    'tokens': '#4A90D9', 'spans': '#E67E22',
    'nodes': '#2ECC71', 'parse_error': '#E74C3C',
}

fig, axes = plt.subplots(2, 2, figsize=(12, 9), dpi=150)
fig.patch.set_facecolor('white')
fig.suptitle('Graph Parser: Operator Types + Fixpoint Iteration',
             fontsize=14, fontweight='bold', y=0.98)

# (a) Parse accuracy: span + node MSE
ax = axes[0, 0]
ax.set_title('Parse Accuracy', fontsize=11, fontweight='bold')
labels = ['Span MSE', 'Node MSE']
values = [span_mse, node_mse]
colors_bar = [PARSER_COLORS['spans'], PARSER_COLORS['nodes']]
bars = ax.bar(labels, values, color=colors_bar, edgecolor='white', linewidth=2)
for bar, val in zip(bars, values):
    ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.002,
            f'{val:.4f}', ha='center', va='bottom', fontsize=11, fontweight='bold')
ax.set_ylabel('MSE')
ax.grid(True, alpha=0.2, axis='y')

# Annotate operators
ax.text(0.98, 0.95, 'tokens->spans: observe\nspans->nodes: bind',
        transform=ax.transAxes, ha='right', va='top', fontsize=9,
        bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5),
        family='monospace')

# (b) Fixpoint convergence: MSE per iteration
ax = axes[0, 1]
ax.set_title('Fixpoint Convergence', fontsize=11, fontweight='bold')
iterations = range(1, len(iter_qualities) + 1)
ax.plot(iterations, iter_qualities, 'o-', color=PARSER_COLORS['spans'],
        lw=2.5, markersize=8, label='span MSE per iteration')
ax.set_xlabel('Iteration')
ax.set_ylabel('Span MSE')
ax.grid(True, alpha=0.2)
ax.legend(fontsize=9)

# Also show quality vs number of iterations
ax2 = ax.twinx()
ax2.plot(range(1, 6), qualities_by_n, 's--', color=PARSER_COLORS['nodes'],
         lw=2, markersize=6, alpha=0.7, label='quality at N iters')
ax2.set_ylabel('Quality (N iterations)', color=PARSER_COLORS['nodes'], fontsize=9)
ax2.tick_params(axis='y', labelcolor=PARSER_COLORS['nodes'])
ax2.legend(fontsize=8, loc='center right')

# (c) Operator-annotated topology
ax = axes[1, 0]
ax.set_title('Operator-Annotated Topology', fontsize=11, fontweight='bold')
# Draw regions as boxes with arrows for operators
H, W = bound.layout.H, bound.layout.W
grid = np.ones((H, W, 3)) * 0.93
for name, color in PARSER_COLORS.items():
    if name not in bound:
        continue
    bf = bound[name]
    r, g, b = int(color[1:3], 16)/255, int(color[3:5], 16)/255, int(color[5:7], 16)/255
    h0, h1 = bf.spec.bounds[2], bf.spec.bounds[3]
    w0, w1 = bf.spec.bounds[4], bf.spec.bounds[5]
    grid[h0:h1, w0:w1] = [r, g, b]
    rp = program.regions.get(name, RegionProgram())
    label = f'{name}\n({rp.family})'
    ax.text((w0 + w1) / 2 - 0.5, (h0 + h1) / 2 - 0.5,
            label, ha='center', va='center', fontsize=5, fontweight='bold', color='white')
ax.imshow(grid, aspect='equal', interpolation='nearest')

# Draw operator arrows
for (src, dst), cp in program.connections.items():
    if src in bound and dst in bound:
        s_bf = bound[src]
        d_bf = bound[dst]
        s_h = (s_bf.spec.bounds[2] + s_bf.spec.bounds[3]) / 2 - 0.5
        s_w = (s_bf.spec.bounds[4] + s_bf.spec.bounds[5]) / 2 - 0.5
        d_h = (d_bf.spec.bounds[2] + d_bf.spec.bounds[3]) / 2 - 0.5
        d_w = (d_bf.spec.bounds[4] + d_bf.spec.bounds[5]) / 2 - 0.5
        ax.annotate(cp.operator, xy=(d_w, d_h), xytext=(s_w, s_h),
                    arrowprops=dict(arrowstyle='->', color='white', lw=2),
                    fontsize=6, color='white', fontweight='bold',
                    ha='center', va='center')
ax.set_xlabel('W'); ax.set_ylabel('H')

# (d) Training curves
ax = axes[1, 1]
ax.set_title('Training Curves', fontsize=11, fontweight='bold')
ax.semilogy(losses_total, color='#2C3E50', lw=1.5, alpha=0.5)
w = 20
smoothed = np.convolve(losses_total, np.ones(w)/w, mode='valid')
ax.semilogy(range(w-1, len(losses_total)), smoothed, color='#E74C3C', lw=2, label='smoothed')

# Also plot convergence over training (last iteration MSE)
last_iter_mse = [ch[-1] for ch in convergence_history]
smoothed_conv = np.convolve(last_iter_mse, np.ones(w)/w, mode='valid')
ax.semilogy(range(w-1, len(last_iter_mse)), smoothed_conv,
            color=PARSER_COLORS['spans'], lw=2, label='final iteration MSE')

ax.set_xlabel('Epoch')
ax.set_ylabel('Loss')
ax.legend(fontsize=9)
ax.grid(True, alpha=0.2)

plt.tight_layout(rect=[0, 0, 1, 0.96])
path = os.path.join(ASSETS, "17_graph_parser.png")
fig.savefig(path, bbox_inches='tight', facecolor='white', dpi=150)
plt.close()
print(f"\n  saved {path}")
