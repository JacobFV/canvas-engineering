"""Multi-objective training for the browser canvas agent.

Three learning signals:
1. Behavioral cloning (BC): supervised action prediction from expert demos
2. Next-page prediction (SSL): predict what the page looks like after an action
3. Task completion reward (RL): REINFORCE-style reward for task completion

Uses canvas-engineering's public API:
- compile_program() with families -> right learning defaults
- RegionScheduler: plan only updates when prediction_error > threshold
- ResidualAccumulator: tracks prediction errors
- AttentionDispatcher: topology-aware attention routing

Usage:
    python research/browser/train.py --epochs 100 --d_model 128
    python research/browser/train.py --mode flat --epochs 100   # flat baseline
    python research/browser/train.py --mode dense --epochs 100  # dense baseline
"""

from __future__ import annotations

import argparse
import json
import math
import sys
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Optional, Set, Tuple

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, Dataset, TensorDataset

# Ensure project root is importable
_CE_ROOT = Path(__file__).resolve().parents[2]
if str(_CE_ROOT) not in sys.path:
    sys.path.insert(0, str(_CE_ROOT))

from canvas_engineering import (
    AttentionDispatcher,
    CanvasTopology,
    Connection,
    RegionScheduler,
    ResidualAccumulator,
    ResidualSpec,
    ClockSpec,
    default_learning,
    FAMILY_DEFAULTS,
)
from canvas_engineering.program import CanvasProgram, RegionProgram

from research.browser.browser_canvas import (
    BrowserAgent,
    build_browser_program,
    get_region_names,
    OBSERVATION_REGIONS,
    STATE_REGIONS,
    MEMORY_REGIONS,
    ACTION_REGIONS,
    RESIDUAL_REGIONS,
)
from research.browser.environment import (
    BrowserEnvironment,
    BrowserAction,
    generate_demonstrations,
    batch_demonstrations,
    encode_instruction,
    encode_screen,
    SCREEN_C, SCREEN_H, SCREEN_W,
    DOM_MAX_ELEMENTS, DOM_FEATURE_DIM,
    NUM_ACTIONS, ACTION_NAMES,
)

RESULTS_DIR = Path(_CE_ROOT) / "research" / "browser" / "results"


# ---- Dataset ----

class BrowserDataset(Dataset):
    """Dataset of (observation, action, next_observation, reward) tuples."""

    def __init__(self, data: Dict[str, torch.Tensor]):
        self.data = data
        self._len = data["screens"].shape[0]

    def __len__(self):
        return self._len

    def __getitem__(self, idx):
        return {k: v[idx] for k, v in self.data.items()}


# ---- Canvas-based browser agent model ----

class BrowserCanvasModel(nn.Module):
    """Browser agent using canvas-engineering topology.

    Architecture:
    1. Input encoders: project screen/DOM/instruction to d_model
    2. Region embeddings: learned per-region embedding
    3. N layers of AttentionDispatcher with browser topology
    4. Action heads: predict action type, coordinates, text
    5. Next-page prediction head (SSL)
    6. Residual tracking: ResidualAccumulator monitors prediction error
    """

    def __init__(
        self,
        d_model: int = 128,
        n_heads: int = 4,
        n_layers: int = 4,
        dropout: float = 0.1,
        instruction_dim: int = 32,
    ):
        super().__init__()
        self.d_model = d_model
        self.n_layers = n_layers

        # Build canvas program
        bound, program = build_browser_program(d_model=d_model)
        self.bound = bound
        self.program = program
        self.region_names = list(bound.schema.layout.regions.keys())
        self.n_regions = len(self.region_names)
        self._name_to_idx = {n: i for i, n in enumerate(self.region_names)}

        # Input encoders
        self.screen_encoder = nn.Sequential(
            nn.Conv2d(SCREEN_C, 32, 3, padding=1),
            nn.ReLU(),
            nn.Conv2d(32, 64, 3, padding=1),
            nn.ReLU(),
            nn.AdaptiveAvgPool2d((4, 4)),  # -> (64, 4, 4) = 1024
            nn.Flatten(),
            nn.Linear(64 * 4 * 4, d_model),
        )

        self.dom_element_encoder = nn.Sequential(
            nn.Linear(DOM_FEATURE_DIM, d_model),
            nn.ReLU(),
            nn.Linear(d_model, d_model),
        )

        self.dom_layout_encoder = nn.Sequential(
            nn.Linear(4, d_model),
            nn.ReLU(),
            nn.Linear(d_model, d_model),
        )

        self.instruction_encoder = nn.Sequential(
            nn.Linear(instruction_dim, d_model),
            nn.ReLU(),
            nn.Linear(d_model, d_model),
        )

        # Learned region embeddings
        self.region_embeddings = nn.Embedding(self.n_regions, d_model)

        # Layer norm
        self.input_norm = nn.LayerNorm(d_model)

        # Build a simple 1-position-per-region layout for the dispatcher.
        # The compile_program layout has multi-position regions (e.g. 8x8),
        # but we operate at the region level: one d_model vector per region.
        from canvas_engineering.canvas import CanvasLayout, RegionSpec

        dispatch_regions = {}
        for i, name in enumerate(self.region_names):
            dispatch_regions[name] = RegionSpec(bounds=(0, 1, i, i + 1, 0, 1))
        dispatch_layout = CanvasLayout(
            T=1, H=self.n_regions, W=1, d_model=d_model,
            regions=dispatch_regions,
        )

        # Attention layers using canvas topology
        topology = program.schema.topology

        self.dispatchers = nn.ModuleList()
        self.layer_norms = nn.ModuleList()
        self.ffn_layers = nn.ModuleList()
        self.ffn_norms = nn.ModuleList()

        for _ in range(n_layers):
            if topology is not None:
                dispatcher = AttentionDispatcher(
                    topology=topology,
                    layout=dispatch_layout,
                    d_model=d_model,
                    n_heads=n_heads,
                    dropout=dropout,
                )
            else:
                # Fallback: dense
                dense_top = CanvasTopology.dense(self.region_names)
                dispatcher = AttentionDispatcher(
                    topology=dense_top,
                    layout=dispatch_layout,
                    d_model=d_model,
                    n_heads=n_heads,
                    dropout=dropout,
                )
            self.dispatchers.append(dispatcher)
            self.layer_norms.append(nn.LayerNorm(d_model))
            self.ffn_layers.append(nn.Sequential(
                nn.Linear(d_model, d_model * 4),
                nn.GELU(),
                nn.Dropout(dropout),
                nn.Linear(d_model * 4, d_model),
                nn.Dropout(dropout),
            ))
            self.ffn_norms.append(nn.LayerNorm(d_model))

        # Action heads
        self.action_type_head = nn.Linear(d_model, NUM_ACTIONS)
        self.coord_head = nn.Linear(d_model, 2)
        self.text_head = nn.Linear(d_model, instruction_dim)

        # Next-page prediction head (SSL)
        self.next_page_head = nn.Sequential(
            nn.Linear(d_model, d_model),
            nn.ReLU(),
            nn.Linear(d_model, SCREEN_C * 4 * 4),  # predict coarse next screen
        )

        # Reward prediction head
        self.reward_head = nn.Sequential(
            nn.Linear(d_model, d_model // 2),
            nn.ReLU(),
            nn.Linear(d_model // 2, 1),
        )

        # Region scheduler
        self.scheduler = RegionScheduler(program)

        # Residual accumulator for prediction error tracking
        residual_names = [n for n in self.region_names
                          if n in RESIDUAL_REGIONS
                          or (n in program.regions and
                              program.regions[n].family == "residual")]
        if not residual_names:
            residual_names = ["_dummy_residual"]
        self.residual_accumulator = ResidualAccumulator(
            residual_names,
            ResidualSpec(kinds=("prediction",), decay=0.9),
        )

        # Track planning frequency
        self.plan_fire_count = 0
        self.total_steps = 0

    def _encode_inputs(
        self,
        screen: torch.Tensor,
        dom_elements: torch.Tensor,
        dom_layout: torch.Tensor,
        instruction: torch.Tensor,
    ) -> torch.Tensor:
        """Encode all inputs and arrange as region tokens.

        Returns: (B, n_regions, d_model) tensor.
        """
        B = screen.shape[0]
        device = screen.device

        # Encode each modality
        screen_emb = self.screen_encoder(screen)     # (B, d_model)
        # Pool DOM elements: mean over elements
        dom_elem_emb = self.dom_element_encoder(dom_elements).mean(dim=1)  # (B, d_model)
        dom_layout_emb = self.dom_layout_encoder(dom_layout).mean(dim=1)   # (B, d_model)
        inst_emb = self.instruction_encoder(instruction)                    # (B, d_model)

        # Build region token sequence
        # Each region gets d_model embedding. We broadcast the input to
        # the right regions and zero-initialize internal/action regions.
        tokens = torch.zeros(B, self.n_regions, self.d_model, device=device)

        # Map encoded inputs to their regions
        input_mapping = {
            "screen.pixels": screen_emb,
            "dom.elements": dom_elem_emb,
            "dom.layout": dom_layout_emb,
            "instruction": inst_emb,
        }

        for name, emb in input_mapping.items():
            if name in self._name_to_idx:
                idx = self._name_to_idx[name]
                tokens[:, idx] = emb

        # Add region embeddings
        region_ids = torch.arange(self.n_regions, device=device)
        tokens = tokens + self.region_embeddings(region_ids).unsqueeze(0)

        return self.input_norm(tokens)

    def forward(
        self,
        screen: torch.Tensor,
        dom_elements: torch.Tensor,
        dom_layout: torch.Tensor,
        instruction: torch.Tensor,
        external_t: int = 0,
    ) -> Dict[str, torch.Tensor]:
        """Forward pass through the browser agent.

        Args:
            screen: (B, C, H, W) screen observation
            dom_elements: (B, max_elem, dom_feat) DOM features
            dom_layout: (B, max_elem, 4) bounding boxes
            instruction: (B, instruction_dim) encoded instruction
            external_t: current environment step (for scheduling)

        Returns:
            Dict with action_logits, coord_pred, text_pred,
            next_page_pred, reward_pred, and active_regions.
        """
        B = screen.shape[0]

        # Encode inputs
        x = self._encode_inputs(screen, dom_elements, dom_layout, instruction)

        # Get active regions from scheduler
        summaries = self.residual_accumulator.summaries()
        active = self.scheduler.step(external_t, summaries=summaries)
        self.total_steps += 1
        if "agent.plan" in active:
            self.plan_fire_count += 1

        # Run through attention layers
        for i in range(self.n_layers):
            residual = x
            x = self.dispatchers[i](x, active_regions=active)
            x = self.layer_norms[i](residual + x)

            residual = x
            x = self.ffn_layers[i](x)
            x = self.ffn_norms[i](residual + x)

        # Read out from specific regions
        outputs = {}

        # Action predictions: from action regions
        action_tokens = []
        for name in ACTION_REGIONS:
            if name in self._name_to_idx:
                idx = self._name_to_idx[name]
                action_tokens.append(x[:, idx])
        if action_tokens:
            action_repr = torch.stack(action_tokens, dim=1).mean(dim=1)
        else:
            action_repr = x.mean(dim=1)

        outputs["action_logits"] = self.action_type_head(action_repr)
        outputs["coord_pred"] = self.coord_head(action_repr)
        outputs["text_pred"] = self.text_head(action_repr)

        # Next-page prediction: from state regions (page_belief)
        belief_idx = self._name_to_idx.get("agent.page_belief")
        if belief_idx is not None:
            belief_repr = x[:, belief_idx]
        else:
            belief_repr = x.mean(dim=1)
        outputs["next_page_pred"] = self.next_page_head(belief_repr)

        # Reward prediction
        global_repr = x.mean(dim=1)
        outputs["reward_pred"] = self.reward_head(global_repr).squeeze(-1)

        # Update residual accumulator with prediction quality
        # (computed in training loop when we have targets)
        outputs["active_regions"] = active
        outputs["hidden_states"] = x

        return outputs


# ---- Flat baseline model ----

class FlatBaselineModel(nn.Module):
    """Flat MLP baseline (no canvas structure)."""

    def __init__(
        self,
        d_model: int = 128,
        n_layers: int = 4,
        dropout: float = 0.1,
        instruction_dim: int = 32,
    ):
        super().__init__()
        self.d_model = d_model
        input_dim = SCREEN_C * SCREEN_H * SCREEN_W + DOM_MAX_ELEMENTS * DOM_FEATURE_DIM + DOM_MAX_ELEMENTS * 4 + instruction_dim

        layers = [nn.Linear(input_dim, d_model), nn.ReLU(), nn.Dropout(dropout)]
        for _ in range(n_layers - 1):
            layers.extend([
                nn.Linear(d_model, d_model),
                nn.ReLU(),
                nn.Dropout(dropout),
            ])
        self.backbone = nn.Sequential(*layers)

        self.action_type_head = nn.Linear(d_model, NUM_ACTIONS)
        self.coord_head = nn.Linear(d_model, 2)
        self.text_head = nn.Linear(d_model, instruction_dim)
        self.next_page_head = nn.Linear(d_model, SCREEN_C * 4 * 4)
        self.reward_head = nn.Linear(d_model, 1)

        self.plan_fire_count = 0
        self.total_steps = 0

    def forward(
        self,
        screen: torch.Tensor,
        dom_elements: torch.Tensor,
        dom_layout: torch.Tensor,
        instruction: torch.Tensor,
        external_t: int = 0,
    ) -> Dict[str, torch.Tensor]:
        B = screen.shape[0]
        flat = torch.cat([
            screen.reshape(B, -1),
            dom_elements.reshape(B, -1),
            dom_layout.reshape(B, -1),
            instruction,
        ], dim=1)

        h = self.backbone(flat)

        self.total_steps += 1
        self.plan_fire_count += 1  # always "plans"

        return {
            "action_logits": self.action_type_head(h),
            "coord_pred": self.coord_head(h),
            "text_pred": self.text_head(h),
            "next_page_pred": self.next_page_head(h),
            "reward_pred": self.reward_head(h).squeeze(-1),
            "active_regions": set(),
            "hidden_states": h.unsqueeze(1),
        }


# ---- Dense baseline model ----

class DenseBaselineModel(nn.Module):
    """Dense transformer baseline (fully connected, no structured topology)."""

    def __init__(
        self,
        d_model: int = 128,
        n_heads: int = 4,
        n_layers: int = 4,
        dropout: float = 0.1,
        instruction_dim: int = 32,
        n_tokens: int = 16,
    ):
        super().__init__()
        self.d_model = d_model
        self.n_tokens = n_tokens

        self.screen_encoder = nn.Sequential(
            nn.Conv2d(SCREEN_C, 32, 3, padding=1),
            nn.ReLU(),
            nn.AdaptiveAvgPool2d((2, 2)),
            nn.Flatten(),
            nn.Linear(32 * 4, d_model),
        )
        self.dom_encoder = nn.Linear(DOM_FEATURE_DIM, d_model)
        self.layout_encoder = nn.Linear(4, d_model)
        self.inst_encoder = nn.Linear(instruction_dim, d_model)

        # Project to n_tokens
        self.input_proj = nn.Linear(d_model * 4, n_tokens * d_model)

        encoder_layer = nn.TransformerEncoderLayer(
            d_model=d_model, nhead=n_heads, dim_feedforward=d_model * 4,
            dropout=dropout, batch_first=True,
        )
        self.transformer = nn.TransformerEncoder(encoder_layer, num_layers=n_layers)

        self.action_type_head = nn.Linear(d_model, NUM_ACTIONS)
        self.coord_head = nn.Linear(d_model, 2)
        self.text_head = nn.Linear(d_model, instruction_dim)
        self.next_page_head = nn.Linear(d_model, SCREEN_C * 4 * 4)
        self.reward_head = nn.Linear(d_model, 1)

        self.plan_fire_count = 0
        self.total_steps = 0

    def forward(
        self,
        screen: torch.Tensor,
        dom_elements: torch.Tensor,
        dom_layout: torch.Tensor,
        instruction: torch.Tensor,
        external_t: int = 0,
    ) -> Dict[str, torch.Tensor]:
        B = screen.shape[0]

        s = self.screen_encoder(screen)
        d = self.dom_encoder(dom_elements).mean(dim=1)
        l = self.layout_encoder(dom_layout).mean(dim=1)
        inst = self.inst_encoder(instruction)

        combined = torch.cat([s, d, l, inst], dim=1)
        tokens = self.input_proj(combined).reshape(B, self.n_tokens, self.d_model)
        tokens = self.transformer(tokens)

        pooled = tokens.mean(dim=1)
        self.total_steps += 1
        self.plan_fire_count += 1

        return {
            "action_logits": self.action_type_head(pooled),
            "coord_pred": self.coord_head(pooled),
            "text_pred": self.text_head(pooled),
            "next_page_pred": self.next_page_head(pooled),
            "reward_pred": self.reward_head(pooled).squeeze(-1),
            "active_regions": set(),
            "hidden_states": tokens,
        }


# ---- Training loop ----

def compute_losses(
    outputs: Dict[str, torch.Tensor],
    batch: Dict[str, torch.Tensor],
    bc_weight: float = 1.0,
    ssl_weight: float = 0.5,
    rl_weight: float = 0.3,
) -> Tuple[torch.Tensor, Dict[str, float]]:
    """Compute multi-objective loss.

    Args:
        outputs: Model outputs from forward pass
        batch: Training batch with targets
        bc_weight: Weight for behavioral cloning loss
        ssl_weight: Weight for next-page prediction loss
        rl_weight: Weight for reward-weighted loss

    Returns:
        (total_loss, loss_dict) tuple
    """
    loss_dict = {}

    # 1. Behavioral cloning: supervised action prediction
    bc_action = F.cross_entropy(
        outputs["action_logits"], batch["action_types"],
    )
    bc_coord = F.mse_loss(
        torch.sigmoid(outputs["coord_pred"]),
        batch["action_coords"],
    )
    bc_text = F.mse_loss(outputs["text_pred"], batch["action_texts"])
    bc_loss = bc_action + bc_coord + 0.5 * bc_text
    loss_dict["bc_action"] = bc_action.item()
    loss_dict["bc_coord"] = bc_coord.item()
    loss_dict["bc_text"] = bc_text.item()
    loss_dict["bc_total"] = bc_loss.item()

    # 2. Next-page prediction (SSL)
    # Predict coarse version of next screen
    next_screen_coarse = F.adaptive_avg_pool2d(batch["next_screens"], (4, 4))
    next_screen_flat = next_screen_coarse.reshape(next_screen_coarse.shape[0], -1)
    ssl_loss = F.mse_loss(outputs["next_page_pred"], next_screen_flat)
    loss_dict["ssl_next_page"] = ssl_loss.item()

    # 3. Task completion reward (RL-style)
    # Use reward-weighted log-likelihood for action prediction
    reward = batch["rewards"]
    # Normalize rewards for stability
    if reward.std() > 1e-6:
        reward_norm = (reward - reward.mean()) / (reward.std() + 1e-8)
    else:
        reward_norm = reward
    log_probs = -F.cross_entropy(
        outputs["action_logits"], batch["action_types"], reduction="none",
    )
    rl_loss = -(reward_norm * log_probs).mean()
    loss_dict["rl_reward"] = rl_loss.item()

    # Reward prediction auxiliary loss
    reward_pred_loss = F.mse_loss(outputs["reward_pred"], reward)
    loss_dict["reward_pred"] = reward_pred_loss.item()

    # Total loss
    total = bc_weight * bc_loss + ssl_weight * ssl_loss + rl_weight * rl_loss + 0.1 * reward_pred_loss
    loss_dict["total"] = total.item()

    return total, loss_dict


def train_epoch(
    model: nn.Module,
    dataloader: DataLoader,
    optimizer: torch.optim.Optimizer,
    epoch: int,
    bc_weight: float = 1.0,
    ssl_weight: float = 0.5,
    rl_weight: float = 0.3,
) -> Dict[str, float]:
    """Train for one epoch."""
    model.train()
    total_losses = {}
    n_batches = 0

    for batch_idx, batch in enumerate(dataloader):
        optimizer.zero_grad()

        outputs = model(
            screen=batch["screens"],
            dom_elements=batch["dom_elements"],
            dom_layout=batch["dom_layouts"],
            instruction=batch["instructions"],
            external_t=epoch * len(dataloader) + batch_idx,
        )

        loss, loss_dict = compute_losses(
            outputs, batch,
            bc_weight=bc_weight,
            ssl_weight=ssl_weight,
            rl_weight=rl_weight,
        )

        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        optimizer.step()

        # Update residual accumulator if the model has one
        if hasattr(model, "residual_accumulator"):
            # Use prediction error for residual tracking
            with torch.no_grad():
                next_screen_coarse = F.adaptive_avg_pool2d(
                    batch["next_screens"], (4, 4),
                )
                pred_error = (
                    outputs["next_page_pred"]
                    - next_screen_coarse.reshape(next_screen_coarse.shape[0], -1)
                ).abs().mean()

            for rn in model.residual_accumulator.region_names:
                if rn != "_dummy_residual":
                    model.residual_accumulator.update(rn, pred_error)

        # Accumulate losses
        for k, v in loss_dict.items():
            total_losses[k] = total_losses.get(k, 0.0) + v
        n_batches += 1

    # Average
    avg_losses = {k: v / max(n_batches, 1) for k, v in total_losses.items()}

    # Add scheduling stats
    if hasattr(model, "total_steps") and model.total_steps > 0:
        avg_losses["plan_fire_rate"] = model.plan_fire_count / max(model.total_steps, 1)

    return avg_losses


def evaluate_model(
    model: nn.Module,
    dataloader: DataLoader,
) -> Dict[str, float]:
    """Evaluate the model on a dataset."""
    model.eval()
    total_losses = {}
    n_batches = 0
    correct_actions = 0
    total_actions = 0
    top3_correct = 0

    with torch.no_grad():
        for batch in dataloader:
            outputs = model(
                screen=batch["screens"],
                dom_elements=batch["dom_elements"],
                dom_layout=batch["dom_layouts"],
                instruction=batch["instructions"],
            )

            _, loss_dict = compute_losses(outputs, batch)

            # Action accuracy
            pred_actions = outputs["action_logits"].argmax(dim=1)
            correct_actions += (pred_actions == batch["action_types"]).sum().item()

            # Top-3 accuracy
            if outputs["action_logits"].shape[1] >= 3:
                top3 = outputs["action_logits"].topk(3, dim=1).indices
                for i in range(batch["action_types"].shape[0]):
                    if batch["action_types"][i] in top3[i]:
                        top3_correct += 1

            total_actions += batch["action_types"].shape[0]

            for k, v in loss_dict.items():
                total_losses[k] = total_losses.get(k, 0.0) + v
            n_batches += 1

    avg_losses = {k: v / max(n_batches, 1) for k, v in total_losses.items()}
    avg_losses["action_accuracy_top1"] = correct_actions / max(total_actions, 1)
    avg_losses["action_accuracy_top3"] = top3_correct / max(total_actions, 1)

    return avg_losses


# ---- Rollout evaluation ----

def evaluate_rollout(
    model: nn.Module,
    n_episodes: int = 50,
    seed: int = 999,
    instruction_dim: int = 32,
) -> Dict[str, float]:
    """Evaluate by running the agent in the environment."""
    model.eval()
    env = BrowserEnvironment(seed=seed)

    results_by_type = {}
    total_reward = 0.0
    total_success = 0
    total_steps = 0

    task_types = ["click", "type", "navigate", "form"]

    for ep in range(n_episodes):
        tt = task_types[ep % len(task_types)]
        obs, task = env.reset(task_type=tt)

        ep_reward = 0.0
        ep_steps = 0

        for step in range(task.max_steps):
            with torch.no_grad():
                inst = encode_instruction(task.instruction, dim=instruction_dim)
                outputs = model(
                    screen=obs.screen.unsqueeze(0),
                    dom_elements=obs.dom_elements.unsqueeze(0),
                    dom_layout=obs.dom_layout.unsqueeze(0),
                    instruction=inst.unsqueeze(0),
                    external_t=step,
                )

                action_type = outputs["action_logits"].argmax(dim=1).item()
                coords = torch.sigmoid(outputs["coord_pred"][0])
                x, y = coords[0].item(), coords[1].item()

                # Decode text for type actions
                text = ""
                if action_type == 2:
                    text_pred = outputs["text_pred"][0]
                    top_idx = text_pred.abs().topk(min(8, text_pred.shape[0])).indices
                    chars = []
                    for idx in top_idx:
                        ch = (idx.item() * 7 + 97) % 128
                        if 32 <= ch < 127:
                            chars.append(chr(ch))
                    text = "".join(chars[:8]) if chars else "a"

                action = BrowserAction(action_type=action_type, x=x, y=y, text=text)

            obs = env.step(action)
            ep_reward += obs.reward
            ep_steps += 1

            if obs.done:
                break

        total_reward += ep_reward
        total_steps += ep_steps
        if ep_reward >= 0.9:
            total_success += 1

        if tt not in results_by_type:
            results_by_type[tt] = {"success": 0, "count": 0, "reward": 0.0}
        results_by_type[tt]["count"] += 1
        results_by_type[tt]["reward"] += ep_reward
        if ep_reward >= 0.9:
            results_by_type[tt]["success"] += 1

    metrics = {
        "avg_reward": total_reward / max(n_episodes, 1),
        "success_rate": total_success / max(n_episodes, 1),
        "avg_steps": total_steps / max(n_episodes, 1),
    }

    for tt, data in results_by_type.items():
        cnt = max(data["count"], 1)
        metrics["success_rate_{}".format(tt)] = data["success"] / cnt
        metrics["avg_reward_{}".format(tt)] = data["reward"] / cnt

    return metrics


# ---- Main training function ----

def train(
    mode: str = "canvas",
    epochs: int = 100,
    d_model: int = 128,
    n_layers: int = 4,
    n_heads: int = 4,
    lr: float = 3e-4,
    batch_size: int = 32,
    n_demos: int = 200,
    instruction_dim: int = 32,
    bc_weight: float = 1.0,
    ssl_weight: float = 0.5,
    rl_weight: float = 0.3,
    seed: int = 42,
    log_interval: int = 5,
) -> Tuple[nn.Module, List[Dict[str, Any]]]:
    """Run the full training pipeline.

    Args:
        mode: "canvas" (structured), "dense" (dense transformer), or "flat" (MLP).
        epochs: Number of training epochs.
        d_model: Model dimension.
        n_layers: Number of layers.
        n_heads: Number of attention heads.
        lr: Learning rate.
        batch_size: Batch size.
        n_demos: Number of expert demonstrations to generate.
        instruction_dim: Instruction encoding dimension.
        bc_weight: Behavioral cloning loss weight.
        ssl_weight: SSL loss weight.
        rl_weight: RL loss weight.
        seed: Random seed.
        log_interval: Epochs between evaluations.

    Returns:
        (trained_model, training_log) tuple.
    """
    torch.manual_seed(seed)
    np.random.seed(seed)

    print("Generating {} expert demonstrations...".format(n_demos))
    demos = generate_demonstrations(n_demos=n_demos, seed=seed)
    success_count = sum(1 for d in demos if d.success)
    print("  Expert success rate: {:.1f}% ({}/{})".format(
        100 * success_count / n_demos, success_count, n_demos,
    ))

    # Convert to training data
    data = batch_demonstrations(demos, d_model=d_model, instruction_dim=instruction_dim)
    print("  Total transitions: {}".format(data["screens"].shape[0]))

    # Train/val split
    n_total = data["screens"].shape[0]
    n_val = max(1, n_total // 5)
    n_train = n_total - n_val

    indices = torch.randperm(n_total)
    train_idx = indices[:n_train]
    val_idx = indices[n_train:]

    train_data = {k: v[train_idx] for k, v in data.items()}
    val_data = {k: v[val_idx] for k, v in data.items()}

    train_dataset = BrowserDataset(train_data)
    val_dataset = BrowserDataset(val_data)

    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, drop_last=True)
    val_loader = DataLoader(val_dataset, batch_size=batch_size)

    # Build model
    print("Building {} model (d_model={}, n_layers={})...".format(mode, d_model, n_layers))
    if mode == "canvas":
        model = BrowserCanvasModel(
            d_model=d_model, n_heads=n_heads, n_layers=n_layers,
            instruction_dim=instruction_dim,
        )
    elif mode == "dense":
        model = DenseBaselineModel(
            d_model=d_model, n_heads=n_heads, n_layers=n_layers,
            instruction_dim=instruction_dim,
        )
    elif mode == "flat":
        model = FlatBaselineModel(
            d_model=d_model, n_layers=n_layers,
            instruction_dim=instruction_dim,
        )
    else:
        raise ValueError("Unknown mode: {}".format(mode))

    n_params = sum(p.numel() for p in model.parameters())
    print("  Parameters: {:,}".format(n_params))

    optimizer = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=0.01)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epochs)

    # Training log
    log = []
    best_val_loss = float("inf")
    start_time = time.time()

    RESULTS_DIR.mkdir(parents=True, exist_ok=True)

    print("Training for {} epochs...".format(epochs))
    for epoch in range(epochs):
        # Reset scheduling counters each epoch
        model.plan_fire_count = 0
        model.total_steps = 0

        train_losses = train_epoch(
            model, train_loader, optimizer, epoch,
            bc_weight=bc_weight, ssl_weight=ssl_weight, rl_weight=rl_weight,
        )
        scheduler.step()

        entry = {
            "epoch": epoch,
            "mode": mode,
            "time": time.time() - start_time,
            "lr": scheduler.get_last_lr()[0],
        }
        entry.update({"train_{}".format(k): v for k, v in train_losses.items()})

        # Evaluate periodically
        if epoch % log_interval == 0 or epoch == epochs - 1:
            val_losses = evaluate_model(model, val_loader)
            entry.update({"val_{}".format(k): v for k, v in val_losses.items()})

            if val_losses.get("total", float("inf")) < best_val_loss:
                best_val_loss = val_losses["total"]

            # Rollout evaluation
            rollout_metrics = evaluate_rollout(
                model, n_episodes=40, seed=epoch * 100 + 999,
                instruction_dim=instruction_dim,
            )
            entry.update({"rollout_{}".format(k): v for k, v in rollout_metrics.items()})

            print("  Epoch {:3d} | train {:.4f} | val {:.4f} | acc {:.2f}% | "
                  "rollout success {:.1f}% | plan rate {:.2f}".format(
                      epoch,
                      train_losses.get("total", 0),
                      val_losses.get("total", 0),
                      100 * val_losses.get("action_accuracy_top1", 0),
                      100 * rollout_metrics.get("success_rate", 0),
                      train_losses.get("plan_fire_rate", 1.0),
                  ))

        log.append(entry)

        # Write JSONL log incrementally
        log_path = RESULTS_DIR / "training_log_{}.jsonl".format(mode)
        with open(str(log_path), "a") as f:
            f.write(json.dumps(entry) + "\n")

    elapsed = time.time() - start_time
    print("Training complete in {:.1f}s".format(elapsed))

    return model, log


# ---- CLI ----

def main():
    parser = argparse.ArgumentParser(description="Train browser canvas agent")
    parser.add_argument("--mode", default="canvas", choices=["canvas", "dense", "flat"])
    parser.add_argument("--epochs", type=int, default=100)
    parser.add_argument("--d_model", type=int, default=128)
    parser.add_argument("--n_layers", type=int, default=4)
    parser.add_argument("--n_heads", type=int, default=4)
    parser.add_argument("--lr", type=float, default=3e-4)
    parser.add_argument("--batch_size", type=int, default=32)
    parser.add_argument("--n_demos", type=int, default=200)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--bc_weight", type=float, default=1.0)
    parser.add_argument("--ssl_weight", type=float, default=0.5)
    parser.add_argument("--rl_weight", type=float, default=0.3)
    args = parser.parse_args()

    model, log = train(
        mode=args.mode,
        epochs=args.epochs,
        d_model=args.d_model,
        n_layers=args.n_layers,
        n_heads=args.n_heads,
        lr=args.lr,
        batch_size=args.batch_size,
        n_demos=args.n_demos,
        seed=args.seed,
        bc_weight=args.bc_weight,
        ssl_weight=args.ssl_weight,
        rl_weight=args.rl_weight,
    )

    # Save final model info
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    summary_path = RESULTS_DIR / "summary_{}.json".format(args.mode)
    with open(str(summary_path), "w") as f:
        json.dump({
            "mode": args.mode,
            "epochs": args.epochs,
            "d_model": args.d_model,
            "n_layers": args.n_layers,
            "n_params": sum(p.numel() for p in model.parameters()),
            "final_train_loss": log[-1].get("train_total", None),
            "final_val_loss": log[-1].get("val_total", None),
            "final_val_accuracy": log[-1].get("val_action_accuracy_top1", None),
            "final_rollout_success": log[-1].get("rollout_success_rate", None),
        }, f, indent=2)

    print("Results saved to {}".format(RESULTS_DIR))


if __name__ == "__main__":
    main()
