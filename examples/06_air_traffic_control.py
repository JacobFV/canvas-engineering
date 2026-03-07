"""Air traffic control: real-time deconfliction for a terminal airspace.

Model an entire terminal radar approach control (TRACON) as a canvas type.
Each aircraft has:
- Radar track (position, altitude, heading, speed)
- Flight plan (input-only context — filed route, destination, type)
- Intent (what the pilot/autopilot is trying to do)
- Predicted trajectory (the safety-critical output)
- A conflict state field that flags potential separation violations

The TRACON controller has:
- Sector state (weather, traffic flow, restricted areas)
- Sequence plan (arrival/departure ordering)
- Clearances (the actual ATC instructions — the primary output)

Design choices:
- Array element "dense": ALL aircraft see ALL other aircraft.
  In ATC, everything is relevant — a conflict between AC 3 and AC 7
  is just as important as one between AC 1 and AC 2.
- Causal temporal: strictly causal for safety. You cannot condition
  on future radar tracks that haven't happened yet.
- Conflict state gets 10x loss weight. A missed conflict = lives at risk.
- Flight plans are input-only. The model doesn't predict them — they're
  filed before the aircraft enters the sector.
"""

from dataclasses import dataclass, field as dc_field
from canvas_engineering import Field, compile_schema, ConnectivityPolicy


@dataclass
class Aircraft:
    """One aircraft in the TRACON."""
    radar_track: Field = Field(1, 6,          # 6 positions: x, y, z, hdg, spd, vrate
                               semantic_type="ADS-B radar track state vector")

    flight_plan: Field = Field(2, 8,          # 16 positions: route, dest, type, wake cat
                               is_output=False,
                               semantic_type="filed IFR flight plan")

    intent: Field = Field(2, 4,               # 8 positions: inferred pilot intent
                          attn="gated",        # optional — not always clear
                          semantic_type="inferred pilot/autopilot intent")

    trajectory: Field = Field(4, 4,           # 16 positions: predicted 60s trajectory
                              loss_weight=5.0,
                              semantic_type="predicted trajectory 60s lookahead")

    conflict: Field = Field(1, 2,             # 2 positions: conflict flag + severity
                            loss_weight=10.0,  # CRITICAL — missed conflict = fatal
                            attn="linear_attention",
                            semantic_type="separation conflict state")


@dataclass
class TRACON:
    """Terminal radar approach control sector."""
    # Sector-level state
    weather: Field = Field(4, 4,              # 16 positions: SIGMET, wind, visibility
                           is_output=False,
                           semantic_type="terminal area weather state")

    sector_load: Field = Field(2, 2,          # 4 positions: traffic density, complexity
                               semantic_type="sector workload metric")

    sequence: Field = Field(4, 8,             # 32 positions: arrival/departure sequence
                            loss_weight=2.0,
                            attn="mamba",      # sequential ordering is inherently temporal
                            semantic_type="arrival departure sequence plan")

    clearance: Field = Field(4, 8,            # 32 positions: ATC clearance instructions
                             loss_weight=3.0,  # these are the commands pilots follow
                             semantic_type="ATC clearance instructions")

    # The aircraft in this sector
    aircraft: list = dc_field(default_factory=list)


# --- Busy TRACON with 12 aircraft ---

tracon = TRACON(
    aircraft=[Aircraft() for _ in range(12)],
)

bound = compile_schema(
    tracon, T=16, H=64, W=64, d_model=512,
    connectivity=ConnectivityPolicy(
        intra="dense",              # controller sees everything
        parent_child="hub_spoke",   # every aircraft reads sector state and vice versa
        array_element="dense",      # EVERY aircraft sees EVERY other aircraft
        temporal="causal",          # strictly causal for safety
    ),
)

print("=== Terminal Radar Approach Control ===")
print(f"Aircraft: 12")
print(f"Fields per aircraft: {sum(1 for n in bound.field_names if 'aircraft[0]' in n)}")
print(f"Total fields: {len(bound.field_names)}")
print(f"Total connections: {len(bound.topology.connections)}")

# Count connection types
intra_ac = sum(1 for c in bound.topology.connections
               if "aircraft[" in c.src and "aircraft[" in c.dst
               and c.src.split("]")[0] == c.dst.split("]")[0])
cross_ac = sum(1 for c in bound.topology.connections
               if "aircraft[" in c.src and "aircraft[" in c.dst
               and c.src.split("]")[0] != c.dst.split("]")[0])
ac_sector = sum(1 for c in bound.topology.connections
                if ("aircraft[" in c.src) != ("aircraft[" in c.dst))

print(f"\nConnection breakdown:")
print(f"  Within each aircraft: {intra_ac}")
print(f"  Cross-aircraft (deconfliction): {cross_ac}")
print(f"  Aircraft <-> sector: {ac_sector}")

# Loss budget
weights = bound.layout.loss_weight_mask("cpu")
total_w = weights.sum().item()
conflict_w = 0
trajectory_w = 0
clearance_w = 0
for name, bf in bound.fields.items():
    indices = bf.indices()
    w = sum(weights[i].item() for i in indices)
    if "conflict" in name:
        conflict_w += w
    elif "trajectory" in name:
        trajectory_w += w
    elif "clearance" in name:
        clearance_w += w

print(f"\nLoss budget (% of total gradient signal):")
print(f"  Conflict detection: {conflict_w/total_w*100:.1f}%")
print(f"  Trajectory prediction: {trajectory_w/total_w*100:.1f}%")
print(f"  ATC clearances: {clearance_w/total_w*100:.1f}%")
print(f"  Everything else: {(1 - (conflict_w+trajectory_w+clearance_w)/total_w)*100:.1f}%")
