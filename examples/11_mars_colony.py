"""Mars colony: autonomous multi-system coordination without Earth relay.

Model an early Mars colony's autonomous control system. Communication delay
to Earth is 4-24 minutes, so the colony must handle emergencies autonomously.

The colony has:
- Habitat modules (life support, power, thermal)
- Rovers (geology, construction, cargo transport)
- A greenhouse (food production, atmosphere recycling)
- An ISRU plant (fuel and oxygen production from regolith)
- EVA astronauts (health monitoring, task planning)

All systems share a colony-level "situation awareness" field that serves
as the central nervous system. The temporal policy is causal — you can't
condition on future sensor readings. Array elements within the rover fleet
use ring connectivity (geographic neighbors share sensor data).

This is the kind of multi-system coordination problem where declaring
the information architecture IS the engineering problem. The canvas type
hierarchy makes the design reviewable and auditable — critical for
human-rated systems.
"""

from dataclasses import dataclass, field as dc_field
from canvas_engineering import Field, compile_schema, ConnectivityPolicy


# ── Subsystem types ──────────────────────────────────────────────────

@dataclass
class LifeSupport:
    """Atmosphere, water, and waste processing."""
    atmosphere: Field = Field(2, 4,            # O2, CO2, N2, pressure, humidity, temp
                              loss_weight=5.0, # life critical
                              semantic_type="habitat atmosphere composition and state")
    water: Field = Field(1, 4,                 # tank levels, recycling rate, quality
                         loss_weight=3.0,
                         semantic_type="water system levels and quality metrics")
    waste: Field = Field(1, 2,
                         semantic_type="waste processing system state")


@dataclass
class PowerSystem:
    """Solar arrays, batteries, RTGs, distribution."""
    generation: Field = Field(2, 4,            # solar output, RTG output, dust level
                              semantic_type="power generation solar/RTG output kW")
    storage: Field = Field(1, 4,               # battery SoC, temperature, health
                           loss_weight=2.0,
                           semantic_type="battery storage SoC and health")
    distribution: Field = Field(2, 4,          # per-subsystem allocation
                                loss_weight=2.0,
                                semantic_type="power distribution allocation per subsystem")


@dataclass
class ThermalControl:
    """Heating, cooling, and thermal regulation."""
    temperatures: Field = Field(2, 4,          # zone temperatures throughout habitat
                                loss_weight=3.0,
                                semantic_type="thermal zone temperatures 8 zones")
    radiator: Field = Field(1, 2,              # radiator state, fluid temps
                            semantic_type="radiator loop state and fluid temperatures")


@dataclass
class HabitatModule:
    """One pressurized habitat module."""
    life_support: LifeSupport = dc_field(default_factory=LifeSupport)
    power: PowerSystem = dc_field(default_factory=PowerSystem)
    thermal: ThermalControl = dc_field(default_factory=ThermalControl)
    structural: Field = Field(1, 4,            # pressure, seal integrity, micrometeorite
                              loss_weight=4.0,
                              semantic_type="structural integrity pressure seals")


@dataclass
class Rover:
    """Autonomous Mars rover."""
    camera: Field = Field(8, 8,                # 64 pos: stereo nav cam
                          semantic_type="stereo navigation camera Mars terrain")
    position: Field = Field(1, 4,              # lat, lon, heading, speed
                            semantic_type="rover GPS-denied position estimate")
    battery: Field = Field(1, 2,               # SoC, temperature
                           loss_weight=2.0,
                           semantic_type="rover battery state")
    task: Field = Field(2, 4, is_output=False, # assigned mission
                        semantic_type="rover mission task assignment")
    plan: Field = Field(4, 4,                  # path plan + actions
                        loss_weight=2.0,
                        attn="mamba",
                        semantic_type="rover autonomous navigation plan")


@dataclass
class Greenhouse:
    """Mars greenhouse for food production."""
    crops: Field = Field(4, 4,                 # growth stage per crop bay
                         semantic_type="crop growth state 16 bays")
    atmosphere: Field = Field(2, 4,            # CO2, O2, humidity for plants
                              loss_weight=2.0,
                              semantic_type="greenhouse atmosphere for crop growth")
    water: Field = Field(1, 4,                 # irrigation state
                         semantic_type="greenhouse irrigation and nutrient state")
    lighting: Field = Field(1, 4,              # LED spectrum and intensity
                            semantic_type="grow light spectrum and intensity")
    harvest: Field = Field(2, 4,               # predicted harvest schedule
                           loss_weight=1.5,
                           semantic_type="predicted crop harvest schedule and yield")


@dataclass
class ISRUPlant:
    """In-Situ Resource Utilization — fuel and oxygen from regolith."""
    feedstock: Field = Field(2, 4,             # regolith hopper level, composition
                             is_output=False,
                             semantic_type="regolith feedstock level and composition")
    electrolysis: Field = Field(2, 4,          # O2 production rate, efficiency
                                loss_weight=3.0,
                                semantic_type="water electrolysis O2 production state")
    sabatier: Field = Field(2, 4,              # CH4 fuel production
                            loss_weight=2.0,
                            semantic_type="Sabatier reactor CH4 fuel production state")
    storage: Field = Field(1, 4,               # O2 and CH4 tank levels
                           loss_weight=3.0,
                           semantic_type="propellant and O2 storage tank levels")


@dataclass
class Astronaut:
    """EVA astronaut health and task state."""
    vitals: Field = Field(2, 4,                # HR, SpO2, BP, body temp
                          loss_weight=5.0,     # crew safety paramount
                          semantic_type="astronaut vital signs HR/SpO2/BP/temp")
    suit: Field = Field(2, 4,                  # suit pressure, O2 remaining, battery
                        loss_weight=4.0,
                        semantic_type="EVA suit pressure O2 battery thermal state")
    location: Field = Field(1, 2,              # position relative to habitat
                            loss_weight=2.0,
                            semantic_type="astronaut position relative to habitat")
    task: Field = Field(2, 4, is_output=False, # assigned EVA task
                        semantic_type="astronaut EVA task assignment and timeline")
    fatigue: Field = Field(1, 2,               # estimated cognitive/physical fatigue
                           loss_weight=3.0,
                           semantic_type="estimated astronaut fatigue level")


@dataclass
class MarsColony:
    """Autonomous Mars colony control system."""
    # Colony-level situation awareness — the "central nervous system"
    situation: Field = Field(8, 8,             # 64 pos: holistic colony state
                             loss_weight=3.0,
                             attn="cross_attention",
                             semantic_type="colony situation awareness integrated state")

    alert_level: Field = Field(1, 4,           # emergency classification
                               loss_weight=8.0, # missed emergency = death
                               semantic_type="colony alert level and emergency classification")

    resource_plan: Field = Field(4, 8,         # 32 pos: resource allocation plan
                                 loss_weight=2.0,
                                 attn="mamba",
                                 semantic_type="colony resource allocation plan 30-sol horizon")

    # Weather (input-only, from orbital/surface observations)
    weather: Field = Field(4, 4, is_output=False,
                           semantic_type="Mars weather dust/wind/temperature/radiation")

    earth_comms: Field = Field(2, 4, is_output=False,  # last uplink from mission control
                               semantic_type="Earth mission control last uplink commands")

    # Subsystems
    habitats: list = dc_field(default_factory=list)
    rovers: list = dc_field(default_factory=list)
    greenhouse: Greenhouse = dc_field(default_factory=Greenhouse)
    isru: ISRUPlant = dc_field(default_factory=ISRUPlant)
    crew: list = dc_field(default_factory=list)


# --- A colony: 2 habitat modules, 4 rovers, 4 crew ---

colony = MarsColony(
    habitats=[HabitatModule(), HabitatModule()],
    rovers=[Rover() for _ in range(4)],
    crew=[Astronaut() for _ in range(4)],
)

bound = compile_schema(
    colony, T=32, H=96, W=96, d_model=512,
    connectivity=ConnectivityPolicy(
        intra="dense",
        parent_child="hub_spoke",      # situation awareness sees everything
        array_element="ring",          # rovers share with geographic neighbors
        temporal="causal",
    ),
)

print("=== Mars Colony Autonomous Control ===")
print(f"Habitats: 2, Rovers: 4, Crew: 4")
print(f"Total fields: {len(bound.field_names)}")
print(f"Total connections: {len(bound.topology.connections)}")
print(f"Canvas: T={bound.layout.T}, {bound.layout.H}x{bound.layout.W}, "
      f"d={bound.layout.d_model}")
print(f"Total positions: {bound.layout.num_positions:,}")
used = sum(bf.num_positions for bf in bound.fields.values())
print(f"Used positions: {used:,} ({used/bound.layout.num_positions*100:.1f}%)")
print()

# Subsystem breakdown
print("Subsystem field counts:")
subsystems = {
    "Colony-level": ["situation", "alert_level", "resource_plan", "weather", "earth_comms"],
    "Habitats": [n for n in bound.field_names if n.startswith("habitats")],
    "Rovers": [n for n in bound.field_names if n.startswith("rovers")],
    "Greenhouse": [n for n in bound.field_names if n.startswith("greenhouse")],
    "ISRU plant": [n for n in bound.field_names if n.startswith("isru")],
    "Crew": [n for n in bound.field_names if n.startswith("crew")],
}
for subsys, fields in subsystems.items():
    total_pos = sum(bound[f].num_positions for f in fields if f in bound)
    print(f"  {subsys}: {len(fields)} fields, {total_pos:,} positions")

# Safety hierarchy
weights = bound.layout.loss_weight_mask("cpu")
total_w = weights.sum().item()
print(f"\nSafety loss hierarchy:")
safety_fields = [
    ("Colony alert", "alert_level", None),
    ("Crew vitals", "vitals", "crew"),
    ("EVA suits", "suit", "crew"),
    ("Atmosphere", "atmosphere", "habitats"),
    ("Structural", "structural", "habitats"),
    ("Disruption risk", "situation", None),
]
for label, keyword, parent in safety_fields:
    matching = [n for n in bound.field_names
                if keyword in n and (parent is None or parent in n)]
    w = sum(sum(weights[i].item() for i in bound[n].indices()) for n in matching)
    if w > 0:
        lw = bound[matching[0]].spec.loss_weight
        print(f"  {label}: {w/total_w*100:.1f}% (weight={lw})")
