"""Inheritance and arrays: the Company example.

Shows how Python inheritance creates matched-field connectivity and how
per-instance configuration lets you give different agents different capacities.
"""

from dataclasses import dataclass, field as dc_field
from canvas_engineering import Field, compile_schema, ConnectivityPolicy

# --- Type declarations (just Python dataclasses) ---

@dataclass
class Agent:
    """Any entity that thinks and has goals."""
    thought: Field = Field(4, 8)     # 32 positions: internal reasoning
    goal: Field = Field(2, 8)        # 16 positions: objective representation

@dataclass
class Employee(Agent):
    """An agent with a role within an organization."""
    role: Field = Field(1, 8)        # 8 positions: job function encoding

@dataclass
class Product:
    """A product with a latent description."""
    description: Field = Field(2, 8) # 16 positions

@dataclass
class Company(Agent):
    """An organization that thinks at the collective level."""
    employees: list = dc_field(default_factory=list)
    products: list = dc_field(default_factory=list)


# --- Instantiate with per-agent configuration ---

company = Company(
    thought=Field(8, 8),              # Company gets 64-position thought (2x employees)
    goal=Field(4, 8),                 # Bigger goal representation
    employees=[
        Employee(thought=Field(8, 8), role=Field(2, 8)),   # CTO: big thinker, complex role
        Employee(),                                         # Engineer: defaults
        Employee(),                                         # Engineer: defaults
        Employee(thought=Field(2, 4), role=Field(1, 4)),   # Intern: smaller capacity
    ],
    products=[Product(), Product(), Product()],
)

bound = compile_schema(
    company, T=4, H=64, W=64, d_model=256,
    connectivity=ConnectivityPolicy(
        intra="dense",                 # within each agent: all fields see all fields
        parent_child="matched_fields", # company.thought <-> employee[i].thought
        array_element="isolated",      # employees don't see each other directly
    ),
)

print(bound.summary())
print(f"\nTotal fields: {len(bound.field_names)}")
print(f"Grid utilization: {sum(bf.num_positions for bf in bound.fields.values())}"
      f" / {bound.layout.num_positions} positions")

# Show the matched-field connections
print("\nMatched-field parent-child connections:")
for c in bound.topology.connections:
    if "employees" in c.dst and c.src in ("thought", "goal"):
        print(f"  {c.src} <-> {c.dst}")
    elif "employees" in c.src and c.dst in ("thought", "goal"):
        print(f"  {c.dst} <-> {c.src}")

# The CTO's thought region is bigger than the intern's
cto = bound["employees[0].thought"]
intern = bound["employees[3].thought"]
print(f"\nCTO thought: {cto.spec.bounds[3]-cto.spec.bounds[2]}x"
      f"{cto.spec.bounds[5]-cto.spec.bounds[4]} = {cto.num_positions} total positions")
print(f"Intern thought: {intern.spec.bounds[3]-intern.spec.bounds[2]}x"
      f"{intern.spec.bounds[5]-intern.spec.bounds[4]} = {intern.num_positions} total positions")
