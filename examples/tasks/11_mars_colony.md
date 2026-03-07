# 11: Mars Colony — Multi-System Autonomous Control

## Purpose
The capstone. The most complex type hierarchy. Show that declaring
70+ fields across 6 subsystems with hub-spoke connectivity creates
a model that handles cascading failures through cross-system reasoning.

## Data
Synthetic Mars colony simulation:
- Life support: atmosphere composition evolving based on crew activity,
  plant photosynthesis, ISRU O2 production, and leak rate
- Power: solar output varying with dust storms, battery SoC, load balancing
- Thermal: heat from equipment, radiation to space, habitat insulation
- Crew: activity patterns, fatigue accumulation, EVA exposure
- Cascading failures: dust storm -> solar drop -> power shortage ->
  life support degradation -> crew hypoxia -> emergency

## Training
- Multi-task prediction on all output fields
- **Cascading failure detection**: special scenarios where failures
  propagate across subsystems. The model must predict downstream effects.
- **Resource optimization**: given current state, predict the optimal
  power allocation across subsystems. This is an RL-like objective
  within the SFT framework: the "expert" allocation is computed by
  a known optimization algorithm, and the model learns to approximate it.
- **Emergency prioritization**: when alert_level is high, the loss on
  crew safety fields (vitals, suit) is dynamically increased. This is
  curriculum learning: normal operation emphasizes efficiency, emergency
  operation emphasizes survival.

## Visualization
`assets/examples/11_mars_colony.png` — large composite:
(a) Colony system diagram (type hierarchy as connected blocks)
(b) Normal operation: all systems nominal, smooth trajectories
(c) Dust storm cascade: solar -> power -> life support -> crew
(d) Cross-system attention: does the model learn the causal chain?
(e) Resource allocation: model vs optimal, under normal and emergency
(f) Emergency response: alert level, crew vitals, life support over time

## Key message
"72 fields. 6 subsystems. 1328 connections. All declared in 200 lines
of Python. The model learns cascading failure dynamics because the type
hierarchy declares the causal structure. Hub-spoke from situation awareness
means every subsystem failure is visible to the central coordinator.
This is what canvas engineering was built for."
