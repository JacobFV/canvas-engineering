# Recording Outline

Record yourself talking through these points naturally. Don't read a script —
just know the beats and talk. We'll edit and match animation later.

---

## Opening: Why This Matters (25 seconds)

**Setting**: Sit in a chair, look at camera. Be genuine.

**Beats to hit**:
- Predicting brain dynamics would be huge for neuroscience/medicine
- What specific thing it would enable (pick one you care about):
  - Understanding how thoughts form and propagate
  - Predicting seizures before they happen
  - Designing better brain-computer interfaces for paralyzed patients
  - Testing drug effects on brain circuits without human trials
  - Understanding how children's brains develop
- "We built a first step toward that"

**Don't say**: anything about the software architecture yet. Pure motivation.

---

## The Problem: Unstructured Neural Networks (25 seconds)

**Beats**:
- Neural networks right now = flat bag of numbers
- The model has to figure out from scratch what each internal position means
- That's like building a brain with no wiring diagram
- What if we could give it structure from the start?

---

## The Solution: Declaring Structure (50 seconds)

**Beats**:
- We built a system where you declare what each region of the model IS
  - This region is visual processing
  - This region is language
  - This region is motor output
  - This region stores memory
  - This region tracks prediction errors
- You declare how they connect — which matches real cortical pathways
- You declare how fast each region updates — visual cortex every frame, planning every 4th frame
- The system compiles all of that into the actual attention masks and training recipes
- It's a compiler — you declare the architecture, it builds the model

**Key phrase to work in** (your own words): something about it being a "type system" or "schema" for the model's internal state

---

## The Brain Model (60 seconds)

**Beats**:
- We mapped 23 cortical regions onto this system
  - Visual cortex: V1, V2, V4, fusiform face area
  - Auditory: primary auditory, Wernicke's area
  - Language: Broca's area, angular gyrus
  - Frontal: prefrontal, premotor, motor
  - Default mode: precuneus, cingulate
- Connected them with 42 known cortical pathways — the actual wiring diagram of the brain
- Used Facebook's TRIBE v2 model to generate real cortical predictions
  - Feed it a text description, it predicts what the brain does
  - We get activation at 20,000 vertices on the cortical surface, per timestep
- The task: predict what happens NEXT in the brain
  - Given activation at time t, predict time t+1
  - This requires the model to learn how activation flows through the pathways
- Result: R-squared 0.84 on real brain data
  - The model predicts cortical dynamics
  - The structured topology helps it learn the right routing

---

## Beyond Brains: Same Architecture, Different Domains (40 seconds)

**Beats**:
- Same system, applied to robot control
  - 4 robots, each with sensors, belief state, action output
  - They coordinate through compressed summaries (not raw sensor data)
  - The topology enforces that robots only see each other through bottleneck representations
- Same system, applied to browser agents
  - Screen observation, DOM state, planning, action
  - The planner only fires when something surprising happens (event-driven)
  - 8x less compute than always planning
- One architecture substrate, three completely different domains

---

## What's Next: Foundation Brain Model (30 seconds)

**Beats**:
- We proved the concept with 23 regions and 72 stimuli
- Next: 500 cortical regions, real fMRI datasets (HCP, Natural Scenes, BOLD5000)
- Full spatiotemporal dynamics across text, audio, and video
- 2-3 weeks on an 8xH100 cluster
- The architecture is built, the pipeline is built
- We just need the compute

---

## Total: ~4 minutes of natural talking

### Recording tips:
- Don't try to be polished. Be yourself.
- If you stumble, just pause and restart that section.
- Record each section separately if easier.
- We only need audio — the video is the 3D animation. 
  (Exception: opening headshot, ~15 seconds of you talking to camera)
- Record in a quiet room with your phone or laptop mic.
- Send me the raw audio files and I'll cut/sync them.
