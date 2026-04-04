# canvas-engineering Video Presentation Plan (v2)

## Format
Single continuous 3D animation with voice-over narration. 4-5 minutes.
The brain is the visual throughline — it transforms from neuroscience
object → engineered architecture → embodied agent controller.

Built with Remotion + Three.js (React Three Fiber) for the unified 3D scene.

## Visual Language Rules
- NO line graphs or bar charts in the video (boring, static)
- ONE continuous 3D scene with parametric transitions
- The brain rotates slowly throughout — regions light up/fade as narration progresses
- Background/context changes smoothly (lab → canvas grid → robot environment)
- Text overlays appear as floating 3D labels, not 2D cards
- Numbers appear as 3D text integrated into the scene
- Camera moves smoothly — dolly, orbit, push-in — never jump cuts

---

## The Unified 3D Animation

### Camera path (continuous):
1. Start: Close-up on rotating brain (dark background, medical/scientific mood)
2. Pull back: Brain becomes part of a canvas grid layout (regions labeled)
3. Push in: Activation flows through cortical pathways (regions pulse in sequence)
4. Pull back further: Brain shrinks, enters a robot's head
5. Wide shot: Robot fleet moving in formation, each with a glowing brain
6. Final: Zoom back to brain, now fully lit, scale numbers floating around it

### Parametric variations (same scene, different states):
- Brain region illumination (which ROIs are lit, intensity, color)
- Connection lines between regions (appear/pulse when pathway is active)
- Background environment (void → grid → physical world)
- Overlay text (floating 3D, fades in/out)
- Robot positions and movements (formation control)

---

## Scene Breakdown

### Opening (0:00 - 0:25)
**Visual**: Headshot/chair shot of Jacob, intercut with:
  - Slowly rotating 3D brain, dark background, subtle glow
  - B-roll: surgeons in OR, fMRI scanner, children playing
  - Brain regions begin to softly pulse

**Narration theme**: Why predicting brain dynamics matters for
neuroscience, healthcare, and understanding human cognition.

### Scene 1: The Problem (0:25 - 0:50)
**Visual**: Brain dissolves into a flat gray grid of identical dots
  (representing an unstructured transformer). All dots same color. Boring.
  Camera orbits around this flat, featureless space.

**Narration theme**: Current neural architectures have no internal structure.

### Scene 2: The Solution (0:50 - 1:40)
**Visual**: The flat grid transforms — dots cluster into colored regions.
  Labels float in: "Visual Cortex", "Language Network", "Motor Cortex".
  Connection lines draw between regions (the topology appears).
  The flat grid lifts into 3D — it's the brain again, but now with
  labeled regions and visible wiring.

**Narration theme**: canvas-engineering declares what each region IS
and how they connect. Type system for latent space.

### Scene 3: The Brain in Action (1:40 - 2:45)
**Visual**: The brain rotates. A stimulus appears (text floating in space):
  "A golden retriever bounding across a field..."

  Activation sequence animates:
  1. V1/V2 lights up bright yellow (visual processing)
  2. Activation flows along ventral stream → V4 → fusiform (object recognition)
  3. Simultaneously: temporal regions pulse (semantic processing)
  4. Prefrontal glows (executive processing)
  5. Broca's area pulses (language formulation)
  
  The connections between regions pulse as activation flows through them.
  Floating text: "R² = 0.838" appears as the prediction accuracy.

**Narration theme**: We mapped 23 cortical regions with 42 known pathways
and trained on real TRIBE v2 brain data. The model predicts cortical dynamics.

### Scene 4: Beyond Brains (2:45 - 3:30)
**Visual**: The brain shrinks smoothly. Camera pulls back to reveal
  it's inside a robot's transparent head. The robot is in a 2D arena.
  
  Three more robots appear, each with a glowing brain.
  They begin moving in formation — toward target positions.
  Obstacles appear. The robots navigate around them.
  
  Connection lines appear between robots (coarse-grained communication).
  Each robot's brain pulses at different rates (scheduler visualization).

  Brief flash: browser window appears, canvas agent clicking elements,
  plan region only firing when prediction error spikes.

**Narration theme**: Same architecture — brain dynamics, robot control,
browser agents. Three domains, one substrate.

### Scene 5: What's Next (3:30 - 4:15)
**Visual**: Camera pushes back in to the brain. It grows.
  More regions appear (23 → 100 → 500). The brain fills with light.
  Connections multiply. The whole cortical surface glows.
  
  Floating text: "500 regions", "8xH100", "2-3 weeks"
  
  The brain rotates one final time, fully illuminated.

**Narration theme**: Foundation brain world model. Real fMRI data.
Full spatiotemporal dynamics. The architecture is built — we need compute.

### Closing (4:15 - 4:30)
**Visual**: Brain settles, slowly rotating. 
  Title card fades in as 3D text: "canvas-engineering"
  URL: github.com/JacobFV/canvas-engineering

---

## Production Pipeline

### Phase 1: Record narration
- Jacob records voice-over from the outline (see outline below)
- Send back transcript → we match animation timing to narration

### Phase 2: Build 3D scene
- React Three Fiber (Three.js in React) for the brain + environment
- Brain mesh: fsaverage5 from nilearn, exported as OBJ/GLB
- Region activation: shader uniforms controlling per-vertex color
- Robot meshes: simple geometric shapes
- Remotion for sequencing and rendering to video

### Phase 3: Compose
- Sync animation keyframes to narration timestamps
- Add floating text overlays
- Add ambient music (subtle, scientific)
- Render at 1080p 30fps

### Phase 4: Polish
- Color grading
- Sound mixing
- Export final MP4

---

## Assets to create for 3D scene
1. Brain mesh (fsaverage5 pial surface → GLB)
2. Per-vertex region labels (for selective illumination)
3. Robot mesh (simple, geometric)
4. Arena/environment (flat plane + obstacles)
5. Canvas grid texture (for the "flat grid" → "structured layout" transition)
