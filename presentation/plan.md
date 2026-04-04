# canvas-engineering Video Presentation Plan

## Format
Interview-style narration with animated visualizations. 3-5 minutes.
Built with Remotion (React-based video framework).

## Target Audience
Technical collaborators, compute sponsors, and researchers who want to understand
what canvas-engineering does and why the brain world model needs GPU time.

---

## Storyboard (6 scenes)

### Scene 1: The Problem (0:00 - 0:30)
**Visual**: Side-by-side comparison — flat transformer (gray blob) vs structured canvas (colored regions)
**Narration**: Explains that current neural architectures treat latent space as unstructured

### Scene 2: The Solution — Canvas Engineering (0:30 - 1:15)
**Visual**: Animated canvas layout building up — regions appearing with labels, connections drawing between them
**Narration**: Declares what each region IS, how it connects, what kind of thing it holds

### Scene 3: The Brain Model (1:15 - 2:30)
**Visual**: 3D brain rotating, regions lighting up in sequence (V1→V2→fusiform, A1→Wernicke→Broca)
**Narration**: How we mapped the canvas to real cortical wiring and trained on TRIBE v2 data

### Scene 4: The Results (2:30 - 3:15)
**Visual**: Learning curves animating, R² climbing, comparison bars growing
**Narration**: Cortical topology R²=0.838, beats flat baselines, learns faster

### Scene 5: Three Research Tracks (3:15 - 4:00)
**Visual**: Triptych — brain activation, browser agent clicking, robot fleet moving
**Narration**: Brain dynamics, browser control, multi-robot coordination

### Scene 6: What's Next — Foundation Brain Model (4:00 - 4:30)
**Visual**: Scaling chart, 8xH100 cluster, brain lighting up fully
**Narration**: 2-3 weeks on 8xH100, 500 regions, real fMRI data

---

## Assets Needed

### From existing results:
1. 3D brain renders (4 views) — research/brain/report/brain_*.png
2. Connectivity matrix — research/brain/results/connectivity_matrix.png
3. Dynamics comparison — research/brain/results/dynamics_comparison.png
4. BCI result — assets/examples/09b_bci_tribe.png
5. Robot trajectory GIF — research/robotics/results/trajectory.gif
6. Scaling analysis — research/robotics/results/scaling_analysis.png
7. Browser planning frequency — research/browser/results/planning_frequency.png
8. Browser comparison — research/browser/results/comparison_multipanel.png
9. Memo figures (4) — research/memo_figures/fig1-4
10. Live progress chart — research/memo_figures/live_progress.png
11. Architecture overview — research/memo_figures/fig1_architecture_overview.png

### Need to create:
12. Animated canvas layout building (Remotion component)
13. Animated brain activation sequence (Remotion component with 3D brain frames)
14. Animated learning curves (Remotion component)
15. Title card / outro card

---

## Interview Script

See script/interview.md
