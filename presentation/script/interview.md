# Interview Script

Record this as a talking-head or voice-over. Casual but precise.
Each section maps to a scene in the storyboard.

---

## Scene 1: The Problem (30 seconds)

> "So here's the thing about neural networks — they treat their internal state
> as a flat bag of numbers. There's no structure. A transformer just has
> positions, and it figures out through training what each position means.
>
> But what if you could tell the model what its internal state means?
> What if you could say: these positions are visual cortex, these are
> language, these are motor output — and this is how they connect?"

---

## Scene 2: The Solution (45 seconds)

> "That's what canvas-engineering does. You declare a typed layout — each
> region has a family: observation, state, memory, action. You declare
> the topology — which regions can attend to which. And you declare the
> dynamics — how fast each region updates, what kind of learning it does,
> and what gets compiled away at deploy time.
>
> It's basically a type system for latent space. The model doesn't
> discover structure — you declare it, and the structure constrains
> what it learns."

---

## Scene 3: The Brain Model (75 seconds)

> "So we took this and built a brain. Literally — we mapped 23 cortical
> regions from the Destrieux atlas onto canvas regions. Visual cortex,
> auditory cortex, Broca's area, Wernicke's area, motor cortex, prefrontal,
> the default mode network — all wired up with 42 known cortical pathways.
>
> Then we used Facebook's TRIBE v2 model to generate real cortical
> predictions — what the brain actually does when you describe a scene,
> play music, or talk about danger. 72 text stimuli, real vertex-level
> predictions on the cortical surface.
>
> The task: given how the brain activates at time t, predict what it does
> at time t+1. This is cortical dynamics prediction — how activation flows
> from V1 through the ventral stream to fusiform for face recognition,
> or from auditory cortex through Wernicke's area to Broca's for language
> production.
>
> And the cortical topology — the wiring diagram — tells the model
> exactly where to route."

---

## Scene 4: The Results (45 seconds)

> "With 23 scalar regions, a flat MLP wins. Too easy — just memorize
> the mapping. But when we scale to 135 features — 8 per brain region —
> the cortical topology model hits R-squared 0.838. And it learns
> fastest: R-squared 0.41 at epoch 40, while the dense model is still
> at 0.06.
>
> The structured wiring is a developmental prior. The brain is born
> with these pathways. Given enough training, a fully connected network
> finds the same solution — but the cortical topology gets there first.
> That's the value: sample efficiency and convergence speed."

---

## Scene 5: Three Tracks (45 seconds)

> "We didn't stop at brains. We built a browser agent — screen
> observation, DOM state, planning, action — where the planner only
> fires 12% of steps, driven by prediction error. Eight times less
> compute than a dense baseline.
>
> And a multi-robot fleet — four robots with lidar, belief state,
> and communication channels, coordinating through coarse-grained
> summaries. The canvas topology means each robot sees the fleet
> through a compressed representation, not raw sensor data.
>
> Three domains, one architecture substrate."

---

## Scene 6: What's Next (30 seconds)

> "The proof of concept works. Structured topology helps with
> convergence and sample efficiency. Now we need to scale.
>
> A foundational brain world model: 500 cortical regions, real fMRI
> datasets, predicting full spatiotemporal dynamics across modalities.
> Two to three weeks on an 8xH100 node.
>
> The architecture is built. The pipeline is built. We just need
> the compute."

---

## Total: ~4 minutes 30 seconds
