# 01: Hello Canvas Types

## Purpose
The "hello world." Teach the reader what Field, compile_schema, and BoundSchema
do in 30 seconds. Synthetic data. This one is allowed to be simple.

## What it does
1. Declare a 3-field type (signal_a, signal_b, output)
2. Compile to canvas
3. Generate synthetic sinusoidal data — signal_a and signal_b are different
   frequency sine waves, output is their product
4. Place data on canvas, run a tiny 1-layer transformer, extract predictions
5. Train for 200 steps, show loss curve
6. Visualize: 2x2 grid — (a) the canvas grid colored by region, (b) the
   input signals, (c) predicted vs true output, (d) loss curve

## Visualization
Single PNG: `assets/examples/01_hello_types.png`
Clean, minimal. The point is "look how little code this took."

## Training
Pure SFT on synthetic signal data. 200 steps, CPU, < 5 seconds.

## Key message
"This is what compile_schema does. Types become regions. Regions become
attention masks. The model learns to combine signals."
