# Wiki

## Vision
The goal of this repository is to provide a clear, visual, and practical understanding of the slider-crank mechanism's position behavior. The focus is on making key effects, especially those that are non-intuitive when an offset is present, easy to see through animation and plots.

## Current Scope
- Visualize the slider-crank position from crank, connecting rod, and offset dimensions.
- Provide animations and plots that show piston position versus crank angle.
- Highlight range-of-motion implications and stroke timing (intake/combustion vs compression/exhaust) when offsets are nonzero.

## Work in Progress
- Con-rod angle evolution and angle range.
- Detecting blocked/invalid configurations and incompatible ranges of motion.
- Velocity analysis using the Point Reference approach.
- Acceleration analysis using the Point Reference approach.

## Way of Working
- Start with visual outputs (animation and plots) to ground the mechanics in intuition.
- Use notebooks for exploration and derivations before folding results into the app/visualization layer.
- Expand the model iteratively: position -> angle range/compatibility -> velocity -> acceleration.
- Keep the approach aligned with the Point Reference methodology referenced in the project's inspiration.

## What Could Be Better
- Clearer documentation that links each notebook to a specific visualization or feature.
- A minimal set of automated checks (linting and basic test coverage for core math).
- Explicit handling of units and parameter validation to prevent invalid inputs.
- Performance improvements for higher-resolution animations or large parameter sweeps.
- More guidance on interpreting plots and connecting them to physical intuition.

## Tools and Foundations
- Primary visualization stack: Dash and Plotly.
- Experiments and derivations live in Jupyter notebooks (e.g., position and symbolic analysis).
- Inspiration and references are documented in the README.

## Current Tech Stack
- Python.
- Dash for the interactive app.
- Plotly for plotting and animation.
- Jupyter notebooks for exploration and derivations.
- Streamlit script (prototype/alternative entry point).
- Dockerfiles for reproducible environments.

## Contributing
- Fork the project, experiment, and propose improvements.
- If you add new analyses, aim to include both a notebook (derivation) and a visualization (intuition).
