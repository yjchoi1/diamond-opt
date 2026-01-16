## Diamond Reconstruction Optimization

Reconstructs a diamond-like polyhedron from measured edge lengths and face structure using nonlinear optimization, then exports plots and 3D visualizations.

### Run
1) Install deps: `python -m pip install torch numpy scipy trimesh matplotlib plotly`  
2) Run: `python main.py`

### Inputs
- Edit `INPUT_CONFIG` in `main.py` to choose the dataset, initial guess, and weights.
- JSON datasets live in this directory (e.g. `diamond_data_experiment.json`).

### Outputs
- `*_reconstructed.json` and `*_reconstructed.obj`
- `optimization_history.png`
- `diamond_reconstruction_static.html`
- `diamond_reconstruction_animation.html`
