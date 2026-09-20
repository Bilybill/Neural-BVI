# Installation

The locally tested research runtime is Python 3.11.15, PyTorch 2.11.0+cu128, Deepwave 0.0.27, NumPy 2.4.4, SciPy 1.17.1 and Matplotlib 3.10.8 on Windows. The public demo also runs on CPU. CI defines Linux and Windows jobs; remote CI execution is only confirmed after pushing the repository.

Create and activate a virtual environment, install your chosen PyTorch build, and run `python -m pip install -e ".[dev]"` from the repository root. The matching console entry point is `neural-bvi`; `python -m neural_bvi` works as well. Install `.[field]` only when extracting the original measured-residual noise bank; review that dependency's own license.

`requirements-tested.txt` records the numerical versions used for local release checks. It is a constraints file rather than a complete transitive lockfile. CUDA wheels and driver requirements depend on your platform. The demo always defaults to CPU; supply `--device cuda` explicitly if desired.

Run `python -m neural_bvi doctor` to print version/device information. If Deepwave cannot load, first test `python -c "import deepwave, torch; print(torch.__version__)"` in the same environment. If CUDA is unavailable, use CPU. For headless plotting the demo selects Matplotlib's Agg backend.

Source checkout commands use paths relative to the repository root. Full research configuration files are delivered in the source repository/archive; the wheel provides the Python modules and demo, not the private artifacts or entire experiment tree.
