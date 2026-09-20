"""Install a built wheel in a fresh venv and run outside the source checkout.

Numerical dependencies are inherited to avoid a second multi-GB torch download.
The project itself must resolve inside the fresh environment, never the checkout.
"""
import argparse
import os
import subprocess
import tempfile
import venv
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--wheel", type=Path)
    args = parser.parse_args()
    candidates = sorted((ROOT / "dist").glob("*.whl"))
    wheel = args.wheel or (candidates[-1] if candidates else None)
    if wheel is None or not wheel.is_file():
        parser.error("Build a wheel with python -m build first")
    with tempfile.TemporaryDirectory(prefix="neural-bvi-wheel-") as temporary:
        directory = Path(temporary)
        environment = directory / "environment"
        venv.EnvBuilder(with_pip=True, system_site_packages=True).create(environment)
        python = environment / ("Scripts/python.exe" if os.name == "nt" else "bin/python")
        env = os.environ.copy()
        env.pop("PYTHONPATH", None)
        def run(*command):
            subprocess.run([str(python), *command], cwd=directory, env=env, check=True)
        run("-m", "pip", "install", "--no-deps", "--force-reinstall", str(wheel.resolve()))
        run("-I", "-c", "import pathlib,sys,neural_bvi,e2e_bvi_gpr; "
            "root=pathlib.Path(sys.prefix); "
            "assert pathlib.Path(neural_bvi.__file__).is_relative_to(root); "
            "assert pathlib.Path(e2e_bvi_gpr.__file__).is_relative_to(root); "
            "print('Wheel imports resolve inside the fresh environment')")
        run("-I", "-m", "neural_bvi", "doctor")
        run("-I", "-m", "neural_bvi", "demo", "--epochs", "2", "--bvi-steps", "1",
            "--output", str(directory / "demo"))
    print("Wheel smoke test passed (inherited numerical dependencies).")


if __name__ == "__main__":
    main()
