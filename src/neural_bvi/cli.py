"""Small, artifact-independent entry points for first-time users."""
import argparse
import importlib.metadata
import json
import platform
from pathlib import Path


def main():
    parser = argparse.ArgumentParser(description="Neural-BVI for GPR inversion")
    sub = parser.add_subparsers(dest="command", required=True)
    sub.add_parser("doctor", help="Print runtime versions and device availability")
    demo = sub.add_parser("demo", help="Train and refine a small procedural scalar-wave problem")
    demo.add_argument("--output", type=Path, default=Path("outputs/demo"))
    demo.add_argument("--seed", type=int, default=7)
    demo.add_argument("--epochs", type=int, default=20)
    demo.add_argument("--bvi-steps", type=int, default=4)
    demo.add_argument("--device", choices=["cpu", "cuda"], default="cpu")
    args = parser.parse_args()
    if args.command == "doctor":
        import torch
        print(json.dumps({"python": platform.python_version(), "platform": platform.system(),
                          "cuda_available": torch.cuda.is_available(),
                          "packages": {p: importlib.metadata.version(p) for p in
                                       ("torch", "numpy", "scipy", "deepwave", "matplotlib")}}, indent=2))
    else:
        from .demo import run_demo
        run_demo(args.output, args.seed, args.epochs, args.bvi_steps, args.device)
