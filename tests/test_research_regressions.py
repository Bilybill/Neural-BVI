"""Keep the original split, noise, backbone and BVI regression checks visible."""
import importlib.util
from pathlib import Path

path = Path(__file__).resolve().parents[1] / "experiments/bvi_e2e/test_publication_pipeline.py"
spec = importlib.util.spec_from_file_location("research_regressions", path)
module = importlib.util.module_from_spec(spec)
spec.loader.exec_module(module)
PublicationProtocolTests = module.PublicationProtocolTests
