"""Public interface to the preserved Neural-BVI research implementation."""
__version__ = "0.1.0"


def run_bvi(*args, **kwargs):
    """Run residual-space BVI; see e2e_bvi_gpr.run_bvi for tensor arguments."""
    from e2e_bvi_gpr import run_bvi as implementation
    return implementation(*args, **kwargs)
