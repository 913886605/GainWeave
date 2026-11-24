"""GainWeave: frozen decoder features and bounded HDR gain refinement."""

__version__ = "0.2.0"


def __getattr__(name):
    if name == "GainWeave":
        from .pipeline import GainWeave
        return GainWeave
    raise AttributeError(name)


__all__ = ["GainWeave", "__version__"]
