
from .matrix import Matrix, Entity, Process


def __getattr__(name):
    # Imports Generator lazily (PEP 562), so importing utils.bsp does not load the LLM libraries
    if name == "Generator":
        from .generator import Generator
        return Generator
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")