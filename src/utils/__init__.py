
from .matrix import Matrix, Entity, Process


def __getattr__(name):
    # Generator (and everything it pulls in -- langchain, transformers,
    # torch, sympy...) is only imported lazily, the first time something
    # actually does `from utils import Generator` (PEP 562). Importing
    # utils.bsp on its own -- bsp.py has no LLM dependency at all -- no
    # longer pays for that entire chain just because it lives in the same
    # package. app.py and the test scripts that import Generator via
    # `from utils.generator import Generator` are unaffected either way.
    if name == "Generator":
        from .generator import Generator
        return Generator
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")