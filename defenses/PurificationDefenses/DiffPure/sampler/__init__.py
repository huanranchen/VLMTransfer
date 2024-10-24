import sys
from utils.imports import LazyModule
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from .edm import EDMStochasticSampler
    from .PreconditionChanger import *

_import_structure = {
    "PreconditionChanger": ["EDM2VP", "VP2EDM"],
    "edm": ["EDMStochasticSampler", "EulerMaruyamaWithExtraLangevin"],
}

sys.modules[__name__] = LazyModule(
    __name__,
    globals()["__file__"],
    _import_structure,
)
