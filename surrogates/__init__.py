from typing import TYPE_CHECKING
from utils import LazyModule
import sys

if TYPE_CHECKING:
    from .VisionEncoders import (
        BlipFeatureExtractor,
        ClipFeatureExtractor,
        EnsembleFeatureExtractor,
        EnsembleFeatureLoss,
        VisionTransformerFeatureExtractor,
    )
    from .get_minigpt4_image_model import (
        get_minigpt4_loss_model,
        get_minigpt4_predict_loss_model,
        get_minigpt4_predict_model,
    )
    from .Blip2 import Blip2VisionModel, Blip2PredictModel
    from .InstructBlip import InstructBlipVisionModel, InstructBlipPredictModel
    from .get_llava_vision_model import get_llava_loss_model, get_llava_predict_loss_model, get_llava_predict_model
    from .DYPVision import HuanranTransformerClipAttackVisionModel, HuanranOpenClipAttackVisionModel
    from .BaseNormModel import BaseNormModel
    from .get_gpt4_vision_model import GPT4PredictModel


_import_structure = {
    "get_minigpt4_image_model": [
        "get_minigpt4_predict_model",
        "get_minigpt4_predict_loss_model",
        "get_minigpt4_loss_model",
    ],
    "Blip2": ["Blip2VisionModel", "Blip2PredictModel"],
    "InstructBlip": ["InstructBlipVisionModel", "InstructBlipPredictModel"],
    "VisionEncoders": [
        "BlipFeatureExtractor",
        "ClipFeatureExtractor",
        "EnsembleFeatureExtractor",
        "EnsembleFeatureLoss",
        "VisionTransformerFeatureExtractor",
    ],
    "get_llava_vision_model": ["get_llava_loss_model", "get_llava_predict_loss_model", "get_llava_predict_model"],
    "DYPVision": ["HuanranTransformerClipAttackVisionModel", "HuanranOpenClipAttackVisionModel"],
    "BaseNormModel": ["BaseNormModel"],
    "get_gpt4_vision_model": ["GPT4PredictModel"],
}
sys.modules[__name__] = LazyModule(
    __name__,
    globals()["__file__"],
    _import_structure,
)
