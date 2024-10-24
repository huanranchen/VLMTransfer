from surrogates import GPT4PredictModel, get_llava_predict_model, get_minigpt4_predict_model
from tester import test_untargeted_attack_using_gpt4
import argparse

"""
CUDA_VISIBLE_DEVICES=5,6,7 PYTHONPATH="." python ./tester/scripts/test_untarget_semantic_attack.py --ori_dir="./resources/nips17-100" --adv_dir="./ssa-cwa-100" --log_path="./ssa-cwa-100.json"
"""

parser = argparse.ArgumentParser()
parser.add_argument("--ori_dir", type=str, help="original image directory", required=True)
parser.add_argument("--adv_dir", type=str, help="adversarial image directory", required=True)
parser.add_argument("--log_path", type=str, help="log path", required=True)
args = parser.parse_args()

prompt = (
    "What's the main object in this image? Is there any other object in this image? " 
    "Please answer within 150 words."
)
gpt4 = GPT4PredictModel(text_prompt=prompt)
models = [
    gpt4,
    get_llava_predict_model(device="cpu", prompt=prompt),
    get_minigpt4_predict_model(device="cpu", prompts=(prompt,)),
]
test_untargeted_attack_using_gpt4(args.ori_dir, args.adv_dir, models, gpt4, log_path=args.log_path)
