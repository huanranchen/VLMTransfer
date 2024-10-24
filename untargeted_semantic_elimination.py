# To eliminate the current semantics
import os
import torch
from surrogates import HuanranTransformerClipAttackVisionModel, HuanranOpenClipAttackVisionModel
from utils import save_image, get_image, get_list_image, save_list_images
from tqdm import tqdm
from attacks import MI_CommonWeakness, SSA_CommonWeakness
from utils.init import forbid_initialization
import argparse
from torchvision import transforms

parser = argparse.ArgumentParser()
parser.add_argument("--input_dir", type=str)
parser.add_argument("--output_dir", type=str)
args = parser.parse_args()
print(vars(args))
forbid_initialization()

# Don't care about this target text. Just for initialization of models.
target_text = "A Bomb."
clip1 = HuanranTransformerClipAttackVisionModel("openai/clip-vit-large-patch14", target_text=target_text)
laion_clip = HuanranOpenClipAttackVisionModel("hf-hub:laion/CLIP-ViT-B-32-laion2B-s34B-b79K", target_text)
laion_clip2 = HuanranOpenClipAttackVisionModel("hf-hub:laion/CLIP-ViT-B-16-laion2B-s34B-b88K", target_text)
laion_clip3 = HuanranOpenClipAttackVisionModel(
    "hf-hub:laion/CLIP-convnext_large_d_320.laion2B-s29B-b131K-ft-soup", target_text
)
laion_clip4 = HuanranOpenClipAttackVisionModel("hf-hub:laion/CLIP-ViT-bigG-14-laion2B-39B-b160k", target_text)
sig_clip = HuanranOpenClipAttackVisionModel("hf-hub:timm/ViT-SO400M-14-SigLIP-384", target_text, resolution=(384, 384))
models = [clip1, laion_clip, laion_clip4, sig_clip, laion_clip2, laion_clip3]


class LossPrinter:
    def __init__(self):
        self.count = 0

    def __call__(self, loss, *args):
        self.count += 1
        if self.count % 100 == 1:
            print(loss)
        return -loss  # Minimize the cosine similarity


attacker = SSA_CommonWeakness(
    models,
    epsilon=16 / 255,
    step_size=1 / 255,
    total_step=50,
    criterion=LossPrinter(),
)

if not os.path.exists(args.output_dir):
    os.mkdir(args.output_dir)
images = get_list_image(args.input_dir)
transform = transforms.Resize((224, 224))
images = [transform(i).cuda() for i in images]
for i, image in enumerate(tqdm(images)):
    for model in models:
        model.change_target_image(image)
    adv_x = attacker(image, torch.tensor([0]))
    save_image(adv_x, os.path.join(args.output_dir, f"{i}.png"))
