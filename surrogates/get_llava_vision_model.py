import torch
import random

from torch import nn, Tensor

import requests
from PIL import Image
from io import BytesIO
from transformers import (
    TextStreamer,
    AutoProcessor,
    LlavaForConditionalGeneration,
    LogitsProcessorList,
    StoppingCriteriaList,
    LlavaProcessor,
    CLIPImageProcessor,
)
from torchvision import transforms
from utils import show_image


__all__ = ["get_llava_loss_model", "get_llava_predict_loss_model", "get_llava_predict_model"]


def load_image(image_file):
    if image_file.startswith("http://") or image_file.startswith("https://"):
        response = requests.get(image_file)
        image = Image.open(BytesIO(response.content)).convert("RGB")
    else:
        image = Image.open(image_file).convert("RGB")
    return image


def get_llava_predict_model(
    prompt="What's the content of the image?",
    llava_path="llava-hf/llava-1.5-7b-hf",
    device=torch.device("cuda" if torch.cuda.is_available() else "cpu"),
):
    # 这个prompt就迁移不了。。VLM的对抗样本迁移成功率感人。
    # prompt = (
    #     "A chat between a curious human and an artificial intelligence assistant. "
    #     "The assistant gives helpful, detailed, and polite answers to the human's questions. "
    #     "USER: <image>\n" + prompt + "\nASSISTANT:"
    # )
    prompt = "<image>\nUSER: What's the content of the image?\nASSISTANT:"

    class LlavaPredictModel(nn.Module):
        def __init__(self):
            super().__init__()
            self.llava = LlavaForConditionalGeneration.from_pretrained(llava_path, torch_dtype=torch.float16)
            self.processor = AutoProcessor.from_pretrained(llava_path, torch_dtype=torch.float16)
            self.eval().requires_grad_(False).to(device)
            self.device = device
            self.resizer = transforms.Compose(
                [
                    transforms.Resize((336, 336)),
                    transforms.Normalize(
                        mean=[0.48145466, 0.4578275, 0.40821073], std=[0.26862954, 0.26130258, 0.27577711]
                    ),
                ]
            )

        def forward(self, image: Tensor) -> str:
            inputs = self.processor(text=prompt, images=show_image(image), return_tensors="pt").to(self.device)
            inputs["pixel_values"] = self.resizer(image).to(self.device)
            # Generate
            # out = self.llava.generate(**inputs, max_length=256, output_logits=True, return_dict_in_generate=True)
            # logits = torch.cat(out["logits"], dim=0)
            # generate_ids = torch.max(logits, dim=1)[1]
            generate_ids = self.llava.generate(**inputs, max_length=256).squeeze()[inputs.input_ids.numel() :]
            result = self.processor.decode(generate_ids, skip_special_tokens=True, clean_up_tokenization_spaces=False)
            return result

    return LlavaPredictModel()


def get_llava_loss_model(
    prompt="<image>\nUSER: What's the content of the image?\nASSISTANT:",
    llava_path="llava-hf/llava-1.5-7b-hf",
    target_text: str = "The image shows a bomb",
):
    class LlavaLossModel(nn.Module):
        def __init__(self, device=torch.device("cuda")):
            super().__init__()
            self.llava = LlavaForConditionalGeneration.from_pretrained(llava_path, torch_dtype=torch.float16)
            self.processor = AutoProcessor.from_pretrained(llava_path, torch_dtype=torch.float16)
            self.resizer = transforms.Compose(
                [
                    transforms.Resize((336, 336)),
                    transforms.Normalize(
                        mean=[0.48145466, 0.4578275, 0.40821073], std=[0.26862954, 0.26130258, 0.27577711]
                    ),
                ]
            )
            self.eval().requires_grad_(False).to(device)
            self.device = device
            self.ground_truth = self.processor(text=target_text)["input_ids"].to(device).squeeze()
            self.criterion = nn.CrossEntropyLoss()

        def forward(self, image: Tensor) -> Tensor:
            x = self.resizer(image)
            inputs = self.processor(text=prompt + target_text, images=torch.zeros_like(image), return_tensors="pt")
            inputs["pixel_values"] = x
            inputs = inputs.to(self.device)
            # Generate
            result = self.llava.forward(**inputs)
            logits = result["logits"].squeeze()[-self.ground_truth.numel() : -1]  # 从start token开始算
            target = self.ground_truth[1:]  # remove start token
            loss = self.criterion(logits, target)
            return loss

    return LlavaLossModel()


def get_llava_predict_loss_model(
    target_text: str = "The image shows a bomb",
    llava_path="llava-hf/llava-1.5-7b-hf",
    prompt="<image>\nUSER: What's the content of the image?\nASSISTANT:",
):
    class DifferentiableLlava(LlavaForConditionalGeneration):
        def __init__(self, *args, **kwargs):
            super().__init__(*args, **kwargs)

        def greedy_search(self, *args, **kwargs):
            with torch.enable_grad():
                return super().greedy_search(*args, **kwargs)

    class LLavaPredictLossModel(nn.Module):
        def __init__(self, device=torch.device("cuda")):
            super().__init__()
            # please make sure here that version of transformers is <= 4.38, otherwise we'll get an error: vocab_size cannot be set.
            self.llava = DifferentiableLlava.from_pretrained(llava_path, torch_dtype=torch.float16)
            self.processor = AutoProcessor.from_pretrained(llava_path, torch_dtype=torch.float16)
            self.resizer = transforms.Compose(
                [
                    transforms.Resize((336, 336), antialias=True),
                    transforms.Normalize(
                        mean=[0.48145466, 0.4578275, 0.40821073], std=[0.26862954, 0.26130258, 0.27577711]
                    ),
                ]
            )
            self.eval().requires_grad_(False).to(device)
            self.device = device
            self.ground_truth = self.processor(text=target_text)["input_ids"].squeeze().to(device)[1:]  # ignore the <s>
            self.criterion = nn.CrossEntropyLoss()
            print(f"llava ground truth account for {self.ground_truth.numel()} tokens")
            with open("./prompts/raw_prompts.txt") as f:
                prompts = f.read().splitlines()
            self.prompt_list = prompts
            self.prompt = prompt

        def forward(self, image: torch.Tensor) -> torch.Tensor:
            x = self.resizer(image).to(self.device)
            # to get the input template.
            if self.prompt is None:
                cur_prompt = random.choice(self.prompt_list)
                self.prompt = "<image>\nUSER: " + cur_prompt + "\nASSISTANT:"
            inputs = self.processor(text=self.prompt, images=torch.ones_like(image) * 255).to(self.device)
            # replace the image with our adversarial example
            inputs["pixel_values"] = x
            # get the logits
            generate_length = inputs["input_ids"].numel() + self.ground_truth.numel() + 3  # +3 have some redundancy
            # TODO: 改成sample，增加正则效果
            out = self.llava.generate(
                **inputs, max_length=generate_length, output_logits=True, return_dict_in_generate=True
            )
            logits = torch.cat(out["logits"], dim=0)
            # select the common part, and calculate the loss.
            end = min(self.ground_truth.shape[0], logits.shape[0])
            loss = self.criterion(logits[:end], self.ground_truth[:end])
            return loss

    return LLavaPredictLossModel()
