import torch
from transformers import (
    InstructBlipProcessor,
    InstructBlipForConditionalGeneration,
)
from torchvision import transforms
from torch import nn
import os
from utils.ImageHandling import show_image

os.environ["TOKENIZERS_PARALLELISM"] = "false"

OPENAI_CLIP_MEAN = [0.48145466, 0.4578275, 0.40821073]
OPENAI_CLIP_STD = [0.26862954, 0.26130258, 0.27577711]
IMAGENET_DEFAULT_MEAN = [0.485, 0.456, 0.406]
IMAGENET_DEFAULT_STD = [0.229, 0.224, 0.225]
IMAGENET_STANDARD_MEAN = [0.5, 0.5, 0.5]
IMAGENET_STANDARD_STD = [0.5, 0.5, 0.5]


__all__ = ["InstructBlipVisionModel", "InstructBlipPredictModel"]


class InstructBlipVisionModel(nn.Module):
    def __init__(
        self,
        prompt="Question: describe the image. Answer:",
        target_text="bomb bomb",
    ):
        super(InstructBlipVisionModel, self).__init__()
        self.normalizer = transforms.Compose(
            [
                transforms.Resize((224, 224)),
                transforms.Normalize(mean=OPENAI_CLIP_MEAN, std=OPENAI_CLIP_STD),
            ]
        )
        self.processor = InstructBlipProcessor.from_pretrained("Salesforce/instructblip-vicuna-7b")
        self.model = InstructBlipForConditionalGeneration.from_pretrained("Salesforce/instructblip-vicuna-7b")
        self.device = torch.device("cuda")
        self.eval().requires_grad_(False)
        self.labels = torch.tensor(self.processor(text=target_text).input_ids).view(1, -1)
        self.prompt = torch.tensor(self.processor(text=prompt).input_ids).view(1, -1)

    def forward(self, x):
        x = torch.clamp(x, min=0, max=1)
        batch_size = x.shape[0]
        inputs = dict(pixel_values=self.normalizer(x).to(self.device))
        # inputs["input_ids"] = self.prompt.repeat(batch_size, 1)
        inputs["input_ids"] = self.labels.repeat(batch_size, 1).to(self.device)
        inputs["labels"] = self.labels.repeat(batch_size, 1).to(self.device)
        # TODO: qformer_input_ids应该是这个吗？？？
        inputs["qformer_input_ids"] = self.labels.repeat(batch_size, 1).to(self.device)
        # inputs["labels"] = self.labels.repeat(batch_size, 1)
        outputs = self.model(**inputs)
        return outputs.loss


class InstructBlipPredictModel(nn.Module):
    def __init__(
        self,
        prompt="Question: describe the image. Answer:",
        target_text="bomb bomb",
    ):
        super(InstructBlipPredictModel, self).__init__()
        self.normalizer = transforms.Compose(
            [
                transforms.Resize((224, 224)),
                transforms.Normalize(mean=OPENAI_CLIP_MEAN, std=OPENAI_CLIP_STD),
            ]
        )
        self.processor = InstructBlipProcessor.from_pretrained("Salesforce/instructblip-vicuna-7b")
        self.model = InstructBlipForConditionalGeneration.from_pretrained("Salesforce/instructblip-vicuna-7b")
        self.device = torch.device("cuda")
        self.eval().requires_grad_(False)
        self.labels = torch.tensor(self.processor(text=target_text).input_ids, device=self.device).view(1, -1)
        self.prompt = torch.tensor(self.processor(text=prompt).input_ids, device=self.device).view(1, -1)
        self.eval().requires_grad_(False).cuda()

    def forward(self, x):
        x = torch.clamp(x, min=0, max=1)
        batch_size = x.shape[0]
        inputs_my = dict(pixel_values=self.normalizer(x).cuda())
        # inputs["input_ids"] = self.prompt.repeat(batch_size, 1)
        inputs = self.processor(images=show_image(x),
                                text="Question: describe the image in detail, at least 5 words. Answer:",
                                return_tensors="pt").to(self.device)
        print(torch.sum((inputs_my["pixel_values"] - inputs["pixel_values"]) ** 2))
        ids = self.model.generate(**inputs)
        text = self.processor.batch_decode(ids, skip_special_tokens=True)[0].strip()
        print(text)
        return text
