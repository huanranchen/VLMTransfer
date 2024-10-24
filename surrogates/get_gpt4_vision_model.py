from torch import nn, Tensor
import os
from openai import OpenAI
import base64
from utils import show_image
import io
import dotenv
from typing import Iterable
from openai.types.chat import ChatCompletionMessageParam

dotenv.load_dotenv(".env")

__all__ = ["GPT4PredictModel"]


class GPT4PredictModel(nn.Module):
    def __init__(
        self,
        max_tokens=200,
        text_prompt="What's the main object in this image? Is there any other object in this image? "
                    "Please answer within 150 words.",
    ):
        super(GPT4PredictModel, self).__init__()
        self.client = OpenAI(
            api_key=os.environ.get("OPENAI_API_KEY"),
            # replace openai with aiproxy
            base_url="https://api.aiproxy.io/v1/",
        )
        self.max_tokens = max_tokens
        self.text_prompt = text_prompt

    @staticmethod
    def encode_image(image: Tensor):
        pil_image = show_image(image)
        buffered = io.BytesIO()
        pil_image.save(buffered, format="PNG")  # Please align the PNG/JPG type with that in forward function
        img_str = base64.b64encode(buffered.getvalue()).decode("utf-8")
        return img_str

    def forward(self, x: Tensor) -> str:
        assert x.shape[0] == 1, "batch size should be 1."
        encoded_image = self.encode_image(x)
        message = [
            {
                "role": "user",
                "content": [
                    {"type": "text", "text": self.text_prompt},
                    {"type": "image_url", "image_url": {"url": f"data:image/png;base64,{encoded_image}"}},
                ],
            },
        ]
        response = self.get_response(message)
        return response

    def get_response(self, message: Iterable[ChatCompletionMessageParam]) -> str:
        response = self.client.chat.completions.create(
            model="gpt-4-vision-preview",
            messages=message,
            max_tokens=150,
        )
        return response.choices[0].message.content
