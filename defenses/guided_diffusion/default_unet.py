from .script_util import create_model_and_diffusion
from .config import model_and_diffusion_defaults
import torch


def get_guided_diffusion_unet(
    pretrained=True,
    resolution=256,
    cond=False,
    use_fp16=False,
    checkpoint_path="./resources/checkpoints/guided_diffusion/256x256_diffusion",
):
    config = model_and_diffusion_defaults()
    config.update(use_fp16=use_fp16)
    path = checkpoint_path + "_uncond.pt" if not cond else checkpoint_path + ".pt"
    if cond is False:
        config["class_cond"] = False
    if resolution == 128:
        config["image_size"] = 128
    elif resolution == 64:
        config["image_size"] = 64
    model, _ = create_model_and_diffusion(**config)
    if pretrained:
        model.load_state_dict(torch.load(path.replace("256", str(resolution))))

    class GuidedDiffusionMeanModel(torch.nn.Module):
        def __init__(self):
            super(GuidedDiffusionMeanModel, self).__init__()
            self.model = model
            self.eval().requires_grad_(False).cuda()
            if use_fp16:
                self.model.convert_to_fp16()

        def forward(self, x, t, y=None):
            if use_fp16:
                x, t = x.to(torch.float32), t.to(torch.float32)
                y = y.to(torch.float32) if y is not None else y
            pre = self.model(x, t, y)[:, :3, :, :]
            return pre.to(torch.float32)

    return GuidedDiffusionMeanModel()
