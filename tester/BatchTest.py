from utils import get_list_image
from typing import Callable


def test_model_output_batch(model: Callable, img_path: str, verbose: bool = True):
    results = []
    images = get_list_image(img_path, sort_key=lambda x: int(x[:-4]))
    for image in images:
        out = model(image)
        if verbose:
            print(out)
            print("-" * 50)
        results.append(out)
    return results
