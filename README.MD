# Transfer Attacks on VLMs

By Huanran Chen

Welcome to the **Transfer Attacks on VLMs** project! This work demonstrates an impressive **95%+ transfer attack success rate** against GPT-4V. In simple terms, we can alter the meaning of an image—such as removing or adding semantics—with over 95% accuracy when targeting GPT-4.

Previous methods achieve nearly 0% transfer rate. We overcome these by

- CWA optimizer. **The key of our success**. Which enables us to ensemble as many models as we want without gradient conflict.
- Attacking vision encoder rather than the whole pipeline. Which make a **significant** difference.

## Project Structure

- **attacks**: Contains the attack algorithms, all implemented by Huanran Chen. These algorithms are similar to those in my other repositories: [Repo 1](https://github.com/huanranchen/AdversarialAttacks), [Repo 2](https://github.com/huanranchen/DiffusionClassifier), and [Repo 3](https://github.com/huanranchen/NoisedDiffusionClassifiers).
- **data**: Provides dataloaders for various datasets like CIFAR, MNIST, ImageNet, and the commonly used NIPS17 dataset (1,000 images).
- **defenses**: Contains DiffPure, also implemented by Huanran Chen. For more details, see the previous repository.
- **surrogates**: Includes pre-packaged surrogate models, designed and implemented by Huanran Chen.
- **tester**: Includes functions for testing, such as `test_acc` and `test_robustness`.
- **utils**: Houses useful utility functions.

## Usage

To get started, first install the latest version of PyTorch along with any other necessary packages. Since the package uses dynamic imports, you only need to install a minimal set of dependencies. For this reason, we do not provide a `requirements.txt` file, as you may only need part of the package.

A minimal example of the required packages is:

```shell
pip3 install torch torchvision torchaudio
pip install transformers opencv-python open_clip_torch
```

There are two main scripts in this project:

- **change_semantic.py**: Changes the semantic content of input images to match a target text.
- **untargeted_semantic_elimination.py**: Removes the current semantics of input images, similar to an untargeted attack, resulting in completely different semantics.

You can follow the instruction inside these files, and use arbitrary number of GPUs to run it (as long as not run to OOM), e.g., 

```shell
CUDA_VISIBLE_DEVICES=4,5,6 python untargeted_semantic_elimination.py --input_dir="/home/chenhuanran2022/work/VLMTransfer/resources/bombs/" --output_dir="./output/"
```

## Citations

This project uses the **SSA-CWA algorithm**, which allows optimization across multiple surrogate models without gradient conflicts. The **SSA method** improves the semantic accuracy of adversarial examples.

If you use this work, please consider citing the following papers:

```bibtex
@inproceedings{chenrethinking,
  title={Rethinking Model Ensemble in Transfer-based Adversarial Attacks},
  author={Chen, Huanran and Zhang, Yichi and Dong, Yinpeng and Yang, Xiao and Su, Hang and Zhu, Jun},
  booktitle={The Twelfth International Conference on Learning Representations},
  year={2023},
}
```

Additionally, this attack has been evaluated in the following papers:

```bibtex
@inproceedings{dongrobust,
  title={How Robust is Google's Bard to Adversarial Image Attacks?},
  author={Dong, Yinpeng and Chen, Huanran and Chen, Jiawei and Fang, Zhengwei and Yang, Xiao and Zhang, Yichi and Tian, Yu and Su, Hang and Zhu, Jun},
  booktitle={R0-FoMo: Robustness of Few-shot and Zero-shot Learning in Large Foundation Models},
  year={2023},
}

@article{zhang2024benchmarking,
  title={Benchmarking Trustworthiness of Multimodal Large Language Models: A Comprehensive Study},
  author={Zhang, Yichi and Huang, Yao and Sun, Yitong and Liu, Chang and Zhao, Zhe and Fang, Zhengwei and Wang, Yifan and Chen, Huanran and Yang, Xiao and Wei, Xingxing and others},
  journal={arXiv preprint arXiv:2406.07057},
  year={2024}
}
```

