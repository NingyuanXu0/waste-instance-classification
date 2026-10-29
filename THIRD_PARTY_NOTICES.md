# Source and third-party notices

The main training, evaluation, dataset and model files are selected from the author's artificial intelligence classification project (`AI Project2`). The HOG model comes from the related `AI Lab4` implementation. Both were developed around course-provided classification and COCO-loading scaffolds. Shared settings, image-level partitions and the HOG command-line entry were added for publication.

`model.py` follows the torchvision AlexNet implementation structure, with seven output classes by default. The upstream [torchvision source](https://github.com/pytorch/vision/blob/main/torchvision/models/alexnet.py) and its [BSD 3-Clause license](third_party/torchvision-LICENSE.txt) provide the architecture attribution and retained reuse terms. The license text was obtained from [pytorch/vision/LICENSE](https://github.com/pytorch/vision/blob/main/LICENSE).

PyTorch, pycocotools, NumPy, Pillow, scikit-learn, scikit-image, matplotlib, seaborn and tqdm are installed as dependencies under their own licenses. Repository-wide reuse terms for the coursework code and publication additions remain unspecified.
