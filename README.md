# Automated mitosis detection in stained histopathological images using Faster R-CNN and stain techniques

Please consider to take a look to the official MIDOG++ repository https://github.com/DeepMicroscopy/MIDOGpp.

## Abstract

Accurate mitosis detection is essential for cancer diagnosis and treatment. Traditional manual counting by pathologists is time-consuming and may cause errors. This research investigates automated mitosis detection in stained histopathological images
using Deep Learning (DL) techniques, particularly object detection models. We propose a two-stage object detection model based on Faster R-CNN to effectively detect mitosis within histopathological images. The stain augmentation and normalization techniques are also applied to address the significant challenge of domain shift in  histopathological image analysis. The experiments are conducted using the MIDOG++ dataset, the most recent dataset from the MIDOG challenge. This research builds on our previous work, in which two one-stage frameworks, in particular on RetinaNet using fastai and PyTorch, are proposed. Our results indicate favorable F1-scores across various scenarios and tumor types, demonstrating the effectiveness of the object detection models. In addition, Faster R-CNN with stain techniques provides the most accurate and reliable mitosis detection, while RetinaNet models exhibit faster performance. Our results highlight the importance of handling domain shifts and the number of mitotic figures for robust diagnostic tools.

## Usage

For our experiments, we employed a Python 3.8.10 environment. If you consider to employ a higher Python version, be aware of warnings and posibble errors in the code developed.

### 1. Download MIDOG++ dataset.

### 2. Install requirements.

### 3. Launch experiments


## Citation
If you use this repository in your research, please cite the following papers:

```bibtex
@article{aubreville2023comprehensive,
  title={A comprehensive multi-domain dataset for mitotic figure detection},
  author={Aubreville, Marc and Wilm, Frauke and Stathonikos, Nikolas and Breininger, Katharina and Donovan, Taryn A and Jabari, Samir and Veta, Mitko and Ganz, Jonathan and Ammeling, Jonas and van Diest, Paul J and others},
  journal={Scientific data},
  volume={10},
  number={1},
  pages={484},
  year={2023},
  publisher={Nature Publishing Group UK London}
}

@inproceedings{jesus2024validating,
  author    = {Jesus García-Salmerón and J. M. García and G. Bernabé and P. González-Férez},
  title     = {Validating {RetinaNet} for the {Object Detection-Based Mitosis Detection} in the {MIDOG} Challenge},
  booktitle = {Proceedings of the 18th International Conference on Practical Applications of Computational Biology \& Bioinformatics (PACBB)},
  series    = {Lecture Notes in Networks and Systems},
  publisher = {Springer Verlag},
  year      = {2024},
  month     = {June},
  address   = {Salamanca, Spain},
  note      = {To be published}
}


```
