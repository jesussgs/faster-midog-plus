# Automated mitosis detection in stained histopathological images using Faster R-CNN and stain techniques

Please consider to take a look to the official MIDOG++ repository (https://github.com/DeepMicroscopy/MIDOGpp).

## Abstract

Accurate mitosis detection is essential for cancer diagnosis and treatment. Traditional manual counting by pathologists is time-consuming and may cause errors. This research investigates automated mitosis detection in stained histopathological images
using Deep Learning (DL) techniques, particularly object detection models. We propose a two-stage object detection model based on Faster R-CNN to effectively detect mitosis within histopathological images. The stain augmentation and normalization techniques are also applied to address the significant challenge of domain shift in  histopathological image analysis. The experiments are conducted using the MIDOG++ dataset, the most recent dataset from the MIDOG challenge. This research builds on our previous work, in which two one-stage frameworks, in particular on RetinaNet using fastai and PyTorch, are proposed. Our results indicate favorable F1-scores across various scenarios and tumor types, demonstrating the effectiveness of the object detection models. In addition, Faster R-CNN with stain techniques provides the most accurate and reliable mitosis detection, while RetinaNet models exhibit faster performance. Our results highlight the importance of handling domain shifts and the number of mitotic figures for robust diagnostic tools.

## Usage

All the code developed is available in the **MIDOGpp-main** folder.

### 1. Download MIDOG++ dataset.

Following the MIDOG++ instructions, you should use ``Setup.ipynb`` to download all the images of the dataset. The dataset comprises 65 GB of images, so the download process will take some time. Feel free to take a break and enjoy a coffee!

Remember to place these images in the ``images`` folder.

### 2. Install requirements.

For our experiments, we utilized a Python 3.8.10 environment. If you choose to use a higher version of Python, please be mindful of potential warnings and possible errors in the developed code.


To install all the required dependencies for our code, you may use pip as follows:

``pip install -r requirements.txt``

### 3. Launch experiments.

Before starting any experiment, it is essential to review and update the following elements:
  - **Wandb project configuration**: Ensure that you update ``configs/all.yaml`` with your wandb group and project settings. Additionally, you must set your wandb API key in the training files for proper execution.
  - **Code path**: The developed code uses  ``/app/MIDOGpp-main`` as the main project directory. Please review all files and update this path to match your custom directory as needed.
    
To start a experiment you only need to launch ``python MIDOGpp-main/main.py``. You can modify this file by commenting or uncommenting the relevant function calls within the main procedure.
  - **Train a specific model**: Simply comment out the call to the current training model and uncomment the call for the desired model.
  - **Single-Domain and Leave-One-Out experiments**: In the training files, you need to select or deselect the tumor type for the images to train your model.

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
