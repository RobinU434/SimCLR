# SimCLR Lightning

## Overview

This repository provides a PyTorch Lightning implementation of the SimCLR (Self-Supervised Contrastive Learning) framework for self-supervised learning of visual representations. SimCLR learns representations by contrasting positive pairs of augmented views of the same data example against negative pairs of different examples.

## Installation

1. **Clone the repository:**
   
   ```bash
   git clone https://github.com/RobinU434/SimCLR.git
   ```

2. **Create a virtual environment:**
   
   ```bash
   python -m venv venv
   source venv/bin/activate
   ```

3. **Install dependencies:** We depend on poetry for python package management. 
   
   ```bash
   poetry install --no-root
   ```

## Training

1. **Model Configuration:**
   Configure the SimCLR model, including the base encoder (e.g., ResNet), projection head, and other hyperparameters in the [config file](./configs/simclr_config.yaml). You can also make a copy of this config file and specify the path to your config file in the CLI.  
   If you would like to develop further features please make sure your altered config file is mirrored in [code](./simclr/utils/config.py). If you don't like to do this manually please checkout [Config2Class](https://github.com/RobinU434/Config2Class). 

2. **Training Script:**
   Run the training script:
   ```bash
   python -m simclr train --config-path <your-custom-config-path>  --save-path <directory-for-checkpoints-or-similar>
   ```
   Please note that the command-line arguments are optional.


**Contributions**

We welcome contributions to this repository. Feel free to submit pull requests or issues.

**License**

This project is licensed under the MIT License.

**Acknowledgements**

We would like to thank the PyTorch Lightning team for providing a flexible and efficient framework for deep learning.

**Beyond the Basics: Advanced Topics**

* [Solo-Learn](https://github.com/vturrisi/solo-learn): Explore the concept of solo-learning, a more efficient variant of SimCLR that uses a single network to learn representations.
* [MoCo](https://github.com/facebookresearch/moco): Implement the Momentum Contrast (MoCo) method, which uses a momentum encoder to stabilize training.
* [BYOL](https://github.com/google-deepmind/deepmind-research/tree/master/byol): Experiment with the Bootstrap Your Own Latent (BYOL) method, which leverages a target network to improve training stability.
* [SwAV](https://github.com/facebookresearch/swav): Implement the SwAV (Swapping Assignment) method, which uses clustering assignments to learn representations.

By diving into these advanced techniques, you can further enhance the performance and versatility of your SimCLR implementation.
