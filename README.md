# AudioFoleyDeepFake Detection

## Overview
AudioFoleyDeepFake is a research project dedicated to the detection of deepfake environmental audio. The project utilizes the power of CLAP audio embeddings to efficiently and accurately discern synthesized environmental sounds from real ones. This repository contains the code and documentation for the detection pipeline developed as part of our research.

## Introduction

This project is based on the research paper titled "Detection of Deepfake Environmental Audio," authored by Hafsa Ouajdi, Oussama Hadder, Modan Tailleur, Mathieu Lagrange, and Laurie M. Heller. The paper introduces a novel approach for detecting fake environmental sounds with a primary focus on Foley sound synthesis. Through extensive research and experimentation, the team developed a detection pipeline leveraging CLAP audio embeddings, which proved to be highly effective in distinguishing between real and synthetic environmental sounds. The proposed method achieved an impressive detection accuracy of 98%, highlighting its potential in combating the challenges posed by deepfake audio technologies.

The findings of this research are going to be presented at the European Signal Processing Conference (EUSIPCO) 2024.

This repository contains the code and documentation for our project on detecting deepfake environmental audio, presented as Python scripts and Jupyter notebooks. The project is structured around key tasks that represent the stages of developing and deploying a deepfake audio detection system:

1. [Data Loader](deepfakeClassifiers/data_loader.py)
   - Prepares the training, evaluation, and validation loaders.
   - Ensures that data is correctly formatted and partitioned for the training, validation and evaluation steps.

2. [Classifier](deepfakeClassifiers/models)
   - Defines the architecture of deep learning classifiers used in the experiments.

3. [Hparams](deepfakeClassifiers/hparams.py)
   - Manages parameters handling for the experiments.
   - Allows for easy adjustment and optimization of hyperparameters to enhance model performance.

4. [Utilities](deepfakeClassifiers/utils.py)
   - Provides classes and functions to train the classifier on embeddings and predict outputs.

5. [examples](deepfakeClassifiers/examples/)
   - Contains Jupyter notebooks with examples to train the model, compute the inference time, and analyze predictions' results.
   - Demonstrates best practices and methodological insights gained from our research.


For a more detailed overview of each component and to understand how they integrate to form a complete detection pipeline, please explore the source files and notebooks provided in this repository.


## Installation

To set up the project environment, follow these steps:

1. **Clone the Repository**
```
git clone https://github.com/mathieulagrange/audioFoleyDeepFake.git
``` 
2. **Installing CUDA**

CUDA depends on the actual gpu of your machine, if you are willing to use the GPU in the presented experiments and want to install cuda, we recommend visiting the [PyTorch official website](https://pytorch.org/) and [Cuda Installation Guide](https://docs.nvidia.com/cuda/cuda-installation-guide-microsoft-windows/index.html) and following the instructions there to install the version of CUDA that matches your system's GPU.

For instance, we performed all our experiments in NVIDIA GeForce RTX 3060 GPU. Therefore the CUDA version installed was 12.4.

3. **Install Dependencies**

Navigate to the project directory and install the required Python packages:
```
pip install -r requirements.txt
``` 

## Dataset

The detection model was evaluated using audio data from the 2023 DCASE challenge task 7 on Foley sound synthesis, which includes over 6 hours of recorded audio and 28 hours of generated audio. The non-fake audio files and the generated audio files from the DCASE 2023 Challenge can be downloaded from [this website](https://zenodo.org/records/8091972)


## Results and Discussion

The detailed results of our experiments are thoroughly discussed in our paper. We achieved high accuracy with the CLAP-2023 method and a Multilayer Perceptron (MLP) model, showing an evaluation and validation accuracy of 98%.

For our "Discussion" section, we selected specific audio examples that are available on [the companion page](./index.html). These selections were made using the embeddings from the CLAP-2023 method and the aforementioned MLP model. The strategy for choosing these sounds was based on their likelihood scores:

- For the False Positive (FP) category, we selected fake sounds that the model incorrectly identified with high confidence.
- For the False Negative (FN) category, we chose fake sounds that the model failed to identify, indicated by low likelihood scores.

For the most curious, some of these examples are available on the companion page.



