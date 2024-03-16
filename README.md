# AudioFoleyDeepFake Detection

## Overview
AudioFoleyDeepFake is a research project dedicated to the detection of deepfake environmental audio. The project utilizes the power of CLAP audio embeddings to efficiently and accurately discern synthesized environmental sounds from real ones. This repository contains the code and documentation for the detection pipeline developed as part of our research.

## Research Paper

This project is based on the research paper titled "Detection of Deepfake Environmental Audio," authored by Hafsa Ouajdi, Oussama Hadder, Modan Tailleur, Mathieu Lagrange, and Laurie M. Heller. The paper introduces a novel approach for detecting fake environmental sounds with a primary focus on Foley sound synthesis. Through extensive research and experimentation, the team developed a detection pipeline leveraging CLAP audio embeddings, which proved to be highly effective in distinguishing between real and synthetic environmental sounds. The proposed method achieved an impressive detection accuracy of 98%, highlighting its potential in combating the challenges posed by deepfake audio technologies.

The findings of this research are going to be presented at the European Signal Processing Conference (EUSIPCO) 2024, showcasing the significant advancements in the field of audio deepfake detection. 


## Installation

To set up the project environment, follow these steps:

1. **Clone the Repository**
```
git clone https://github.com/mathieulagrange/audioFoleyDeepFake.git
``` 
2. **Installing PyTorch**

PyTorch requires a specific installation depending on your system's CUDA version. We recommend visiting the [PyTorch official website](https://pytorch.org/) and following the instructions there to install the version of PyTorch that matches your system's CUDA version.

For example, to install PyTorch with CUDA 12.1 (), you would run:
```bash
pip install torch==2.1.1+cu121
```
3. **Install Dependencies**
Navigate to the project directory and install the required Python packages:
```
pip install -r requirements.txt
``` 


