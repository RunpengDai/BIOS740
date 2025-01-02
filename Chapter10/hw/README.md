# Chapter10: Image Segmentation and Registration

## Note for Instructor (Should be deleted when published in Canvas)

1. The solution file in provided in the same folder, such as `Task1.1_Segmentation_solution.ipynb`, which contains the codes as well as output of these codes.
2. `Task1.2/dataset_conversion.py` should also be downloaded for students. Please make sure it is not present in the repository.
3. For the working virtual environment, script and dataset, please refer to the folder `/work/users/y/u/yuukias/Class-Homework-Full`. 

## Description

This project consists of two major tasks: 

+ Task1: Image segmentation 
+ Task2: Image registration

Both tasks utilize the deep learning techniques to deal with real-word medical imaging problems. If possible, you might be able to request and access the GPU partition on Longleaf to facilitate the training of models. Please refer to  [this link](https://help.rc.unc.edu/gpu/) for more information. 

### Note for Task1

The task1 can be further divided into three sub-tasks, which demonstrates the roadmap from the vanilla U-Net that is introduced in 2015 to the recently-proposed foundation model in these two years.

Some helper functions have been provided for you to save the time, which can be found in `utils/Task1_utils.py`.


## Setup

### Create Virtual Environment

Through creation of virtual environment (which is usually done through [Conda](https://anaconda.org/anaconda/conda)), you can isolate the dependencies of this project from other projects and avoid the nasty problem of incompatible versions. 

```bash
conda create --prefix=<your_env_name> python=3.9.6
conda activate ./<your_env_name>
```

### Install Dependencies

A list of requirements is provided in `requirements.txt`, which should be able to provide frequently-used packages such as `numpy`, `matplotlib` and `torch`. The project and may be refined in the future. **You still need to manually downloaded certain packages for task1.2 and task2.**

```bash
pip install -r requirements.txt
```
### Expectations

Please refer to each Jupyter Notebook for details.

You are expected to finish the following code for each task:

1. Task1.1: Finish the structure of U-Net, evaluate performance using Dice and JaccardIndex
2. Task1.2: Install and configure nnU-Net, compare the performance of nnU-Net and U-Net
3. Task1.3: Configure Segment Anything, learn how to use Python to make inference with pre-trained checkpoint.
4. Task2: Conduct image registration using VoxelMorph on brain MRI according to the example on MNIST.
