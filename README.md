# Tensor Decomposition Based Attention Module for Spiking Neural Networks
![](https://img.shields.io/badge/docker_image-√-green)
![](https://img.shields.io/badge/docker_file-√-green)
![](https://img.shields.io/badge/pip-√-green)

Haoyu Deng<sup>a</sup>, Ruijie Zhu<sup>b</sup>, Xuerui Qiu<sup>a</sup>, Yule Duanu<sup>a</sup>, Malu Zhang<sup>a,†</sup> and Liang-Jian Deng<sup>a,†</sup>

<sup>a</sup>University of Electronic Science and Technology of China, 611731, China

<sup>b</sup>University of California, Santa Cruz, 95064, The United States

<sup>†</sup>Corresponding authors
![img.png](img.png)


This is the official repository for paper *Tensor Decomposition Based Attention Module for Spiking Neural Networks* 

paper:[pdf](https://arxiv.org/pdf/2310.14576.pdf)


## CIFAR10/CIFAR100
### How to Run

#### 1. Prepare environment
A Docker environment is strongly recommended, but you can also use pip to prepare the environment.
If you are not familiar with Docker, you can refer to the following link to know about docker:

[English Tutorial](https://towardsdatascience.com/build-and-run-a-docker-container-for-your-machine-learning-model-60209c2d7a7f)

[中文教程](https://zhuanlan.zhihu.com/p/31772428)
##### 1) Docker file
A docker file is provided in `env` directory. You can build the docker image and run the container with the following commands:

```bash
docker build -t pfa ./env
```

##### 2) Docker image
We provide pre-build docker image on docker hub. You can pull the image with the following command:

```bash
docker pull risingentropy409/pfa
```
The environment is ready to use after you have the image.

##### 3) pip
Use the following command to setup environment with pip:

```bash
pip install -r requirements.txt
```
<font color="red">NOTE: If your cuda version is above 12.0, you may modify cupy-cuda11x in requirements.txt to cupy-cuda12x</font>

### 2.Run the code
Modify `CIFAR/main.py` according to your needs and run:
```bash
python main.py
```

## FMNIST
This code is modified from the [example](https://github.com/fangwei123456/spikingjelly/blob/0.0.0.0.12/spikingjelly/clock_driven/examples/conv_fashion_mnist.py) code of the [spikingjelly](https://github.com/fangwei123456/spikingjelly/) framework. We appreciate the authors' great contribution.

### How to run

Just run:
```bash
python run.py
```

For more configuration, please refer to the code. The code is simple and clear.

## Generation Tasks (FSVAE)
We conduct generation tasks based on [FSVAE](https://github.com/kamata1729/FullySpikingVAE). The modified code is in the `Generation` folder. To run it, please refer to `Generation/README.md`

## Citation
If you think our work is useful, please give us a worm citation!
```
@article{DENG2024111780,
title = {Tensor decomposition based attention module for spiking neural networks},
journal = {Knowledge-Based Systems},
volume = {295},
pages = {111780},
year = {2024},
issn = {0950-7051},
doi = {https://doi.org/10.1016/j.knosys.2024.111780},
url = {https://www.sciencedirect.com/science/article/pii/S0950705124004143},
author = {Haoyu Deng and Ruijie Zhu and Xuerui Qiu and Yule Duan and Malu Zhang and Liang-Jian Deng},
keywords = {Spiking neural network, Attention mechanism, Tensor decomposition, Neuromorphic computing}
}
```