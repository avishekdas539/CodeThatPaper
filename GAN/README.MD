# 🧠 Generative Adversarial Network (GAN) on MNIST

This repository contains a custom implementation of a Generative Adversarial Network (GAN) trained on the MNIST dataset. The architecture and training procedure closely follow the original GAN paper by Ian Goodfellow et al., with a few practical improvements for training stability.

## 📌 Overview

A GAN consists of two neural networks:
- **Generator**: Learns to map random noise vectors to plausible images.
- **Discriminator**: Learns to distinguish between real images (from the dataset) and fake images (produced by the generator).

These two models compete in a MiniMax game:
> The generator tries to fool the discriminator, while the discriminator tries not to be fooled.

This adversarial process helps the generator learn to create realistic images.

---

## 🗂️ Project Structure

- `1. Data Preprocessing`: Loads and normalizes MNIST dataset images to the range [-1, 1].
- `2. Model Definitions`: Custom classes for Generator, Discriminator, and GAN with modular and configurable architecture.
- `3. Training Setup`: Defines optimizers, loss functions, accuracy metrics, and image-saving callbacks.
- `4. Training Loop`: Trains the GAN for a user-defined number of epochs using a custom `train_step()`.

---

## 🧱 Architecture

### Generator
- Fully connected layers with increasing units.
- Uses `ReLU` activation and `BatchNormalization`.
- Final layer uses `sigmoid` to output pixel values between 0 and 1.
- Outputs reshaped to (28, 28, 1).

### Discriminator
- Fully connected layers with decreasing units.
- Uses `ReLU` and `Dropout` for regularization.
- Final layer uses `sigmoid` to predict real/fake probability.

### Training Strategy
- **Label Smoothing**: Applies soft labels to real/fake images to prevent overconfidence.
- **Multiple Generator Updates**: Trains the generator multiple times per discriminator step to balance convergence.
- **Accuracy Metric**: Custom discriminator accuracy metric included.


## 📝 Reference

- [Original GAN Paper (Goodfellow et al., 2014)](https://arxiv.org/abs/1406.2661)
- [MNIST Dataset](http://yann.lecun.com/exdb/mnist/)

---

## 📷 Example Generated Images Grid

### 1. GAN-Tensorflow with Mode Collapse

![Test](/GAN/imgs/GAN-Loop.gif)

### 2. GAN-Pytorch with `LeakyRelu` and `BatchNorm`

![Test](/GAN/imgs/GAN-Pytorch.gif)

---

## 📌 TODO

- Add FID/IS score evaluation
- Add Conditional GAN version
- Train on Fashion-MNIST or CIFAR-10
