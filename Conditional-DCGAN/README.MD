# 📄 **About the Original Paper: Conditional GAN (Mirza & Osindero, 2014)**

The **Conditional GAN** was first introduced by *Mehdi Mirza and Simon Osindero* in their paper:
**“Conditional Generative Adversarial Nets” (2014)**
**[arXiv:1411.1784](https://arxiv.org/abs/1411.1784)**

### 🔑 Key Idea:

Unlike standard GANs that generate data solely from noise, **Conditional GANs (cGANs)** incorporate **side information** such as **class labels** to condition both the Generator and the Discriminator.

**In formal terms:**

* **Generator (G)** learns to model `P(x | y)`: generate data `x` given label `y`.
* **Discriminator (D)** receives both image `x` and label `y`, and learns to classify whether the pair `(x, y)` is real or fake.

This allows controlled image generation—e.g., generating a specific digit from MNIST or a specific object class from CIFAR.

---

## 🔄 **Differences from the Original Paper**

Your implementation builds on this idea, but adapts it for more **complex, high-resolution scene images** from the **AID dataset**. Here's how it differs and improves:

| Feature                 | Original cGAN (2014)   | Your Implementation                                                   |
| ----------------------- | ---------------------- | --------------------------------------------------------------------- |
| **Dataset**             | MNIST / Toy data       | AID scene classification (high-res images, 256×256)                   |
| **Generator Input**     | Concatenated `[z, y]`  | Label embedding + linear projection + convolutional blocks            |
| **Discriminator Input** | Concatenated `[x, y]`  | Dual path for image and label, merged via projection                  |
| **Image Size**          | 28×28 or small         | 256×256 (large, high-res)                                             |
| **Upsampling**          | Fully connected layers | Bilinear upsampling + convolutional layers                            |
| **Loss**                | Binary cross-entropy   | BCE with **label smoothing** (for better convergence)                 |
| **Label Conditioning**  | Basic concat           | **Separate embedding layers**, condition injection at multiple stages |
| **Visualization**       | Minimal                | Intermediate image saving + **GIF visualization**                     |

---

## 🧠 Why These Changes?

* **High-resolution images** require deeper architectures and better upsampling strategies.
* **Scene data** is more complex than digits; thus, label conditioning needs to be injected more effectively.
* **Label smoothing** in the loss improves training stability by preventing the discriminator from becoming overconfident.
* **Bilinear upsampling** is less prone to checkerboard artifacts than transpose convolutions.

## Generated Images Evolve

![gif](/Conditional-DCGAN/imgs/generation%20progress.gif)