# DCGAN: Deep Convolutional Generative Adversarial Network for Car Image Generation

This project implements a Deep Convolutional Generative Adversarial Network (DCGAN) to generate realistic car images, inspired by the foundational work by Radford et al.

---

## 🧠 Overview
The DCGAN architecture comprises two primary components

- **Generator**: Transforms random noise vectors into synthetic images
- **Discriminator**: Evaluates images to distinguish between real and generated samples
 The models are trained adversarially, with the generator aiming to produce images that can deceive the discriminator, and the discriminator striving to accurately identify real versus fake images

---

## 🗂️ Dataset
 The model is trained on the [Stanford Cars Dataset](http://kaggle.com/datasets/jessicali9530/stanford-cars-dataset/versions/2)


 Images are resized to 64x64 pixels and normalized to a range of [-1, 1] to facilitate stable trainin.

---

## ⚙️ Architecture Details

### Generator
 The generator starts with a 100-dimensional latent vector and employs a series of transposed convolutional layers to upscale the input to a 64x64x3 image. Each layer is followed by batch normalization and ReLU activations, except for the output layer, which uses a Tanh activation functio.

### Discriminator
 The discriminator is a convolutional neural network that downsamples the input image through successive convolutional layers, each followed by batch normalization and LeakyReLU activations. The final layer outputs a single scalar value, passed through a Sigmoid activation to represent the probability of the input being a real imag.
 Both networks are initialized with weights drawn from a normal distribution, as recommended in the DCGAN pape.

---

## 🧪 Training Procedure

- **Loss Function**:  Binary Cross-Entropy Loss is used for both generator and discriminatr.
- **Optimizers**:  Adam optimizer with learning rate 0.0002, β1=0.5, and β2=0.99.
- **Training Duration**:  200 epocs.
- **Batch Size**:  128
- **Label Smoothing**:  Implemented to improve training stability. Real labels are set to 0.95, and fake labels to 0.5.

 During training, the discriminator is updated to better distinguish real from fake images, and the generator is updated to produce more realistic images that can fool the discriminatr.

---

## 📊 Visualizatin

 The training process includes periodic visualization of generated images to monitor progress. Every 5 epochs, a grid of generated images.

 Additionally, a GIF (`DC-GAN.gif`) is created from the saved images to provide an animated view of the generator's improvement over tme.

![gif](/DCGAN/imgs/DC-GAN%20(1).gif)


### Randomly Generated Samples
![sample1](/DCGAN/imgs/generated_1.png)
![sample2](/DCGAN/imgs/generated_2.png)
![sample3](/DCGAN/imgs/generated_3.png)
![sample4](/DCGAN/imgs/generated_4.png)
![sample5](/DCGAN/imgs/generated_5.png)
---

## 📚 References

- Radford, A., Metz, L., & Chintala, S. (2015). *Unsupervised Representation Learning with Deep Convolutional Generative Adversarial Networks*. arXiv preprint arXiv:1511.06434. 

- Dumoulin, V., & Visin, F. (2016). *A guide to convolution arithmetic for deep learning*. arXiv preprint arXiv:1603.07285.

---

## 📝 Notes

- The code is designed to run in a Kaggle environment. Adjust file paths as necessary for other environments.
- Ensure that the Stanford Cars Dataset is properly downloaded and accessible at the specified path.
- For best results, monitor GPU usage to optimize batch size and training speed.
---

 This implementation provides a practical example of training a DCGAN on a real-world dataset, offering insights into generative modeling and adversarial training techiques. 