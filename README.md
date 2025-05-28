# 🧠 Deep Learning with PyTorch: Generative Adversarial Network (DCGAN)
![image](https://github.com/user-attachments/assets/c47f82f6-f400-4a2b-a8f3-e1541454bb5d)

## Project Overview
This project walks through the process of:

- Preparing the MNIST dataset with data augmentation.

- Building a Generator to create synthetic handwritten digits from random noise.

- Building a Discriminator to distinguish between real and generated images.

- Defining loss functions and optimizers.

- Training the GAN using alternating updates.

- Visualizing generated outputs after each epoch.

## Built using:

PyTorch — Deep learning framework for building and training neural networks

Torchvision — For dataset loading and image transformations

Matplotlib — To visualize generated images

TQDM — For progress bars during training

Torchsummary — To summarize model architecture

## Key Insights:
- #### The Model Can Learn to Create Handwritten Digits
We used a special type of neural network called a DCGAN that can generate new, fake handwritten digits that look like real ones from the MNIST dataset.

- #### Good Starting Weights Help Training
We initialized the model's weights in a smart way (using a normal distribution), which helped the model learn faster and more smoothly.

- #### Generator Output Uses Tanh Activation
We used a Tanh function at the end of the Generator so the images it creates match the expected pixel range. This made training more stable.

- #### LeakyReLU Helps the Discriminator Learn Better
Instead of using regular ReLU, we used LeakyReLU in the Discriminator. This allowed the model to learn from more inputs and avoid getting stuck during training.

- #### Optimizers Matter a Lot in GANs
We used the Adam optimizer with specific settings (beta1 = 0.5, beta2 = 0.99) to keep training stable. These settings helped the Generator and Discriminator learn at the right pace.

- #### Training Both Models Together is a Balancing Act
GANs are tricky because two models are learning at the same time. We trained them step-by-step and watched their loss values to make sure neither one got too strong.

- #### Image Rotation Makes the Model Smarter
By slightly rotating the MNIST images during training, we helped the model see more variations. This made the Generator produce more interesting and diverse digits.

- #### Seeing Results After Each Epoch is Helpful
We displayed generated images after each training round (epoch). This helped us quickly check if the Generator was improving or if something was going wrong.

## Project Structure
- Configurations: Set hyperparameters and device settings.

- Dataset Loading: Download and preprocess the MNIST dataset.

- Discriminator: CNN to classify real vs. fake digits.

- Generator: CNN-based model that transforms noise into images.

- Loss Functions: Binary cross-entropy losses for real and fake classifications.

- Optimizers: Adam optimizers for both networks.

- Training Loop: GAN training over multiple epochs.

- Visualization: Displays generated digit samples during training.

## Project Codes

### ⚙️ Configuration
These hyperparameters are standard for GAN training, with beta_1=0.5 helping stabilize training and reduce oscillations.
![image](https://github.com/user-attachments/assets/b562cac6-3519-415f-a072-7da9dd752567)

### 📦 Data Preparation
Random rotations introduce variation, helping the model learn more robust features.
![image](https://github.com/user-attachments/assets/41e0b53e-d677-4260-b963-edb74337ad64)

### 🧱 Discriminator Network
The Discriminator classifies images as real or fake.
![image](https://github.com/user-attachments/assets/dec967c1-1d12-4a77-a5b3-362a388e8798)

Design notes:
- Uses convolution layers to capture spatial patterns.
- LeakyReLU ensures gradient flow even for negative values.
- Outputs a single value indicating real or fake.

### 🧠 Generator Network
The Generator creates images from random noise vectors.
![image](https://github.com/user-attachments/assets/bc4d2dda-63a4-4d72-965e-eb2c754f413f)

Design notes:
- Upsamples noise into images through transposed convolutions.
- Uses ReLU activations to build features progressively.
- Final layer uses Tanh to output images scaled between -1 and 1.

### 📉 Loss Functions
The GAN uses binary cross-entropy loss
![image](https://github.com/user-attachments/assets/127bc67b-0c90-49d1-aea3-c3323acf6305)

- Discriminator tries to classify real images as 1 and fake images as 0.
- Generator tries to fool Discriminator, making fake images be classified as 1.

### 🔁 Training Loop
We train both models iteratively:
![image](https://github.com/user-attachments/assets/d6d7afd4-a8f9-41eb-bb7c-f20b93183cb3)

Training steps:
- Update Discriminator weights to better detect real/fake images.
- Update Generator weights to produce more realistic images.
- Alternate training ensures balance.

### 🧪 Results
After training, the Generator can create realistic handwritten digit images from random noise. Here’s a sample output after final training:
![image](https://github.com/user-attachments/assets/d2d209cd-caa5-4dc4-a234-683c8e25aedb)

#
🚀 Harness the power of deep learning and generative models to unlock creative AI solutions — build smarter, learn faster, and push the boundaries of what’s possible!
