## CNN Image Recognition

The Problem I Chose

Image recognition is one of the most widely studied problems in artificial intelligence.
My goal was to build a Convolutional Neural Network (CNN) capable of learning visual patterns directly from image data and classifying objects into categories.

To do this, I trained models on the CIFAR-10 dataset, which contains 60,000 colour images in 10 classes (e.g. airplane, cat, ship, truck, etc.).
The project demonstrates how a CNN can learn hierarchical visual features — edges, textures, shapes — from raw pixels, without manual feature engineering.

## What I Did

- Experimented with two deep-learning frameworks:

  - PyTorch: implemented initial model and training loop manually.

  - Keras/TensorFlow: built a modular, callback-driven architecture for the final version.

- Data preprocessing and augmentation:

  - Normalised pixel values to [0,1].

  - Applied random flips, shifts, and rotations to improve generalisation.

- Built a CNN architecture from scratch:

  - Multiple convolution–batchnorm–pooling blocks.

  - Dense layers with dropout for regularisation.

  - Softmax output for multi-class classification.

- Added training controls:

  - Early stopping (monitors validation accuracy).

  - Learning-rate reduction on plateau.

  - Model checkpointing (saves best epoch).

- Visualised training behaviour:

  - Accuracy and loss curves for train vs validation sets.

  - Confusion matrix for detailed class-wise performance.

- Tested real-world inference:

  - Predicted the label of a custom input image (cat.jpg) using the trained model.

## Model Architecture

The final CNN model included:

  - 3 convolutional blocks (Conv → BatchNorm → MaxPooling → Dropout)

  - A dense block with 256 neurons and dropout (0.5)

  - A softmax output layer for 10 classes

This balance of depth, normalization, and dropout achieved a good mix of expressive power and regularisation.

## The Outcome

- The model steadily improved over training, peaking around 78–80 % validation accuracy after 18–20 epochs.

- Early stopping restored the best weights to avoid overfitting beyond that point.

- Confusion matrices showed well-balanced performance across most classes, with slightly lower precision for visually similar categories (cat/dog, deer/horse).

- On custom test images, predictions were consistent and confident.

This accuracy is strong for a CNN trained from scratch on CIFAR-10 without transfer learning, matching expectations for mid-depth architectures.

🛠 Tech Stack

Python · TensorFlow / Keras · PyTorch · NumPy · Matplotlib · Seaborn · scikit-learn
