# Image Classifier Project

## Overview
This project implements a deep learning-based image classifier capable of identifying different flower species. The classifier is built using a **pre-trained neural network (VGG16)** from `torchvision.models`. The project consists of two main components:

1. **Development Notebook (Jupyter Notebook):** Used for data preprocessing, model training, evaluation, and saving/loading model checkpoints.
2. **Command-Line Application:** Provides scripts (`train.py` & `predict.py`) to train a model and classify images using command-line arguments.

---


## Part 1 - Development Notebook (Jupyter Notebook)
This section details the model development process inside a Jupyter Notebook.

### 1. Package Imports
- The notebook imports all necessary libraries such as `torch`, `torchvision`, `matplotlib`, `PIL`, and `numpy`.
- Random seeds are set for reproducibility.

### 2. Data Preprocessing & Augmentation
- **Data Augmentation:** `torchvision.transforms` is used to apply random scaling, rotations, mirroring, and cropping to the training dataset.
- **Data Normalization:** Images are normalized to match the input format expected by pre-trained models.
- **Data Batching:** `torchvision.datasets` and `torch.utils.data.DataLoader` are used for efficient batching.

### 3. Loading the Data
- The dataset is split into **training, validation, and testing sets** using `torchvision.datasets.ImageFolder`.
- Data is loaded using `DataLoader` for faster data processing and shuffling.

### 4. Pretrained Neural Network
- A **pretrained VGG16 model** is loaded from `torchvision.models`.
- **Feature extraction layers are frozen**, so only the classifier is trained.

### 5. Feedforward Classifier
- A new classifier is attached to the pre-trained network.
- The classifier includes **fully connected layers, ReLU activation, and dropout** to prevent overfitting.

### 6. Training the Model
- **Backpropagation & optimization:**
  - Uses **cross-entropy loss** and **Adam optimizer** for training.
  - Only the classifier’s parameters are updated; feature extraction layers remain unchanged.

### 7. Model Evaluation
- The trained model is tested on a **separate dataset** to measure accuracy.
- **Validation loss and accuracy** are printed after each epoch during training.

### 8. Saving & Loading Checkpoints
- The trained model is **saved as a checkpoint**, including:
  - Model architecture
  - Trained weights
  - Class-to-index dictionary
  - Hyperparameters
- The `load_checkpoint()` function successfully reloads the model.

### 9. Image Processing
- The `process_image()` function converts a **PIL image** into a **tensor** for model inference.

### 10. Class Prediction
- The `predict()` function takes an **image path** and a trained model checkpoint, then returns:
  - **Top K most probable classes**
  - **Associated probabilities**

### 11. Sanity Checking with Matplotlib
- The **Jupyter Notebook generates a Matplotlib figure** displaying:
  - The **input image**
  - The **top 5 predicted classes**
  - Their **actual flower names** using a JSON category mapping file.

---

## Part 2 - Command-Line Application
This section provides **CLI scripts** for model training and inference.

### 1. `train.py` - Training the Model
This script trains a deep learning model and saves the trained model as a checkpoint.

#### Features
- Users can specify the **dataset directory** (e.g., `python train.py data_dir`).
- **Training loss, validation loss, and validation accuracy** are displayed during training.
- **Multiple architectures** supported (`vgg16`, `resnet18`).
- **Customizable hyperparameters**:
  - Learning rate (`--learning_rate 0.01`)
  - Hidden units (`--hidden_units 512`)
  - Epochs (`--epochs 20`)
- **GPU support** (`--gpu` enables CUDA acceleration).

#### Example Usage
```bash
python train.py flowers --arch vgg16 --learning_rate 0.01 --hidden_units 512 --epochs 10 --gpu
```
This trains the model on the **flowers dataset** and saves a checkpoint.

---

### 2. `predict.py` - Classifying Images
This script loads a trained model and classifies an image.

#### Features
- Reads an image and a checkpoint, then predicts the **most likely class**.
- Users can specify **Top-K predictions** (`--top_k`).
- Loads a **JSON file** to map class numbers to real flower names.
- Supports **GPU inference** for faster predictions (`--gpu`).

#### Example Usage
```bash
python predict.py /path/to/image checkpoint.pth --top_k 3 --category_names cat_to_name.json --gpu
```
This command:
- Loads `checkpoint.pth`
- Predicts the **top 3 most probable classes**
- Displays **real flower names** (from `cat_to_name.json`)


This README serves as a complete guide to the project.

---

## How to Use

### Clone the Repository
```bash
git clone https://github.com/your-username/image-classifier.git
cd image-classifier
```

### Train the Model
```bash
python train.py flowers --arch vgg16 --learning_rate 0.01 --hidden_units 512 --epochs 10 --gpu
```

### Predict an Image
```bash
python predict.py /path/to/image checkpoint.pth --top_k 3 --category_names cat_to_name.json --gpu
```

This README provides all necessary details for running the project.

