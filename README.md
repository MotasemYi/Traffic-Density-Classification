# Traffic-Density-Classification
TRAFFIC DENSITY CLASSIFICATION

This study aims to automatically analyze urban traffic congestion using image processing and deep learning techniques. It provides a data-driven solution to one of the biggest challenges of modern smart cities: transportation management.

PROBLEM DEFINITION & MOTIVATION
WHY THIS PROBLEM?

Accurate detection of traffic density is critical for intelligent transportation systems and city planning. Traditional sensors are expensive and require maintenance. Camera-based systems, however, offer a low-cost solution by utilizing existing infrastructure.

IMPORTANCE OF THE PROJECT

This study tests the performance of machine learning in complex visual scenes using real-world data. It aims to overcome the limitations of manual observation by providing continuous, objective 24/7 traffic density analysis.

NATURE OF THE PROBLEM & TARGET OUTPUTS

Technically, this project is a Multi-Class Classification problem. Each road image is assigned to one of five different traffic density levels depending on complexity and number of vehicles.

INPUT

Road images captured under different angles and lighting conditions.

PROCESS

Feature extraction and training using ResNet18 and Random Forest algorithms.

OUTPUT

A five-class density label:
Empty, Low, Medium, High, Traffic Jam

MACHINE LEARNING METHODS USED

Two different methods were selected, covering both traditional and modern deep learning approaches.

CNN (RESNET18) – Reasons for Selection

CNN is one of the most successful deep learning methods for image processing and classification.

CNNs automatically learn spatial features such as edges, textures, and shapes.

Pre-trained models like ResNet18 allow transfer learning, enabling higher accuracy with less data.

Data augmentation techniques were used to prevent overfitting.

CNNs are widely considered the most effective approach for image classification in the literature.

RANDOM FOREST – Reasons for Selection

Random Forest is a strong and stable classical ML algorithm, included for comparison.

Effective on small- to medium-scale datasets.

Fast training and low computational cost.

Naturally robust against overfitting.

Provides a simpler structure, allowing a comparison between deep learning and traditional ML.

COMPARATIVE ANALYSIS OF METHODS
Method	Advantages	Disadvantages
CNN	Automatically learns spatial features; high accuracy	Long training time; requires GPU power
Random Forest	Fast training; interpretable; low hardware requirements	Cannot fully capture pixel spatial relationships; limited performance

CNN models preserve neighbor relations between pixels via convolution layers, while Random Forest flattens images and loses this information.

CNN IMPLEMENTATION

Implemented using PyTorch and Torchvision.

Based on the ResNet18 architecture using pre-trained ImageNet weights (transfer learning).

Input images resized to 224×224, normalized, and augmented (rotation, horizontal flip).

Loss function: CrossEntropyLoss

Optimizer: Adam

Best validation model saved as best_cnn_model.pth

Evaluation metrics: confusion matrix, precision, recall, weighted F1-score.

RANDOM FOREST IMPLEMENTATION

Implemented using scikit-learn.

Images resized to 64×64 and flattened into 1D vectors.

Features scaled using StandardScaler.

Model trained with 200 trees (n_estimators = 200).

Evaluation metrics identical to the CNN model.

Model and scaler saved as rf_model.joblib and rf_scaler.joblib.

DATASET & CLASS STRUCTURE

The dataset consists of carefully labeled images representing real-world traffic scenarios.

TRAINING – Model learning

VALIDATION – Hyperparameter tuning

TEST – Generalization performance

TOTAL DATA – All images and labels

Dataset Description

The dataset was created for traffic density classification. Images were collected from open-source internet platforms and selected to represent different traffic states.

The dataset contains 5 classes:
Empty, Low, Medium, High, Traffic Jam

Data split into: training, validation, testing.

DATA PREPROCESSING & AUGMENTATION
CNN Pipeline

Resize to 224×224

Normalize using ImageNet mean & standard deviation

Augmentation: random horizontal flip, rotation

Random Forest Pipeline

Resize to 64×64

Flatten to a 1D vector

Normalize using StandardScaler

TRAINING PARAMETERS & METRICS
Metrics:

Accuracy: Overall correct predictions

F1-Score: Balance between precision and recall

Confusion Matrix: Shows misclassifications

Hyperparameters

CNN:

Learning rate: 0.001

Batch size: 16

Optimizer: Adam

Epochs: 10

Random Forest:

n_estimators: 200

random_state: 42

max_depth: unlimited

PERFORMANCE COMPARISON

Results show that the deep learning model significantly outperforms the classical ML model in processing complex visual data.

Accuracy (%) | Training Time (Minutes)
The CNN achieved approximately 92% accuracy, clearly outperforming the Random Forest model.
