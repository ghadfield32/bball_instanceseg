Basketball Instance Segmentation and Tracking
Project Overview

This project aims to perform instance segmentation and tracking on basketball videos. It leverages a Mask R-CNN model, integrated with ByteTrack for object tracking, to identify and track basketballs and hoops in video footage. The project is structured modularly, with separate scripts for different functionalities like model training, data preprocessing, and video processing.
Getting Started
Prerequisites

    Python 3.6 or higher
    PyTorch 1.10.0 or higher
    torchvision
    CUDA (for GPU acceleration)

Installation

    Clone the GitHub repository:

    bash

git clone https://github.com/ghadfield32/bball_instanceseg

Change to the cloned directory:

bash

cd bball_instanceseg/

Install required Python packages:

    pip install -r requirements.txt

Cloning ByteTrack Repository

    Clone the ByteTrack GitHub repository:

    bash

git clone https://github.com/ifzhang/ByteTrack.git

Install ByteTrack requirements and set up the package:

arduino

    cd ByteTrack
    pip3 install -r requirements.txt
    python3 setup.py develop
    pip3 install cython
    pip3 install 'git+https://github.com/cocodataset/cocoapi.git#subdirectory=PythonAPI'
    pip3 install cython_bbox

Data Preparation

    Change the current working directory to the project directory:

    lua

os.chdir("/content/bball_instanceseg")

Download utility files from torchvision:

scss

    download_files(urls)

Training the Model

    Create a modular directory structure and write utility scripts for various functionalities (e.g., data loading, model utilities, visualization):

    scss

create_directory("going_modular")

Train the model using the train.py script:

css

    python train.py --api_key YOUR_API_KEY \
                    --workspace YOUR_WORKSPACE \
                    --project_name YOUR_PROJECT_NAME \
                    --project_folder_name YOUR_PROJECT_FOLDER_NAME \
                    --version YOUR_DATASET_VERSION \
                    --hidden_layer 256 \
                    --lr 0.005 \
                    --num_epochs 10 \
                    --threshold 0.6 \
                    --video_url "YOUR_VIDEO_URL" \
                    --video_name "YOUR_VIDEO_NAME" \
                    --mode train

Video Processing

    Process a video for object detection and tracking:

    css

    python train.py --mode process_video \
                    --video_url "VIDEO_URL" \
                    --video_name "VIDEO_FILENAME"

Feel free to customize this README by adding more details about your project, such as the dataset used, the model's architecture, and any additional instructions or notes.

Model Architecture

The project utilizes a Mask R-CNN model, a state-of-the-art model for object instance segmentation. Mask R-CNN extends Faster R-CNN by adding a branch for predicting segmentation masks on each Region of Interest (RoI), in parallel with the existing branch for classification and bounding box regression.

Key components of the Mask R-CNN model used in this project:

    Backbone: ResNet-50 with Feature Pyramid Network (FPN) as the backbone network. ResNet-50 is a 50-layer deep convolutional network, and FPN enhances feature extraction by combining low-resolution, semantically strong features with high-resolution, semantically weak features.
    Region Proposal Network (RPN): Generates region proposals for object detection.
    RoI Align: Extracts a small feature map from each RoI.
    Classification and Bounding Box Regression Head: Determines the class of each RoI and refines their bounding boxes.
    Mask Prediction Head: Predicts segmentation masks for each instance in parallel.

Hyperparameters

Hyperparameters are crucial for training neural networks effectively. Key hyperparameters used in your project include:

    Hidden Layer Size for Mask Predictor: Specified as hidden_layer in the train.py script. It's the size of the hidden layer in the Mask R-CNN's mask predictor. In your code, it's set to 256 by default.

    Learning Rate (lr): Controls how much to update the model's weights by at the end of each batch. In your script, it's set to 0.005. A smaller learning rate might make the training more stable, but it could also slow down convergence.

    Number of Epochs (num_epochs): The number of times the entire training dataset is passed forward and backward through the neural network. In your script, it's set to 10. More epochs can lead to a better-trained model but also increase the risk of overfitting.

    Momentum: Used in the SGD optimizer, it helps accelerate gradients vectors in the right direction, leading to faster converging. It's set to 0.9 in your script.

    Weight Decay (weight_decay): Adds a regularization term to the loss to prevent overfitting. It's set to 0.0005 in your script.

    Step Size and Gamma for Learning Rate Scheduler: step_size and gamma are used in the learning rate scheduler (StepLR). step_size is the period for the learning rate decay, and gamma is the multiplicative factor of learning rate decay. These are set to 3 and 0.1 respectively in your script.

    Detection Threshold (threshold): Used during video processing to filter out detections with low confidence scores. It's set to 0.6 in your script.

Note

The choice of these hyperparameters can significantly affect the model's performance. They are often tuned based on the specific characteristics of the dataset and the problem at hand. Experimenting with different values for these parameters might be necessary to achieve optimal results.