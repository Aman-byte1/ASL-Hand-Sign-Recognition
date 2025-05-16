# ASL Hand Sign Recognition

This project is a web application built with Streamlit to recognize American Sign Language (ASL) hand signs from uploaded images using a deep learning model implemented in Keras.

## Table of Contents

* [Introduction](https://gemini.google.com/app/24bc5777f3947ede#introduction "null")
* [Features](https://gemini.google.com/app/24bc5777f3947ede#features "null")
* [Requirements](https://gemini.google.com/app/24bc5777f3947ede#requirements "null")
* [Setup](https://gemini.google.com/app/24bc5777f3947ede#setup "null")
* [Usage](https://gemini.google.com/app/24bc5777f3947ede#usage "null")
* [Model](https://gemini.google.com/app/24bc5777f3947ede#model "null")
* [Model Training](https://gemini.google.com/app/24bc5777f3947ede#model-training "null")
* [Supported Signs](https://gemini.google.com/app/24bc5777f3947ede#supported-signs "null")

## Introduction

This application provides a user-friendly interface that allows users to upload an image containing an American Sign Language (ASL) hand sign. The application then processes the image and provides a prediction of the corresponding English letter. It utilizes a deep learning model built with Keras for the recognition task, OpenCV for image processing, and Streamlit for creating the interactive web interface.

## Features

* **Interactive Web Interface:** Easy-to-use interface built with Streamlit.
* **Image Upload:** Supports uploading hand sign images in JPG, JPEG, and PNG formats.
* **Deep Learning Recognition:** Employs a Keras-based deep learning model for accurate hand sign classification.
* **Visual Output:** Displays the uploaded image with the predicted ASL sign clearly annotated.

## Requirements

To set up and run this project, ensure you have Python installed (version 3.6 or higher is recommended). You will also need the following Python libraries:

* `streamlit`: For building the web application interface.
* `opencv-python` (or `opencv-contrib-python`): For image loading and processing.
* `numpy`: For numerical operations, especially with image data.
* `tensorflow`: The backend for Keras, used for building and running the deep learning model.
* `Pillow` (PIL): Used by Streamlit for handling image files.
* `scikit-learn`: Used in the training script for splitting the dataset.

You can install these required libraries using pip:

```
pip install streamlit opencv-python numpy tensorflow Pillow scikit-learn

```

**Note:** If you encounter issues with `opencv-python`, try installing `opencv-contrib-python` instead: `pip install opencv-contrib-python`.

## Setup

1. **Clone the Repository (Optional):** If this project is in a Git repository, clone it to your local machine:
   ```
   git clone [YOUR_REPOSITORY_URL]
   cd [YOUR_REPOSITORY_DIRECTORY]

   ```
2. **Save Application Code:** Save the Streamlit application Python code (the code for the web interface) as a file, for example, `app.py`.
3. **Save Training Code:** Save the model training Python code (the script you provided earlier) as `main.py`.
4. **Obtain the Dataset:** Download the ASL alphabet image dataset. You will need to update the `dataset_root` variable within the `main.py` script to point to the correct root directory where your dataset is stored on your system.
5. **Train the Model:** Run the `main.py` script from your terminal. This script will load the dataset, preprocess the images, train the Keras model, and save the trained model to an HDF5 file named `asl_model.h5`.
   ```
   python main.py

   ```
6. **Verify Model Path:** Ensure that the `MODEL_PATH` variable in your Streamlit application code (`app.py`) is correctly set to the location of the generated `asl_model.h5` file.

## Usage

1. Open your terminal or command prompt.
2. Navigate to the directory where you have saved `app.py` and the trained `asl_model.h5` file.
3. Run the Streamlit application using the following command:
   ```
   streamlit run app.py

   ```
4. This command will start the Streamlit server and open the application in your default web browser.
5. In the web application, use the file uploader labeled "Choose an image" to select a JPG, JPEG, or PNG image containing an ASL hand sign.
6. The application will process the image and display it along with the predicted ASL sign.

Screenshot of the main application page.

## Model

The core of this application is a deep learning model implemented using the Keras API (typically running on a TensorFlow backend). The model is loaded from the `asl_model.h5` file. It is a Convolutional Neural Network (CNN) designed to classify images of ASL hand signs. Based on the `preprocess_frame` function and the model architecture in `main.py`, the model expects input images to be resized to 100x100 pixels with 3 color channels (RGB). The output layer provides probabilities for each supported ASL sign.

## Model Training

The deep learning model used in this application is trained separately using the `main.py` script. This script performs the following steps:

1. Loads images and their corresponding labels from the specified dataset directory (`dataset_root`).
2. Resizes and normalizes the image data.
3. Converts the string labels into a numerical, one-hot encoded format suitable for training.
4. Splits the dataset into training and testing sets.
5. Defines and compiles a CNN model architecture.
6. Trains the CNN model on the training data.
7. Evaluates the trained model on the testing data to report accuracy.
8. Saves the trained model to `asl_model.h5`.

The training dataset is expected to contain images organized by class (letter), with approximately 100 images available for each supported ASL letter.

To train the model, execute the `main.py` script from your terminal:

```
python main.py

```

Ensure the `dataset_root` variable in `main.py` is correctly set to your dataset's location before running the script.

## Supported Signs

The trained model is capable of recognizing the following American Sign Language hand signs, which correspond to the English alphabet letters:

A, B, C, D, E, F, G, H, I, J, K, L, M, N, O, P, Q, R, S, T, U, V, W, X, Y, Z
