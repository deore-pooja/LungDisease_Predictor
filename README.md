# LungDisease_Predictor

# Lung Image Classification

# Project Overview
This project aims to develop a machine learning model for classifying lung images into different categories, such as lung adenocarcinoma (lung_aca), normal lung (lung_n), and lung squamous cell carcinoma (lung_scc). The project is implemented using Python and various machine learning libraries, including NumPy, Pandas, Matplotlib, scikit-learn, OpenCV, and TensorFlow with Keras.

# Key Features

Lung cancer is one of the leading causes of cancer-related deaths worldwide. Early detection can significantly improve the prognosis and survival rates of patients. This project leverages convolutional neural networks (CNNs) to analyze lung images and predict the presence of lung cancer.

1) Data Preprocessing: Tools and scripts for preparing lung image datasets for model training.

2) Model Training: Implementation of CNNs using Keras for training on lung image datasets.

3) Evaluation: Methods to evaluate model performance using various metrics.

4) Visualization: Functions to visualize the model architecture and results.

# Installations

git clone https://github.com/deore-pooja/LungDisease_Predictor.git

cd LungDisease_predictor

pip install -r requirements.txt

# Dataset

The dataset used in this project is the "lung_image_sets.zip" file, which contains the lung images organized into the following categories:

https://drive.google.com/file/d/1W2WczHGw1Ng8bn0BfdIEhB2R3SHepYe_/view?usp=drive_link

lung_aca: Lung adenocarcinoma images

lung_n: Normal lung images

lung_scc: Lung squamous cell carcinoma images

For Model Testing : https://github.com/deore-pooja/LungDisease_Predictor/tree/82c9c2d17e20bba144d234945edeaee47397bb2a/For%20Model%20Testing

is for model testing. with the help of these images you can check that your model is actually working or not. You can also test with different images that are not in the folder repository.

For Verification : https://github.com/deore-pooja/LungDisease_Predictor/tree/82c9c2d17e20bba144d234945edeaee47397bb2a/For%20Verification

is for your verification. You can check if the prediction of disease is correct or not. These folders are the original folders of the (For Model Testing) folder and both have the same data. but they have different names.


# Project Structure
The project is organized as a Jupyter Notebook, with the following steps:

1) Importing Libraries: The necessary Python libraries are imported for the project.
   
2) Extracting the Data Set: The "lung_image_sets.zip" file is extracted, and the available classes are listed.
   
3) Data Exploration: The dataset is explored, and sample images are displayed.
   
4) Data Preprocessing: The images are preprocessed, including resizing, normalization, and splitting the data into training and validation sets.
   
5) Model Definition: A machine learning model, such as a convolutional neural network (CNN), is defined for the lung image classification task.
   
6) Model Training: The model is trained on the preprocessed data, and the training process is monitored.
   
7) Model Evaluation: The trained model's performance is evaluated using various metrics, such as accuracy, precision, recall, and F1-score.
   
8) Model Deployment: The trained model is saved, and a discussion is provided on how to deploy the model for real-world use.

# UI Demo

![Screenshot 2024-07-09 144604](https://github.com/deore-pooja/LungDisease_Predictor/assets/158804349/e0dc2b60-f7a2-444f-8e4d-413bb011daf1)

![Screenshot 2024-07-09 144621](https://github.com/deore-pooja/LungDisease_Predictor/assets/158804349/984d4376-023f-4992-a3eb-353fd6e8ef52)

![Screenshot 2024-07-09 144709](https://github.com/deore-pooja/LungDisease_Predictor/assets/158804349/47e831c0-f446-4f90-995b-0bc31f37beac)

## Video

https://github.com/deore-pooja/LungDisease_Predictor/assets/158804349/abec4cda-136c-4c16-942a-69908beae08d

# Requirements
To run this project, you will need the following:

Python 3.x

Jupyter Notebook

The following Python libraries:

For UI : Gradio

NumPy

Pandas

Matplotlib

scikit-learn

OpenCV

TensorFlow with Keras

# Project Flow

1) Data Preparation: Ensure the lung image datasets are placed in the data/ directory.

2) Training: Use the provided notebooks and scripts to train the model.

3) Evaluation: Run evaluation scripts to assess the model's performance.

4) Prediction: Use the trained model to predict lung cancer on new sample images.

# Usage

1) Clone the repository or copy the Jupyter Notebook file to your local machine.

2) Install the required Python libraries.

3) Open the Jupyter Notebook file and run the cells to execute the project.

# Contribution
If you have any suggestions, improvements, or find any issues, please feel free to create a new issue or submit a pull request.
