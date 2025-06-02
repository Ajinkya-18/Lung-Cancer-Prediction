# Lung Cancer Prediction using Machine Learning

This project aims to build a ternary classification model to predict the level of susceptibility of lung cancer in patients using clinical and behavioral features. The objective is to support early diagnosis and assist healthcare professionals through data-driven insights.

## Dataset

The dataset used in this project is publicly available on Kaggle:  
https://www.kaggle.com/datasets/thedevastator/cancer-patients-and-air-pollution-a-new-link

The dataset has been included in this repository for convenience. All credit for the dataset goes to the original contributors on Kaggle.

## ML Models Implemented

- Logistic Regression
- Support Vector Machine (SVM)
- Random Forest
- K-Nearest Neighbors (KNN)
- Decision Trees

Evaluation includes accuracy, precision, recall, F1-score, and confusion matrix.

## Tech Stack

- Python 3.10
- Jupyter Notebook
- scikit-learn
- pandas, numpy, matplotlib, seaborn

## Project Structure

Lung-Cancer-Prediction/
├── data/
│   └── lung_cancer.csv
├── models/
│   └── pickle files of trained models
├── notebooks/
│   └── lung_cancer_prediction.ipynb
├── reports/
│   ├── classification_report.png
│   └── confusion_matrix.png
├── src/
│   └── empty
├── environment.yml
├── requirements.txt
├── README.md
└── LICENSE


## 🛠️ Environment Setup

You can recreate the environment using either of the following methods:

### Using `environment.yml`
conda env create -f environment.yml
conda activate lung-cancer-prediction

### Using requirements.txt
python -m venv venv

### Activate it:

Windows:
.\venv\Scripts\activate

Linux/macOS:
source venv/bin/activate

### Install the dependencies:
pip install -r requirements.txt
