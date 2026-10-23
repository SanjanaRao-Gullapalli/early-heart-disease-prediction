# Heart Disease Risk Prediction Using Machine Learning

## Project Overview

This project implements a machine learning-based heart disease risk prediction system using the Cleveland Heart Disease Dataset.

It evaluates multiple machine learning classification algorithms and provides an interactive web application built with Streamlit. Users can enter clinical parameters to receive a model-generated risk estimate, view a risk category, review their entered information, and read general lifestyle and clinical suggestions.

**This project is developed for academic and educational purposes. It is not a medical diagnostic tool.**

## Project Objectives

- Explore and preprocess heart disease data.
- Train and evaluate multiple machine learning classification algorithms.
- Compare models using standard classification metrics.
- Identify the best-performing model according to evaluation accuracy.
- Develop an interactive Streamlit application for heart disease risk estimation.
- Display risk estimates, patient input summaries, and general lifestyle suggestions.

## Technologies Used

- **Programming Language:** Python
- **Data Processing:** Pandas, NumPy
- **Machine Learning:** Scikit-learn
- **Web Application:** Streamlit
- **Model Serialization:** Joblib

## Project Structure

```text
early-heart-disease-prediction/
│
├── a1.py
├── finalmain.py
├── best_logistic_model.pkl
├── processed.cleveland.data
└── README.md
```

### File Descriptions

| File | Description |
|---|---|
| `finalmain.py` | Trains and evaluates multiple machine learning classification models. |
| `a1.py` | Streamlit application for entering clinical parameters and viewing predictions. |
| `best_logistic_model.pkl` | Saved Logistic Regression pipeline loaded by the Streamlit application. |
| `processed.cleveland.data` | Cleveland Heart Disease Dataset used by the evaluation script. |
| `README.md` | Project documentation and setup instructions. |

## Dataset

This project uses the **Cleveland Heart Disease Dataset**, which contains clinical attributes used to study the relationship between patient characteristics and heart disease.

The features include:

- Age
- Sex
- Chest pain type
- Resting blood pressure
- Serum cholesterol
- Fasting blood sugar
- Resting electrocardiographic results
- Maximum heart rate achieved
- Exercise-induced angina
- ST depression (`oldpeak`)
- Slope of the peak exercise ST segment
- Number of major vessels (`ca`)
- Thalassemia-related information (`thal`)

The target variable is converted into a binary classification label: absence or presence of heart disease.

## Machine Learning Models

The evaluation script compares the following classification algorithms:

1. Logistic Regression
2. Support Vector Machine (SVM)
3. Linear Support Vector Machine
4. Random Forest
5. Extra Trees
6. Gradient Boosting
7. AdaBoost
8. K-Nearest Neighbors (KNN)
9. Gaussian Naive Bayes
10. Decision Tree

### Model Evaluation Results

In the current evaluation run, Logistic Regression achieved the highest accuracy among the evaluated models.

| Evaluation Metric | Logistic Regression |
|---|---:|
| Accuracy | 90.67% |
| Precision | 91.18% |
| Recall | 88.57% |
| F1-score | 89.86% |
| ROC-AUC | 94.57% |

These results are from the train-test split used by the evaluation script. Results may vary depending on the dataset, preprocessing, split, and model configuration.

**Model consistency note:** The Streamlit application loads `best_logistic_model.pkl`. The relationship between this saved model and the model evaluated by `finalmain.py` should be verified before treating these evaluation metrics as the deployed application's performance.

## Streamlit Application Features

The application offers the following functionality:

### 1. Clinical Input Form

Users can enter clinical parameters, including age, sex, chest pain type, blood pressure, cholesterol, ECG results, maximum heart rate, and other relevant features.

### 2. Heart Disease Risk Prediction 

The application uses the saved Logistic Regression pipeline to generate a model-based estimate from the entered values.

### 3. Risk Likelihood and Category

The application displays the estimated model probability and a risk category based on its configured decision threshold.

The displayed probability should not be interpreted as a clinically validated probability of having heart disease.

### 4. Patient Summary

The application presents a summary of the values entered by the user.

### 5. Lifestyle and Clinical Suggestions

The results page provides general suggestions, such as seeking professional medical evaluation, engaging in appropriate physical activity, following a heart-healthy diet, and managing stress.

These suggestions are general educational guidance, not individualized medical advice.

### 6. Medical Disclaimer

The application reminds users that its predictions do not replace professional medical diagnosis or treatment.

## Installation and Execution

Follow these steps to run the project locally.

### Prerequisites

- Python installed on your computer
- Internet access to install the required packages

### Step 1: Clone the Repository

Open a terminal and run:

```bash
git clone https://github.com/SanjanaRao-Gullapalli/early-heart-disease-prediction.git
cd early-heart-disease-prediction
```

### Step 2: Install Dependencies

Install the Python packages required by the project:

```bash
python -m pip install numpy pandas scikit-learn streamlit joblib
```

### Step 3: Run Model Evaluation

Run the evaluation script:

```bash
python finalmain.py
```

The script loads the Cleveland dataset, trains and evaluates the configured classification models, and displays their performance metrics.

### Step 4: Launch the Streamlit Application

Run:

```bash
streamlit run a1.py
```

Streamlit should open the application in your default web browser. If it does not, follow the local URL displayed in the terminal.

**Important:** The application requires `best_logistic_model.pkl` to be present in the project directory. Running `finalmain.py` does not currently regenerate this saved model.

## Limitations and Medical Disclaimer

This project is intended for academic, educational, and demonstration purposes only.

- Predictions depend on the saved model and the quality and compatibility of the input data.
- The reported evaluation metrics do not establish clinical effectiveness.
- The model's output probability is not necessarily calibrated to represent an individual's actual probability of having heart disease.
- The system must not be used as a substitute for professional medical advice, diagnosis, or treatment.

Always consult a qualified healthcare professional for medical evaluation and decisions.

## Author

**Sanjana Rao Gullapalli**

GitHub: [SanjanaRao-Gullapalli](https://github.com/SanjanaRao-Gullapalli)
