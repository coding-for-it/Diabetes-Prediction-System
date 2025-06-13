# Diabetes Prediction System

This project implements a machine learning-based **Diabetes Prediction System** using the PIMA Indian Diabetes dataset. It explores various classification models to predict whether a patient is likely to have diabetes based on diagnostic measurements.

## 📊 Dataset

The dataset includes the following health-related features:
- Pregnancies
- Glucose
- Blood Pressure
- Skin Thickness
- Insulin
- BMI
- Diabetes Pedigree Function
- Age
- Outcome (Target: 1 = diabetic, 0 = non-diabetic)

## 🧰 Libraries Used

- `pandas` – for data manipulation
- `numpy` – for numerical operations
- `matplotlib`, `seaborn` – for data visualization
- `scikit-learn` – for data preprocessing, model training, and evaluation

## 🔹 Features

✅ **Comprehensive Exploratory Data Analysis (EDA)**  
✅ **Clean and Preprocessed Data** (handled missing values, duplicates, and scaling)  
✅ **Model Evaluation:** Logistic Regression, Decision Tree, and Random Forest  
✅ **Performance Metrics:** Accuracy, Classification Report, and Confusion Matrix  
✅ **Visualizations:** Distribution, Pairplot, Heatmap of correlations, and Model Evaluation charts  

## 🔍 Project Workflow

### 1. Data Cleaning
- Zeros in certain health-related fields are replaced with median values to handle invalid entries.

### 2. Exploratory Data Analysis (EDA)
- Visualizations such as heatmaps and class distribution charts help understand relationships and feature importance.

### 3. Feature Scaling
- StandardScaler is used to normalize the feature set for improved model performance.

### 4. Model Training
Three different models are trained:
- Logistic Regression
- Decision Tree Classifier
- Random Forest Classifier

### 5. Model Evaluation
- Evaluation is done using:
  - Accuracy Score
  - Confusion Matrix
  - Classification Report
 

### Tech Stack

- **Python 3**
- **Pandas**, **Numpy**
- **Scikit-learn**
- **Matplotlib**, **Seaborn**
- **Jupyter Notebook**

## 🧪 How to Run

1. Clone the repository:
   ```bash
   git clone https://github.com/coding-for-it/Diabetes-Prediction-System.git
   cd Diabetes-Prediction-System

