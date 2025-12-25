# AutoML Streamlit Classifier Documentation

## Overview

The **AutoML Streamlit Classifier** is a comprehensive, user-friendly web application designed to automate the Machine Learning pipeline. Built with Streamlit and powered by Llama Groq, it guides users from raw data upload to model deployment, offering intelligent insights and robust visualization tools along the way.

## Key Features

### 1. User Interface & Navigation

- **Progress Tracking**: A sidebar progress bar visualizes the user's journey through the 7-step pipeline.
- **Modern Design**: A custom "Light Blue" theme with polished CSS styling for headers, buttons, and progress bars.
- **AI Tutor Chat**: An always-available AI assistant in the sidebar that answers questions about the dataset, models, or ML concepts in real-time.

### 2. Data Ingestion

- **Upload Data**: Supports CSV file uploads.
- **Data Preview**: Displays the raw dataset for initial inspection.

### 3. Exploratory Data Analysis (EDA)

- **Descriptive Statistics**: Automatically calculates mean, median, standard deviation, min/max for numerical columns.
- **Correlation Heatmap**: Visualizes relationships between numerical variables to identify collinearity.
- **Distribution Plots**:
  - **Histograms & Boxplots** for numerical data.
  - **Count Plots** for categorical data.
- **Pairplot**: Generates a scatter matrix to visualize interactions between multiple variables.
- **3D Scatter Plot**: Interactive 3D visualization for analyzing three numerical variables simultaneously.

### 4. Automated Issue Detection

- **Missing Values Analysis**: Identifies columns with null values and calculates the percentage of missing data.
- **Duplicate Detection**: Scans for and displays duplicate rows.
- **Outlier Detection**: Uses the Interquartile Range (IQR) method to flag potential statistical outliers in numerical columns.

### 5. Intelligent Preprocessing

- **AI Suggestions**: The AI analyzes the dataset and recommends specific cleaning actions (e.g., "Impute Age with Mean").
- **Missing Value Imputation**:
  - **Numeric**: Mean, Median.
  - **Categorical**: Mode.
  - **General**: Drop Rows.
- **Action Logging**: All preprocessing steps are logged and fed into the final report for reproducibility.
- **Schema View**: Allows users to inspect column data types before and after transformation.

### 6. Model Training

- **Target Selection**: Users can easily select the target variable (Y) and features (X).
- **Train/Test Split**: Configurable slider to set the test set size (10% - 50%).
- **Multi-Model Training**: Automatically trains and compares six standard classifiers:
  - Logistic Regression
  - Decision Tree
  - Random Forest
  - Support Vector Machine (SVM)
  - K-Nearest Neighbors (KNN)
  - Naive Bayes
- **Real-time Progress**: Displays a progress bar and status updates during training.

### 7. Evaluation & Interpretation

- **Performance Leaderboard**: A ranked table of models based on F1 Score, Accuracy, Precision, and Recall. The best model is highlighted.
- **Confusion Matrix**: Visualizes the performance of the champion model.
- **Feature Importance**: Bar charts showing which features contributed most to the model's predictions.
- **Prediction Playground**: An interactive form allowing users to input custom values and get real-time predictions from the best model.
- **Model Export**: One-click download of the trained model as a `.pkl` file.

### 8. Comprehensive Reporting

- **AI Executive Summary**: Generates a professional, narrative summary of the project, explaining data transformations, model choices, and results.
- **PDF Generation**: Creates a downloadable PDF report containing:
  - Project Summary
  - Data Transformation Metrics
  - Model Leaderboard
  - Conclusion & Recommendations
- **Footer**: Custom footer "Made in NUST - SEECS" on generated reports.

## AI Integration

- **Engine**: Llama Groq.
- **Capabilities**:
  - Context-aware chat (knows the dataset shape, missing values, and best model).
  - Generates code-free explanations for complex ML concepts.
  - Writes the final project report based on the user's specific actions.

## Technical Stack

- **Frontend**: Streamlit
- **Data Manipulation**: Pandas, NumPy
- **Machine Learning**: Scikit-learn
- **Visualization**: Matplotlib, Seaborn, Plotly
- **AI/LLM**: Google Generative AI (Gemini)
- **Reporting**: FPDF
