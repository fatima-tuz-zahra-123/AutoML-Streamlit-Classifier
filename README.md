# AutoML Streamlit Classifier

## Overview
This is an automated machine learning (AutoML) application built with Streamlit. It allows users to upload a dataset, perform exploratory data analysis (EDA), detect data quality issues, preprocess the data, train multiple classification models, evaluate their performance, and generate a final report.

## Features
1.  **Upload Data**: Support for CSV file uploads. Includes a complex sample dataset generator for testing.
2.  **EDA**: 
    *   Descriptive statistics.
    *   Correlation heatmap.
    *   Distribution plots (Histograms, Boxplots).
    *   Pairplots.
    *   **New**: Interactive 3D Scatter Plots.
3.  **Issue Detection**:
    *   Missing values analysis.
    *   Duplicate rows detection.
    *   Outlier detection using IQR method.
4.  **Preprocessing**:
    *   Missing value imputation (Mean, Median, Mode, Drop).
    *   Categorical encoding (Label Encoding, One-Hot Encoding).
    *   Feature scaling (StandardScaler, MinMaxScaler).
    *   **New**: Ability to convert numeric columns to categorical.
5.  **Model Training**:
    *   Train multiple models: Logistic Regression, Decision Tree, Random Forest, SVM, KNN, Naive Bayes.
    *   Real-time training status updates.
6.  **Evaluation**:
    *   Performance leaderboard (Accuracy, F1 Score, Precision, Recall).
    *   Confusion Matrix.
    *   **New**: Feature Importance visualization.
    *   **New**: Prediction Playground for "What-If" analysis.
    *   Model export (.pkl).
7.  **Report**: Generate a PDF summary of the experiment.

## Installation

1.  Clone the repository.
2.  Create a virtual environment:
    ```bash
    python3 -m venv venv
    source venv/bin/activate
    ```
3.  Install dependencies:
    ```bash
    pip install -r app/requirements.txt
    ```

## Usage

Run the Streamlit app:
```bash
streamlit run app/app.py
```

## Testing with Complex Data
To test the robustness of the application, you can use the built-in "Load Sample Test Data (Complex)" button on the Upload page. This dataset includes:
*   Missing values (NaNs).
*   Outliers.
*   Duplicate rows.
*   Mixed data types.

## Project Structure
*   `app/app.py`: Main application entry point.
*   `app/eda.py`: Exploratory Data Analysis module.
*   `app/issue_detection.py`: Data quality check module.
*   `app/preprocessing.py`: Data cleaning and transformation module.
*   `app/modeling.py`: Model training module.
*   `app/evaluation.py`: Model evaluation and visualization module.
*   `app/report.py`: PDF report generation module.

## Recent Updates (Asma Branch)
- **Navigation Fix**: Resolved `StreamlitAPIException` by implementing callback functions for sidebar navigation buttons.
- **AI Model Upgrade**: Updated Gemini API integration to use `gemini-2.0-flash` model.
- **Enhanced Reporting**: 
    - Rewrote `generate_report` to provide comprehensive, explainable AI insights.
    - Added reasoning capabilities to AI analysis (explaining *why* a model performed best).
    - Integrated AI insights into the downloadable PDF report.
- **UI/UX Improvements**:
    - Added "Thinking..." spinners for visual feedback during AI operations.
    - Refined AI persona to be friendly, professional, and emoji-free.
    - Removed emojis from navigation buttons for a cleaner look.
