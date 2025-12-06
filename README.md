# AutoML Streamlit Classifier

**An intelligent, end-to-end Machine Learning pipeline powered by Streamlit and Google Gemini AI.**

This application automates the process of building a Machine Learning classifier. It guides users through every step of the data science lifecycle—from raw data upload to model deployment—making ML accessible to everyone.

---

## Key Features

### 1. Data Ingestion

- **Easy Upload**: Drag and drop your CSV files directly into the app.
- **Instant Preview**: View your raw data immediately to understand its structure.

### 2. Issue Detection

- **Automated Scanning**: The app automatically scans your dataset for common quality issues.
- **Detailed Insights**: Detects missing values, duplicate rows, and statistical outliers (using IQR).

### 3. Intelligent Preprocessing

- **AI-Powered Suggestions**: Get recommendations on how to clean your data.
- **Flexible Cleaning**:
  - **Impute Missing Values**: Fill gaps with Mean, Median, or Mode.
  - **Encode Categoricals**: Convert text labels to numbers automatically.
  - **Scale Features**: Normalize numerical data for better model performance.
- **Schema View**: Track how your data types change during processing.

### 4. Exploratory Data Analysis (EDA)

- **Interactive Visualizations**:
  - **Histograms & Boxplots**: Understand data distribution.
  - **Correlation Heatmaps**: Spot relationships between variables.
  - **3D Scatter Plots**: Explore complex interactions in 3D space.
- **Descriptive Stats**: Get instant summary statistics (Mean, Std, Min, Max).

### 5. Model Training

- **Multi-Model Comparison**: Automatically trains and evaluates 6 powerful classifiers:
  - Logistic Regression
  - Decision Tree
  - Random Forest
  - Support Vector Machine (SVM)
  - K-Nearest Neighbors (KNN)
  - Naive Bayes
- **Customizable Split**: Adjust the Train/Test split ratio (10-50%) with a simple slider.

### 6. Evaluation & Interpretation

- **Leaderboard**: A ranked table of models based on Accuracy, F1 Score, Precision, and Recall.
- **Visual Metrics**:
  - **Confusion Matrix**: See exactly where the model gets confused.
  - **Feature Importance**: Discover which variables drive predictions.
- **Prediction Playground**: Test the best model with your own custom inputs in real-time.

### 7. Comprehensive Reporting

- **PDF Report Generation**: Download a professional project report containing:
  - Executive Summary (written by AI).
  - Data Transformation Logs.
  - Model Performance Metrics.
  - Visualizations (Feature Importance, Confusion Matrix).
- **AI Insights**: The integrated Gemini AI explains your results in plain English.

---

## AI Assistant (Gemini 2.0)

The app features a built-in **AI Tutor** in the sidebar.

- **Context-Aware**: It knows your dataset and current progress.
- **Educational**: Ask it to explain "What is F1 Score?" or "Why did Random Forest win?"
- **Interactive**: The robot mascot wakes up when you chat!

---

## Installation & Setup

### Prerequisites

- Python 3.8+
- A Google Gemini API Key (configured in `app/app.py` or via environment variables)

### Steps

1. **Clone the Repository**

   ```bash
   git clone https://github.com/fatima-tuz-zahra-123/AutoML-Streamlit-Classifier.git
   cd AutoML-Streamlit-Classifier
   ```

2. **Create a Virtual Environment**

   ```bash
   python -m venv venv
   source venv/bin/activate  # On Windows: venv\Scripts\activate
   ```

3. **Install Dependencies**

   ```bash
   pip install -r app/requirements.txt
   ```

4. **Run the Application**
   ```bash
   streamlit run app/app.py
   ```

---

## Project Structure

```
AutoML-Streamlit-Classifier/
├── app/
│   ├── app.py                 # Main application entry point
│   ├── ai_assistant.py        # Gemini AI integration logic
│   ├── create_test_data.py    # Utility to generate dummy data
│   ├── eda.py                 # Exploratory Data Analysis module
│   ├── evaluation.py          # Model evaluation metrics & plots
│   ├── issue_detection.py     # Data quality checks
│   ├── modeling.py            # Model training logic
│   ├── preprocessing.py       # Data cleaning & transformation
│   ├── report.py              # PDF report generation
│   ├── utils.py               # Helper functions
│   └── requirements.txt       # Python dependencies
├── DOCUMENTATION.md           # Detailed project documentation
└── README.md                  # This file
```

---

## 👥 Credits

**Made in NUST - SEECS**

- **Developers**: Asma & Fatima
- **Tech Stack**: Streamlit, Scikit-learn, Pandas, Google Gemini

---

_Happy Modeling!_ 🚀
