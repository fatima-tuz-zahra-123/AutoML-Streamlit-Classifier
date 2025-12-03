import streamlit as st
import pandas as pd

# Import your custom modules directly
import utils
import eda
import issue_detection
import preprocessing
import modeling
import evaluation
import report

# --- PAGE CONFIGURATION ---
st.set_page_config(
    page_title="AutoML Classifier",
    page_icon="🤖",
    layout="wide",
    initial_sidebar_state="expanded"
)

# --- SESSION STATE INITIALIZATION ---
# We use session_state to persist data between pages (tabs)
if 'raw_data' not in st.session_state:
    st.session_state['raw_data'] = None
if 'clean_data' not in st.session_state:
    st.session_state['clean_data'] = None
if 'model_results' not in st.session_state:
    st.session_state['model_results'] = None
if 'best_model' not in st.session_state:
    st.session_state['best_model'] = None

# --- SIDEBAR NAVIGATION ---
st.sidebar.title("🔍 AutoML System")
page = st.sidebar.radio(
    "Navigate",
    ["1. Upload Data", "2. EDA", "3. Issue Detection", "4. Preprocessing", "5. Model Training", "6. Evaluation", "7. Report"]
)

st.sidebar.markdown("---")
st.sidebar.info("CS-245 Semester Project")

# --- PAGE ROUTING ---

# 1. UPLOAD DATA
if page == "1. Upload Data":
    st.title("📂 Upload Your Dataset")
    uploaded_file = st.file_uploader("Upload CSV file", type=["csv"])
    
    if uploaded_file:
        # Use utils to load data
        df = utils.load_data(uploaded_file)
        
        if df is not None:
            st.session_state['raw_data'] = df
            st.success("File uploaded successfully!")
            
            # Show basic metadata using utils
            utils.display_metadata(df)
            
            st.subheader("Raw Data Preview")
            st.dataframe(df.head())

# 2. EXPLORATORY DATA ANALYSIS
elif page == "2. EDA":
    st.title("📊 Exploratory Data Analysis")
    if st.session_state['raw_data'] is not None:
        # Call functions from eda.py
        eda.run_eda(st.session_state['raw_data'])
    else:
        st.warning("Please upload a dataset in Step 1.")

# 3. ISSUE DETECTION
elif page == "3. Issue Detection":
    st.title("⚠️ Data Quality Issues")
    if st.session_state['raw_data'] is not None:
        # Call functions from issue_detection.py
        issue_detection.detect_issues(st.session_state['raw_data'])
    else:
        st.warning("Please upload a dataset in Step 1.")

# 4. PREPROCESSING
elif page == "4. Preprocessing":
    st.title("🛠️ Preprocessing")
    if st.session_state['raw_data'] is not None:
        # Pass the raw data, get back cleaned data
        # Note: preprocessing.py should handle the UI for user approval
        clean_df = preprocessing.run_preprocessing(st.session_state['raw_data'])
        
        # If user clicked "Apply" inside the module, save to session state
        if clean_df is not None:
            st.session_state['clean_data'] = clean_df
    else:
        st.warning("Please upload a dataset in Step 1.")

# 5. MODEL TRAINING
elif page == "5. Model Training":
    st.title("🧠 Model Training")
    if st.session_state['clean_data'] is not None:
        # modeling.py handles model selection and training loop
        results, best_model = modeling.train_models(st.session_state['clean_data'])
        
        if results is not None:
            st.session_state['model_results'] = results
            st.session_state['best_model'] = best_model
    else:
        st.warning("Please preprocess your data in Step 4.")

# 6. EVALUATION
elif page == "6. Evaluation":
    st.title("📈 Model Evaluation")
    if st.session_state['model_results'] is not None:
        evaluation.show_dashboard(st.session_state['model_results'], st.session_state['best_model'])
    else:
        st.warning("Please train models in Step 5.")

# 7. REPORT
elif page == "7. Report":
    st.title("📝 Final Report")
    if st.session_state['model_results'] is not None:
        report.generate_report(
            st.session_state['raw_data'], 
            st.session_state['clean_data'], 
            st.session_state['model_results']
        )
    else:
        st.warning("Complete the pipeline to generate a report.")