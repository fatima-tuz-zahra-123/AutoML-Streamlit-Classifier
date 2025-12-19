import streamlit as st
import pandas as pd

# Import your custom modules
import utils
import eda
import issue_detection
import preprocessing
import modeling
import evaluation
import report
import ai_assistant  # New AI Module

# --- PAGE CONFIGURATION ---
st.set_page_config(
    page_title="AutoML Classifier",
    layout="wide",
    initial_sidebar_state="expanded"
)

# --- SESSION STATE INITIALIZATION ---
if 'experience_level' not in st.session_state:
    st.session_state['experience_level'] = 'Beginner'

# --- CUSTOM CSS FOR MODERN UI ---
st.markdown("""
    <style>
    /* Main Background */
    .main {
        background-color: #FFFFFF;
    }
    /* Headers */
    h1, h2, h3 {
        color: #2962FF;
        font-family: 'Helvetica Neue', sans-serif;
        font-weight: 700;
    }
    /* Buttons */
    .stButton>button {
        background-color: #2962FF;
        color: white;
        border-radius: 8px;
        border: none;
        padding: 0.5rem 1rem;
        font-weight: 600;
    }
    .stButton>button:hover {
        background-color: #0039CB;
        color: white;
    }
    /* Metrics */
    [data-testid="stMetricValue"] {
        color: #2962FF;
    }
    /* Sidebar Radio Button Color Fix */
    div[role="radiogroup"] > label > div:first-child {
        background-color: #2962FF !important;
        border-color: #2962FF !important;
    }
    div[data-testid="stMarkdownContainer"] p {
        font-size: 1.05rem;
    }
    /* Sidebar Expander Header Color - Robust Selector */
    [data-testid="stSidebar"] [data-testid="stExpander"] details > summary {
        background-color: #E3F2FD !important;
        color: #0D47A1 !important;
        border-radius: 8px;
    }
    [data-testid="stSidebar"] [data-testid="stExpander"] details > summary:hover {
        background-color: #BBDEFB !important;
    }
    [data-testid="stSidebar"] [data-testid="stExpander"] details > summary p {
        color: #0D47A1 !important;
        font-weight: 600;
    }
    [data-testid="stSidebar"] [data-testid="stExpander"] details > summary svg {
        fill: #0D47A1 !important;
        color: #0D47A1 !important;
    }
    /* Progress Bar Color Override */
    .stProgress > div > div > div > div {
        background-color: #2962FF;
    }
    /* Reduce whitespace for HR */
    hr {
        margin-top: 0.5rem !important;
        margin-bottom: 0.5rem !important;
    }
    /* Robot Animation */
    @keyframes sleep {
        0% { transform: scale(1) translateY(0px); opacity: 0.7; }
        50% { transform: scale(1.02) translateY(2px); opacity: 0.5; }
        100% { transform: scale(1) translateY(0px); opacity: 0.7; }
    }
    @keyframes happy-bounce {
        0%, 100% { transform: translateY(0); }
        25% { transform: translateY(-4px) rotate(-5deg); }
        50% { transform: translateY(0) rotate(5deg); }
        75% { transform: translateY(-2px) rotate(-3deg); }
    }
    .robot-avatar {
        font-size: 2rem;
        text-align: center;
        margin-bottom: 5px;
        display: inline-block;
        cursor: pointer;
    }
    .robot-sleeping {
        animation: sleep 3s infinite ease-in-out;
        filter: grayscale(0.6);
    }
    .robot-waking {
        animation: happy-bounce 2s infinite ease-in-out;
        filter: drop-shadow(0 0 8px rgba(41, 98, 255, 0.4));
    }
    </style>
    """, unsafe_allow_html=True)

# --- SESSION STATE INITIALIZATION ---
if 'experience_level' not in st.session_state:
    st.session_state['experience_level'] = 'Beginner'
if 'raw_data' not in st.session_state:
    st.session_state['raw_data'] = None
if 'clean_data' not in st.session_state:
    st.session_state['clean_data'] = None
if 'model_results' not in st.session_state:
    st.session_state['model_results'] = None
if 'best_model' not in st.session_state:
    st.session_state['best_model'] = None
if 'current_page' not in st.session_state:
    st.session_state['current_page'] = "1. Upload Data"
if 'groq_api_key' not in st.session_state:
    # Load API Key from secrets
    if "GROQ_API_KEY" in st.secrets:
        st.session_state['groq_api_key'] = st.secrets["GROQ_API_KEY"]
        ai_assistant.configure_genai(st.session_state['groq_api_key'])
    else:
        st.error("Groq API Key not found. Please add GROQ_API_KEY to .streamlit/secrets.toml")

# --- SIDEBAR NAVIGATION ---
st.sidebar.title("✧ AutoML System")

# Experience Level Selection
st.sidebar.subheader("Experience Level")

def on_level_change():
    st.toast(f"AI Persona updated to: {st.session_state['experience_level']}", icon="🧠")

st.sidebar.select_slider(
    "Select your knowledge level:",
    options=["Beginner", "Moderate", "Expert"],
    key='experience_level',
    on_change=on_level_change,
    help="Slide to adjust how the AI explains concepts to you."
)

st.sidebar.markdown("---")

# Progress Tracker
steps = {
    "1. Upload Data": "raw_data",
    "2. Issue Detection": "raw_data",
    "3. Preprocessing": "clean_data",
    "4. EDA": "clean_data",
    "5. Model Training": "model_results",
    "6. Evaluation": "model_results",
    "7. Report": "model_results"
}

# Determine progress
progress_count = 0
for step, state_key in steps.items():
    if st.session_state.get(state_key) is not None:
        progress_count += 1

progress_percent = int((progress_count / len(steps)) * 100)
st.sidebar.progress(progress_percent)
st.sidebar.caption(f"Pipeline Progress: {progress_percent}%")

# Navigation Logic
step_list = list(steps.keys())

def update_page():
    # Callback to sync radio button with session state
    pass 

def go_next(next_page):
    st.session_state['current_page'] = next_page

page = st.sidebar.radio(
    "Navigate",
    step_list,
    key="current_page",
    on_change=update_page
)

# Proceed Next Button Logic
current_index = step_list.index(page)
if current_index < len(step_list) - 1:
    next_step = step_list[current_index + 1]
    if st.sidebar.button(f"Proceed to {next_step.split('. ')[1]}", on_click=go_next, args=(next_step,)):
        pass

st.sidebar.markdown("---")
st.sidebar.info("ML Project by Asma and Fatima")

# --- AI CHATBOT (SIDEBAR) ---
st.sidebar.markdown("---")

if 'chat_open' not in st.session_state:
    st.session_state['chat_open'] = False

def toggle_chat():
    st.session_state['chat_open'] = not st.session_state['chat_open']

# Robot Display
if not st.session_state['chat_open']:
    # Sleeping State
    st.sidebar.markdown("""
    <div style="text-align: center;">
        <div class="robot-avatar robot-sleeping">🤖💤</div>
        <p style="font-size: 0.8rem; color: #666;"><i>Activate AI Assistant</i></p>
    </div>
    """, unsafe_allow_html=True)
    if st.sidebar.button("Start", key="wake_btn", use_container_width=True, on_click=toggle_chat):
        pass

else:
    # Active State
    st.sidebar.markdown("""
    <div style="text-align: center;">
        <div class="robot-avatar robot-waking">🤖✨</div>
        <p style="font-size: 0.9rem; color: #2962FF; font-weight: bold;">How can I assist?</p>
    </div>
    """, unsafe_allow_html=True)
    
    if st.sidebar.button("Close Chat", key="close_btn", use_container_width=True, on_click=toggle_chat):
        pass

    # Chat Interface (Only visible when open)
    st.sidebar.markdown("### 💬 Chat with Data")
    if st.session_state.get('groq_api_key'):
        ai_assistant.configure_genai(st.session_state['groq_api_key'])
    
    with st.sidebar.form(key='ai_chat_form'):
        user_query = st.text_area("", height=100, placeholder="e.g., What is the best model?")
        submit_button = st.form_submit_button(label='Send Question')

    if submit_button and user_query:
        with st.spinner("Thinking..."):
            response = ai_assistant.chat_response(user_query, st.session_state)
            st.sidebar.info(response)

# --- DEBUG LOGS (ALWAYS VISIBLE) ---
st.sidebar.markdown("---")
with st.sidebar.expander("🛠️ API Debug Logs", expanded=False):
    if 'api_logs' in st.session_state and st.session_state['api_logs']:
        for log in reversed(st.session_state['api_logs']):
            st.text(log)
    else:
        st.text("No API calls yet.")
    
    if st.sidebar.button("Clear Logs"):
        st.session_state['api_logs'] = []
        st.rerun()
        
    if st.sidebar.button("⚠️ Clear AI Cache"):
        ai_assistant.clear_cache()
        st.rerun()

# --- PAGE ROUTING ---

# 1. UPLOAD DATA
if page == "1. Upload Data":
    st.title("Upload Your Dataset")
    
    # AI Guide
    st.info(ai_assistant.get_step_guidance("1. Upload Data"))
    
    st.markdown("Welcome to the **AutoML Classifier**! Upload your CSV file to get started. This system will guide you through EDA, cleaning, and training machine learning models.")
    
    uploaded_file = st.file_uploader("Upload CSV file", type=["csv"])
    
    if uploaded_file:
        df = utils.load_data(uploaded_file)
        if df is not None:
            st.session_state['raw_data'] = df
            # Reset preprocessing state when new data is uploaded
            if 'working_df' in st.session_state:
                del st.session_state['working_df']
            st.success("File uploaded successfully!")
    
    st.markdown("---")
    st.markdown("**OR** use a sample dataset:")
    
    # Add a button to load sample data for testing
    if st.button("Load Sample Test Data (Complex)"):
        try:
            df = pd.read_csv("app/complex_test_dataset.csv")
            st.session_state['raw_data'] = df
            if 'working_df' in st.session_state:
                del st.session_state['working_df']
            # Clear previous AI analysis when new data is loaded
            if 'dataset_intro' in st.session_state:
                st.session_state['dataset_intro'] = None
            st.success("Loaded complex test dataset!")
        except FileNotFoundError:
            st.error("Test data not found. Please run the generation script first.")

    # --- COMMON DATA DISPLAY LOGIC ---
    if st.session_state.get('raw_data') is not None:
        df = st.session_state['raw_data']
        
        # AI Introduction
        st.markdown("### Dataset Overview")
        with st.expander("See Dataset Breakdown (AI Analysis)"):
            if 'dataset_intro' not in st.session_state:
                st.session_state['dataset_intro'] = None
            
            if st.button("Analyze Dataset with AI"):
                with st.spinner("AI is analyzing the dataset..."):
                    st.session_state['dataset_intro'] = ai_assistant.get_dataset_introduction(df)
            
            if st.session_state['dataset_intro']:
                st.markdown(st.session_state['dataset_intro'])
        
        utils.display_metadata(df)
        
        with st.expander("View Raw Data Preview", expanded=True):
            st.dataframe(df.head())

# 2. ISSUE DETECTION
elif page == "2. Issue Detection":
    st.title("Data Quality Issues")
    
    # AI Guide
    st.info(ai_assistant.get_step_guidance("2. Issue Detection"))
    
    if st.session_state['raw_data'] is not None:
        issue_detection.detect_issues(st.session_state['raw_data'])
    else:
        st.warning("Please upload a dataset in Step 1.")

# 3. PREPROCESSING
elif page == "3. Preprocessing":
    st.title("Preprocessing")
    
    # AI Guide
    st.info(ai_assistant.get_step_guidance("3. Preprocessing"))
    
    if st.session_state['raw_data'] is not None:
        # We will implement this module next
        clean_df = preprocessing.run_preprocessing(st.session_state['raw_data'])
        if clean_df is not None:
            st.session_state['clean_data'] = clean_df
    else:
        st.warning("Please upload a dataset in Step 1.")

# 4. EDA
elif page == "4. EDA":
    st.title("Exploratory Data Analysis")
    
    # AI Guide
    st.info(ai_assistant.get_step_guidance("4. EDA"))
    
    if st.session_state['clean_data'] is not None:
        # AI Insights for EDA
        with st.spinner("AI is looking for patterns..."):
            insights = ai_assistant.interpret_eda(st.session_state['clean_data'])
        if insights:
            with st.expander("🤖 AI Insights for EDA", expanded=True):
                for i in insights:
                    st.write(i)
                    
        eda.run_eda(st.session_state['clean_data'])
    else:
        st.warning("Please preprocess your data in Step 3.")

# 5. MODEL TRAINING
elif page == "5. Model Training":
    st.title("Model Training")
    
    # AI Guide
    st.info(ai_assistant.get_step_guidance("5. Model Training"))
    
    if st.session_state['clean_data'] is not None:
        results, best_model = modeling.train_models(st.session_state['clean_data'])
        if results is not None:
            st.session_state['model_results'] = results
            st.session_state['best_model'] = best_model
    else:
        st.warning("Please preprocess your data in Step 4.")

# 6. EVALUATION
elif page == "6. Evaluation":
    st.title("Model Evaluation")
    
    # AI Guide
    st.info(ai_assistant.get_step_guidance("6. Evaluation"))
    
    if st.session_state['model_results'] is not None:
        evaluation.show_dashboard(st.session_state['model_results'], st.session_state['best_model'])
    else:
        st.warning("Please train models in Step 5.")

# 7. REPORT
elif page == "7. Report":
    st.title("Final Report")
    
    # AI Guide
    st.info(ai_assistant.get_step_guidance("7. Report"))
    
    if st.session_state['model_results'] is not None:
        # Get preprocessing log if available
        preprocessing_log = st.session_state.get('preprocessing_log', [])
        
        report.generate_report(
            st.session_state['raw_data'], 
            st.session_state['clean_data'], 
            st.session_state['model_results'],
            st.session_state['best_model'],
            preprocessing_log
        )
    else:
        st.warning("Please train models in Step 5.")