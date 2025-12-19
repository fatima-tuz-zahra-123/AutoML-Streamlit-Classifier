import streamlit as st
import pandas as pd
import base64
import os

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
    @import url('https://fonts.googleapis.com/css2?family=Inter:wght@400;500;600;700&display=swap');
    
    /* Global Font */
    * {
        font-family: 'Inter', -apple-system, BlinkMacSystemFont, sans-serif !important;
    }
    
    /* Main Background */
    .main {
        background-color: #FFFFFF;
    }
    
    /* Headers */
    h1, h2, h3 {
        color: #2962FF;
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
    
    /* Selectbox styling */
    [data-testid="stSelectbox"] label {
        color: #2962FF !important;
        font-weight: 600;
    }
    
    /* Info/Warning messages - black text */
    .stAlert, [data-testid="stAlert"] {
        color: #000000 !important;
    }
    .stAlert p, [data-testid="stAlert"] p {
        color: #000000 !important;
    }
    
    /* Sample data buttons - equal sizing */
    .sample-btn-container .stButton > button {
        min-height: 60px;
        font-size: 0.85rem;
    }
    
    /* Remove all orange/red colors */
    * {
        --primary-color: #2962FF !important;
    }
    
    /* Sidebar Radio Button Color Fix */
    div[role="radiogroup"] > label > div:first-child {
        background-color: #2962FF !important;
        border-color: #2962FF !important;
    }
    
    /* Sidebar Expander Header */
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
    
    /* Progress Bar - Blue */
    .stProgress > div > div > div > div {
        background-color: #2962FF !important;
    }
    
    /* All Links - Blue */
    a {
        color: #2962FF !important;
    }
    a:hover {
        color: #0039CB !important;
    }
    
    /* Expander - Blue on hover */
    [data-testid="stExpander"] summary:hover {
        color: #2962FF !important;
    }
    
    /* Error/Warning to Info Blue */
    .stAlert {
        background-color: #E3F2FD !important;
        color: #0D47A1 !important;
    }
    
    /* Reduce whitespace for HR */
    hr {
        margin-top: 0.5rem !important;
        margin-bottom: 0.5rem !important;
    }
    
    /* AI Assistant Icon Animation */
    @keyframes gentle-pulse {
        0%, 100% { transform: scale(1); opacity: 0.85; }
        50% { transform: scale(1.05); opacity: 1; }
    }
    @keyframes active-glow {
        0%, 100% { transform: translateY(0); box-shadow: 0 0 10px rgba(41, 98, 255, 0.3); }
        50% { transform: translateY(-2px); box-shadow: 0 0 20px rgba(41, 98, 255, 0.5); }
    }
    .ai-avatar {
        font-size: 2rem;
        text-align: center;
        margin-bottom: 5px;
        display: inline-block;
        cursor: pointer;
    }
    .ai-sleeping {
        animation: gentle-pulse 3s infinite ease-in-out;
    }
    .ai-active {
        animation: active-glow 2s infinite ease-in-out;
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

st.sidebar.selectbox(
    "Select your level:",
    options=["Beginner", "Moderate", "Expert"],
    key='experience_level',
    on_change=on_level_change,
    label_visibility="collapsed"
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

# --- AI CHATBOT (SIDEBAR) ---
if 'chat_open' not in st.session_state:
    st.session_state['chat_open'] = False

def toggle_chat():
    st.session_state['chat_open'] = not st.session_state['chat_open']

# Load robot gif as base64
def get_robot_gif_base64():
    gif_path = os.path.join(os.path.dirname(__file__), "ROBOT.gif")
    if os.path.exists(gif_path):
        with open(gif_path, "rb") as f:
            return base64.b64encode(f.read()).decode()
    return None

robot_gif_b64 = get_robot_gif_base64()
robot_img_tag = f'<img src="data:image/gif;base64,{robot_gif_b64}" width="80" />' if robot_gif_b64 else '🤖'

# AI Assistant Display
if not st.session_state['chat_open']:
    # Inactive State
    st.sidebar.markdown(f"""
    <div style="text-align: center;">
        <div class="ai-avatar ai-sleeping">
            {robot_img_tag}
        </div>
        <p style="font-size: 0.8rem; color: #2962FF;"><i>Activate AI Assistant</i></p>
    </div>
    """, unsafe_allow_html=True)
    if st.sidebar.button("Start", key="wake_btn", use_container_width=True, on_click=toggle_chat):
        pass

else:
    # Active State
    st.sidebar.markdown(f"""
    <div style="text-align: center;">
        <div class="ai-avatar ai-active">
            {robot_img_tag}
        </div>
        <p style="font-size: 0.9rem; color: #2962FF; font-weight: bold;">How can I assist?</p>
    </div>
    """, unsafe_allow_html=True)
    
    if st.sidebar.button("Close Chat", key="close_btn", use_container_width=True, on_click=toggle_chat):
        pass

    # Chat Interface (Only visible when open)
    st.sidebar.markdown("### 💬 Chat with Data")
    if st.session_state.get('groq_api_key'):
        ai_assistant.configure_genai(st.session_state['groq_api_key'])
    
    user_question = st.sidebar.text_input("Ask anything about your data...")
    if user_question:
        with st.sidebar:
            with st.spinner("Thinking..."):
                # Build context for chat_response
                context_state = {
                    'raw_data': st.session_state.get('raw_data'),
                    'clean_data': st.session_state.get('clean_data'),
                    'model_results': st.session_state.get('model_results'),
                    'groq_api_key': st.session_state.get('groq_api_key')
                }
                response = ai_assistant.chat_response(user_question, context_state)
                st.markdown(response)

st.sidebar.markdown("---")
st.sidebar.info("⸜(｡˃ ᵕ ˂ )⸝♡ Made by Asma and Fatima ⊹ ࣪ ˖")

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
    st.markdown("**OR** choose a sample dataset:")
    
    # Sample datasets from the sample data folder
    sample_datasets = [
        ("🌸 Iris", "Iris.csv"),
        ("❤️ Heart", "heart.csv"),
        ("🏦 Bank", "BankChurners.csv"),
        ("🎮 Games", "vgsales.csv"),
        ("📊 2017", "2017.csv")
    ]
    
    cols = st.columns(5)
    for idx, (name, filename) in enumerate(sample_datasets):
        with cols[idx]:
            if st.button(name, key=f"sample_{filename}", use_container_width=True):
                try:
                    sample_path = os.path.join(os.path.dirname(os.path.dirname(__file__)), "sample data", filename)
                    df = pd.read_csv(sample_path)
                    st.session_state['raw_data'] = df
                    if 'working_df' in st.session_state:
                        del st.session_state['working_df']
                    if 'dataset_intro' in st.session_state:
                        st.session_state['dataset_intro'] = None
                    st.success(f"Loaded {filename}!")
                    st.rerun()
                except FileNotFoundError:
                    st.error(f"File {filename} not found.")

    # --- COMMON DATA DISPLAY LOGIC ---
    if st.session_state.get('raw_data') is not None:
        df = st.session_state['raw_data']
        
        # AI Introduction
        st.markdown("### Dataset Overview")
        with st.expander("See Dataset Breakdown (AI Analysis)"):
            if 'dataset_intro' not in st.session_state:
                st.session_state['dataset_intro'] = None
            
            if st.button("Analyze Dataset with AI", key="analyze_dataset_btn"):
                with st.spinner("AI is analyzing the dataset..."):
                    st.session_state['dataset_intro'] = ai_assistant.get_dataset_introduction(df)
                st.rerun()
            
            if st.session_state['dataset_intro']:
                st.markdown(st.session_state['dataset_intro'])
        
        utils.display_metadata(df)

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