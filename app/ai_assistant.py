import streamlit as st
import pandas as pd
import numpy as np
from groq import Groq
import os
import time
import re
from datetime import datetime
import io

class RateLimitError(Exception):
    pass

def log_to_ui(message):
    """Logs messages to the Streamlit session state for debugging."""
    print(f"[AI DEBUG] {message}")
    if 'api_logs' not in st.session_state:
        st.session_state['api_logs'] = []
    st.session_state['api_logs'].append(f"{datetime.now().strftime('%H:%M:%S')} - {message}")

def configure_genai(api_key):
    """
    Configures the Groq API client.
    """
    if api_key:
        return Groq(api_key=api_key)
    return None

def get_persona_instruction():
    """
    Returns the persona instruction based on the user's experience level.
    """
    level = st.session_state.get('experience_level', 'Beginner')
    if level == 'Expert':
        return "You are an expert colleague. Be concise, use technical jargon, focus on advanced insights, and assume deep knowledge. Do not over-explain basics."
    elif level == 'Moderate':
        return "You are a helpful mentor. Balance simplicity with technical terms. Explain 'why' things happen, assuming basic knowledge of ML concepts."
    else: # Beginner
        return "You are a patient teacher. Explain simply, use analogies, avoid jargon where possible (or explain it), and be encouraging."

def get_gemini_response(prompt, client=None):
    """
    Helper to get response from Groq with Meta Llama with retry logic.
    """
    if client is None:
        api_key = st.session_state.get('groq_api_key')
        if not api_key:
            return "⚠️ **API Key Missing**. Please add your Groq API key."
        client = Groq(api_key=api_key)
    
    log_to_ui(f"Initiating API Call. Prompt length: {len(prompt)} chars")
    
    for attempt in range(3): # Try 3 times
        try:
            log_to_ui(f"Attempt {attempt + 1}/3...")
            # Using Meta Llama 3.3 70B model
            response = client.chat.completions.create(
                model="llama-3.3-70b-versatile",
                messages=[
                    {"role": "user", "content": prompt}
                ],
                temperature=0.7,
                max_tokens=2048
            )
            log_to_ui("API Call Success!")
            return response.choices[0].message.content
        except Exception as e:
            error_msg = str(e)
            log_to_ui(f"Error on attempt {attempt + 1}: {error_msg}")
            
            if "429" in error_msg or "rate_limit" in error_msg.lower():
                if attempt < 2: # Don't sleep on the last attempt
                    wait_time = 10 * (attempt + 1) # Progressive backoff: 10s, 20s
                    log_to_ui(f"Rate Limit hit. Sleeping for {wait_time}s...")
                    st.toast(f"AI is busy (Rate Limit). Retrying in {wait_time}s...", icon="⏳")
                    time.sleep(wait_time) 
                    continue
                else:
                    log_to_ui("Max retries reached. Returning Rate Limit message.")
                    return "⚠️ **AI Busy (Rate Limit)**. Please wait a minute and try again."
            elif "401" in error_msg or "authentication" in error_msg.lower():
                return f"⚠️ **Authentication Error**: Invalid API key. Please check your Groq API key."
            else:
                return f"AI Error: {error_msg}"
    return "AI Error: Request failed."

@st.cache_data(show_spinner=False)
def _cached_ai_call_v2(prompt, api_key):
    """
    Internal function to cache AI responses.
    """
    client = configure_genai(api_key)
    response = get_gemini_response(prompt, client)
    if "AI Busy (Rate Limit)" in response:
        raise RateLimitError(response)
    return response

def clear_cache():
    """Clears the function cache to force new API calls."""
    st.cache_data.clear()
    log_to_ui("Cache cleared manually.")

def get_eda_insights(df):
    """Generates EDA insights using the AI by sending a summary instead of the full DataFrame."""
    try:
        # Check if Groq is configured
        api_key = st.session_state.get('groq_api_key')
        if not api_key:
            return "Please configure Groq API Key for AI-powered insights."

        # Limit data to avoid "Request Entity Too Large" error
        # Only use first 20 columns max
        df_limited = df.iloc[:, :20] if len(df.columns) > 20 else df

        # Create a string buffer to capture df.info() output
        buffer = io.StringIO()
        df_limited.info(buf=buffer)
        info_str = buffer.getvalue()

        persona = get_persona_instruction()

        prompt = f"""
        {persona}
        
        You are an expert data analyst. Based on the following data summary, provide key exploratory data analysis (EDA) insights.
        Focus on interesting patterns, potential data quality issues for a classification model, and relationships between variables.

        Data Head (first 5 rows, first 20 columns):
        {df_limited.head().to_string()}

        Descriptive Statistics:
        {df_limited.describe(include='all').to_string()}

        Column Info (Data Types & Non-Null Counts):
        {info_str}

        Provide a bulleted list of your top 5-7 insights. Keep the language simple and clear.
        Do not use emojis or markdown headers.
        """
        log_to_ui("Requesting EDA Insights (Checking Cache...)")
        try:
            return _cached_ai_call_v2(prompt, api_key)
        except RateLimitError as e:
            return str(e)
        except Exception as e:
            log_to_ui(f"An unexpected error occurred in get_eda_insights: {e}")
            return f"An unexpected error occurred while generating insights: {e}"

    except Exception as e:
        log_to_ui(f"Error preparing data for EDA insights: {e}")
        return f"Error preparing data for AI analysis: {e}"

def get_dataset_introduction(df):
    """
    Generates a comprehensive introductory explanation of the dataset.
    Uses Groq if available, otherwise falls back to heuristics.
    Produces professional, report-ready content.
    """
    n_rows, n_cols = df.shape
    num_cols = df.select_dtypes(include=['number']).columns.tolist()
    cat_cols = df.select_dtypes(include=['object', 'category']).columns.tolist()
    missing_total = df.isnull().sum().sum()
    missing_pct = (missing_total / (n_rows * n_cols)) * 100 if n_rows * n_cols > 0 else 0
    
    # Check if Groq is configured
    api_key = st.session_state.get('groq_api_key')
    if api_key:
        # Limit data to avoid "Request Entity Too Large" error
        # Only use first 20 columns max for the sample
        df_limited = df.iloc[:, :20] if len(df.columns) > 20 else df
        
        # Create a summary for the AI
        summary = df_limited.head().to_string()
        dtypes = df_limited.dtypes.to_string()
        stats = df_limited.describe().to_string()
        
        persona = get_persona_instruction()
        
        prompt = f"""
        {persona}
        
        You are writing the introduction section of a university-level machine learning project report.
        Write a comprehensive, professional introduction for this dataset that would fit in an academic report.
        
        Dataset Information:
        - Shape: {n_rows} rows, {n_cols} columns
        - Numerical columns ({len(num_cols)}): {', '.join(num_cols[:10])}{'...' if len(num_cols) > 10 else ''}
        - Categorical columns ({len(cat_cols)}): {', '.join(cat_cols[:10])}{'...' if len(cat_cols) > 10 else ''}
        - Missing values: {missing_total} total ({missing_pct:.1f}% of data)
        
        Columns & Types (first 20):
        {dtypes}
        
        First 5 rows (sample, first 20 columns):
        {summary}
        
        Descriptive Statistics:
        {stats}
        
        Your introduction should include:
        1. A clear description of what this dataset appears to represent (the domain/context)
        2. An explanation of what each column likely represents
        3. The potential target variable for classification
        4. The type of machine learning problem this represents
        5. Any initial observations about data quality or interesting patterns
        6. How this dataset could be useful for predictive modeling
        
        Write in a clear, academic tone suitable for a university project report.
        Use proper paragraphs, not bullet points.
        Do not use emojis or markdown headers.
        Keep it between 200-300 words.
        """
        log_to_ui("Requesting Dataset Introduction (Checking Cache...)")
        try:
            return _cached_ai_call_v2(prompt, api_key)
        except RateLimitError as e:
            return str(e)

    # Fallback Heuristic - More comprehensive version
    intro = f"""
**Dataset Overview**

This dataset contains {n_rows:,} observations (rows) and {n_cols} variables (columns), providing a substantial foundation for machine learning analysis.

**Data Structure**

The dataset comprises {len(num_cols)} numerical features and {len(cat_cols)} categorical features. The numerical columns include: {', '.join(num_cols[:5])}{'...' if len(num_cols) > 5 else ''}. The categorical columns include: {', '.join(cat_cols[:5]) if cat_cols else 'None detected'}{'...' if len(cat_cols) > 5 else ''}.

**Data Quality**

The dataset has {missing_total:,} missing values, representing approximately {missing_pct:.1f}% of the total data. {'This indicates a relatively clean dataset.' if missing_pct < 5 else 'Some data cleaning may be required to handle these missing values.'}

**Classification Potential**

As this is a classification application, the goal is to predict a categorical outcome. Potential target variables may include columns with distinct categories or labels. Look for columns that represent outcomes, classes, or groups (such as 'Species', 'Churn', 'Outcome', or 'Target').

**Next Steps**

Proceed to the Issue Detection step to identify and address any data quality issues before preprocessing and model training.
    """
    return intro.strip()

def get_step_guidance(step):
    """
    Returns guidance for each step based on experience level.
    """
    level = st.session_state.get('experience_level', 'Beginner')
    
    # Define content for each level
    content = {
        "Beginner": {
            "1. Upload Data": "👋 **Welcome!** Start by uploading your CSV file. I'll help you understand what's inside.",
            "2. Issue Detection": "🧹 **Cleanup Time!** Real data is messy. Let's find missing values and duplicates.",
            "3. Preprocessing": "⚙️ **Preparation:** Computers need numbers, not words. We'll convert text to numbers and fill in blanks.",
            "4. EDA": "📊 **Exploration:** Let's look at charts to see patterns in your data.",
            "5. Model Training": "🧠 **Training:** We will teach the computer to recognize patterns using different algorithms.",
            "6. Evaluation": "📝 **Grading:** Let's see how well our models performed (like a test score!).",
            "7. Report": "🎉 **Done!** Here is a summary of your project to download and share."
        },
        "Moderate": {
            "1. Upload Data": "📂 **Data Ingestion:** Upload your dataset to initialize the pipeline. We'll perform a schema check.",
            "2. Issue Detection": "🔍 **Data Quality Check:** We'll scan for nulls, duplicates, and outliers that could skew the model.",
            "3. Preprocessing": "🛠️ **Feature Engineering:** Handle imputation, encoding (One-Hot/Label), and scaling for optimal model performance.",
            "4. EDA": "📈 **EDA:** Analyze distributions and correlations to understand feature relationships.",
            "5. Model Training": "🤖 **Model Selection:** Train standard classifiers (Logistic Regression, RF, SVM) and compare baselines.",
            "6. Evaluation": "📉 **Metrics:** Analyze Accuracy, F1-Score, and Confusion Matrices to validate performance.",
            "7. Report": "📄 **Documentation:** Generate a comprehensive report of the pipeline and results."
        },
        "Expert": {
            "1. Upload Data": "📥 **Initialize Pipeline:** Load raw data for schema inference and initial profiling.",
            "2. Issue Detection": "⚠️ **Sanity Check:** Audit data integrity. Check for sparsity, high cardinality, and statistical outliers.",
            "3. Preprocessing": "🔧 **Transformation:** Configure imputation strategies, dimensionality reduction, and normalization techniques.",
            "4. EDA": "📊 **Statistical Analysis:** Inspect distributions, multicollinearity, and variance.",
            "5. Model Training": "🚀 **Training Loop:** Execute training across selected algorithms with default hyperparameters.",
            "6. Evaluation": "📐 **Performance Analysis:** Review precision-recall trade-offs and misclassification errors.",
            "7. Report": "📑 **Executive Summary:** Export technical documentation and model artifacts."
        }
    }
    
    # Fallback to Beginner if level not found
    selected_content = content.get(level, content["Beginner"])
    return selected_content.get(step, "")

def interpret_eda(df):
    """
    Provides specific insights based on EDA.
    Uses Groq if available.
    """
    # Check if Groq is configured
    api_key = st.session_state.get('groq_api_key')
    if api_key:
        # Limit the data sent to avoid "Request Entity Too Large" error
        # Only use first 20 columns max for summary
        numeric_df = df.select_dtypes(include=['number'])
        if len(numeric_df.columns) > 20:
            numeric_df = numeric_df.iloc[:, :20]
        
        summary = numeric_df.describe().to_string()
        corr = numeric_df.corr().round(2).to_string()
        
        persona = get_persona_instruction()
        
        prompt = f"""
        {persona}
        Analyze these statistics and correlations from a dataset.
        
        Statistics (first 20 numeric columns):
        {summary}
        
        Correlations:
        {corr}
        
        Provide 3-5 key insights about the data distribution, potential outliers, and relationships between variables.
        Do not use emojis.
        Explain your reasoning for each insight.
        """
        log_to_ui("Requesting EDA Interpretation (Checking Cache...)")
        try:
            return [_cached_ai_call_v2(prompt, api_key)] # Return as a list to match existing structure
        except RateLimitError as e:
            return [str(e)]

    # Fallback Heuristic
    insights = []
    
    # Correlation Insight
    numeric_df = df.select_dtypes(include=['number'])
    if not numeric_df.empty:
        corr_matrix = numeric_df.corr().abs()
        # Find high correlations (excluding diagonal)
        upper = corr_matrix.where(np.triu(np.ones(corr_matrix.shape), k=1).astype(bool))
        high_corr = [column for column in upper.columns if any(upper[column] > 0.8)]
        
        if high_corr:
            insights.append(f"High Correlation Detected: The columns {high_corr} are very strongly correlated with other features. This might mean they contain redundant information.")
        else:
            insights.append("Balanced Features: No extremely high correlations (> 0.8) were found between numerical features. This is good!")

    # Class Balance Insight (if we can guess a target)
    # Heuristic: Column with few unique values (2-10) might be target
    potential_targets = [col for col in df.columns if 2 <= df[col].nunique() <= 10]
    if potential_targets:
        col = potential_targets[0] # Pick first one as example
        counts = df[col].value_counts(normalize=True)
        if counts.max() > 0.8:
            insights.append(f"Imbalanced Classes: If '{col}' is your target, be careful! {counts.max():.1%} of the data belongs to one class. You might need F1-Score instead of Accuracy.")
            
    return insights

def get_preprocessing_suggestions(df):
    """
    Analyzes the dataframe and suggests preprocessing steps.
    """
    api_key = st.session_state.get('groq_api_key')
    if not api_key:
        return "Please configure Groq API Key for suggestions."

    # Limit data to avoid "Request Entity Too Large" error
    # Only use first 30 columns max
    df_limited = df.iloc[:, :30] if len(df.columns) > 30 else df
    
    summary = df_limited.describe().to_string()
    missing = df_limited.isnull().sum().to_string()
    dtypes = df_limited.dtypes.to_string()
    
    prompt = f"""
    {get_persona_instruction()}
    Analyze this dataset summary and suggest preprocessing steps.
    
    Data Types (first 30 columns):
    {dtypes}
    
    Missing Values:
    {missing}
    
    Statistics:
    {summary}
    
    Suggest what to do for:
    1. Missing Values (Imputation vs Dropping)
    2. Categorical Encoding (One-Hot vs Label)
    3. Scaling (Standard vs MinMax)
    
    Explain WHY for each suggestion. Keep it concise and professional. No emojis.
    """
    log_to_ui("Requesting Preprocessing Suggestions (Checking Cache...)")
    try:
        return _cached_ai_call_v2(prompt, api_key)
    except RateLimitError as e:
        return str(e)

def get_modeling_suggestions(df, target_col):
    """
    Suggests models based on data characteristics.
    """
    api_key = st.session_state.get('groq_api_key')
    if not api_key:
        return "Please configure Gemini API Key for suggestions."

    n_rows, n_cols = df.shape
    target_type = df[target_col].dtype
    target_unique = df[target_col].nunique()
    
    prompt = f"""
    {get_persona_instruction()}
    We are solving a classification problem.
    
    Dataset Info:
    - Rows: {n_rows}
    - Columns: {n_cols}
    - Target Variable: '{target_col}' (Type: {target_type}, Unique Values: {target_unique})
    
    Suggest which algorithms might work best and why.
    Consider:
    - Logistic Regression (Linear?)
    - Random Forest (Complex/Non-linear?)
    - SVM (Small/Medium data?)
    - KNN (Instance based?)
    
    Provide a brief recommendation strategy. Keep it concise and professional. No emojis.
    """
    log_to_ui("Requesting Modeling Suggestions (Checking Cache...)")
    try:
        return _cached_ai_call_v2(prompt, api_key)
    except RateLimitError as e:
        return str(e)

def chat_response(user_query, context_state):
    """
    Chatbot response. Uses Gemini if available, else rule-based.
    """
    # Check if Gemini is configured
    if 'groq_api_key' in context_state and context_state['groq_api_key']:
        api_key = context_state['groq_api_key']
        # Build context string
        context_str = ""
        if context_state['raw_data'] is not None:
            df = context_state['raw_data']
            context_str += f"Dataset Shape: {df.shape}\nColumns: {df.columns.tolist()}\n"
            context_str += f"Missing Values: {df.isnull().sum().sum()}\n"
        
        if context_state.get('model_results') is not None:
            best = context_state['model_results'].loc[context_state['model_results']['F1 Score'].idxmax()]['Model']
            context_str += f"Best Model: {best}\n"

        prompt = f"""
        {get_persona_instruction()}
        Current Context:
        {context_str}
        
        User Question: {user_query}
        
        Answer the question clearly.
        Keep your response short (max 3-4 sentences) unless the user explicitly asks for a detailed explanation.
        If it's about the data, use the context provided.
        Do not use emojis. Keep the tone friendly and professional.
        """
        log_to_ui("Requesting Chat Response (Checking Cache...)")
        try:
            return _cached_ai_call_v2(prompt, api_key)
        except RateLimitError as e:
            return str(e)

    # Fallback Rule-Based
    user_query = user_query.lower()
    
    if "missing" in user_query:
        if context_state['raw_data'] is not None:
            missing = context_state['raw_data'].isnull().sum().sum()
            return f"Your dataset has a total of {missing} missing values. You should check the 'Issue Detection' page."
        return "Please upload a dataset first."
        
    elif "best model" in user_query:
        if context_state['model_results'] is not None:
            best = context_state['model_results'].loc[context_state['model_results']['F1 Score'].idxmax()]['Model']
            return f"Based on the training results, the **{best}** appears to be the best model."
        return "You haven't trained any models yet. Go to 'Model Training'!"
        
    elif "outlier" in user_query:
        return "Outliers are data points that are far away from others. We use the IQR method (Interquartile Range) to detect them in the 'Issue Detection' tab."
        
    elif "hello" in user_query or "hi" in user_query:
        return "Hello! I'm your AI Data Science Tutor. Ask me about your dataset, models, or what to do next! (Add a Gemini API Key in the sidebar for smarter answers!)"
        
    else:
        return "I'm a simple AI. Try asking about 'missing values', 'best model', or 'outliers'. To unlock my full potential, please add a Gemini API Key in the sidebar!"
