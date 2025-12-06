import streamlit as st
import pandas as pd
import numpy as np
import google.generativeai as genai
import os

def configure_genai(api_key):
    """
    Configures the Gemini API.
    """
    if api_key:
        genai.configure(api_key=api_key)
        return True
    return False

def get_gemini_response(prompt):
    """
    Helper to get response from Gemini.
    """
    try:
        model = genai.GenerativeModel('gemini-2.0-flash')
        response = model.generate_content(prompt)
        return response.text
    except Exception as e:
        return f"AI Error: {str(e)}"

def get_dataset_introduction(df):
    """
    Generates an introductory explanation of the dataset.
    Uses Gemini if available, otherwise falls back to heuristics.
    """
    n_rows, n_cols = df.shape
    num_cols = df.select_dtypes(include=['number']).columns.tolist()
    cat_cols = df.select_dtypes(include=['object', 'category']).columns.tolist()
    
    # Check if Gemini is configured
    if 'gemini_api_key' in st.session_state and st.session_state['gemini_api_key']:
        # Create a summary for the AI
        summary = df.head().to_string()
        dtypes = df.dtypes.to_string()
        stats = df.describe().to_string()
        
        prompt = f"""
        You are an expert Data Science Teacher. Introduce this dataset to a student.
        Here is the data summary:
        Shape: {n_rows} rows, {n_cols} columns.
        Columns & Types:
        {dtypes}
        
        First 5 rows:
        {summary}
        
        Statistics:
        {stats}
        
        Explain what this dataset seems to be about, what the columns represent, and what kind of classification problem we might solve with it. 
        Keep it encouraging and educational. Do not use emojis. Keep the tone friendly and professional, like a helpful colleague. 
        Make it concise and simple for a beginner, but include expert concepts where relevant.
        Explain your reasoning: "I deduced this because..."
        """
        return get_gemini_response(prompt)

    # Fallback Heuristic
    intro = f"""
    ### Dataset Introduction
    Hello! I've analyzed your uploaded dataset. Here is what I found:
    
    - **Size:** The dataset contains **{n_rows} rows** (samples) and **{n_cols} columns** (features).
    - **Structure:** It is a mix of **{len(num_cols)} numerical** features (like {', '.join(num_cols[:3])}...) and **{len(cat_cols)} categorical** features (like {', '.join(cat_cols[:3])}...).
    
    **Goal:** 
    We are likely trying to predict one of the categorical columns (Classification) or a numerical value (Regression). 
    Since this is a **Classifier** app, you should look for a column that represents a category or label (e.g., 'Survived', 'Churn', 'Species') to be your **Target Variable**.
    """
    return intro

def get_step_guidance(step):
    """
    Returns teacher-like guidance for each step.
    """
    guidance = {
        "1. Upload Data": """
        **Teacher's Note:**
        The first step in any Machine Learning project is getting your data ready. 
        Upload your CSV file here. Once uploaded, I'll take a quick look and tell you what's inside!
        """,
        
        "2. EDA": """
        **Teacher's Note:**
        **Exploratory Data Analysis (EDA)** is like interviewing your data. 
        We want to understand:
        1. How is the data distributed? (Histograms)
        2. Are there relationships between variables? (Correlations/Scatter Plots)
        3. Are there any weird patterns?
        
        Look at the charts below and try to spot trends!
        """,
        
        "3. Issue Detection": """
        **Teacher's Note:**
        Real-world data is rarely clean. It often has:
        - **Missing Values:** Empty cells.
        - **Duplicates:** Repeated rows.
        - **Outliers:** Values that are way too high or too low.
        
        We need to find these issues now so we can fix them in the next step.
        """,
        
        "4. Preprocessing": """
        **Teacher's Note:**
        This is the most critical step! Machine Learning models are like math equations—they need numbers, not words.
        
        **Your Tasks:**
        1. **Impute Missing Values:** Fill in the blanks (e.g., with the average).
        2. **Encode Categories:** Turn text (e.g., "Red", "Blue") into numbers (e.g., 0, 1).
        3. **Scale Features:** Make sure big numbers (like Salary) don't dominate small numbers (like Age).
        """,
        
        "5. Model Training": """
        **Teacher's Note:**
        Now the magic happens! We will train multiple algorithms to learn patterns in your data.
        
        - **Logistic Regression:** Good for simple linear relationships.
        - **Random Forest:** Great for complex data, uses many "Decision Trees".
        - **SVM:** Good for finding clear boundaries between classes.
        
        We will split your data into **Training** (to learn) and **Testing** (to check performance).
        """,
        
        "6. Evaluation": """
        **Teacher's Note:**
        How well did we do?
        
        - **Accuracy:** Overall percentage correct.
        - **F1 Score:** A balanced score (better if classes are uneven).
        - **Confusion Matrix:** Shows exactly where the model got confused (e.g., predicted 'Yes' when it was 'No').
        """,
        
        "7. Report": """
        **Teacher's Note:**
        Great job! You've built a full ML pipeline. 
        This page generates a professional report summarizing everything we found. You can download it and share it!
        """
    }
    return guidance.get(step, "")

def interpret_eda(df):
    """
    Provides specific insights based on EDA.
    Uses Gemini if available.
    """
    # Check if Gemini is configured
    if 'gemini_api_key' in st.session_state and st.session_state['gemini_api_key']:
        summary = df.describe().to_string()
        corr = df.select_dtypes(include=['number']).corr().to_string()
        
        prompt = f"""
        You are an expert Data Analyst. Analyze these statistics and correlations from a dataset.
        
        Statistics:
        {summary}
        
        Correlations:
        {corr}
        
        Provide 3-5 key insights about the data distribution, potential outliers, and relationships between variables.
        Be specific and educational. Do not use emojis.
        Explain your reasoning for each insight. For example: "I noticed X is high, which suggests Y..."
        """
        response = get_gemini_response(prompt)
        return [response] # Return as a list to match existing structure

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

def chat_response(user_query, context_state):
    """
    Chatbot response. Uses Gemini if available, else rule-based.
    """
    # Check if Gemini is configured
    if 'gemini_api_key' in context_state and context_state['gemini_api_key']:
        # Build context string
        context_str = ""
        if context_state['raw_data'] is not None:
            df = context_state['raw_data']
            context_str += f"Dataset Shape: {df.shape}\nColumns: {df.columns.tolist()}\n"
            context_str += f"Missing Values: {df.isnull().sum().sum()}\n"
        
        if context_state['model_results'] is not None:
            best = context_state['model_results'].loc[context_state['model_results']['F1 Score'].idxmax()]['Model']
            context_str += f"Best Model: {best}\n"

        prompt = f"""
        You are a helpful AI Data Science Tutor for a student project.
        Current Context:
        {context_str}
        
        User Question: {user_query}
        
        Answer the question clearly and concisely. If it's about the data, use the context provided.
        Do not use emojis. Keep the tone friendly and professional.
        Explain your reasoning step-by-step. "I checked the data and found..."
        """
        return get_gemini_response(prompt)

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
