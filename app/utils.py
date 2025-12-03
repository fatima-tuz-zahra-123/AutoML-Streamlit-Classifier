import streamlit as st
import pandas as pd
import io

def load_data(uploaded_file):
    """
    Loads CSV data and handles basic errors.
    """
    try:
        df = pd.read_csv(uploaded_file)
        return df
    except Exception as e:
        st.error(f"Error loading file: {e}")
        return None

def display_metadata(df):
    """
    Displays row count, column count, and types.
    """
    col1, col2, col3 = st.columns(3)
    col1.metric("Rows", df.shape[0])
    col2.metric("Columns", df.shape[1])
    
    # Count categorical vs numerical
    num_cols = df.select_dtypes(include=['number']).shape[1]
    cat_cols = df.select_dtypes(include=['object', 'category']).shape[1]
    col3.metric("Numerical / Categorical", f"{num_cols} / {cat_cols}")

    # Show column types expander
    with st.expander("View Column Types"):
        st.dataframe(df.dtypes.astype(str), height=200)