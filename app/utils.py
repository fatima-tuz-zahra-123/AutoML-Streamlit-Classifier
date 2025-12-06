import streamlit as st
import pandas as pd

def load_data(uploaded_file):
    """
    Loads CSV data and handles basic errors.
    Satisfies FR-1, FR-2.
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
    Satisfies FR-2, FR-3.
    """
    col1, col2, col3 = st.columns(3)
    col1.metric("Rows", df.shape[0])
    col2.metric("Columns", df.shape[1])
    
    # Count categorical vs numerical
    num_cols = df.select_dtypes(include=['number']).shape[1]
    cat_cols = df.select_dtypes(include=['object', 'category']).shape[1]
    col3.metric("Numerical / Categorical", f"{num_cols} / {cat_cols}")

    with st.expander("View Column Types"):
        st.dataframe(df.dtypes.astype(str))