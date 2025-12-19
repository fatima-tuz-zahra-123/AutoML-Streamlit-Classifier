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
    Displays comprehensive dataset overview including row count, column count, types, and preview.
    Satisfies FR-2, FR-3.
    """
    # Basic metrics
    col1, col2, col3 = st.columns(3)
    col1.metric("Rows", f"{df.shape[0]:,}")
    col2.metric("Columns", df.shape[1])
    
    # Count categorical vs numerical
    num_cols = df.select_dtypes(include=['number']).shape[1]
    cat_cols = df.select_dtypes(include=['object', 'category']).shape[1]
    col3.metric("Numerical / Categorical", f"{num_cols} / {cat_cols}")
    
    # Missing values overview
    total_missing = df.isnull().sum().sum()
    missing_pct = (total_missing / (df.shape[0] * df.shape[1])) * 100 if df.shape[0] * df.shape[1] > 0 else 0
    st.caption(f"Missing Values: {total_missing:,} ({missing_pct:.1f}% of total data)")

    # Column Types Table
    with st.expander("📊 Column Types & Details", expanded=True):
        # Create a comprehensive column info table
        col_info = pd.DataFrame({
            'Column Name': df.columns,
            'Data Type': df.dtypes.astype(str).values,
            'Non-Null Count': df.count().values,
            'Null Count': df.isnull().sum().values,
            'Unique Values': [df[col].nunique() for col in df.columns],
            'Sample Value': [str(df[col].dropna().iloc[0]) if len(df[col].dropna()) > 0 else 'N/A' for col in df.columns]
        })
        st.dataframe(col_info, use_container_width=True, hide_index=True)
    
    # Raw Data Preview
    with st.expander("📋 Raw Data Preview (First 10 Rows)", expanded=True):
        st.dataframe(df.head(10), use_container_width=True)