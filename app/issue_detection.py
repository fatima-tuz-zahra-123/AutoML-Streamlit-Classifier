import streamlit as st
import pandas as pd
import numpy as np

def detect_issues(df):
    """
    Scans the dataset for quality issues.
    Satisfies FR-9, FR-10, FR-11.
    """
    st.header("Data Quality Assessment")

    # --- FR-9: Missing Values Analysis [cite: 100] ---
    st.subheader("1. Missing Values")
    missing_count = df.isnull().sum()
    missing_percent = (df.isnull().sum() / len(df)) * 100
    
    # Create a summary dataframe
    missing_df = pd.DataFrame({
        'Missing Count': missing_count,
        'Percentage (%)': missing_percent
    })
    # Filter to show only columns with missing values
    missing_df = missing_df[missing_df['Missing Count'] > 0]
    
    if not missing_df.empty:
        st.warning(f"Found {len(missing_df)} columns with missing values.")
        st.dataframe(missing_df)
    else:
        st.success("No missing values detected.")

    st.markdown("---")

    # --- FR-10: Duplicate Rows Detection [cite: 101] ---
    st.subheader("2. Duplicate Rows")
    duplicate_count = df.duplicated().sum()
    
    if duplicate_count > 0:
        st.warning(f"Found {duplicate_count} duplicate rows.")
        if st.checkbox("Show Duplicate Rows Example"):
            st.dataframe(df[df.duplicated()].head())
    else:
        st.success("No duplicate rows detected.")

    st.markdown("---")

    # --- FR-11: Outlier Detection (IQR Method) [cite: 102] ---
    st.subheader("3. Outlier Detection (IQR Method)")
    
    numeric_cols = df.select_dtypes(include=['float64', 'int64']).columns
    outlier_summary = []

    for col in numeric_cols:
        Q1 = df[col].quantile(0.25)
        Q3 = df[col].quantile(0.75)
        IQR = Q3 - Q1
        
        lower_bound = Q1 - 1.5 * IQR
        upper_bound = Q3 + 1.5 * IQR
        
        outliers = df[(df[col] < lower_bound) | (df[col] > upper_bound)]
        num_outliers = len(outliers)
        
        if num_outliers > 0:
            outlier_summary.append({
                'Column': col,
                'Outliers Count': num_outliers,
                'Percentage (%)': round((num_outliers / len(df)) * 100, 2)
            })

    if outlier_summary:
        st.warning(f"Potential outliers detected in {len(outlier_summary)} columns.")
        st.dataframe(pd.DataFrame(outlier_summary))
    else:
        st.success("No statistical outliers detected in numerical columns.")