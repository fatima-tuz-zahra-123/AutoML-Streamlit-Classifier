import streamlit as st
import pandas as pd
from sklearn.preprocessing import StandardScaler, MinMaxScaler, LabelEncoder

def run_preprocessing(df):
    """
    Allows user to fix issues and preprocess data.
    Satisfies FR-12 to FR-15.
    """
    st.header("Data Preprocessing")
    
    # --- STATE MANAGEMENT ---
    if 'working_df' not in st.session_state:
        st.session_state['working_df'] = df.copy()
    
    if 'preprocessing_log' not in st.session_state:
        st.session_state['preprocessing_log'] = []

    # Display feedback from previous action if any
    if 'preprocessing_feedback' in st.session_state:
        st.success(st.session_state['preprocessing_feedback'])
        del st.session_state['preprocessing_feedback']

    if st.button("Reset Preprocessing"):
        st.session_state['working_df'] = df.copy()
        st.session_state['preprocessing_log'] = []
        st.rerun()

    df_clean = st.session_state['working_df']
    
    # --- AI SUGGESTIONS ---
    if 'gemini_api_key' in st.session_state and st.session_state['gemini_api_key']:
        with st.expander("🤖 AI Preprocessing Suggestions", expanded=False):
            if st.button("Get AI Suggestions"):
                with st.spinner("Analyzing data for preprocessing suggestions..."):
                    import ai_assistant
                    suggestions = ai_assistant.get_preprocessing_suggestions(df_clean)
                    st.markdown(suggestions)
    
    # --- CURRENT DATA SCHEMA VIEW ---
    with st.expander("🔍 View Current Data Schema (Columns & Types)", expanded=False):
        st.write(f"**Shape:** {df_clean.shape[0]} rows, {df_clean.shape[1]} columns")
        st.dataframe(pd.DataFrame({
            'Column': df_clean.columns,
            'Type': df_clean.dtypes.astype(str)
        }).reset_index(drop=True), use_container_width=True)
    
    # --- 1. HANDLE MISSING VALUES (FR-12) ---
    st.subheader("1. Missing Value Imputation")
    
    # Show all missing values
    missing_summary = df_clean.isnull().sum()
    missing_summary = missing_summary[missing_summary > 0]
    
    if not missing_summary.empty:
        st.error(f"The dataset has missing values in {len(missing_summary)} columns.")
        st.dataframe(pd.DataFrame({'Missing Count': missing_summary, 'Percentage': (missing_summary/len(df_clean)*100).round(2)}))
        
        col_to_impute = st.selectbox("Select Column to Impute", missing_summary.index)
        strategy = st.selectbox("Imputation Method", ["Mean", "Median", "Mode", "Drop Rows"])
        
        if st.button("Apply Imputation"):
            if strategy == "Drop Rows":
                old_len = len(df_clean)
                df_clean.dropna(subset=[col_to_impute], inplace=True)
                new_len = len(df_clean)
                msg = f"Dropped {old_len - new_len} rows with missing values in '{col_to_impute}'."
                st.session_state['preprocessing_log'].append(msg)
            elif strategy == "Mode":
                mode_val = df_clean[col_to_impute].mode()[0]
                df_clean[col_to_impute] = df_clean[col_to_impute].fillna(mode_val)
                msg = f"Imputed '{col_to_impute}' with Mode: {mode_val}"
                st.session_state['preprocessing_log'].append(msg)
            else:
                # Ensure column is numeric for Mean/Median
                if pd.api.types.is_numeric_dtype(df_clean[col_to_impute]):
                    if strategy == "Mean":
                        fill_val = df_clean[col_to_impute].mean()
                    else:
                        fill_val = df_clean[col_to_impute].median()
                    df_clean[col_to_impute] = df_clean[col_to_impute].fillna(fill_val)
                    msg = f"Imputed '{col_to_impute}' with {strategy}: {fill_val:.2f}"
                    st.session_state['preprocessing_log'].append(msg)
                else:
                    st.error(f"Cannot apply {strategy} to non-numeric column '{col_to_impute}'. Please use 'Mode' or 'Drop Rows'.")
                    return None
            
            st.session_state['preprocessing_feedback'] = msg
            st.session_state['working_df'] = df_clean
            st.rerun()
    else:
        st.success("No missing values found!")

    st.markdown("---")

    # --- 2. ENCODING CATEGORICAL VARIABLES (FR-13) ---
    st.subheader("2. Feature Encoding")
    
    # Allow user to convert numeric columns to categorical if needed
    with st.expander("Advanced: Convert Numeric to Categorical"):
        all_cols = df_clean.columns.tolist()
        cols_to_convert = st.multiselect("Select numeric columns to treat as categorical (e.g. Year, ZipCode)", all_cols)
        if st.button("Convert Selected to String"):
            for col in cols_to_convert:
                df_clean[col] = df_clean[col].astype(str)
            st.session_state['preprocessing_feedback'] = f"Converted {cols_to_convert} to string/categorical."
            st.session_state['preprocessing_log'].append(f"Converted columns {cols_to_convert} to categorical type.")
            st.session_state['working_df'] = df_clean
            st.rerun()

    # Allow selection of ANY column for target, but prefer categorical/int
    all_columns = df_clean.columns.tolist()
    categorical_cols = df_clean.select_dtypes(include=['object', 'category']).columns.tolist()
    
    # Target Selection
    target_col = st.selectbox("Select Target Variable (Will be Label Encoded)", all_columns, index=len(all_columns)-1)
    
    # One-Hot Encoding Selection (Exclude Target)
    potential_categorical = [c for c in all_columns if c != target_col and (c in categorical_cols or df_clean[c].nunique() < 20)]
    features_to_encode = st.multiselect("Select Categorical Features to One-Hot Encode", potential_categorical)
    
    if st.button("Apply Encoding"):
        msg = ""
        # Label Encode Target if it's object or user wants to ensure it's int
        if df_clean[target_col].dtype == 'object' or not pd.api.types.is_numeric_dtype(df_clean[target_col]):
            le = LabelEncoder()
            df_clean[target_col] = le.fit_transform(df_clean[target_col].astype(str))
            msg += f"Target '{target_col}' Label Encoded. "
        
        # One-Hot Encode Features
        if features_to_encode:
            # Check for high cardinality
            for col in features_to_encode:
                if df_clean[col].nunique() > 50:
                    st.warning(f"Warning: Column '{col}' has {df_clean[col].nunique()} unique values. One-Hot Encoding this will create many columns.")
            
            # Use dtype=int to ensure 0/1 instead of True/False
            df_clean = pd.get_dummies(df_clean, columns=features_to_encode, drop_first=True, dtype=int)
            msg += f"One-Hot Encoded: {features_to_encode}"
        
        if not msg:
            msg = "No encoding applied (Target was already numeric and no features selected)."
        else:
            st.session_state['preprocessing_log'].append(msg)

        st.session_state['preprocessing_feedback'] = msg
        st.session_state['working_df'] = df_clean
        st.rerun()

    st.markdown("---")

    # --- 3. DROP COLUMNS (New) ---
    st.subheader("3. Drop Unwanted Columns")
    st.caption("Remove columns that are not useful for training (e.g., IDs, Names, Dates with too many unique values).")
    cols_to_drop = st.multiselect("Select Columns to Drop", df_clean.columns)
    if st.button("Drop Selected Columns"):
        if cols_to_drop:
            df_clean.drop(columns=cols_to_drop, inplace=True)
            st.session_state['preprocessing_feedback'] = f"Dropped columns: {cols_to_drop}"
            st.session_state['preprocessing_log'].append(f"Dropped columns: {cols_to_drop}")
            st.session_state['working_df'] = df_clean
            st.rerun()

    st.markdown("---")

    # --- 4. FEATURE SCALING (FR-14) ---
    st.subheader("4. Feature Scaling")
    scaler_type = st.radio("Select Scaling Method", ["None", "StandardScaler", "MinMaxScaler"])
    
    if scaler_type != "None":
        numeric_cols = df_clean.select_dtypes(include=['float64', 'int64']).columns.tolist()
        # Exclude target if it was just encoded to int, usually we scale inputs not target
        # For simplicity, user selects columns to scale
        cols_to_scale = st.multiselect("Select Columns to Scale", numeric_cols)
        
        if st.button("Apply Scaling") and cols_to_scale:
            try:
                if scaler_type == "StandardScaler":
                    scaler = StandardScaler()
                else:
                    scaler = MinMaxScaler()
                
                df_clean[cols_to_scale] = scaler.fit_transform(df_clean[cols_to_scale])
                st.session_state['preprocessing_feedback'] = f"Applied {scaler_type} to {cols_to_scale}"
                st.session_state['preprocessing_log'].append(f"Applied {scaler_type} to columns: {cols_to_scale}")
                
                st.session_state['working_df'] = df_clean
                st.rerun()
            except ValueError as e:
                st.error(f"Scaling failed! Error: {e}")
                st.warning("Tip: Scaling usually fails if there are missing values (NaNs). Please check '1. Missing Value Imputation' above.")
            except Exception as e:
                st.error(f"An error occurred: {e}")

    st.markdown("---")

    # --- FR-15: FINAL SAVE ---
    st.subheader("5. Finalize Preprocessing")
    st.caption("Once you are happy with the changes above, click below to save this dataset for Model Training.")
    
    with st.expander("Preview Processed Data"):
        st.dataframe(df_clean.head())

    if st.button("Save & Proceed to Training"):
        st.success("Dataset saved! Proceed to the Model Training page.")
        return df_clean  # Returns the cleaned dataframe to app.py
    
    return None # Return None if not finalized