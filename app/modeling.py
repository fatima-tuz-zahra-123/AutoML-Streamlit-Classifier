import streamlit as st
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC
from sklearn.neighbors import KNeighborsClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score

def train_models(df):
    """
    Trains multiple classification models.
    Satisfies FR-16, FR-17, FR-18, FR-19.
    """
    st.header("Model Training Configuration")
    
    # --- FR-16: Target Selection ---
    columns = df.columns.tolist()
    target_col = st.selectbox("Select Target Variable (Y)", columns, index=len(columns)-1)
    
    # Feature Selection (X)
    feature_cols = [c for c in columns if c != target_col]
    
    # --- AI SUGGESTIONS ---
    if 'gemini_api_key' in st.session_state and st.session_state['gemini_api_key']:
        with st.expander("🤖 AI Model Training Suggestions", expanded=False):
            if st.button("Get AI Model Suggestions"):
                with st.spinner("Analyzing data for model suggestions..."):
                    import ai_assistant
                    suggestions = ai_assistant.get_modeling_suggestions(df, target_col)
                    st.markdown(suggestions)

    # --- FR-17: Data Splitting ---
    test_size = st.slider("Test Set Size", 0.1, 0.5, 0.2, 0.05)
    
    if st.button("Start Training"):
        st.write("### Training in progress...")
        progress_bar = st.progress(0)
        status_text = st.empty()
        
        # Prepare Data
        X = df[feature_cols]
        y = df[target_col]
        
        # Check for non-numeric columns in X
        non_numeric_cols = X.select_dtypes(exclude=['number']).columns.tolist()
        if non_numeric_cols:
            st.error(f"The following feature columns are not numeric: {non_numeric_cols}. Please encode them in the Preprocessing step.")
            progress_bar.empty()
            return None, None

        # Split Data
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=test_size, random_state=42)
        
        # Define Models (FR-18)
        models = {
            "Logistic Regression": LogisticRegression(max_iter=1000),
            "Decision Tree": DecisionTreeClassifier(),
            "Random Forest": RandomForestClassifier(),
            "SVM": SVC(probability=True),
            "K-Nearest Neighbors": KNeighborsClassifier(),
            "Naive Bayes": GaussianNB()
        }
        
        results = []
        trained_models = {}
        
        # Loop through models
        for i, (name, model) in enumerate(models.items()):
            status_text.text(f"Training {name}...")
            try:
                # Train
                model.fit(X_train, y_train)
                
                # Predict
                y_pred = model.predict(X_test)
                
                # Evaluate
                acc = accuracy_score(y_test, y_pred)
                f1 = f1_score(y_test, y_pred, average='weighted')
                prec = precision_score(y_test, y_pred, average='weighted', zero_division=0)
                rec = recall_score(y_test, y_pred, average='weighted', zero_division=0)
                
                results.append({
                    "Model": name,
                    "Accuracy": acc,
                    "F1 Score": f1,
                    "Precision": prec,
                    "Recall": rec
                })
                
                # Store the actual trained model object for later use
                trained_models[name] = model
                
            except Exception as e:
                st.error(f"Failed to train {name}: {e}")
            
            # Update progress
            progress_bar.progress((i + 1) / len(models))
        
        progress_bar.empty()
        status_text.text("Training Complete!")
        st.success("All models trained successfully!")
        
        # Convert results to DataFrame
        results_df = pd.DataFrame(results)
        
        # Identify Best Model (based on F1 Score)
        best_model_name = results_df.loc[results_df['F1 Score'].idxmax()]['Model']
        best_model_obj = trained_models[best_model_name]
        
        st.subheader("Leaderboard")
        st.dataframe(results_df.sort_values(by="F1 Score", ascending=False))
        
        st.success(f"Best Model Selected: **{best_model_name}**")
        
        # Return results and the best model object (needed for Evaluation tab)
        return results_df, {
            "name": best_model_name,
            "model": best_model_obj,
            "X_test": X_test,
            "y_test": y_test
        }
    
    return None, None