import streamlit as st
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay

def show_dashboard(results_df, best_model_data):
    """
    Displays the model evaluation dashboard.
    Satisfies FR-20, FR-21, FR-22.
    """
    st.header("Model Evaluation Dashboard")
    
    # --- FR-20: Performance Leaderboard ---
    st.subheader("1. Model Performance Leaderboard")
    
    # Highlight the best model row (max F1 Score)
    best_idx = results_df['F1 Score'].idxmax()
    
    def highlight_best_row(row):
        if row.name == best_idx:
            return ['background-color: #e6f3ff'] * len(row)
        return [''] * len(row)

    st.dataframe(results_df.sort_values(by="F1 Score", ascending=False).style.apply(highlight_best_row, axis=1))
    
    # --- FR-21: Confusion Matrix for Best Model ---
    st.subheader(f"2. Confusion Matrix ({best_model_data['name']})")
    
    model = best_model_data['model']
    X_test = best_model_data['X_test']
    y_test = best_model_data['y_test']
    
    # Predict
    y_pred = model.predict(X_test)
    
    # Generate Matrix
    cm = confusion_matrix(y_test, y_pred)
    
    # Plot
    fig_cm, ax_cm = plt.subplots()
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', ax=ax_cm)
    ax_cm.set_xlabel('Predicted')
    ax_cm.set_ylabel('Actual')
    st.pyplot(fig_cm)
    
    # --- FR-22: Download Model ---
    st.subheader("3. Export Model")
    import pickle
    
    # Serialize model
    model_pkl = pickle.dumps(model)
    
    st.download_button(
        label="Download Best Model (.pkl)",
        data=model_pkl,
        file_name=f"best_model_{best_model_data['name']}.pkl",
        mime="application/octet-stream"
    )

    st.markdown("---")

    # --- Feature Importance (New Feature) ---
    st.subheader("4. Feature Importance")
    if hasattr(model, 'feature_importances_'):
        importances = model.feature_importances_
        feature_names = X_test.columns
        feature_imp_df = pd.DataFrame({'Feature': feature_names, 'Importance': importances})
        feature_imp_df = feature_imp_df.sort_values(by='Importance', ascending=False)
        
        fig_imp, ax_imp = plt.subplots()
        sns.barplot(x='Importance', y='Feature', data=feature_imp_df, ax=ax_imp, palette='Blues_r')
        ax_imp.set_title(f"Feature Importance for {best_model_data['name']}")
        st.pyplot(fig_imp)
    elif hasattr(model, 'coef_'):
        # For linear models
        importances = model.coef_[0]
        feature_names = X_test.columns
        feature_imp_df = pd.DataFrame({'Feature': feature_names, 'Importance': importances})
        feature_imp_df['Abs_Importance'] = feature_imp_df['Importance'].abs()
        feature_imp_df = feature_imp_df.sort_values(by='Abs_Importance', ascending=False)
        
        fig_imp, ax_imp = plt.subplots()
        sns.barplot(x='Importance', y='Feature', data=feature_imp_df, ax=ax_imp, palette='Blues_r')
        ax_imp.set_title(f"Feature Coefficients for {best_model_data['name']}")
        st.pyplot(fig_imp)
    else:
        st.info(f"Feature importance not available for {best_model_data['name']}.")

    st.markdown("---")

    # --- Prediction Playground (New Feature) ---
    st.subheader("5. Prediction Playground")
    st.caption("Test the model with your own values!")
    
    with st.form("prediction_form"):
        input_data = {}
        cols = st.columns(2)
        for i, col_name in enumerate(X_test.columns):
            # Try to infer type from X_test
            if pd.api.types.is_numeric_dtype(X_test[col_name]):
                min_val = float(X_test[col_name].min())
                max_val = float(X_test[col_name].max())
                mean_val = float(X_test[col_name].mean())
                input_data[col_name] = cols[i % 2].number_input(f"{col_name}", value=mean_val)
            else:
                # Should be encoded, but if not...
                unique_vals = X_test[col_name].unique()
                input_data[col_name] = cols[i % 2].selectbox(f"{col_name}", unique_vals)
        
        submit_btn = st.form_submit_button("Predict")
        
        if submit_btn:
            input_df = pd.DataFrame([input_data])
            prediction = model.predict(input_df)[0]
            
            if hasattr(model, "predict_proba"):
                proba = model.predict_proba(input_df)[0]
                confidence = max(proba) * 100
                st.success(f"Prediction: **{prediction}** (Confidence: {confidence:.2f}%)")
            else:
                st.success(f"Prediction: **{prediction}**")