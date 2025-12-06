import streamlit as st
from fpdf import FPDF
import base64
import pandas as pd
import ai_assistant
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import confusion_matrix
import os

def generate_ai_insights(raw_df, clean_df, results_df, best_model_data, preprocessing_log):
    """
    Generates 'AI-powered' insights based on heuristics and data analysis.
    Uses Gemini if available for a comprehensive report.
    """
    # Extract Feature Importance
    model = best_model_data['model']
    X_test = best_model_data['X_test']
    feature_names = X_test.columns.tolist()
    
    feature_importance_dict = {}
    if hasattr(model, 'feature_importances_'):
        importances = model.feature_importances_
        feature_importance_dict = dict(zip(feature_names, importances))
    elif hasattr(model, 'coef_'):
        importances = model.coef_[0]
        feature_importance_dict = dict(zip(feature_names, importances))
    
    # Sort and get top 5
    sorted_features = sorted(feature_importance_dict.items(), key=lambda item: abs(item[1]), reverse=True)[:5]
    feature_imp_str = "\n".join([f"- {k}: {v:.4f}" for k, v in sorted_features])

    # Check if Gemini is configured
    if 'gemini_api_key' in st.session_state and st.session_state['gemini_api_key']:
        best_model_row = results_df.loc[results_df['F1 Score'].idxmax()]
        
        # Format preprocessing log
        log_str = "\n".join([f"- {item}" for item in preprocessing_log]) if preprocessing_log else "No specific preprocessing steps recorded."

        prompt = f"""
        You are an expert Data Scientist writing a final project report.
        
        Project Summary:
        - Original Data: {raw_df.shape}
        - Cleaned Data: {clean_df.shape}
        - Best Model: {best_model_row['Model']} (F1 Score: {best_model_row['F1 Score']:.3f})
        
        Preprocessing Actions Taken:
        {log_str}
        
        Top 5 Important Features (based on model weights/importance):
        {feature_imp_str}
        
        Full Results:
        {results_df.to_string()}
        
        Write a comprehensive executive summary of this project.
        1. Explain the data transformation (cleaning/preprocessing) in detail.
        2. Analyze the model performance comparison.
        3. Explain WHY the best model likely won.
        4. **Feature Importance Analysis**: Discuss the top features listed above. Explain what they likely represent and why they might be important for prediction in this context.
        5. Provide a final conclusion and recommendation.
        
        Keep the tone professional, educational, and encouraging. Do not use emojis.
        Do not use markdown headers (like ## or ###). Use plain text with clear paragraph breaks.
        Explain your reasoning clearly.
        """
        response = ai_assistant.get_gemini_response(prompt)
        return [response], sorted_features

    # Fallback Heuristic
    insights = []
    
    # 1. Data Transformation Insights
    rows_dropped = raw_df.shape[0] - clean_df.shape[0]
    cols_dropped = raw_df.shape[1] - clean_df.shape[1]
    
    if rows_dropped > 0:
        insights.append(f"Data Cleaning: {rows_dropped} rows were removed during preprocessing (likely due to missing values). This ensures the model trains on high-quality data.")
    else:
        insights.append("Data Cleaning: No rows were dropped, indicating a complete dataset or successful imputation.")
        
    if cols_dropped > 0:
        insights.append(f"Feature Selection: {cols_dropped} columns were removed. Reducing dimensionality helps prevent overfitting and speeds up training.")
    
    # 2. Model Performance Insights
    best_model_row = results_df.loc[results_df['F1 Score'].idxmax()]
    best_model_name = best_model_row['Model']
    best_f1 = best_model_row['F1 Score']
    best_acc = best_model_row['Accuracy']
    
    insights.append(f"Champion Model: The **{best_model_name}** is the top performer with an F1 Score of **{best_f1:.3f}**.")
    
    if best_f1 > 0.9:
        insights.append("Excellent Performance: The model shows exceptional predictive power. It is ready for deployment in critical systems.")
    elif best_f1 > 0.75:
        insights.append("Strong Performance: The model is reliable for most business use cases.")
    else:
        insights.append("Moderate Performance: The model has room for improvement. Consider collecting more data or engineering new features.")
        
    # 3. Comparative Analysis
    worst_model_row = results_df.loc[results_df['F1 Score'].idxmin()]
    diff = best_f1 - worst_model_row['F1 Score']
    insights.append(f"Model Comparison: The best model outperformed the lowest performing model ({worst_model_row['Model']}) by **{diff:.1%}**. This highlights the importance of testing multiple algorithms.")

    return insights, sorted_features

def generate_report(raw_df, clean_df, results_df, best_model_data, preprocessing_log=None):
    """
    Generates an interactive report and offers a PDF download.
    Satisfies Final Report User Story.
    """
    if preprocessing_log is None:
        preprocessing_log = []

    st.header("Final Project Report")
    st.markdown("This report summarizes the entire AutoML pipeline, from raw data to the final model.")
    
    # --- ON-SCREEN REPORT ---
    
    # 1. Dataset Overview
    st.subheader("1. Dataset Transformation")
    col1, col2 = st.columns(2)
    with col1:
        st.metric("Original Rows", raw_df.shape[0])
        st.metric("Original Columns", raw_df.shape[1])
    with col2:
        st.metric("Processed Rows", clean_df.shape[0])
        st.metric("Processed Columns", clean_df.shape[1])
        
    # 2. AI Insights
    st.subheader("2. 🤖 AI Insights & Analysis")
    st.info("These insights are generated based on the analysis of your data pipeline and model results.")
    
    with st.spinner("AI is analyzing your results..."):
        insights, top_features = generate_ai_insights(raw_df, clean_df, results_df, best_model_data, preprocessing_log)
        for insight in insights:
            st.markdown(insight)
            
    # 3. Feature Importance
    st.subheader("3. Key Drivers (Feature Importance)")
    st.caption("These features had the most impact on the model's predictions.")
    if top_features:
        fi_df = pd.DataFrame(top_features, columns=['Feature', 'Importance'])
        st.dataframe(fi_df)
    else:
        st.info("Feature importance not available for this model type.")
        
    # 4. Leaderboard
    st.subheader("4. Model Leaderboard")
    st.dataframe(results_df.style.highlight_max(axis=0, color='#e6f3ff'))
    
    # --- PDF GENERATION ---
    st.markdown("---")
    st.subheader("5. Export Report")
    st.write("Download a formal PDF version of this report for your records.")
    
    if st.button("Generate PDF Report"):
        try:
            # Initialize PDF
            pdf = FPDF()
            pdf.set_auto_page_break(auto=True, margin=15)
            pdf.add_page()
            
            # --- Title Page ---
            pdf.set_font("Arial", 'B', 24)
            pdf.cell(0, 20, txt="AutoML Classification Report", ln=True, align='C')
            pdf.ln(10)
            
            pdf.set_font("Arial", size=12)
            pdf.cell(0, 8, txt="Generated by AutoML Streamlit App", ln=True, align='C')
            pdf.set_font("Arial", 'I', 10)
            pdf.cell(0, 8, txt="Made by Asma Imran and Fatima Tuz Zahra", ln=True, align='C')
            pdf.cell(0, 8, txt="NUST, Pakistan - Machine Learning Project", ln=True, align='C')
            pdf.ln(20)
            
            # --- 1. Dataset Overview ---
            pdf.set_font("Arial", 'B', 16)
            pdf.cell(0, 10, txt="1. Dataset Overview", ln=True, align='L')
            pdf.ln(5)
            
            pdf.set_font("Arial", size=11)
            pdf.cell(0, 8, txt=f"Original Dataset: {raw_df.shape[0]} rows, {raw_df.shape[1]} columns", ln=True)
            pdf.cell(0, 8, txt=f"Processed Dataset: {clean_df.shape[0]} rows, {clean_df.shape[1]} columns", ln=True)
            pdf.ln(10)
            
            # --- 2. AI Analysis & Insights ---
            pdf.set_font("Arial", 'B', 16)
            pdf.cell(0, 10, txt="2. AI Analysis & Insights", ln=True, align='L')
            pdf.ln(5)
            
            pdf.set_font("Arial", size=11)
            for insight in insights:
                # Clean text
                clean_text = insight.replace("**", "").replace("🚀 ", "").replace("✅ ", "").replace("⚠️ ", "")
                
                # Split into paragraphs
                paragraphs = clean_text.split('\n')
                for p in paragraphs:
                    if p.strip():
                        # Check if it's a header (heuristic)
                        if len(p) < 50 and p.strip().endswith(':'):
                             pdf.set_font("Arial", 'B', 12)
                             pdf.ln(5)
                             pdf.multi_cell(0, 8, txt=p.strip())
                             pdf.set_font("Arial", size=11)
                        else:
                             pdf.multi_cell(0, 6, txt=p.strip())
                             pdf.ln(2)
            pdf.ln(10)

            # --- 3. Feature Importance ---
            pdf.add_page()
            pdf.set_font("Arial", 'B', 16)
            pdf.cell(0, 10, txt="3. Key Drivers (Feature Importance)", ln=True, align='L')
            pdf.ln(5)
            pdf.set_font("Arial", size=11)
            pdf.multi_cell(0, 8, txt="The chart below shows the top features that influenced the model's predictions. Higher bars indicate greater importance.")
            pdf.ln(5)
            
            # Generate Feature Importance Plot
            if top_features:
                try:
                    feat_names = [x[0] for x in top_features]
                    feat_vals = [x[1] for x in top_features]
                    
                    plt.figure(figsize=(8, 4))
                    sns.barplot(x=feat_vals, y=feat_names, palette='viridis')
                    plt.title("Top 5 Feature Importances")
                    plt.xlabel("Importance Score")
                    plt.tight_layout()
                    plt.savefig("temp_feat_imp.png")
                    plt.close()
                    
                    pdf.image("temp_feat_imp.png", x=10, w=170)
                    os.remove("temp_feat_imp.png")
                    pdf.ln(5)
                except Exception as e:
                    pdf.cell(0, 8, txt=f"Could not generate plot: {e}", ln=True)

            if top_features:
                for feat, imp in top_features:
                    pdf.cell(0, 8, txt=f"- {feat}: {imp:.4f}", ln=True)
            else:
                pdf.cell(0, 8, txt="Feature importance not available for this model.", ln=True)
            pdf.ln(10)

            # --- 4. Model Performance Leaderboard ---
            pdf.set_font("Arial", 'B', 16)
            pdf.cell(0, 10, txt="4. Model Performance Leaderboard", ln=True, align='L')
            pdf.ln(5)
            
            # Generate Confusion Matrix Plot
            try:
                model = best_model_data['model']
                X_test = best_model_data['X_test']
                y_test = best_model_data['y_test']
                y_pred = model.predict(X_test)
                cm = confusion_matrix(y_test, y_pred)
                
                plt.figure(figsize=(6, 5))
                sns.heatmap(cm, annot=True, fmt='d', cmap='Blues')
                plt.title(f"Confusion Matrix ({best_model_data['name']})")
                plt.ylabel('Actual')
                plt.xlabel('Predicted')
                plt.tight_layout()
                plt.savefig("temp_cm.png")
                plt.close()
                
                pdf.image("temp_cm.png", x=50, w=100)
                os.remove("temp_cm.png")
                pdf.ln(5)
                pdf.set_font("Arial", size=11)
                pdf.multi_cell(0, 8, txt="The Confusion Matrix above visualizes the performance of the best model. The diagonal elements represent correctly classified instances, while off-diagonal elements represent errors.")
                pdf.ln(5)
            except Exception as e:
                pdf.cell(0, 8, txt=f"Could not generate confusion matrix: {e}", ln=True)

            pdf.set_font("Arial", 'B', 10)
            # Table Header, txt="4. Model Performance Leaderboard", ln=True, align='L')
            pdf.ln(5)
            
            pdf.set_font("Arial", 'B', 10)
            # Table Header
            col_width = 45
            pdf.cell(col_width, 10, "Model", 1)
            pdf.cell(col_width, 10, "Accuracy", 1)
            pdf.cell(col_width, 10, "F1 Score", 1)
            pdf.ln()
            
            # Table Rows
            pdf.set_font("Arial", size=10)
            for index, row in results_df.iterrows():
                pdf.cell(col_width, 10, str(row['Model']), 1)
                pdf.cell(col_width, 10, f"{row['Accuracy']:.4f}", 1)
                pdf.cell(col_width, 10, f"{row['F1 Score']:.4f}", 1)
                pdf.ln()
            
            pdf.ln(10)
            
            # --- 5. Conclusion ---
            best_model_name = results_df.loc[results_df['F1 Score'].idxmax()]['Model']
            pdf.set_font("Arial", 'B', 16)
            pdf.cell(0, 10, txt="5. Conclusion", ln=True, align='L')
            pdf.ln(5)
            
            pdf.set_font("Arial", size=11)
            pdf.multi_cell(0, 8, txt=f"Based on the evaluation metrics, the {best_model_name} model demonstrated the strongest performance. It achieved the highest F1 Score, indicating a good balance between precision and recall. This model is recommended for deployment.")
            
            # Output PDF to bytes
            pdf_bytes = pdf.output(dest='S').encode('latin-1')
            b64 = base64.b64encode(pdf_bytes).decode()
            href = f'<a href="data:application/octet-stream;base64,{b64}" download="AutoML_Report.pdf">Download PDF Report</a>'
            st.markdown(href, unsafe_allow_html=True)
            
        except Exception as e:
            st.error(f"Error generating PDF: {e}")
            pdf.ln(5)
            
            # 2. Insights
            pdf.set_font("Arial", 'B', 14)
            pdf.cell(200, 10, txt="2. AI Analysis & Insights", ln=True, align='L')
            pdf.set_font("Arial", size=11)
            for insight in insights:
                # Strip markdown asterisks for PDF
                clean_text = insight.replace("**", "").replace("🚀 ", "").replace("✅ ", "").replace("⚠️ ", "")
                pdf.multi_cell(0, 8, txt=f"- {clean_text}")
            pdf.ln(5)
            
            # 3. Model Performance
            pdf.set_font("Arial", 'B', 14)
            pdf.cell(200, 10, txt="3. Model Performance Leaderboard", ln=True, align='L')
            pdf.set_font("Arial", size=10)
            
            # Table Header
            pdf.cell(40, 10, "Model", 1)
            pdf.cell(30, 10, "Accuracy", 1)
            pdf.cell(30, 10, "F1 Score", 1)
            pdf.ln()
            
            # Table Rows
            for index, row in results_df.iterrows():
                pdf.cell(40, 10, str(row['Model']), 1)
                pdf.cell(30, 10, f"{row['Accuracy']:.3f}", 1)
                pdf.cell(30, 10, f"{row['F1 Score']:.3f}", 1)
                pdf.ln()
            
            pdf.ln(10)
            
            # 4. Conclusion
            best_model_name = results_df.loc[results_df['F1 Score'].idxmax()]['Model']
            pdf.set_font("Arial", 'B', 14)
            pdf.cell(200, 10, txt="4. Conclusion", ln=True, align='L')
            pdf.set_font("Arial", size=12)
            pdf.multi_cell(0, 10, txt=f"The best performing model for this dataset was {best_model_name}. It achieved the highest F1 Score and is recommended for deployment.")
            
            # Output PDF to bytes
            pdf_bytes = pdf.output(dest='S').encode('latin-1')
            b64 = base64.b64encode(pdf_bytes).decode()
            href = f'<a href="data:application/octet-stream;base64,{b64}" download="AutoML_Report.pdf">Download PDF Report</a>'
            st.markdown(href, unsafe_allow_html=True)
            
        except Exception as e:
            st.error(f"Error generating PDF: {e}")