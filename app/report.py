import streamlit as st
from fpdf import FPDF
import base64
import pandas as pd
import ai_assistant

def generate_ai_insights(raw_df, clean_df, results_df):
    """
    Generates 'AI-powered' insights based on heuristics and data analysis.
    Uses Gemini if available for a comprehensive report.
    """
    # Check if Gemini is configured
    if 'gemini_api_key' in st.session_state and st.session_state['gemini_api_key']:
        best_model_row = results_df.loc[results_df['F1 Score'].idxmax()]
        
        prompt = f"""
        You are an expert Data Scientist writing a final project report.
        
        Project Summary:
        - Original Data: {raw_df.shape}
        - Cleaned Data: {clean_df.shape}
        - Best Model: {best_model_row['Model']} (F1 Score: {best_model_row['F1 Score']:.3f})
        
        Full Results:
        {results_df.to_string()}
        
        Write a comprehensive executive summary of this project.
        1. Explain the data transformation (cleaning/preprocessing).
        2. Analyze the model performance comparison.
        3. Explain WHY the best model likely won (e.g., Random Forest handles complex non-linear data better than Logistic Regression).
        4. Provide a final conclusion and recommendation.
        
        Keep the tone professional, educational, and encouraging. Do not use emojis.
        Explain your reasoning clearly: "We can deduce that..."
        """
        response = ai_assistant.get_gemini_response(prompt)
        return [response]

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

    return insights

def generate_report(raw_df, clean_df, results_df):
    """
    Generates an interactive report and offers a PDF download.
    Satisfies Final Report User Story.
    """
    st.header("📊 Final Project Report")
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
        insights = generate_ai_insights(raw_df, clean_df, results_df)
        for insight in insights:
            st.markdown(insight)
        
    # 3. Leaderboard
    st.subheader("3. Model Leaderboard")
    st.dataframe(results_df.style.highlight_max(axis=0, color='lightgreen'))
    
    # --- PDF GENERATION ---
    st.markdown("---")
    st.subheader("4. Export Report")
    st.write("Download a formal PDF version of this report for your records.")
    
    if st.button("Generate PDF Report"):
        try:
            # Initialize PDF
            pdf = FPDF()
            pdf.add_page()
            pdf.set_font("Arial", size=12)
            
            # Title
            pdf.set_font("Arial", 'B', 16)
            pdf.cell(200, 10, txt="AutoML Classification Report", ln=True, align='C')
            pdf.ln(10)
            
            # 1. Dataset Overview
            pdf.set_font("Arial", 'B', 14)
            pdf.cell(200, 10, txt="1. Dataset Overview", ln=True, align='L')
            pdf.set_font("Arial", size=12)
            pdf.cell(200, 10, txt=f"Original Rows: {raw_df.shape[0]}, Columns: {raw_df.shape[1]}", ln=True)
            pdf.cell(200, 10, txt=f"Processed Rows: {clean_df.shape[0]}, Columns: {clean_df.shape[1]}", ln=True)
            pdf.ln(5)
            
            # 2. Insights
            pdf.set_font("Arial", 'B', 14)
            pdf.cell(200, 10, txt="2. AI Analysis & Insights", ln=True, align='L')
            pdf.set_font("Arial", size=11)
            for insight in insights:
                # Strip markdown asterisks for PDF
                clean_text = insight.replace("**", "").replace("🚀 ", "").replace("✅ ", "").replace("⚠️ ", "")
                # Handle newlines from Gemini
                for line in clean_text.split('\n'):
                    if line.strip():
                        pdf.multi_cell(0, 8, txt=line)
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