import streamlit as st
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt

def run_eda(df):
    """
    Executes the Exploratory Data Analysis page logic.
    """
    st.subheader("1. Descriptive Statistics")
    st.write(df.describe())
    st.caption("Shows Mean, Median, Std Dev, Min, and Max.")

    st.markdown("---")
    
    st.subheader("2. Correlation Analysis")
    numeric_df = df.select_dtypes(include=['float64', 'int64'])
    
    if not numeric_df.empty:
        corr_matrix = numeric_df.corr()
        fig_corr, ax_corr = plt.subplots(figsize=(10, 8))
        sns.heatmap(corr_matrix, annot=True, cmap='RdBu_r', fmt=".2f", linewidths=0.5, ax=ax_corr, center=0)
        st.pyplot(fig_corr)
        st.caption("Red indicates positive correlation, Blue indicates negative correlation.")
    else:
        st.info("No numerical columns found for correlation heatmap.")

    st.markdown("---")

    st.subheader("3. Distribution Visualization")
    cols = df.columns.tolist()
    if cols:
        selected_col = st.selectbox("Select a column to visualize distribution:", cols)
        if selected_col:
            col1, col2 = st.columns(2)
            with col1:
                st.write(f"**Histogram of {selected_col}**")
                fig_hist, ax_hist = plt.subplots()
                if pd.api.types.is_numeric_dtype(df[selected_col]):
                    sns.histplot(df[selected_col], kde=True, ax=ax_hist, color='skyblue')
                else:
                    sns.countplot(y=df[selected_col], ax=ax_hist, palette='viridis', order=df[selected_col].value_counts().index)
                st.pyplot(fig_hist)
            with col2:
                if pd.api.types.is_numeric_dtype(df[selected_col]):
                    st.write(f"**Boxplot of {selected_col}**")
                    fig_box, ax_box = plt.subplots()
                    sns.boxplot(x=df[selected_col], ax=ax_box, color='lightgreen')
                    st.pyplot(fig_box)
                else:
                    st.write(f"**Value Counts of {selected_col}**")
                    st.dataframe(df[selected_col].value_counts())

    st.markdown("---")

    st.subheader("4. Pairplot (Scatter Matrix)")
    if st.checkbox("Show Pairplot (May be slow for large datasets)"):
        if not numeric_df.empty:
            if len(numeric_df.columns) > 1:
                st.write("Visualizing relationships between numerical features...")
                fig_pair = sns.pairplot(numeric_df.dropna().sample(min(500, len(df))), diag_kind='kde')
                st.pyplot(fig_pair.fig)
            else:
                st.info("Need at least 2 numerical columns for pairplot.")
        else:
            st.info("No numerical columns for pairplot.")

    st.markdown("---")

    st.subheader("5. 3D Scatter Plot")
    st.caption("Visualize relationships between three numerical variables.")
    
    if len(numeric_df.columns) >= 3:
        col1, col2, col3 = st.columns(3)
        x_col = col1.selectbox("X Axis", numeric_df.columns, index=0)
        y_col = col2.selectbox("Y Axis", numeric_df.columns, index=1)
        z_col = col3.selectbox("Z Axis", numeric_df.columns, index=2)
        
        color_col = st.selectbox("Color by (Optional)", [None] + df.columns.tolist())
        
        if st.button("Generate 3D Plot"):
            import plotly.express as px
            fig = px.scatter_3d(df.sample(min(1000, len(df))), x=x_col, y=y_col, z=z_col, color=color_col)
            st.plotly_chart(fig, use_container_width=True)
    else:
        st.info("Need at least 3 numerical columns for 3D scatter plot.")