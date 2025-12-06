import pandas as pd
import numpy as np
import random
from datetime import datetime, timedelta

def generate_complex_dataset(n_rows=1000):
    np.random.seed(42)
    random.seed(42)

    # 1. Numerical Features
    # Age: Normal distribution, but with some outliers and missing values
    age = np.random.normal(loc=35, scale=10, size=n_rows)
    age = np.append(age, [120, 150, -5]) # Outliers
    age = age[:n_rows] # Trim back to size
    age[np.random.choice(n_rows, size=50, replace=False)] = np.nan # Missing

    # Income: Log-normal (skewed), with missing
    income = np.random.lognormal(mean=10.5, sigma=0.8, size=n_rows)
    income[np.random.choice(n_rows, size=30, replace=False)] = np.nan

    # Credit Score: Correlated with Income
    credit_score = (income / 200) + np.random.normal(300, 50, n_rows)
    credit_score = np.clip(credit_score, 300, 850)

    # 2. Categorical Features
    # Education: Ordinal
    education_levels = ['High School', 'Bachelor', 'Master', 'PhD', 'High School', 'Bachelor']
    education = np.random.choice(education_levels, size=n_rows)

    # City: Nominal
    cities = ['New York', 'Los Angeles', 'Chicago', 'Houston', 'Phoenix', 'Unknown']
    city = np.random.choice(cities, size=n_rows, p=[0.3, 0.2, 0.15, 0.15, 0.1, 0.1])

    # Employment: Categorical with missing
    employment = np.random.choice(['Employed', 'Unemployed', 'Self-Employed'], size=n_rows)
    employment[np.random.choice(n_rows, size=100, replace=False)] = np.nan # Missing as NaN
    
    # 3. Date Feature
    # Join Date
    start_date = datetime(2015, 1, 1)
    dates = [start_date + timedelta(days=random.randint(0, 3000)) for _ in range(n_rows)]

    # 4. Target Variable (Binary Classification)
    # Probability of default depends on Credit Score and Age
    # Lower credit score -> Higher default probability
    prob_default = 1 / (1 + np.exp((credit_score - 600) / 50)) 
    default = [1 if random.random() < p else 0 for p in prob_default]

    # Create DataFrame
    df = pd.DataFrame({
        'Customer_ID': range(1001, 1001 + n_rows), # ID column (should be dropped usually)
        'Age': age,
        'Annual_Income': income,
        'Credit_Score': credit_score,
        'Education': education,
        'City': city,
        'Employment_Status': employment,
        'Join_Date': dates,
        'Loan_Default': default
    })

    # Add Duplicate Rows
    df = pd.concat([df, df.iloc[:20]], ignore_index=True)

    return df

if __name__ == "__main__":
    print("Generating complex dataset...")
    df = generate_complex_dataset(1000)
    df.to_csv('app/complex_test_dataset.csv', index=False)
    print(f"Dataset created: app/complex_test_dataset.csv with shape {df.shape}")
