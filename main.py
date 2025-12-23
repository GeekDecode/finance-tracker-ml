# PHASE 1: Data Ingestion and Robust Cleaning

import pandas as pd
import os 
import sys 
import json
import sqlite3
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.ensemble import IsolationForest

CSV_FILE = 'Transactions.csv'
JSON_FILE = 'categories.json'
DB_FILE = 'finance_tracker.db'
TABLE_NAME = 'transactions_data'

PROJECT_ROOT = os.path.dirname(os.path.abspath(__file__))
csv_file_path = os.path.join(PROJECT_ROOT, CSV_FILE)
json_file_path = os.path.join(PROJECT_ROOT, JSON_FILE)
database_path = os.path.join(PROJECT_ROOT, DB_FILE)


print("Starting Data Load and Cleaning...")

if not os.path.exists(csv_file_path):
    print(f"FATAL ERROR: Transaction file not found at path: {csv_file_path}")
    print("Please ensure your CSV file is named 'Transactions.csv' and is in the project root.")
    sys.exit(1) # Exit if the core data file is missing
else:
    try:
        df = pd.read_csv(csv_file_path)
    
        df['Amount'] = df['Amount'].astype(str).str.replace(r'[^\d\.\-]', '', regex=True)
        df['Amount'] = pd.to_numeric(df['Amount'], errors='coerce')
        
        df['Date'] = pd.to_datetime(df['Date'], infer_datetime_format=True)
        
        df.dropna(subset=['Amount'], inplace=True)
        
        print("\n--- SUCCESS: Data Loaded and Cleaned! ---")
        print("Data Types after cleaning:")
        print(df.dtypes)
        print("\nFirst few rows after cleaning:")
        print(df[['Date', 'Transaction Name', 'Amount']].head()) 
        
    except Exception as e:
        print(f"An error occurred while reading or processing the CSV file: {e}")
        sys.exit(1)


# PHASE 2: Categorization & Anomaly Detection

try:
    with open(json_file_path, 'r') as f:
        category_map = json.load(f)
except FileNotFoundError:
    print(f"FATAL ERROR: Configuration file {JSON_FILE} not found at {json_file_path}. Cannot categorize.")
    sys.exit(1)


def assign_category(description):
    if pd.isna(description):
        return 'Missing'
    
    desc = str(description).upper()
    for category, keywords in category_map.items():
        for keyword in keywords:
            if keyword in desc:
                return category
    return 'Miscellaneous'

df['Category'] = df['Transaction Name'].apply(assign_category) 

print("\n--- Starting Anomaly Detection (Isolation Forest) ---")
X = df[['Amount']].abs().values 


iso_forest= IsolationForest(contamination=0.01, random_state=42)
iso_forest.fit(X)

df['Anomaly_Flag']=iso_forest.predict(X)
df['Anomaly']=np.where(df['Anomaly_Flag']==-1,'Yes','No')

anomalies=df[df['Anomaly']=='Yes'].sort_values(by='Amount', ascending=False)

print(f"Total Anomalies Detected: {len(anomalies)}")
if not anomalies.empty:
    print("\nTop Anomalies to Review:")
    print(anomalies[['Date', 'Transaction Name', 'Amount', 'Category', 'Anomaly']].head())

print("\n--- SUCCESS: Categorization and ML Complete! ---")
print("Here are the first 10 rows with the new Category column:")
print(df[['Date', 'Transaction Name', 'Amount', 'Category']].head(10))



# PHASE 3: SQL Database Storage (SQLite)

print("\n--- Starting SQL Storage ---")

try:
    # Use the robust database path
    conn= sqlite3.connect(database_path)
    df.to_sql(TABLE_NAME, conn, if_exists='replace', index=False)

    check_query= f"SELECT COUNT(*) FROM {TABLE_NAME}"
    count = pd.read_sql(check_query, conn).iloc[0,0]

    conn.close()

    print(f"SUCCESS: data written to '{DB_FILE}' in table '{TABLE_NAME}'.")
    print(f"Total records stored: {count}")

except Exception as e:
    print(f"FATAL ERROR in SQL storage: {e}")
    sys.exit(1)

# PHASE 4: Visualization

print("\n--- Starting Data Analysis and Visualization ---")

try:

    conn=sqlite3.connect(database_path)
    
    breakdown_query= f"""
    SELECT Category, SUM(Amount) AS Total_Amount
    FROM {TABLE_NAME}
    GROUP BY CATEGORY
    ORDER BY Total_Amount DESC;
    """
    df_breakdown= pd.read_sql(breakdown_query, conn)

    # Query for Monthly Trends
    monthly_query= f"""
    SELECT strftime('%Y-%m' , Date) AS Year_Month, SUM(Amount) AS Net_Flow
    FROM {TABLE_NAME}
    GROUP BY Year_Month
    ORDER BY Year_Month;
    """
    df_monthly= pd.read_sql(monthly_query, conn)

    conn.close()

except Exception as e:
    print(f"FATAL ERROR in Visualization Query: {e}")
    sys.exit(1)


df_plot= df_breakdown[df_breakdown['Total_Amount'].abs()>100]
plt.figure(figsize=(10,7))
plt.pie(
    df_plot['Total_Amount'].abs(),
    labels=df_plot['Category'],
    autopct='%.1f%%', # Corrected format string
    startangle=90,
    wedgeprops={'edgecolor':'black'}
)
plt.title("Money Breakdown")
plt.axis('equal')
plt.show()

plt.figure(figsize=(12, 6))
sns.lineplot(
    x='Year_Month',
    y='Net_Flow',
    data=df_monthly,
    marker='o'
)
plt.title('Net Money Flow Over Time (Income vs. Spending)')
plt.xlabel('Month')
plt.ylabel('Net Flow Amount')
plt.grid(True)
plt.xticks(rotation=45, ha='right')
plt.tight_layout()
plt.show()

print("\n--- Visualizations Generated! ---")