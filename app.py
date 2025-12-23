from flask import Flask, render_template, request, redirect, url_for
import sqlite3
import pandas as pd
import json
import os
import numpy as np # For Anomaly Detection
from sklearn.ensemble import IsolationForest # For Anomaly Detection
from werkzeug.utils import secure_filename

app = Flask(__name__)

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
DB_PATH = os.path.join(BASE_DIR, 'finance_tracker.db')

UPLOAD_FOLDER = 'temp_uploads'
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER
if not os.path.exists(UPLOAD_FOLDER):
    os.makedirs(UPLOAD_FOLDER)


def process_file(uploaded_csv_path):
    json_file_path = os.path.join(BASE_DIR, 'categories.json')

    try:
        with open(json_file_path, 'r') as f:
            category_map = json.load(f)

        df = pd.read_csv(uploaded_csv_path)

        df['Amount'] = df['Amount'].astype(str).str.replace(r'[^\d\.\-]', '', regex=True)
        df['Amount'] = pd.to_numeric(df['Amount'], errors='coerce')
        df['Date'] = pd.to_datetime(df['Date'], infer_datetime_format=True, errors='coerce')
        df.dropna(subset=['Amount', 'Date'], inplace=True)
        
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

        X = df[['Amount']].abs().values
        iso_forest = IsolationForest(contamination=0.01, random_state=42)
        iso_forest.fit(X)
        df['Anomaly'] = np.where(iso_forest.predict(X) == -1, 'Yes', 'No')

        conn = sqlite3.connect(DB_PATH)
        df.to_sql('transactions_data', conn, if_exists='replace', index=False)
        conn.close()
        
        return True # Success
    
    except Exception as e:
        print(f"File Processing Error: {e}")
        return False # Failure


def get_category_breakdown():
    conn = sqlite3.connect(DB_PATH)
    breakdown_query = """
    SELECT 
        Category, 
        SUM(Amount) AS Total_Amount
    FROM transactions_data
    GROUP BY Category
    HAVING ABS(SUM(Amount)) > 100
    ORDER BY Total_Amount DESC 
    """
    try:
        df_breakdown = pd.read_sql(breakdown_query, conn)
        conn.close()
        return df_breakdown.to_dict('records')
    except Exception as e:
        conn.close()
        print(f"Database Query Error: {e}")
        return pd.DataFrame()
    
def get_monthly_trends():
    conn = sqlite3.connect(DB_PATH)
    monthly_query = """
    SELECT
        strftime('%Y-%m', Date) AS Year_Month,
        SUM(Amount) AS Net_Flow
    FROM transactions_data
    GROUP BY Year_Month
    ORDER BY Year_Month;
    """
    df_monthly = pd.read_sql(monthly_query, conn)
    conn.close()
    
    return {
        'labels': df_monthly['Year_Month'].tolist(),
        'data': df_monthly['Net_Flow'].tolist()
    }


def get_anomalies():
    conn = sqlite3.connect(DB_PATH)
    anomalies_query = f"""
    SELECT
        Date,
        "Transaction Name",
        Amount,
        Category
    FROM transactions_data
    WHERE Anomaly = 'Yes'
    ORDER BY Amount DESC;
    """
    df_anomalies = pd.read_sql(anomalies_query, conn)
    conn.close()

    return df_anomalies.to_dict('records')

@app.route('/', methods=['GET'])
def dashboard():
    """Renders the main dashboard page with visualizations."""
    breakdown_chart_data = get_category_breakdown()
    monthly_chart_data = get_monthly_trends()
    anomaly_table_data = get_anomalies()

    return render_template(
        'dashboard.html', 
        breakdown=json.dumps(breakdown_chart_data),
        monthly=json.dumps(monthly_chart_data),
        anomalies=anomaly_table_data,
        page_title="Financial Dashboard"
    )


@app.route('/upload', methods=['POST'])
def upload_file():
    """Handles the file upload from the HTML form and triggers processing."""
    
    if 'file' not in request.files:
        return redirect(url_for('dashboard'))

    file = request.files['file']

    if file.filename == '' or not file.filename.endswith('.csv'):
        return redirect(url_for('dashboard'))

    if file:
        filename = secure_filename(file.filename)
        save_path = os.path.join(app.config['UPLOAD_FOLDER'], filename)
        file.save(save_path) 
        
        success = process_file(save_path) 
        os.remove(save_path) 

        if success:
            return redirect(url_for('dashboard'))
        else:
            return "Error processing file. Please check file format and column names."
    
    return redirect(url_for('dashboard'))


if __name__ == '__main__':
    print("Ensure main.py has been run once to initialize the database.")
    app.run(debug=True)