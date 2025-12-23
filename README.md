""Where Is My Money Going?"" (Full-Stack Financial Analyzer)


This project is a unique, full-stack data science application designed to transform raw bank statement data into actionable financial insights, complete with an interactive web dashboard and a Machine Learning component for anomaly detection.Based on a suggestion for a unique portfolio project, this demonstrates proficiency in data pipeline construction, SQL database management, statistical analysis, machine Learning, and web deployment (Flask).

Key FeaturesAutomated Data Pipeline (ETL): Cleans, structures, and categorizes messy, real-world financial transaction data using Python and Pandas.
Custom Categorization: Uses rule-based logic to assign transactions into custom buckets (e.g., 'Investment/Transfer', 'Income/Credit').
SQL Database Persistence: Stores the cleaned and categorized data in an SQLite database for efficient querying.
Machine Learning (Anomaly Detection): Implements the Isolation Forest model from Scikit-learn to automatically flag unusually large or rare transactions requiring review.
Interactive Web Dashboard (Flask): Serves dynamic visualizations and analysis tables on a local web server.

Technology Used:
    Backend & Core : LogicPython, Flask 
                        [Application routing and business logic.]
    Data Processing :Pandas, NumPy 
                        [Data cleaning, transformation, and manipulation.]
    Machine Learning : Scikit-learn (Isolation Forest) 
                        [Unsupervised anomaly detection.]
    Database : SQLite  
                        [Persistent storage and complex SQL querying.]
    Visualization : Chart.js, HTML/CSS 
                        [Dynamic, interactive charts displayed in the browser.]

How to Run the Project Prerequisites:
    Python 3.8+
    Install required libraries:Bashpip install pandas flask scikit-learn matplotlib
Steps:
    Place Your Data: Ensure your transaction file, named Transactions.csv, is in the root project directory, matching the file path used in the code.
    Run the Data Pipeline (main.py): This step cleans the data, runs the Anomaly Detection, and populates the finance_tracker.db database.Bashpython main.py (Output will confirm data cleaning, category assignment, and anomaly detection.)
    Run the Web Application (app.py): This starts the Flask server.Bashpython app.py
    View the Dashboard: Open your web browser and navigate to the address provided in the terminal (usually http://127.0.0.1:5000/).

"Key Insights from the Dashboard"
The dashboard provides two main views:
    1. Financial Health ChartsCategory Breakdown (Pie Chart): Shows the distribution of total money flow across all defined categories.
       Monthly Net Flow (Line Chart): Tracks the flow of money (Income minus Spending) over time to identify seasonal trends or major shifts in financial stability.
    2. ML-Driven Anomaly TableA dedicated section (powered by Isolation Forest) highlights transactions that deviate significantly from the norm   (e.g., the large investment/transfer flagged in the test data). This is the key "value-add" feature for a resume project.