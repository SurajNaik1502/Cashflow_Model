from flask import Flask, request, render_template, jsonify, make_response
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from xgboost import XGBRegressor
import csv
import io
import os
import pickle

app = Flask(__name__)
# Set the upload folder to 'Dataset'
app.config['UPLOAD_FOLDER'] = 'Dataset'
os.makedirs(app.config['UPLOAD_FOLDER'], exist_ok=True)

# Dictionary to store trained models for each group
trained_models = {}

# Function for training the forecasting models for all valid groups
def train_forecasting_models(df):
    global trained_models
    grouped_df = df.groupby(['project', 'ledger', 'company']).size()
    valid_groups = grouped_df[grouped_df > 37].index

    for group in valid_groups:
        specific_group_data = df[
            (df['project'] == group[0]) & 
            (df['ledger'] == group[1]) & 
            (df['company'] == group[2])
        ].copy()

        specific_group_data['date'] = pd.to_datetime(specific_group_data['date'])
        specific_group_data.sort_values('date', inplace=True)
        specific_group_data['target'] = specific_group_data['dr_cr']

        num_lags = 10
        for i in range(1, num_lags + 1):
            specific_group_data[f'lag_{i}'] = specific_group_data['dr_cr'].shift(i)
        
        specific_group_data.dropna(inplace=True)
        feature_columns = [col for col in specific_group_data.columns if col.startswith('lag_')]
        X = specific_group_data[feature_columns]
        y = specific_group_data['target']
        
        X_train, _, y_train, _ = train_test_split(X, y, test_size=0.2, shuffle=False)

        model = XGBRegressor(objective='reg:squarederror')
        model.fit(X_train, y_train)

        # Save the trained model
        trained_models[group] = {
            'model': model,
            'last_data': X.iloc[-1].values
        }

# Function to generate forecasts for all groups
def generate_forecasts(start_date, end_date):
    forecast_results = []
    forecast_dates = pd.date_range(start=start_date, end=end_date, freq='D')

    for group, data in trained_models.items():
        model = data['model']
        last_data = data['last_data'].copy()

        group_forecast = []
        for _ in forecast_dates:
            next_value = model.predict(last_data.reshape(1, -1))[0]
            group_forecast.append(float(next_value))
            last_data = np.roll(last_data, shift=-1)
            last_data[-1] = next_value

        for date, forecast in zip(forecast_dates, group_forecast):
            forecast_results.append({
                "project": group[0],
                "ledger": group[1],
                "company": group[2],
                "date": date.strftime('%Y-%m-%d'),
                "forecast": forecast
            })

    return forecast_results

@app.route('/')
def home():
    return render_template('train.html')

@app.route('/train', methods=['POST'])
def train():
    if 'file' not in request.files:
        return "Error: No file uploaded. Please upload a CSV file.", 400

    file = request.files['file']

    if file.filename == '':
        return "Error: No selected file. Please select a CSV file to upload.", 400

    if file and file.filename.endswith('.csv'):
        file_path = os.path.join(app.config['UPLOAD_FOLDER'], file.filename)
        file.save(file_path)

        # Load the CSV file into a DataFrame
        df = pd.read_csv(file_path)
        train_forecasting_models(df)
        return "Model training completed for all valid groups.", 200
    else:
        return "Error: Invalid file type. Please upload a valid CSV file.", 400

@app.route('/forecast', methods=['POST'])
def forecast():
    start_date = request.form.get('start_date')
    end_date = request.form.get('end_date')
    output_format = request.form.get('output_format')

    if not start_date or not end_date or not output_format:
        return "Error: Please provide valid start_date, end_date, and output_format parameters.", 400

    if not trained_models:
        return "Error: No trained models found. Please upload a dataset and train the models first.", 400

    forecasted_values = generate_forecasts(start_date, end_date)

    if output_format == 'csv':
        # Create a CSV file in memory
        output = io.StringIO()
        writer = csv.DictWriter(output, fieldnames=["project", "ledger", "company", "date", "forecast"])
        writer.writeheader()
        writer.writerows(forecasted_values)
        csv_content = output.getvalue()
        output.close()

        # Return CSV as a file download
        response = make_response(csv_content)
        response.headers["Content-Disposition"] = "attachment; filename=forecast.csv"
        response.headers["Content-Type"] = "text/csv"
        return response
    elif output_format == 'json':
        # Return JSON response
        response = make_response(jsonify(forecasted_values))
        response.headers["Content-Disposition"] = "attachment; filename=forecast.json"
        response.headers["Content-Type"] = "application/json"
        return response
    else:
        return "Error: Invalid output format. Please specify 'csv' or 'json'.", 400

if __name__ == '__main__':
    app.run(debug=True, port=8000)
