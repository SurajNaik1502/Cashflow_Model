from flask import Flask, request, render_template, jsonify, make_response
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from xgboost import XGBRegressor
import csv
import io
import os

app = Flask(__name__)
# Set the upload folder to 'Dataset'
app.config['UPLOAD_FOLDER'] = 'Dataset'
os.makedirs(app.config['UPLOAD_FOLDER'], exist_ok=True)

# Function for training the forecasting model
def train_forecasting_model(df, input_group, start_date, end_date):
    grouped_df = df.groupby(['project', 'ledger', 'company']).size()
    filtered_groups = grouped_df[grouped_df > 37]

    if input_group in filtered_groups.index:
        specific_group_data = df[
            (df['project'] == input_group[0]) & 
            (df['ledger'] == input_group[1]) & 
            (df['company'] == input_group[2])
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

        # Prepare for forecasting
        last_data = X.iloc[-1].values.reshape(1, -1)
        forecast = []

        forecast_dates = pd.date_range(start=start_date, end=end_date, freq='D')
        for _ in forecast_dates:
            next_value = model.predict(last_data)[0]
            forecast.append(float(next_value))  # Convert to float
            last_data = np.roll(last_data, shift=-1)
            last_data[-1] = next_value
        
        return [{"date": date.strftime('%Y-%m-%d'), "forecast": value} for date, value in zip(forecast_dates, forecast)]

    return None

@app.route('/')
def home():
    return render_template('index.html')

@app.route('/forecast', methods=['POST'])
def forecast():
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

        project = request.form.get('project')
        ledger = request.form.get('ledger')
        company = request.form.get('company')
        start_date = request.form.get('start_date')
        end_date = request.form.get('end_date')
        output_format = request.form.get('output_format')

        if not project or not ledger or not company or not start_date or not end_date or not output_format:
            return "Error: Please provide valid project, ledger, company, start_date, end_date, and output_format parameters.", 400

        input_group = (project, ledger, company)
        forecasted_values = train_forecasting_model(df, input_group, start_date, end_date)
        
        if forecasted_values is not None:
            if output_format == 'csv':
                # Create a CSV file in memory
                output = io.StringIO()
                writer = csv.DictWriter(output, fieldnames=["date", "forecast"])
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
                # Return JSON response as a file download
                response = make_response(jsonify(forecasted_values))
                response.headers["Content-Disposition"] = "attachment; filename=forecast.json"
                response.headers["Content-Type"] = "application/json"
                return response
            else:
                return "Error: Invalid output format. Please specify 'csv' or 'json'.", 400
        else:
            return "Error: The specified group was not found or there was an issue with the data.", 404
    else:
        return "Error: Invalid file type. Please upload a valid CSV file.", 400

if __name__ == '__main__':
    app.run(debug=True, port=8000)
