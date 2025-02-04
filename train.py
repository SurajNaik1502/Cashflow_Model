from flask import Flask, request, jsonify, make_response 
import pandas as pd 
import numpy as np 
from sklearn.model_selection import train_test_split 
from xgboost import XGBRegressor 
import csv 
import io 
import os 
import joblib 

app = Flask(__name__) 

# Set the upload folder to 'Dataset' 
app.config['UPLOAD_FOLDER'] = 'Dataset' 
os.makedirs(app.config['UPLOAD_FOLDER'], exist_ok=True) 
MODEL_PATH = "forecasting_model.pkl" 

def train_and_save_model(df): 
    grouped_df = df.groupby(['project', 'ledger', 'company']).size() 
    filtered_groups = grouped_df[grouped_df > 37] 

    models = {} 
    for group in filtered_groups.index: 
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

        models[group] = model 

    joblib.dump(models, MODEL_PATH) 

def forecast_from_saved_model(input_group, start_date, end_date): 
    if not os.path.exists(MODEL_PATH): 
        return None 

    models = joblib.load(MODEL_PATH) 

    if input_group in models: 
        model = models[input_group] 
        forecast_dates = pd.date_range(start=start_date, end=end_date, freq='D') 

        lag_values = np.zeros(10) 
        forecast = [] 
        for _ in forecast_dates: 
            next_value = model.predict(lag_values.reshape(1, -1))[0] 
            forecast.append(float(next_value))
            lag_values = np.roll(lag_values, shift=-1) 
            lag_values[-1] = next_value 

        return [{"date": date.strftime('%Y-%m-%d'), "forecast": value} for date, value in zip(forecast_dates, forecast)] 

    return None 

@app.route('/train', methods=['POST']) 
def train_model():
    if 'file' not in request.files: 
        return "Error: No file uploaded. Please upload a CSV file.", 400 

    file = request.files['file'] 

    if file.filename == '': 
        return "Error: No selected file. Please select a CSV file to upload.", 400 

    if file and file.filename.endswith('.csv'): 
        file_path = os.path.join(app.config['UPLOAD_FOLDER'], file.filename) 
        file.save(file_path) 

        df = pd.read_csv(file_path) 
        train_and_save_model(df) 

        return "Model training and saving completed successfully.", 200 
    else: 
        return "Error: Invalid file type. Please upload a valid CSV file.", 400 

@app.route('/forecast', methods=['POST']) 
def forecast():
    data = request.get_json() 

    project = data.get('project') 
    ledger = data.get('ledger') 
    company = data.get('company') 
    start_date = data.get('start_date') 
    end_date = data.get('end_date') 
    output_format = data.get('output_format') 

    if not project or not ledger or not company or not start_date or not end_date or not output_format: 
        return jsonify({"error": "Please provide valid project, ledger, company, start_date, end_date, and output_format parameters."}), 400 

    input_group = (project, ledger, company) 
    forecasted_values = forecast_from_saved_model(input_group, start_date, end_date) 
    
    if forecasted_values is not None: 
        if output_format == 'csv': 
            output = io.StringIO() 
            writer = csv.DictWriter(output, fieldnames=["date", "forecast"]) 
            writer.writeheader() 
            writer.writerows(forecasted_values) 
            csv_content = output.getvalue() 
            output.close() 

            response = make_response(csv_content) 
            response.headers["Content-Disposition"] = "attachment; filename=forecast.csv" 
            response.headers["Content-Type"] = "text/csv" 
            return response 
        elif output_format == 'json':
            response = make_response(jsonify(forecasted_values)) 
            response.headers["Content-Disposition"] = "attachment; filename=forecast.json" 
            response.headers["Content-Type"] = "application/json" 
            return response 
        else: 
            return jsonify({"error": "Invalid output format. Please specify 'csv' or 'json'."}), 400 
    else: 
        return jsonify({"error": "The specified group was not found or the model is not trained."}), 404 

if __name__ == '__main__': 
    app.run(debug=True, port=8000) 
