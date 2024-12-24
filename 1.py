from flask import Flask, jsonify, request
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split, GridSearchCV
from xgboost import XGBRegressor
from sklearn.metrics import mean_squared_error, r2_score     

app = Flask(__name__)

def train_forecasting_model(input_group):
    # Load your CSV file
    df = pd.read_csv('ledger.csv')
    
    # Group by 'project', 'ledger', and 'company', and get the count of rows in each group
    grouped_df = df.groupby(['project', 'ledger', 'company']).size()
    
    # Filter groups where the count is greater than 37
    filtered_groups = grouped_df[grouped_df > 37]
    
    # Check if the input group exists in the filtered groups
    if input_group in filtered_groups.index:
        # Filter the data for this specific group
        specific_group_data = df[(df['project'] == input_group[0]) & (df['ledger'] == input_group[1]) & (df['company'] == input_group[2])].copy()

        if 'date' in specific_group_data.columns and 'dr_cr' in specific_group_data.columns:
            # Convert 'date' column to datetime if necessary
            specific_group_data.loc[:, 'date'] = pd.to_datetime(specific_group_data['date'])
            
            # Sort by date
            specific_group_data.sort_values('date', inplace=True)
            
            # Feature engineering: Use 'dr_cr' as the target variable
            specific_group_data.loc[:, 'target'] = specific_group_data['dr_cr']
            
            # Add lag features
            num_lags = 10  # Number of lags to create
            for i in range(1, num_lags + 1):
                specific_group_data.loc[:, f'lag_{i}'] = specific_group_data['dr_cr'].shift(i)
            
            # Add rolling averages
            specific_group_data.loc[:, 'roll_avg_3'] = specific_group_data['dr_cr'].rolling(window=3).mean()
            specific_group_data.loc[:, 'roll_avg_7'] = specific_group_data['dr_cr'].rolling(window=7).mean()
            specific_group_data.loc[:, 'roll_avg_14'] = specific_group_data['dr_cr'].rolling(window=14).mean()
            
            # Drop missing values that result from shifting and rolling
            specific_group_data.dropna(inplace=True)
            
            # Define the features (X) and target (y)
            feature_columns = [col for col in specific_group_data.columns if col.startswith('lag_') or col.startswith('roll_avg_')]
            X = specific_group_data[feature_columns]
            y = specific_group_data['target']
            
            # Split data into training and testing
            X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, shuffle=False)
            
            # Hyperparameter tuning for XGBoost
            param_grid = {
                'n_estimators': [50, 100, 200],
                'learning_rate': [0.01, 0.1, 0.2],
                'max_depth': [3, 5, 7],
            }
            
            model = XGBRegressor(objective='reg:squarederror')
            grid_search = GridSearchCV(estimator=model, param_grid=param_grid, cv=3, scoring='neg_mean_squared_error', verbose=1)
            grid_search.fit(X_train, y_train)
            
            # Best model from grid search
            best_model = grid_search.best_estimator_
            
            # Forecast the next 100 days
            last_data = X.iloc[-1].values.reshape(1, -1)
            forecast = []
            for i in range(100):
                next_value = best_model.predict(last_data)[0]
                forecast.append(next_value)
                
                # Update the last_data with the new predicted value
                last_data = np.roll(last_data, shift=-1, axis=1)
                last_data[0, -1] = next_value
                
            # Convert forecast values to Python float (native float type)
            forecast = [float(value) for value in forecast]
            
            # Generate the next 100 dates
            last_date = specific_group_data['date'].iloc[-1]
            forecast_dates = pd.date_range(start=last_date + pd.Timedelta(days=1), periods=100, freq='D')
            
            # Combine forecast dates and values into a list of dictionaries (Date: Forecast)
            forecast_results = [
                {"date": forecast_dates[i].strftime('%Y-%m-%d'), "forecast": forecast[i]} 
                for i in range(100)
            ]
            
            # Return the list of forecast results
            return forecast_results
    return None

@app.route('/forecast', methods=['GET'])
def forecast():
    # Get the input_group from the query parameters
    project = request.args.get('project')
    ledger = request.args.get('ledger')
    company = request.args.get('company')

    # project= 'D-HO'
    # ledger='Thane Bharat Sahakari Bank Ltd-A/C 450'
    # company= 'DREAMZ Pvt Ltd.'
    
    print(f"Received request: {project}, {ledger}, {company}")
    
    # Validate input
    if project=='' or ledger=='' or company=='':
        return jsonify({'error': 'Please provide project, ledger, and company parameters.'}), 400

    input_group = (project, ledger, company)
    print(input_group)
    
    # Train the model and get the forecasted values
    forecasted_values = train_forecasting_model(input_group)
    
    if forecasted_values is not None:
        # Return the forecasted values as a JSON response
        return jsonify({'forecasting': forecasted_values})
    else:
        return jsonify({'error': 'The specified group was not found or there was an issue with the data.'}), 404

if __name__ == '__main__':
    app.run(debug=True , port=8000)
