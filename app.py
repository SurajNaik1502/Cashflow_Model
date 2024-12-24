from flask import Flask, request, jsonify, render_template
from flask_cors import CORS
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split, GridSearchCV
from xgboost import XGBRegressor

app = Flask(__name__)
CORS(app)

def train_forecasting_model(input_group):
    # Load your CSV file
    df = pd.read_csv('Dataset/ledger.csv')
    
    grouped_df = df.groupby(['project', 'ledger', 'company']).size()
    filtered_groups = grouped_df[grouped_df > 37]
    
    if input_group in filtered_groups.index:
        specific_group_data = df[(df['project'] == input_group[0]) & (df['ledger'] == input_group[1]) & (df['company'] == input_group[2])].copy()

        if 'date' in specific_group_data.columns and 'dr_cr' in specific_group_data.columns:
            specific_group_data.loc[:, 'date'] = pd.to_datetime(specific_group_data['date'])
            specific_group_data.sort_values('date', inplace=True)
            specific_group_data.loc[:, 'target'] = specific_group_data['dr_cr']

            num_lags = 10
            for i in range(1, num_lags + 1):
                specific_group_data.loc[:, f'lag_{i}'] = specific_group_data['dr_cr'].shift(i)
            specific_group_data.loc[:, 'roll_avg_3'] = specific_group_data['dr_cr'].rolling(window=3).mean()
            specific_group_data.loc[:, 'roll_avg_7'] = specific_group_data['dr_cr'].rolling(window=7).mean()
            specific_group_data.loc[:, 'roll_avg_14'] = specific_group_data['dr_cr'].rolling(window=14).mean()

            specific_group_data.dropna(inplace=True)
            feature_columns = [col for col in specific_group_data.columns if col.startswith('lag_') or col.startswith('roll_avg_')]
            X = specific_group_data[feature_columns]
            y = specific_group_data['target']
            
            X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, shuffle=False)

            param_grid = {
                'n_estimators': [50, 100, 200],
                'learning_rate': [0.01, 0.1, 0.2],
                'max_depth': [3, 5, 7],
            }

            model = XGBRegressor(objective='reg:squarederror')
            grid_search = GridSearchCV(estimator=model, param_grid=param_grid, cv=3, scoring='neg_mean_squared_error', verbose=1)
            grid_search.fit(X_train, y_train)
            best_model = grid_search.best_estimator_

            last_data = X.iloc[-1].values.reshape(1, -1)
            forecast = []
            for i in range(100):
                next_value = best_model.predict(last_data)[0]
                forecast.append(next_value)
                last_data = np.roll(last_data, shift=-1, axis=1)
                last_data[0, -1] = next_value
            forecast = [float(value) for value in forecast]
            last_date = specific_group_data['date'].iloc[-1]
            forecast_dates = pd.date_range(start=last_date + pd.Timedelta(days=1), periods=100, freq='D')
            forecast_results = [
                {"date": forecast_dates[i].strftime('%Y-%m-%d'), "forecast": forecast[i]} 
                for i in range(100)
            ]
            return forecast_results
    return None

@app.route('/')
def home():
    return render_template('index.html')

@app.route('/forecast', methods=['GET', 'POST'])
def forecast():
    if request.method == 'POST':
        project = request.form.get('project')
        ledger = request.form.get('ledger')
        company = request.form.get('company')
    else:  # For GET requests
        project = request.args.get('project')
        ledger = request.args.get('ledger')
        company = request.args.get('company')

    if not project or not ledger or not company:
        return jsonify({'error': 'Please provide project, ledger, and company parameters.'}), 400

    input_group = (project, ledger, company)
    forecasted_values = train_forecasting_model(input_group)
    if forecasted_values is not None:
        return jsonify({'forecast': forecasted_values})
    else:
        return jsonify({'error': 'The specified group was not found or there was an issue with the data.'}), 404

if __name__ == '__main__':
    app.run(debug=True, port=8000)


