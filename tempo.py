from flask import Flask, request, jsonify
import pandas as pd
import numpy as np
import base64
from io import BytesIO
from sklearn.model_selection import train_test_split, GridSearchCV
from xgboost import XGBRegressor
from sklearn.metrics import mean_squared_error, r2_score

app = Flask(__name__)

@app.route('/process', methods=['POST'])
def process_group():
    try:
        data = request.json
        csv_file = data.get('csv_file')
        input_group = tuple(data.get('input_group'))

        df = pd.read_csv(BytesIO(base64.b64decode(csv_file)))

        grouped_df = df.groupby(['project', 'ledger', 'company']).size()
        filtered_groups = grouped_df[grouped_df > 37]

        if input_group in filtered_groups.index:
            specific_group_data = df[
                (df['project'] == input_group[0]) & 
                (df['ledger'] == input_group[1]) & 
                (df['company'] == input_group[2])
            ].copy()

            if 'date' in specific_group_data.columns and 'dr_cr' in specific_group_data.columns:
                specific_group_data['date'] = pd.to_datetime(specific_group_data['date'])
                specific_group_data.sort_values('date', inplace=True)
                specific_group_data['target'] = specific_group_data['dr_cr']

                for i in range(1, 11):
                    specific_group_data[f'lag_{i}'] = specific_group_data['dr_cr'].shift(i)

                specific_group_data['roll_avg_3'] = specific_group_data['dr_cr'].rolling(window=3).mean()
                specific_group_data['roll_avg_7'] = specific_group_data['dr_cr'].rolling(window=7).mean()
                specific_group_data['roll_avg_14'] = specific_group_data['dr_cr'].rolling(window=14).mean()
                specific_group_data.dropna(inplace=True)

                feature_columns = [col for col in specific_group_data.columns if col.startswith('lag_') or col.startswith('roll_avg_')]
                X = specific_group_data[feature_columns]
                y = specific_group_data['target']

                X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, shuffle=False)

                param_grid = {'n_estimators': [50, 100, 200], 'learning_rate': [0.01, 0.1, 0.2], 'max_depth': [3, 5, 7]}
                model = XGBRegressor(objective='reg:squarederror')
                grid_search = GridSearchCV(estimator=model, param_grid=param_grid, cv=3, scoring='neg_mean_squared_error', verbose=0)
                grid_search.fit(X_train, y_train)

                best_model = grid_search.best_estimator_
                y_pred = best_model.predict(X_test)

                mse = mean_squared_error(y_test, y_pred)
                rmse = np.sqrt(mse)
                r2 = r2_score(y_test, y_pred)

                last_data = X.iloc[-1].values.reshape(1, -1)
                forecast = []
                for _ in range(100):
                    next_value = best_model.predict(last_data)[0]
                    forecast.append(next_value)
                    last_data = np.roll(last_data, shift=-1, axis=1)
                    last_data[0, -1] = next_value

                forecast_dates = pd.date_range(start=specific_group_data['date'].iloc[-1] + pd.Timedelta(days=1), periods=100, freq='D')

                forecast_response = {
                    'forecast_dates': forecast_dates.strftime('%Y-%m-%d').tolist(),
                    'forecast_values': [float(value) for value in forecast],
                    'metrics': {
                        'mse': float(mse),
                        'rmse': float(rmse),
                        'r2': float(r2)
                    }
                }
                return jsonify(forecast_response)
            else:
                return jsonify({'error': "The required columns 'date' or 'dr_cr' are missing in the dataset."}), 400
        else:
            return jsonify({'error': f"The group {input_group} is not present in the filtered data."}), 404

    except Exception as e:
        return jsonify({'error': str(e)}), 500

if __name__ == '__main__':
    app.run(debug=True)
