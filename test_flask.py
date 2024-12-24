import requests

url = "http://127.0.0.1:8000/forecast"
params = {
    'project': 'D-HO',
    'ledger': 'Thane Bharat Sahakari Bank Ltd-A/C 450',
    'company': 'DREAMZ Pvt Ltd.'
}

response = requests.get(url, params=params)
print(response.json())  # This will print the forecasted data

