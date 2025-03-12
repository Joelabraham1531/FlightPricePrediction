import numpy as np
import pandas as pd
import pickle
from flask import Flask, request, jsonify, render_template

app = Flask(__name__)
model = pickle.load(open('model.pkl', 'rb'))

# One-hot encoding reference from training
columns = ['Total_Stops', 'Journey_Day', 'Journey_Month', 'Dep_Hour', 'Dep_Minute',
           'Arrival_Hour', 'Arrival_Minute', 'Duration_Minutes', 'Airline_Air India',
           'Airline_GoAir', 'Airline_IndiGo', 'Airline_Jet Airways',
           'Airline_Jet Airways Business', 'Airline_Multiple carriers',
           'Airline_Multiple carriers Premium economy', 'Airline_SpiceJet',
           'Airline_Trujet', 'Airline_Vistara', 'Airline_Vistara Premium economy',
           'Source_Chennai', 'Source_Delhi', 'Source_Kolkata', 'Source_Mumbai',
           'Destination_Cochin', 'Destination_Delhi', 'Destination_Hyderabad',
           'Destination_Kolkata', 'Destination_New Delhi']

@app.route('/')
def home():
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():
    features = [x for x in request.form.values()]

    # Extract values from input
    source_value = features[0]
    dest_value = features[1]
    date_value = features[2]
    airline_value = features[3]
    stops_value = int(features[4])
  

    # Extract datetime values
    date_time = pd.to_datetime(date_value, format="%Y-%m-%dT%H:%M")
    day, month, hour, minute = date_time.day, date_time.month, date_time.hour, date_time.minute

    # One-hot encode categorical values
    categorical_values = {
        'Airline': ['Air India', 'GoAir', 'IndiGo', 'Jet Airways', 'Jet Airways Business',
                    'Multiple carriers', 'Multiple carriers Premium economy', 'SpiceJet',
                    'Trujet', 'Vistara', 'Vistara Premium economy'],
        'Source': ['Chennai', 'Delhi', 'Kolkata', 'Mumbai'],
        'Destination': ['Cochin', 'Delhi', 'Hyderabad', 'Kolkata', 'New Delhi']
    }
    
    # Create feature vector with one-hot encoding
    feature_dict = {col: 0 for col in columns}
    feature_dict['Total_Stops'] = stops_value
    feature_dict['Journey_Day'] = day
    feature_dict['Journey_Month'] = month
    feature_dict['Dep_Hour'] = hour
    feature_dict['Dep_Minute'] = minute
    feature_dict['Arrival_Hour'] = hour  # Assuming arrival hour is same for now
    feature_dict['Arrival_Minute'] = minute  # Assuming arrival minute is same
    

    # Set one-hot encoding values
    if airline_value in categorical_values['Airline']:
        feature_dict[f'Airline_{airline_value}'] = 1
    if source_value in categorical_values['Source']:
        feature_dict[f'Source_{source_value}'] = 1
    if dest_value in categorical_values['Destination']:
        feature_dict[f'Destination_{dest_value}'] = 1

    # Convert feature_dict to dataframe
    pred_features = pd.DataFrame([feature_dict])
    
    # Ensure features are in correct order
    pred_features = pred_features[columns]
    
    # Predict price
    # Predict price
    prediction = model.predict(pred_features)
    output = round(prediction[0], 2)

# Increase price for Business class
    if request.form.get("Class") == "Business":
        output = round(output * 1.5, 2)  # Increase by 30%

    return render_template('index.html', pred=f'The Flight Fare is: INR {output}')

    

if __name__ == '__main__':
    app.run(debug=True)