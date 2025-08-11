# realtime_anomaly.py

import joblib
import numpy as np
import pandas as pd
import shap

# Load model and input features

model = joblib.load("Models/SI_xgb_model.pkl")
feature_names = [ 'OxEnRa', 'BlFuPeIn', 'EnOxFl', 'CoBlFl', 'BlFuBoGaIn',
       'ThCoTe', 'EnOxPr', 'ToPrDr', 'HoBlPr', 'AcBlVe', 'CoBlTe', 'HoBlTe',
       'ToTe', 'BlHu', 'FoSI', 'FoSI_lag1', 'FoSI_lag2', 'FoSI_lag3',
       'ThCoTe_diff', 'FoSI_rolling3'
]

# Load SHAP explainer
explainer = shap.Explainer(model)

def detect_anomaly(input_data, threshold=0.10):
    """
    Detects if the predicted SI is deviating more than threshold from actual.
    :param input_data: DataFrame with same features as training
    :param threshold: allowed deviation (e.g., 0.10 = 10%)
    :return: dict with prediction, actual (if given), is_anomaly, top contributing feature
    """
    # Ensure column order
    data = input_data[feature_names]

    # Predict
    prediction = model.predict(data)[0]

    # SHAP values
    shap_values = explainer(data)
    top_feature_index = np.abs(shap_values.values[0]).argmax()
    top_feature_name = feature_names[top_feature_index]

    result = {
        "prediction": prediction,
        "top_contributing_feature": top_feature_name
    }

    # Optional: include actual and anomaly flag if actual SI is passed
    if "SI" in input_data.columns:
        actual = input_data["SI"].values[0]
        deviation = abs(prediction - actual) / actual
        result["actual"] = actual
        result["deviation"] = deviation
        result["is_anomaly"] = deviation > threshold

    return result

# Example usage
if __name__ == "__main__":
    # Load 1-row test sample for simulation
    test_input = pd.read_csv("Data/sample_input.csv")  # Must include SI column if you want deviation

    result = detect_anomaly(test_input)
    print("Real-time Anomaly Detection Result:")
    print(result)
