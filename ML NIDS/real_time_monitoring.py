import pandas as pd
import joblib
from sklearn.preprocessing import LabelEncoder, StandardScaler

def preprocess_data_for_prediction(new_data, label_encoder, scaler):
    new_data.fillna(0, inplace=True)

    new_data['IPV4_SRC_ADDR'] = label_encoder.transform(new_data['IPV4_SRC_ADDR'])
    new_data['IPV4_DST_ADDR'] = label_encoder.transform(new_data['IPV4_DST_ADDR'])
    new_data['TCP_FLAGS'] = label_encoder.transform(new_data['TCP_FLAGS'])

    numerical_features = ['L4_SRC_PORT', 'L4_DST_PORT', 'PROTOCOL', 'L7_PROTO', 'IN_BYTES', 'OUT_BYTES', 'IN_PKTS', 'OUT_PKTS', 'FLOW_DURATION_MILLISECONDS']
    new_data[numerical_features] = scaler.transform(new_data[numerical_features])

    return new_data

def predict_with_model(model, preprocessed_data):
    # Make predictions using the trained model
    predictions = model.predict(preprocessed_data)
    return predictions

def get_new_data():
    columns = ['L4_SRC_PORT', 'L4_DST_PORT', 'PROTOCOL', 'L7_PROTO', 'IN_BYTES', 'OUT_BYTES', 'IN_PKTS', 'OUT_PKTS', 'FLOW_DURATION_MILLISECONDS', 'IPV4_SRC_ADDR', 'IPV4_DST_ADDR', 'TCP_FLAGS']
    new_data = pd.DataFrame([[1234, 5678, 6, 2, 1000, 500, 10, 5, 1000, '192.168.1.1', '192.168.2.1', 'SYN']], columns=columns)
    return new_data
    
def main():
    # Load the saved model, label encoder, and scaler
    model_path = "C:/Users/eguhlin/Classes/CAPSTONE/Development/ML NIDS/NB15/model/random_forest_model.joblib"
    label_encoder_path = "C:/Users/eguhlin/Classes/CAPSTONE/Development/ML NIDS/NB15/model/label_encoder.joblib"
    scaler_path = "C:/Users/eguhlin/Classes/CAPSTONE/Development/ML NIDS/NB15/model/scaler.joblib"

    model = joblib.load(model_path)
    label_encoder = joblib.load(label_encoder_path)
    scaler = joblib.load(scaler_path)

    new_data = get_new_data()

    # Preprocess the new data for prediction
    preprocessed_data = preprocess_data_for_prediction(new_data, label_encoder, scaler)

    # Make predictions using the model
    predictions = predict_with_model(model, preprocessed_data)

    # Print or use the predictions as needed
    print("Predictions:", predictions)

if __name__ == "__main__":
    main()
