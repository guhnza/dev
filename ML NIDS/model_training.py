import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, classification_report
from sklearn.preprocessing import LabelEncoder, StandardScaler
import joblib

def load_data(dataset_path):
    df = pd.read_csv(dataset_path)
    return df

def preprocess_data(df):
    df.fillna(0, inplace=True)

    label_encoder = LabelEncoder()
    df['IPV4_SRC_ADDR'] = label_encoder.fit_transform(df['IPV4_SRC_ADDR'])
    df['IPV4_DST_ADDR'] = label_encoder.fit_transform(df['IPV4_DST_ADDR'])
    df['TCP_FLAGS'] = label_encoder.fit_transform(df['TCP_FLAGS'])

    scaler = StandardScaler()
    numerical_features = ['L4_SRC_PORT', 'L4_DST_PORT', 'PROTOCOL', 'L7_PROTO', 'IN_BYTES', 'OUT_BYTES', 'IN_PKTS', 'OUT_PKTS', 'FLOW_DURATION_MILLISECONDS']
    df[numerical_features] = scaler.fit_transform(df[numerical_features])

    return df, label_encoder, scaler

def split_data(df, label_encoder):
    X = df.drop(['Label', 'Attack'], axis=1)
    y = label_encoder.transform(df['Label'])
    return train_test_split(X, y, test_size=0.2, random_state=42)

def train_model(X_train, y_train):
    model = RandomForestClassifier()
    model.fit(X_train, y_train)
    return model

def evaluate_model(model, X_test, y_test):
    y_pred = model.predict(X_test)
    accuracy = accuracy_score(y_test, y_pred)
    report = classification_report(y_test, y_pred)
    return accuracy, report

def save_model(model, label_encoder, scaler):
    model_path = "C:/Users/eguhlin/Classes/CAPSTONE/Development/ML NIDS/NB15/model/random_forest_model.joblib"
    label_encoder_path = "C:/Users/eguhlin/Classes/CAPSTONE/Development/ML NIDS/NB15/model/label_encoder.joblib"
    scaler_path = "C:/Users/eguhlin/Classes/CAPSTONE/Development/ML NIDS/NB15/model/scaler.joblib"

    joblib.dump(model, model_path)
    joblib.dump(label_encoder, label_encoder_path)
    joblib.dump(scaler, scaler_path)

    print(f"Model saved to {model_path}")
    print(f"Label Encoder saved to {label_encoder_path}")
    print(f"Scaler saved to {scaler_path}")

def main():
    dataset_path = "C:/Users/eguhlin/Classes/CAPSTONE/Development/ML NIDS/NB15/data/preprocessed_NF-UNSW-NB15.csv"
    df = load_data(dataset_path)
    
    df, label_encoder, scaler = preprocess_data(df)
    
    X_train, X_test, y_train, y_test = split_data(df, label_encoder)
    
    model = train_model(X_train, y_train)
    
    accuracy, report = evaluate_model(model, X_test, y_test)
    
    print(f"Accuracy: {accuracy}")
    print("Classification Report:")
    print(report)
    
    save_model(model, label_encoder, scaler)

if __name__ == "__main__":
    main()
