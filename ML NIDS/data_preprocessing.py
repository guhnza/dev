import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, classification_report
from sklearn.preprocessing import LabelEncoder, StandardScaler
import joblib

#Load the NF-UNSW-NB15 dataset
dataset_path = "C:/Users/eguhlin/Classes/CAPSTONE/Development (Coding)/ML NIDS/NB15/data/NF-UNSW-NB15.csv"
df = pd.read_csv(dataset_path)

#Preprocess the data
df.fillna(0, inplace=True)

#Encode categorical variables using LabelEncoder
label_encoder = LabelEncoder()
df['IPV4_SRC_ADDR'] = label_encoder.fit_transform(df['IPV4_SRC_ADDR'])
df['IPV4_DST_ADDR'] = label_encoder.fit_transform(df['IPV4_DST_ADDR'])
df['TCP_FLAGS'] = label_encoder.fit_transform(df['TCP_FLAGS'])

#Scale numerical features
scaler = StandardScaler()
numerical_features = ['L4_SRC_PORT', 'L4_DST_PORT', 'PROTOCOL', 'L7_PROTO', 'IN_BYTES', 'OUT_BYTES', 'IN_PKTS', 'OUT_PKTS', 'FLOW_DURATION_MILLISECONDS']
df[numerical_features] = scaler.fit_transform(df[numerical_features])

#Split the dataset into training and testing sets
X = df.drop(['Label', 'Attack'], axis=1)
y = label_encoder.fit_transform(df['Label'])

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

model = RandomForestClassifier()
model.fit(X_train, y_train)

#Make predictions on the test set
y_pred = model.predict(X_test)

#Evaluate the model
accuracy = accuracy_score(y_test, y_pred)
report = classification_report(y_test, y_pred)

#Print the results
print(f"Accuracy: {accuracy}")
print("Classification Report:")
print(report)

#Save the preprocessed data to a CSV file
preprocessed_dataset_path = "C:/Users/eguhlin/Classes/CAPSTONE/Development (Coding)/ML NIDS/NB15/data/preprocessed_NF-UNSW-NB15.csv"
df.to_csv(preprocessed_dataset_path, index=False)

#Print a message indicating the CSV file creation
print(f"Preprocessed data saved to {preprocessed_dataset_path}")
