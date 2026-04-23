import pandas as pd
import pickle
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
from sklearn.preprocessing import LabelEncoder

# Load data
df = pd.read_csv("kaggle1.csv")

# Preprocess
df["Sleep Disorder"] = df["Sleep Disorder"].fillna("None")
df["BMI Category"] = df["BMI Category"].replace("Normal Weight","Normal")

df[['Systolic BP','Diastolic BP']] = df['Blood Pressure'].str.split('/',expand=True)
df[['Systolic BP','Diastolic BP']] = df[['Systolic BP','Diastolic BP']].astype(int)

df.drop(['Blood Pressure','Person ID','Diastolic BP'],axis=1,inplace=True)

# Encode categorical
encoders = {}
for col in ['Gender','Occupation','BMI Category','Sleep Disorder']:
    le = LabelEncoder()
    df[col] = le.fit_transform(df[col])
    encoders[col] = le

# Split
X = df.drop("Stress Level",axis=1)
y = df["Stress Level"]

X_train,X_test,y_train,y_test = train_test_split(X,y,test_size=0.2,random_state=42)

# Train model
model = RandomForestRegressor(n_estimators=200,random_state=42)
model.fit(X_train,y_train)

# Save model + encoders + column order
pickle.dump((model, encoders, X.columns), open("model.pkl","wb"))

print("Model saved successfully!")