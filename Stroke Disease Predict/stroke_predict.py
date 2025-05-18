import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeRegressor
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score
from sklearn.utils import resample
import joblib 

import warnings
warnings.filterwarnings("ignore")

df= pd.read_csv('Stroke Disease Predict/healthcare-dataset-stroke-data.csv')
df.drop(columns=['id'], inplace=True)
print(df.head())
df.info()
print(df.describe())

sns.countplot(x='stroke', data=df)
print(df['stroke'].value_counts())
plt.title('Count of Stroke vs No Stroke')
plt.show()

#Downsampling
df_majority = df[df.stroke==0]
df_minority = df[df.stroke==1]
df_majority_downsampled = resample(df_majority,
                                   replace=False,     
                                   n_samples=249,    
                                   random_state=42) 

df = pd.concat([df_majority_downsampled, df_minority])

sns.countplot(x='stroke', data=df)
print(df['stroke'].value_counts())
plt.title('Count of Stroke vs No Stroke')
plt.show()


#Missing Value : DecisionTreeRegressor
#print(df.isnull().sum())

DT_bmi_pipe = Pipeline(steps=[
    ("scale", StandardScaler()), # Standardize the data
    ("dtr", DecisionTreeRegressor()) # Decision Tree Regressor
])

X=df[["gender","age","bmi"]].copy()

X.gender=X.gender.replace({"Male":0,"Female":1,"Other":-1}).astype(np.uint8)

missing=X[X.bmi.isna()]

X=X[~X.bmi.isna()]
y=X.pop("bmi")
DT_bmi_pipe.fit(X,y)

predicted_bmi=pd.Series(DT_bmi_pipe.predict(missing[["gender","age"]]),index=missing.index)
df.loc[missing.index,"bmi"]=predicted_bmi

#print(df.isnull().sum())

# Training 

df["gender"] = df["gender"].replace({"Male": 0, "Female":1, "Other":-1}).astype(np.uint8)
df["Residence_type"] = df["Residence_type"].replace({"Rural": 0, "Urban":1}).astype(np.uint8)
df["work_type"] = df["work_type"].replace({"Private": 0, "Self-employed":1, "Govt_job":2, "children":-1, "Never_worked":-2}).astype(np.uint8)
df["smoking_status"] = df["smoking_status"].replace({ "never smoked":0,"formerly smoked": 1, "smokes":2,"Unknown":-1}).astype(np.uint8)
df["ever_married"] = df["ever_married"].replace({"No": 0, "Yes":1}).astype(np.uint8)

X = df[["gender", "age", "hypertension", "heart_disease", "work_type", "avg_glucose_level", "bmi","smoking_status","ever_married" ]].copy()
y = df["stroke"]

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.1, random_state=42)

logreg_pipe = Pipeline(steps=[("scale", StandardScaler()), ("LR", LogisticRegression())])

# model training
logreg_pipe.fit(X_train, y_train)

# modelin testi
y_pred = logreg_pipe.predict(X_test)

print("Accuracy: ", accuracy_score(y_test, y_pred))
print("Confusion Matrix: \n", confusion_matrix(y_test, y_pred))
print("Classification Report: \n", classification_report(y_test, y_pred))

joblib.dump(logreg_pipe, 'Stroke Disease Predict/stroke_model.pkl')