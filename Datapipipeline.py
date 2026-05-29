import pandas as pd;
import numpy as np;
#load dataset
url = "https://raw.githubusercontent.com/datasciencedojo/datasets/master/titanic.csv"
df = pd.read_csv(url)
#first five columns data 
df.head()
#counting the no.of missing values in each column
df.isnull().sum()
#in data science we usually remove columns with huge missing values
#in this dataset the cabin column have huge missing values 
#drop cabin columns
df = df.drop("Cabin",axis=1)
#filling null values with median,mode of the columns age and embarked
df["Age"].mean()
df["Age"] = df["Age"].fillna(df["Age"].mean())
df.isnull().sum()
df["Embarked"] = df["Embarked"].fillna(df["Embarked"].mode()[0])
df.isnull().sum()
#remove unnecessary columns which are not useful for predictions
df = df.drop(["Name","Ticket"],axis = 1)
from sklearn.preprocessing import StandardScaler
scaler = StandardScaler()
num_cols = ["Age", "Fare"]
df[num_cols] = scaler.fit_transform(df[num_cols])
#ml models only understand numerical values
df = pd.get_dummies(df, columns=["Sex", "Embarked"])
df.head()
df.info()
X = df.drop("Survived", axis=1)
y = df["Survived"]
print(X.shape)
print(y.shape)
#train-test split
from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.2,
    random_state=42
)
print(X_train.shape)
print(X_test.shape)
print(y_train.shape)
print(y_test.shape)
