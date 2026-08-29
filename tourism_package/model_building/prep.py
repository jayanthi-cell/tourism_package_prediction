import pandas as pd
from sklearn.model_selection import train_test_split

df = pd.read_csv("tourism_package/data/tourism.csv")

df.drop(columns=["Unnamed: 0","CustomerID"], axis=1, inplace=True)

df["Gender"] = df["Gender"].replace({"Male": "1", "Female": "2", "Fe Male" : "2"})

df["MaritalStatus"].replace("Unmarried", "Single", inplace=True)

#Fixing the data types
cols = df.select_dtypes(['object'])
cols.columns

for i in cols.columns:
    df[i] = df[i].astype('category')

# Cap only the extreme outliers (e.g., 99th percentile) rather than IQR bound
upper_cap = df['NumberOfTrips'].quantile(0.99)
df['NumberOfTrips'] = df['NumberOfTrips'].clip(upper=upper_cap)
print(f"NumberOfTrips capped at {upper_cap}")

X = df.drop(columns=["ProdTaken"])
y = df["ProdTaken"]

Xtrain, Xtest, ytrain, ytest = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)

Xtrain.to_csv("Xtrain.csv", index=False)
Xtest.to_csv("Xtest.csv", index=False)
ytrain.to_csv("ytrain.csv", index=False)
ytest.to_csv("ytest.csv", index=False)

print("Data prepared: train/test splits written.")
print("Type values kept as:", sorted(X["Type"].unique()))
