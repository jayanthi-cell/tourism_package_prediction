import pandas as pd
from sklearn.model_selection import train_test_split

df = pd.read_csv("tourism_package/data/tourism.csv")

df.drop(columns=["Unnamed: 0", "CustomerID"], axis=1, inplace=True)

# Fix typo only — keep human-readable labels so they match app.py's input values
df["Gender"] = df["Gender"].replace({"Fe Male": "Female"})
df["MaritalStatus"] = df["MaritalStatus"].replace({"Unmarried": "Single"})

# Cap only the extreme outliers (99th percentile)
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
