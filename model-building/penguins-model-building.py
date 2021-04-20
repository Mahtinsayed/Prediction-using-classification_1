import pandas as pd

# Ordinal feature encoding
# https://www.kaggle.com/pratik1120/penguin-dataset-eda-classification-and-clustering
df = penguins.copy()
target = 'Savings'
encode = ['Earning','Expense']

for col in encode:
    dummy = pd.get_dummies(df[col], prefix=col)
    df = pd.concat([df,dummy], axis=1)
    del df[col]

def target_encode(val):
    return target_mapper[val]

df['Savings'] = df['Savings'].apply(target_encode)

# Separating X and y
X = df.drop('Savings', axis=1)
Y = df['Savings']

# Build random forest model
from sklearn.ensemble import RandomForestClassifier
clf = RandomForestClassifier()
clf.fit(X, Y)


