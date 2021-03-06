import streamlit as st
import pandas as pd
import numpy as np
import pickle
from sklearn.ensemble import RandomForestClassifier
from sklearn import preprocessing

st.write("""
# Earning Prediction App
This app predicts your earning with AI """)

st.sidebar.header('User Input Features')



# Collects user input features into dataframe
uploaded_file = st.sidebar.file_uploader("Upload your input CSV file", type=["csv"])
if uploaded_file is not None:
    input_df = pd.read_csv(uploaded_file)
else:
    def user_input_features():
        
        
        Earning = st.sidebar.slider('Earning', 0,1000000)
        Expense = st.sidebar.slider('Expense', 0,1000000)
        
        data = {'Earning': Earning,
                'Expense': Expense}
               
        features = pd.DataFrame(data, index=[0])
        return features
    input_df = user_input_features()

# Combines user input features with entire penguins dataset
# This will be useful for the encoding phase
penguins_raw = pd.read_csv('penguins_cleaned.csv')
penguins = penguins_raw.drop(columns=['Savings'])
df = pd.concat([input_df,penguins],axis=0)

# Encoding of ordinal features
# https://www.kaggle.com/pratik1120/penguin-dataset-eda-classification-and-clustering
encode = ['Earning','Expense']
for col in encode:
    dummy = pd.get_dummies(df[col], prefix=col)
    df = pd.concat([df,dummy], axis=1)
    del df[col]
df = df[:1] # Selects only the first row (the user input data)




# Displays the user input features
st.subheader('User Input features')

if uploaded_file is not None:
    st.write(df)
else:
    st.write(' To get more accurate prediction upload up to 6 months data. Currently using example input parameters (shown below).')
    st.write(df)

  
clf = pd.read_csv('penguins_cleaned.csv')
clf = RandomForestClassifier()


df = np.array(df)
features = df[:,0]
labels = df[:,1:2]

classes=np.array(labels)
idx=labels==classes[0]
x=labels[df[:,1:2:1],:]
y=features[df[:,0:0],:]
if len(y.todense())> 0:
    if len(x.todense())> 0:
    
  
y = train['Selector']
clf.fit(train[features], train['Selector'])


# Apply model to make predictions
prediction = clf.predict(df[features])
prediction_proba= clf.predict_proba(df[features])




st.subheader('prediction')

st.write(penguins_Savings[prediction])

st.subheader('Prediction Probability')
st.write(prediction_proba)
