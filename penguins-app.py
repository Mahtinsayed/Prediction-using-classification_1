import streamlit as st
import pandas as pd
import numpy as np
import pickle
from sklearn.ensemble import RandomForestClassifier

st.write("""
# Earning Prediction Calculator

This calculator predicts your earning with Artfictial Intelligience """)



st.sidebar.header('User financial datas ')

# Collects user input features into dataframe
uploaded_file = st.sidebar.file_uploader("Upload your input CSV file", type=["csv"])
if uploaded_file is not None:
    input_df = pd.read_csv(uploaded_file)
else:
    def user_input_features():
        Proffesion_Type = st.sidebar.selectbox('Proffesion Type',('Businessmen','Employee','Freelancer'))
          
        Monthly_Earning = st.sidebar.slider('Monthly Earning', 0,1000000)
        Monthly_Expense = st.sidebar.slider('Monthly Expense', 0,1000000)
       
        data = {'Monthly_Earning': Monthly Earning,
                'Monthly_Expense': Monthly Expense,}
               
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
encode = ['Monthly Earning','Monthly Expense']
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
    st.write('Upload up to 6 month financial datas to get more ccurate result. Currently using example input parameters (shown below).')
    st.write(df)



# Apply model to make predictions
prediction = load_clf.predict(df)
prediction_proba = load_clf.predict_proba(df)


st.subheader('Prediction')

st.write(Estimated_Savings[prediction])

st.subheader('Prediction Probability')
st.write(prediction_proba)
