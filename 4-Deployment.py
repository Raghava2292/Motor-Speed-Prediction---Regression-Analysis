# -*- coding: utf-8 -*-
"""
Created on Fri Jul 14 15:19:29 2023

@author: Raghava Varanasi
"""
## Import the libraries
import pandas as pd
import streamlit as st 
from pickle import load
from keras.models import load_model

# Title of the website
st.title('Model Deployment: Motor Speed Prediction')

# Choosing between predicting a single observation or an entire dataset
st.subheader('Single Prediction or Dataset Prediction')
predict_option = st.radio('Select One Option:', ('Single Prediction', 'Dataset Prediction'))

# Choosing a regressor
st.subheader('Select the Regression Model')
regressor = st.selectbox('Available Models arranged in descending order of their prediction accuracies: ', ('AdaBoost', 'Decision Tree', 'Random Forest', 'Bagging', 'KNN', 'Stacking', 'Gradient Boost', 'Neural Networks', 'Cubic Equation'))

# Loading the model based on the user selection
if regressor == 'AdaBoost':
    loaded_model = load(open('AdaBoost_model.sav', 'rb'))
    st.subheader('AdaBoost Regression Model')
    st.markdown('Model Accuracy   --   **99.999%**')
elif regressor == 'Random Forest':
    loaded_model = load(open('Random_Forest.sav', 'rb'))
    st.subheader('Random Forest Regression Model')
    st.markdown('Model Accuracy   --   **99.979%**')
elif regressor == 'Decision Tree':
    loaded_model = load(open('Decision_Tree.sav', 'rb'))
    st.subheader('Decision Tree Regression Model')
    st.markdown('Model Accuracy   --   **99.996%**')
elif regressor == 'Bagging':
    loaded_model = load(open('Bagging.sav', 'rb'))
    st.subheader('Bagging Regression Model')
    st.markdown('Model Accuracy   --   **99.979%**')
elif regressor == 'Gradient Boost':
    loaded_model = load(open('Gradient_Boosting.sav', 'rb'))
    st.subheader('Gradient Boost Regression Model')
    st.markdown('Model Accuracy   --   **99.452%**')
elif regressor == 'Stacking':
    loaded_model = load(open('Stacking.sav', 'rb'))
    st.subheader('Stacking Regression Model')
    st.markdown('Model Accuracy   --   **99.830%**')
elif regressor == 'KNN':
    loaded_model = load(open('KNN.sav', 'rb'))
    st.subheader('KNN Regression Model')
    st.markdown('Model Accuracy   --   **99.919%**')
elif regressor == 'Cubic Equation':
    loaded_model = load(open('Cubic.sav', 'rb'))
    st.subheader('Cubic Equation Regression Model')
    st.markdown('Model Accuracy   --   **96.586%**')
elif regressor == 'Neural Networks':
    loaded_model = load_model('ANN.h5')
    st.subheader('Neural Network Regression Model')
    st.markdown('Model Accuracy   --   **99.599%**')

# Single prediction
if predict_option == 'Single Prediction':
    st.sidebar.header('User Input Parameters')
    
    u_d = st.sidebar.number_input('Voltage D Component')
    u_q = st.sidebar.number_input('Voltage Q Component')
    i_d = st.sidebar.number_input('Current D Component')
    pm = st.sidebar.number_input("Permanent Magnet Temperature")
    data = {'u_d':u_d,
            'u_q':u_q,
            'i_d':i_d,
            'pm':pm}
    df = pd.DataFrame(data,index = [0])
        
    st.subheader('User Input parameters')
    st.write(df)
    st.subheader('Make Prediction:')
    if st.button('Predict'):
        if regressor in ['AdaBoost', 'Decision Tree', 'Random Forest', 'Bagging', 'KNN', 'Stacking', 'Gradient Boost', 'Neural Networks']:
            st.subheader(f'Predicted Result - {regressor} Regressor')
            st.markdown(f'The Motor Speed for the given voltage and current parameters is **:green[{(loaded_model.predict(df)[0]).round(6)}]**')
        elif regressor == 'Cubic Equation':
            df['i_d_squared'] = df['i_d'] * df['i_d']
            df['u_q_squared'] = df['u_q'] * df['u_q']
            df['i_d_cube'] = df['i_d'] * df['i_d'] * df['i_d']
            df['u_q_cube'] = df['u_q'] * df['u_q'] * df['u_q']
            st.subheader(f'Predicted Result - {regressor} Regressor')
            st.markdown(f'The Motor Speed for the given voltage and current parameters is **:green[{(loaded_model.predict(df)[0]).round(6)}]**')

# Dataset prediction
else:
    st.subheader('Upload the dataset')
    uploaded_file = st.file_uploader("Choose a file")

    if uploaded_file is not None: 
        df = pd.read_csv(uploaded_file)
        df = df[['u_d', 'u_q', 'i_d', 'pm']]
        
        st.subheader('Make Predictions:')
        if st.button('Predict'):
            if regressor in ['AdaBoost', 'Decision Tree', 'Random Forest', 'Bagging', 'KNN', 'Stacking', 'Gradient Boost', 'Neural Networks']:
                prediction = loaded_model.predict(df)
                st.subheader(f'Predicted Result - {regressor} Regressor')
                df['Motor_speed'] = prediction
                st.write(df)
            elif regressor == 'Cubic Equation':
                df['i_d_squared'] = df['i_d'] * df['i_d']
                df['u_q_squared'] = df['u_q'] * df['u_q']
                df['i_d_cube'] = df['i_d'] * df['i_d'] * df['i_d']
                df['u_q_cube'] = df['u_q'] * df['u_q'] * df['u_q']
                prediction = loaded_model.predict(df)
                st.subheader(f'Predicted Result - {regressor} Regressor')
                df['Motor_speed'] = prediction
                st.write(df)
            st.subheader('Download Predictions:')
            st.download_button('Download Predictions', data = df.to_csv().encode('utf-8'), file_name=f'Precited Data - {regressor} Regressor.csv', mime='text/csv')


        












    