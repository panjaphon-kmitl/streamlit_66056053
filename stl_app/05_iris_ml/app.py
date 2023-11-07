import streamlit as st
import pandas as pd
import pickle
from rfc_func import random_forrest_classifier

# ---- Intro ----
st.title('Iris Classifier 🌸')
st.write("This app uses 4 inputs to predict the variety of iris using "
         "a model built on the Iris dataset. Use the form below"
         " to get started!")


# ---- Data source selection ----
st.header('1. Let\'s input the data for model training')
data_source_options = [None, 'Data provided by this app', 'My own dataset']
data_source = st.selectbox(
    'What data source do you want to use?',
    options=data_source_options,
    index=0
)


# If data source = provided data -> read the prepared pickle files
if data_source == data_source_options[1]:

    # Read data from pickle
    rf_pickle = open('pickle_files/random_forest_iris.pickle', 'rb')
    map_pickle = open('pickle_files/output_iris.pickle', 'rb')
    rfc = pickle.load(rf_pickle)
    uniques = pickle.load(map_pickle)
    rf_pickle.close()

# If user wants to upload their own data -> read and train data
elif data_source == data_source_options[2]:

    # Upload
    iris_file = st.file_uploader('Upload your own iris data')

    # Train
    if iris_file is not None:
        df = pd.read_csv(iris_file)
        rfc, uniques, score = random_forrest_classifier(
            df=df,
            x=['sepal.length', 'sepal.width', 'petal.length', 'petal.width'],
            y='variety'
        )
        st.write('We trained a Random Forest model on these data,'
                 ' it has a score of {}! Use the '
                 'inputs below to try out the model.'.format(round(score, 2)))