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


# ---- Data preparation ----
ready_for_prediction = False

# If data source = provided data -> read the prepared pickle files
if data_source == data_source_options[1]:

    # Read data from pickle
    rf_pickle = open('files/pickle/random_forest_iris.pickle', 'rb')
    map_pickle = open('files/pickle/output_iris.pickle', 'rb')
    rfc = pickle.load(rf_pickle)
    uniques = pickle.load(map_pickle)
    rf_pickle.close()
    ready_for_prediction = True

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
        ready_for_prediction = True


# ---- Make prediction ----
if ready_for_prediction:
    st.header('2. What does your iris look like? Enter below to find out what variety it might be')

    # Form
    with st.form('user_inputs'):
        sepal_length = st.number_input('Sepal Length', min_value=0.0, value=0.0, step=0.1)
        sepal_width = st.number_input('Sepal Width', min_value=0.0, value=0.0, step=0.1)
        petal_length = st.number_input('Petal Length', min_value=0.0, value=0.0, step=0.1)
        petal_width = st.number_input('Petal Width', min_value=0.0, value=0.0, step=0.1)
        submit = st.form_submit_button()

    # If form button is pressed -> make prediction
    if submit:
        new_prediction = rfc.predict([[sepal_length, sepal_width, petal_length, petal_width]])
        prediction_variety = uniques[new_prediction][0]
        st.write('We predict your iris is of the **{}** variety'.format(prediction_variety))