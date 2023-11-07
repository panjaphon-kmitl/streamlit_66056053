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