import streamlit as st
import pandas as pd
import pickle
from rfc_func import random_forrest_classifier

# ---- Intro ----
st.title('Iris Classifier 🌸')
st.write("This app uses 4 inputs to predict the variety of iris using "
         "a model built on the Iris dataset. Use the form below"
         " to get started!")