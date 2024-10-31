import streamlit as st
import numpy as np
import pandas as pd
import joblib
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

# Streamlit UI
st.title('Diet and Recipe Recommendation System')

# User inputs
weight = st.number_input('Enter your weight (kg):', min_value=0.0, step=0.1)
height = st.number_input('Enter your height (cm):', min_value=0.0, step=0.1)
bmi = weight / ((height / 100) ** 2) if height > 0 else 0
st.write(f'Your BMI is: {bmi:.2f}')

# Allow multiple selections for health conditions
health_conditions = st.multiselect('Select your health condition(s):',
                                   ["No Non-Communicable Disease", "Diabetic", "High Blood Pressure", "High Cholesterol"])

wellness_goal = st.selectbox('Select your wellness goal:',
                             ["No goals", "Lose Fat", "Gain Muscle"])
