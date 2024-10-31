import streamlit as st
import numpy as np
import pandas as pd
import joblib
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

# Streamlit UI
# Display an image at the top
# Center and resize the image using HTML & CSS
st.markdown(
    """
    <div style="display: flex; justify-content: center;">
        <img src="https://img.freepik.com/free-vector/main-food-groups-macronutrients-vector_1308-130027.jpg?t=st=1730365974~exp=1730369574~hmac=72044aa7f1e2f1012e0497ad08724dcb894287c97946d34f69ddf08d8f017c0b&w=740" alt="Healthy Eating" width="250">
    </div>
    """,
    unsafe_allow_html=True
)

#Streamlit App title
st.title('🍅🧀Diet and Recipe Recommendation System🥑🥬')

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
