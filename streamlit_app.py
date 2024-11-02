import streamlit as st
import numpy as np
import pandas as pd
import joblib
from sklearn.feature_extraction.text import TfidfVectorizer
import pickle
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

# Streamlit App title
st.title('🍅🧀MyHealthMyFood🥑🥬')

# User inputs
gender = st.selectbox("Select your gender", ["Female", "Male"])
weight = st.number_input("Enter your weight (kg)", min_value=30, max_value=200, value=70)
height = st.number_input("Enter your height (cm)", min_value=100, max_value=250, value=160)
age = st.number_input("Enter your age (years)", min_value=1, max_value=100, value=30)
health_condition = st.selectbox("Select your health condition", 
                                 ["No Non-Communicable Disease", "Diabetic", "High Blood Pressure", "High Cholesterol"])
wellness_goal = None

# Conditional wellness goal input
if health_condition == "No Non-Communicable Disease":
    wellness_goal = st.selectbox("Select your wellness goal", ["Maintain Weight", "Lose Weight", "Muscle Gain"])

# Button to calculate and get recommendations
if st.button("Get Recommendations"):
    daily_calories = calculate_caloric_needs(gender, weight, height, age)
    nutrient_requirements = calculate_nutrient_requirements(daily_calories, health_condition, weight)
    
    input_data = np.array([
        nutrient_requirements['Calories'],
        nutrient_requirements['Protein'],
        nutrient_requirements['Fats'],
        nutrient_requirements['Carbohydrates'],
        nutrient_requirements['Sodium'],
        nutrient_requirements['Cholesterol'],
        nutrient_requirements['SaturatedFats']
    ])
    
    recommendations = recommend_food(input_data, health_condition)
    st.write("Recommended food items:")
    st.write(recommendations)
