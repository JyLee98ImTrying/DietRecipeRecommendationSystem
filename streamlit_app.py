import streamlit as st
import numpy as np
import pandas as pd
import joblib
from sklearn.feature_extraction.text import TfidfVectorizer
import pickle
from sklearn.metrics.pairwise import cosine_similarity

df = pd.read_csv("Data/HealthData.csv")

# Load models
try:
    with open('kmeans.pkl', 'rb') as f:
        kmeans_model = pickle.load(f)
    st.write("KMeans model loaded successfully.")
    
    with open('rf_classifier.pkl', 'rb') as f:
        rf_model = pickle.load(f)
    st.write("Random Forest model loaded successfully.")

    with open('scaler.pkl', 'rb') as f:
        scaler = pickle.load(f)
    st.write("Scaler loaded successfully.")

except FileNotFoundError as e:
    st.error(f"Error loading model: {e}")
# Function to calculate daily caloric needs
def calculate_caloric_needs(gender, weight, height, age):
    if gender == "Female":
        BMR = 655 + (9.6 * weight) + (1.8 * height) - (4.7 * age)
    else:
        BMR = 66 + (13.7 * weight) + (5 * height) - (6.8 * age)
    return BMR

# Function to calculate nutrient requirements based on health condition
def calculate_nutrient_requirements(daily_calories, health_condition, weight):
    nutrient_requirements = {
        'Calories': daily_calories,
        'Protein': 0,
        'Fats': 0,
        'Carbohydrates': 0,
        'Sodium': 0,
        'Cholesterol': 0,
        'SaturatedFats': 0
    }
    
    if health_condition == "Diabetic":
        nutrient_requirements['Protein'] = 0.8 * weight
        nutrient_requirements['Fats'] = 0.25 * daily_calories
        nutrient_requirements['Carbohydrates'] = 0.6 * daily_calories
        nutrient_requirements['Sodium'] = 2000
        nutrient_requirements['Cholesterol'] = 200
    elif health_condition == "High Blood Pressure":
        nutrient_requirements['Protein'] = 0.5 * daily_calories / 4
        nutrient_requirements['Fats'] = 0.25 * daily_calories / 9
        nutrient_requirements['Carbohydrates'] = 0.6 * daily_calories / 4
        nutrient_requirements['Sodium'] = 1000
        nutrient_requirements['Cholesterol'] = 300
    elif health_condition == "High Cholesterol":
        nutrient_requirements['Protein'] = 0.5 * daily_calories / 4
        nutrient_requirements['Fats'] = daily_calories * 0.15 / 9
        nutrient_requirements['SaturatedFats'] = daily_calories * 0.10 / 9
        nutrient_requirements['Sodium'] = 2000
        nutrient_requirements['Cholesterol'] = 200
    
    return nutrient_requirements

# Define recommendation function
def recommend_food(input_data, health_condition=None):
    input_data_scaled = scaler.transform([input_data])
    predicted_cluster = rf_classifier.predict(input_data_scaled)[0]
    cluster_data = df[df['Cluster'] == predicted_cluster][nutrient_features]
    cluster_data_scaled = scaler.transform(cluster_data)
    similarities = cosine_similarity(input_data_scaled, cluster_data_scaled).flatten()
    cluster_data['Similarity'] = similarities
    top_recommendations = cluster_data.sort_values(by="Similarity", ascending=False).head(10)
    recommended_food_names = df.loc[top_recommendations.index, 'Name']
    return recommended_food_names


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
