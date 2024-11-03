import streamlit as st
import numpy as np
import pandas as pd
import joblib
from sklearn.feature_extraction.text import TfidfVectorizer
import pickle
from sklearn.metrics.pairwise import cosine_similarity
import gdown

st.cache_data.clear()

# Function to download CSV from Google Drive
def download_file_from_gdrive(file_id, output_file):
    gdown.download(f'https://drive.google.com/uc?id={file_id}', output_file, quiet=False)

# Replace with your Google Drive file ID
file_id = '1qle68mxmhtaF5NPBV1VregS-dz-Q9sDG'
download_file_from_gdrive(file_id, 'df_MHMF.csv')

try:
    df = pd.read_csv("df_MHMF.csv", delimiter=',', encoding='utf-8', on_bad_lines='skip')
    st.write("Data loaded successfully.")
    st.write(df.head())  # Display first few rows to confirm structure
except Exception as e:
    st.write("Error loading file:", e)
    # Additional troubleshooting: print first few lines of the file to investigate
    with open("RecipeData.csv", 'r') as file:
        content = file.readlines()
        st.write("First few lines of the file:", content[:5])

# Proceed if df is loaded correctly
if 'df' in locals():
    st.write("Columns in DataFrame:", df.columns.tolist())

    # Ensure 'Cluster' column is present, otherwise assign clusters
    if 'Cluster' not in df.columns:
        # Prepare features for clustering
        features = df[['Calories', 'ProteinContent', 'FatContent', 
                       'CarbohydrateContent', 'SodiumContent', 
                       'CholesterolContent', 'SaturatedFatContent']].values
        cluster_labels = kmeans_model.predict(features)
        df['Cluster'] = cluster_labels

# Load models
try:
    with open('kmeans (1).pkl', 'rb') as f:
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
def recommend_food(input_data, df):
    # Scale the input data using the preloaded scaler
    input_data_scaled = scaler.transform([input_data])

    # Use KMeans to find the closest cluster to the user's input
    cluster_label = kmeans_model.predict(input_data_scaled)[0]
    
    # Filter the dataset to get food items in the predicted cluster
    cluster_data = df[df['Cluster'] == cluster_label]
    
    # Scale the nutrient features of the filtered data
    cluster_features_scaled = scaler.transform(cluster_data[['Calories', 'ProteinContent', 'FatContent', 
                                                             'CarbohydrateContent', 'SodiumContent', 
                                                             'CholesterolContent', 'SaturatedFatContent']])
    
    # Calculate cosine similarity between the user's input and each food item in the cluster
    similarities = cosine_similarity(input_data_scaled, cluster_features_scaled).flatten()
    
    # Add similarity scores to the cluster data and sort by similarity
    cluster_data = cluster_data.copy()  # To avoid modifying original df
    cluster_data['Similarity'] = similarities
    recommended_foods = cluster_data.sort_values(by='Similarity', ascending=False)

    # Use Random Forest classifier to further filter recommendations based on health condition or wellness goal
    rf_predictions = rf_model.predict(cluster_features_scaled)
    recommended_foods['Classification'] = rf_predictions

    # Filter recommended foods based on the Random Forest classification
    # For simplicity, assume a binary classification where '1' is the preferred class
    # (You may adjust this based on your model’s specific classes)
    final_recommendations = recommended_foods[recommended_foods['Classification'] == 1]

    # Return top 5 recommendations
    return final_recommendations[['FoodName', 'Calories', 'ProteinContent', 'FatContent', 
                                  'CarbohydrateContent', 'SodiumContent', 'CholesterolContent', 
                                  'SaturatedFatContent', 'Similarity']].head(5)


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
    
    recommendations = recommend_food(input_data, df)
    st.write("Recommended food items:")
    st.write(recommendations)
