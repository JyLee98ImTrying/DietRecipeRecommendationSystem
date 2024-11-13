import streamlit as st
import numpy as np
import pandas as pd
import joblib
from sklearn.feature_extraction.text import TfidfVectorizer
import pickle
from sklearn.metrics.pairwise import cosine_similarity
from pathlib import path
import os

# Load the dataset
file_path = Path(__file__).parent / 'df_DR.csv'
df = pd.read_csv(file_path, 
                 dtype={'Cluster': float}, 
                 encoding='utf-8')

st.write("Current working directory:", os.getcwd())
st.write("Files in current directory:", os.listdir())

# Clear cache to ensure fresh data loading
st.cache_data.clear()

def load_models():
    try:
        model_files = {
            'kmeans': 'kmeans.pkl',
            'rf_classifier': 'rf_classifier.pkl',
            'scaler': 'scaler.pkl'
        }
        
        models = {}
        for name, file in model_files.items():
            with open(file, 'rb') as f:
                models[name] = pickle.load(f)
        
        # Store models in session state
        st.session_state['models'] = models
        return models
    except Exception as e:
        st.error(f"Error loading models: {str(e)}")
        return None

# Function to calculate daily caloric needs
def calculate_caloric_needs(gender, weight, height, age):
    if gender == "Female":
        BMR = 655 + (9.6 * weight) + (1.8 * height) - (4.7 * age)
    else:
        BMR = 66 + (13.7 * weight) + (5 * height) - (6.8 * age)
    return BMR

def recommend_food(input_data, df, models):
    try:
        # Debug: Print DataFrame info
        st.write("DataFrame shape:", df.shape)
        st.write("DataFrame dtypes:", df.dtypes)
        st.write("First few rows:")
        st.write(df.head())
        
        if df is None or df.empty:
            st.error("DataFrame is empty or None")
            return pd.DataFrame()

        
        # Debug: Print input data
        st.write("Input features:", input_data)

        # Ensure input_data is 2D
        input_data_reshaped = input_data.reshape(1, -1)

        # Scale the input data
        input_data_scaled = models['scaler'].transform(input_data_reshaped)

        # Debug: Print scaled input
        st.write("Scaled input:", input_data_scaled)

        # Get cluster prediction
        cluster_label = models['kmeans'].predict(input_data_scaled)[0]
        st.write(f"Assigned cluster: {cluster_label}")

        # Debug: Print column names
        print(df.columns)

        # Debug: Print cluster distribution
        if 'Cluster' in df.columns:
            df['Cluster']=pd.to_numeric(df['Cluster'], errors='coerce')
            cluster_dist = df['Cluster'].value_counts()
            st.write("Cluster distribution in dataset:", cluster_dist)
        else:
            st.warning("The 'Cluster' column is not found in the DataFrame.")

        # Filter dataset
        if 'Cluster' in df.columns:
            cluster_data = df[df['Cluster'] == cluster_label].copy()
            st.write(f"Number of items in selected cluster: {len(cluster_data)}")
        else:
            st.warning("The 'Cluster' column is not found in the DataFrame.")
            return pd.DataFrame()

        # Ensure feature columns exist
        required_columns = ['Calories', 'ProteinContent', 'FatContent', 'CarbohydrateContent', 'SodiumContent', 'CholesterolContent', 'SaturatedFatContent']

        # Scale features
        cluster_features = cluster_data[required_columns]
        cluster_features_scaled = models['scaler'].transform(cluster_features)

        # Calculate similarities
        similarities = cosine_similarity(input_data_scaled, cluster_features_scaled).flatten()

        # Add similarity scores and sort
        cluster_data['Similarity'] = similarities

        # Get RF predictions
        rf_predictions = models['rf_classifier'].predict(cluster_features_scaled)
        cluster_data['Classification'] = rf_predictions

        # Filter and sort by similarity
        final_recommendations = cluster_data[cluster_data['Classification'] == 1].sort_values(by='Similarity', ascending=False)

        # If no recommendations after classification, return top similar items
        if final_recommendations.empty:
            st.warning("No items passed classification. Returning most similar items instead.")
            final_recommendations = cluster_data.sort_values(by='Similarity', ascending=False)

        return final_recommendations[['Name', 'Calories', 'ProteinContent', 'FatContent', 'CarbohydrateContent', 'SodiumContent', 'CholesterolContent', 'SaturatedFatContent', 'Similarity']].head(5)

    except Exception as e:
        st.error(f"Error in recommendation process: {str(e)}")
        st.write("Full error details:", e)
        st.write("DataFrame columns:", df.columns)
        return pd.DataFrame()

# Streamlit UI
st.title('🍅🧀MyHealthMyFood🥑🥬')

# Load data and models first
df = pd.read_csv('df_DR.csv')
models = load_models()

if df is not None and models is not None:
    # User inputs
    gender = st.selectbox("Select your gender", ["Female", "Male"])
    weight = st.number_input("Enter your weight (kg)", min_value=30, max_value=200, value=70)
    height = st.number_input("Enter your height (cm)", min_value=100, max_value=250, value=160)
    age = st.number_input("Enter your age (years)", min_value=1, max_value=100, value=30)
    health_condition = st.selectbox("Select your health condition", 
                                  ["No Non-Communicable Disease", "Diabetic", "High Blood Pressure", "High Cholesterol"])
    
    if health_condition == "No Non-Communicable Disease":
        wellness_goal = st.selectbox("Select your wellness goal", 
                                   ["Maintain Weight", "Lose Weight", "Muscle Gain"])
    
if st.button("Get Recommendations"):
    daily_calories = calculate_caloric_needs(gender, weight, height, age)
    
    # Adjust input features based on daily caloric needs
    protein_grams = 0.8 * weight  # 0.8g per kg of body weight
    fat_calories = 0.25 * daily_calories  # 25% of daily calories
    carb_calories = 0.55 * daily_calories  # 55% of daily calories
    
    # Convert calories to grams for macronutrients
    fat_grams = fat_calories / 9  # 9 calories per gram of fat
    carb_grams = carb_calories / 4  # 4 calories per gram of carb
    
    # Create scaled-down input features for single meal recommendation
    meal_fraction = 0.3  # Assuming this is for a single meal (30% of daily values)
    input_features = np.array([
        daily_calories * meal_fraction,  # Calories per meal
        protein_grams * meal_fraction,   # Protein grams per meal
        fat_grams * meal_fraction,       # Fat grams per meal
        carb_grams * meal_fraction,      # Carb grams per meal
        2000 * meal_fraction,            # Sodium (mg) per meal
        200 * meal_fraction,             # Cholesterol (mg) per meal
        (fat_grams * 0.3) * meal_fraction # Saturated fats (30% of total fats) per meal
    ])
    
    recommendations = recommend_food(input_features, df, models)
    
    if not recommendations.empty:
        st.write("Recommended food items:")
        st.write(recommendations)
    else:
        st.warning("No recommendations found. Please try different inputs.")
