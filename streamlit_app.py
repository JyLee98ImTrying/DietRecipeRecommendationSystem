import streamlit as st
import numpy as np
import pandas as pd
import joblib
from sklearn.feature_extraction.text import TfidfVectorizer
import pickle
from sklearn.metrics.pairwise import cosine_similarity
import gdown

# Clear cache to ensure fresh data loading
st.cache_data.clear()

# Add error handling and logging
def load_data():
    try:
        # Function to download CSV from Google Drive
        def download_file_from_gdrive(file_id, output_file):
            url = f'https://drive.google.com/uc?id={file_id}'
            gdown.download(url, output_file, quiet=False)

        file_id = '1qle68mxmhtaF5NPBV1VregS-dz-Q9sDG'
        csv_path = 'df_MHMF.csv'
        download_file_from_gdrive(file_id, csv_path)
        
        df = pd.read_csv(csv_path, delimiter=',', encoding='utf-8', on_bad_lines='skip')
        st.session_state['df'] = df
        return df
    except Exception as e:
        st.error(f"Error loading data: {str(e)}")
        return None

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
                st.write(f"{name} model loaded successfully.")
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

# Modified recommend_food function with better error handling
def recommend_food(input_data, df, models):
    try:
        # Ensure input_data is 2D
        input_data_reshaped = input_data.reshape(1, -1)
        
        # Scale the input data
        input_data_scaled = models['scaler'].transform(input_data_reshaped)
        
        # Get cluster prediction
        cluster_label = models['kmeans'].predict(input_data_scaled)[0]
        
        # Filter dataset
        cluster_data = df[df['Cluster'] == cluster_label].copy()
        
        if cluster_data.empty:
            st.warning("No matching foods found in the selected cluster.")
            return pd.DataFrame()
        
        # Ensure feature columns exist
        required_columns = ['Calories', 'ProteinContent', 'FatContent', 
                          'CarbohydrateContent', 'SodiumContent', 
                          'CholesterolContent', 'SaturatedFatContent']
        
        missing_columns = [col for col in required_columns if col not in cluster_data.columns]
        if missing_columns:
            st.error(f"Missing columns in dataset: {missing_columns}")
            return pd.DataFrame()
            
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
        
        # Filter and return recommendations
        final_recommendations = cluster_data[cluster_data['Classification'] == 1]
        return final_recommendations[['FoodName', 'Calories', 'ProteinContent', 'FatContent', 
                                    'CarbohydrateContent', 'SodiumContent', 'CholesterolContent', 
                                    'SaturatedFatContent', 'Similarity']].head(5)
                                    
    except Exception as e:
        st.error(f"Error in recommendation process: {str(e)}")
        return pd.DataFrame()

# Streamlit UI
st.title('🍅🧀MyHealthMyFood🥑🥬')

# Load data and models first
df = load_data()
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
        
        # Create input data array
        input_features = np.array([
            daily_calories,  # Calories
            0.8 * weight,    # Protein (default value)
            0.25 * daily_calories,  # Fats (default value)
            0.55 * daily_calories,  # Carbohydrates (default value)
            2000,           # Sodium (default value)
            200,            # Cholesterol (default value)
            20             # SaturatedFats (default value)
        ])
        
        recommendations = recommend_food(input_features, df, models)
        
        if not recommendations.empty:
            st.write("Recommended food items:")
            st.write(recommendations)
        else:
            st.warning("No recommendations found. Please try different inputs.")
else:
    st.error("Unable to load necessary data and models. Please check the file paths and try again.")
