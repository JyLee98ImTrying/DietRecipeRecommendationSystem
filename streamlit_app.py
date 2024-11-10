# Import all required libraries at the top
import streamlit as st
import pandas as pd
import numpy as np
import requests
import io
import pickle
from sklearn.metrics.pairwise import cosine_similarity

# Function definitions
def load_data():
    try:
        # Modify URL to force download
        url = 'https://www.dropbox.com/scl/fi/vasid7x99si4l40311m4q/df_MHMF.csv?dl=1'
        
        # Download the file content
        response = requests.get(url)
        if response.status_code != 200:
            st.error(f"Failed to download file: Status code {response.status_code}")
            return None
            
        # Read CSV with specific handling for complex fields
        df = pd.read_csv(
            io.StringIO(response.content.decode('utf-8')),
            delimiter=',',
            encoding='utf-8',
            on_bad_lines='skip',
            quoting=1,
            escapechar='\\',
            doublequote=True
        )
        
        # Clean up the Cluster column - ensure it's numeric
        if 'Cluster' in df.columns:
            df['Cluster'] = df['Cluster'].str.strip() if df['Cluster'].dtype == 'object' else df['Cluster']
            df['Cluster'] = pd.to_numeric(df['Cluster'], errors='coerce')
            df['Cluster'] = df['Cluster'].fillna(1)
            df['Cluster'] = df['Cluster'].astype(int)
            
            st.write("Unique clusters found:", df['Cluster'].unique())
            st.write("Cluster distribution:", df['Cluster'].value_counts())
        else:
            st.error("'Cluster' column not found. Available columns:", df.columns.tolist())
        
        return df
        
    except Exception as e:
        st.error(f"Error loading data: {str(e)}")
        st.write("Full error details:", e)
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
        return models
    except Exception as e:
        st.error(f"Error loading models: {str(e)}")
        return None

def calculate_caloric_needs(gender, weight, height, age):
    if gender == "Female":
        BMR = 655 + (9.6 * weight) + (1.8 * height) - (4.7 * age)
    else:
        BMR = 66 + (13.7 * weight) + (5 * height) - (6.8 * age)
    return BMR

def recommend_food(input_data, df, models):
    try:
        st.write("Input features:", input_data)
        
        input_data_reshaped = input_data.reshape(1, -1)
        input_data_scaled = models['scaler'].transform(input_data_reshaped)
        
        cluster_label = models['kmeans'].predict(input_data_scaled)[0]
        st.write(f"Assigned cluster: {cluster_label}")
        
        cluster_data = df[df['Cluster'] == cluster_label].copy()
        st.write(f"Number of items in selected cluster: {len(cluster_data)}")
        
        if cluster_data.empty:
            st.warning(f"No items found in cluster {cluster_label}")
            return pd.DataFrame()
            
        required_columns = ['Name', 'Calories', 'ProteinContent', 'FatContent', 
                          'CarbohydrateContent', 'SodiumContent', 'CholesterolContent', 
                          'SaturatedFatContent']
        
        cluster_features = cluster_data[required_columns[1:]]  # Exclude Name
        cluster_features_scaled = models['scaler'].transform(cluster_features)
        
        similarities = cosine_similarity(input_data_scaled, cluster_features_scaled).flatten()
        cluster_data['Similarity'] = similarities
        
        # Sort by similarity
        recommendations = cluster_data.sort_values(by='Similarity', ascending=False)
        
        return recommendations[required_columns + ['Similarity']].head(5)
                                    
    except Exception as e:
        st.error(f"Error in recommendation process: {str(e)}")
        st.write("Full error details:", e)
        return pd.DataFrame()

# Main Streamlit app
st.title('🍅🧀MyHealthMyFood🥑🥬')

# Load data and models
df = load_data()
models = load_models()

if df is not None and models is not None:
    # User inputs
    gender = st.selectbox("Select your gender", ["Female", "Male"])
    weight = st.number_input("Enter your weight (kg)", min_value=30, max_value=200, value=70)
    height = st.number_input("Enter your height (cm)", min_value=100, max_value=250, value=160)
    age = st.number_input("Enter your age (years)", min_value=1, max_value=100, value=30)
    
    if st.button("Get Recommendations"):
        daily_calories = calculate_caloric_needs(gender, weight, height, age)
        
        # Calculate meal portions
        meal_fraction = 0.3  # 30% of daily values
        protein_grams = 0.8 * weight * meal_fraction
        fat_calories = 0.25 * daily_calories
        carb_calories = 0.55 * daily_calories
        
        fat_grams = (fat_calories / 9) * meal_fraction
        carb_grams = (carb_calories / 4) * meal_fraction
        
        input_features = np.array([
            daily_calories * meal_fraction,  # Calories per meal
            protein_grams,                   # Protein grams per meal
            fat_grams,                       # Fat grams per meal
            carb_grams,                      # Carb grams per meal
            2000 * meal_fraction,            # Sodium (mg) per meal
            200 * meal_fraction,             # Cholesterol (mg) per meal
            (fat_grams * 0.3)                # Saturated fats per meal
        ])
        
        recommendations = recommend_food(input_features, df, models)
        
        if not recommendations.empty:
            st.write("Recommended food items:")
            st.write(recommendations)
        else:
            st.warning("No recommendations found. Please try different inputs.")
else:
    st.error("Unable to load necessary data and models. Please check the error messages above.")
