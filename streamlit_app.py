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
        # Direct download link for Dropbox - ensure it's the direct download link
        url = 'https://www.dropbox.com/scl/fi/vasid7x99si4l40311m4q/df_MHMF.csv?dl=1'
        
        # Set up headers to mimic a browser request
        headers = {
            'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/91.0.4472.124 Safari/537.36'
        }
        
        # Download the file content
        response = requests.get(url, headers=headers)
        
        # Debug information
        st.write("Response status code:", response.status_code)
        st.write("Response content type:", response.headers.get('content-type', 'No content type'))
        
        if response.status_code != 200:
            st.error(f"Failed to download file: Status code {response.status_code}")
            return None
            
        # Check if we got HTML instead of CSV
        content_type = response.headers.get('content-type', '').lower()
        if 'text/html' in content_type:
            st.error("Received HTML instead of CSV data. Please check the download URL.")
            return None
            
        # Try to read the first few bytes to check if it's CSV
        try:
            content_start = response.content[:100].decode('utf-8')
            if '<!DOCTYPE' in content_start or '<html' in content_start:
                st.error("Received HTML content instead of CSV data. Please verify the download URL.")
                return None
        except:
            pass
            
        # Read CSV with specific handling for complex fields
        try:
            df = pd.read_csv(
                io.BytesIO(response.content),
                delimiter=',',
                encoding='utf-8',
                on_bad_lines='skip'
            )
            
            # Display the first few rows and columns for debugging
            st.write("DataFrame head:", df.head())
            st.write("DataFrame columns:", df.columns.tolist())
            
            # Clean up the Cluster column - ensure it's numeric
            if 'Cluster' in df.columns:
                df['Cluster'] = df['Cluster'].str.strip() if df['Cluster'].dtype == 'object' else df['Cluster']
                df['Cluster'] = pd.to_numeric(df['Cluster'], errors='coerce')
                df['Cluster'] = df['Cluster'].fillna(1)
                df['Cluster'] = df['Cluster'].astype(int)
                
                st.write("Unique clusters found:", df['Cluster'].unique())
                st.write("Cluster distribution:", df['Cluster'].value_counts())
            else:
                st.error(f"'Cluster' column not found. Available columns: {df.columns.tolist()}")
                st.error("Please ensure the CSV file contains the required 'Cluster' column.")
                return None
            
            return df
            
        except pd.errors.EmptyDataError:
            st.error("The CSV file appears to be empty.")
            return None
        except pd.errors.ParserError as e:
            st.error(f"Error parsing CSV file: {str(e)}")
            return None
            
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

# Function to calculate daily caloric needs
def calculate_caloric_needs(gender, weight, height, age):
    if gender == "Female":
        BMR = 655 + (9.6 * weight) + (1.8 * height) - (4.7 * age)
    else:
        BMR = 66 + (13.7 * weight) + (5 * height) - (6.8 * age)
    return BMR

def recommend_food(input_data, df, models):
    try:
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
        
        # Debug: Print cluster distribution
        cluster_dist = df['Cluster'].value_counts()
        st.write("Cluster distribution in dataset:", cluster_dist)
        
        # Filter dataset
        cluster_data = df[df['Cluster'] == cluster_label].copy()
        st.write(f"Number of items in selected cluster: {len(cluster_data)}")
        
        if cluster_data.empty:
            # If no exact cluster match, take nearest cluster
            unique_clusters = df['Cluster'].unique()
            if len(unique_clusters) > 0:
                # Get cluster centroids
                cluster_centers = models['kmeans'].cluster_centers_
                # Find nearest cluster
                distances = cosine_similarity(input_data_scaled, cluster_centers)
                nearest_cluster = unique_clusters[distances.argmax()]
                st.write(f"No matches in original cluster. Using nearest cluster: {nearest_cluster}")
                cluster_data = df[df['Cluster'] == nearest_cluster].copy()
            else:
                st.warning("No clusters found in the dataset.")
                return pd.DataFrame()
        
        # Ensure feature columns exist
        required_columns = ['Calories', 'ProteinContent', 'FatContent', 
                          'CarbohydrateContent', 'SodiumContent', 
                          'CholesterolContent', 'SaturatedFatContent']
        
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
        final_recommendations = cluster_data[cluster_data['Classification'] == 1].sort_values(
            by='Similarity', ascending=False
        )
        
        # If no recommendations after classification, return top similar items
        if final_recommendations.empty:
            st.warning("No items passed classification. Returning most similar items instead.")
            final_recommendations = cluster_data.sort_values(by='Similarity', ascending=False)
        
        return final_recommendations[['Name', 'Calories', 'ProteinContent', 'FatContent', 
                                    'CarbohydrateContent', 'SodiumContent', 'CholesterolContent', 
                                    'SaturatedFatContent', 'Similarity']].head(5)
                                    
    except Exception as e:
        st.error(f"Error in recommendation process: {str(e)}")
        st.write("Full error details:", e)
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
