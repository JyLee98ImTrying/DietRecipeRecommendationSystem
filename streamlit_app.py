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


def load_data():
    try:
        # Convert Dropbox share link to direct download link
        url = 'https://www.dropbox.com/scl/fi/vasid7x99si4l40311m4q/df_MHMF.csv?dl=1'
        
        # Load the data
        df = pd.read_csv(url, delimiter=',', encoding='utf-8')
        
        # Debug information
        st.write("DataFrame Info:")
        st.write(df.info())
        
        st.write("\nColumn names (with lengths):")
        for col in df.columns:
            st.write(f"'{col}' - Length: {len(col)}")
        
        st.write("\nFirst few rows of 'Cluster' column (if exists):")
        if 'Cluster' in df.columns:
            st.write(df['Cluster'].head())
        else:
            # Check for similar column names
            similar_cols = [col for col in df.columns if 'cluster' in col.lower()]
            st.write("Columns containing 'cluster' (case-insensitive):", similar_cols)
            
            # Print all column names for debugging
            st.write("\nAll column names:")
            for idx, col in enumerate(df.columns):
                st.write(f"{idx}. {col!r}")  # Using repr to show hidden characters
        
        # Try to load the column with different methods
        try:
            cluster_col = df.get('Cluster')
            st.write("\nUsing df.get('Cluster'):", cluster_col is not None)
        except Exception as e:
            st.write("Error with df.get:", str(e))
            
        try:
            cluster_col = df.iloc[:, df.columns.get_loc('Cluster')]
            st.write("\nUsing get_loc:", cluster_col.head())
        except Exception as e:
            st.write("Error with get_loc:", str(e))
        
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
        
        # Defensive check for Cluster column
        if 'Cluster' not in df.columns:
            st.error("Cluster column missing from DataFrame")
            # Try to find the correct column name
            cluster_cols = [col for col in df.columns if 'cluster' in col.lower()]
            if cluster_cols:
                st.write("Found potential cluster columns:", cluster_cols)
                # Use the first matching column
                df['Cluster'] = df[cluster_cols[0]]
            else:
                raise KeyError("No cluster column found in the dataset")
        
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
if df is not None:
    st.write("Data loaded successfully!")
    st.write("Number of rows:", len(df))
    st.write("Number of columns:", len(df.columns))
    
    # Display sample of the data
    st.write("\nSample of loaded data:")
    st.write(df.head())
else:
    st.error("Failed to load data. Please check the error messages above.")

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
