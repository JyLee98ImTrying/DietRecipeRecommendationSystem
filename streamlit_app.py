import streamlit as st
import pandas as pd
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
import pickle
from sklearn.metrics.pairwise import cosine_similarity
import os

def load_dataset():
    try:
        # Debug: Print current directory and files
        st.write("Current directory contents:", os.listdir())
        
        # Load the dataset
        df = pd.read_csv('df_DR.csv')
        
        # Debug information
        st.write("DataFrame loaded successfully")
        st.write("Shape:", df.shape)
        st.write("Columns:", list(df.columns))
        
        return df
    except Exception as e:
        st.error(f"Error loading dataset: {str(e)}")
        return None

def load_models():
    try:
        models = {}
        model_files = ['kmeans.pkl', 'rf_classifier.pkl', 'scaler.pkl']
        
        for file in model_files:
            with open(file, 'rb') as f:
                model_name = file.replace('.pkl', '')
                models[model_name] = pickle.load(f)
        
        st.success("Models loaded successfully")
        return models
    except Exception as e:
        st.error(f"Error loading models: {str(e)}")
        st.write("Current directory:", os.getcwd())
        st.write("Files available:", os.listdir())
        return None

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
        
        # Filter dataset
        cluster_data = df[df['Cluster'] == cluster_label].copy()
        st.write(f"Number of items in selected cluster: {len(cluster_data)}")
        
        if cluster_data.empty:
            st.warning("No items found in the selected cluster.")
            return pd.DataFrame()
            
        # Ensure feature columns exist
        required_columns = ['Calories', 'ProteinContent', 'FatContent', 'CarbohydrateContent', 
                          'SodiumContent', 'CholesterolContent', 'SaturatedFatContent']
        
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

def main():
    st.title("Diet Recipe Recommendation System")
    
    # Load data and models
    df = load_dataset()
    if df is not None:
        st.write("Dataset preview:")
        st.write(df.head())
    else:
        st.error("Failed to load dataset")
        return
        
    models = load_models()
    if models is None:
        st.error("Failed to load models")
        return
    
    # Create input fields for user preferences
    st.subheader("Enter Your Nutritional Preferences")
    
    calories = st.number_input("Calories", min_value=0.0, max_value=2000.0, value=500.0)
    protein = st.number_input("Protein (g)", min_value=0.0, max_value=100.0, value=20.0)
    fat = st.number_input("Fat (g)", min_value=0.0, max_value=100.0, value=15.0)
    carbs = st.number_input("Carbohydrates (g)", min_value=0.0, max_value=200.0, value=50.0)
    sodium = st.number_input("Sodium (mg)", min_value=0.0, max_value=2000.0, value=500.0)
    cholesterol = st.number_input("Cholesterol (mg)", min_value=0.0, max_value=300.0, value=50.0)
    saturated_fat = st.number_input("Saturated Fat (g)", min_value=0.0, max_value=50.0, value=5.0)
    
    if st.button("Get Recommendations"):
        # Prepare input data
        input_features = np.array([calories, protein, fat, carbs, sodium, cholesterol, saturated_fat])
        
        # Get recommendations
        recommendations = recommend_food(input_features, df, models)
        
        if not recommendations.empty:
            st.subheader("Recommended Recipes:")
            st.write(recommendations)
        else:
            st.warning("No recommendations found. Please try different inputs.")

if __name__ == "__main__":
    main()
