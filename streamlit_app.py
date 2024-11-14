import streamlit as st
import numpy as np
import pandas as pd
import joblib
from sklearn.feature_extraction.text import TfidfVectorizer
import pickle
from sklearn.metrics.pairwise import cosine_similarity

# Clear cache to ensure fresh data loading
st.cache_data.clear()

# Load data
def load_data():
    try:
        url = 'https://raw.githubusercontent.com/JyLee98ImTrying/DietRecipeRecommendationSystem/main/df_sample.csv'
        df = pd.read_csv(url, delimiter=',', encoding='utf-8', on_bad_lines='skip')
        
        # Add clustering step here after loading the data
        if 'Cluster' not in df.columns and 'kmeans' in st.session_state.get('models', {}):
            features = df[['Calories', 'ProteinContent', 'FatContent', 
                           'CarbohydrateContent', 'SodiumContent', 
                           'CholesterolContent', 'SaturatedFatContent']]
            scaled_features = st.session_state['models']['scaler'].transform(features)
            df['Cluster'] = st.session_state['models']['kmeans'].predict(scaled_features)
        
        st.session_state['df'] = df
        return df
    except Exception as e:
        st.error(f"Error loading data: {str(e)}")
        return None

# Load models
def load_models():
    try:
        model_files = {'kmeans': 'kmeans.pkl', 'rf_classifier': 'rf_classifier.pkl', 'scaler': 'scaler.pkl'}
        models = {}
        for name, file in model_files.items():
            with open(file, 'rb') as f:
                models[name] = pickle.load(f)
        st.session_state['models'] = models
        return models
    except Exception as e:
        st.error(f"Error loading models: {str(e)}")
        return None

# Recommendation function
def recommend_food(input_data, df, models, start_idx=0, num_items=5):
    try:
        input_data_reshaped = input_data.reshape(1, -1)
        input_data_scaled = models['scaler'].transform(input_data_reshaped)
        cluster_label = models['kmeans'].predict(input_data_scaled)[0]
        cluster_data = df[df['Cluster'] == cluster_label].copy()
        
        if cluster_data.empty:
            unique_clusters = df['Cluster'].unique()
            if len(unique_clusters) > 0:
                cluster_centers = models['kmeans'].cluster_centers_
                distances = cosine_similarity(input_data_scaled, cluster_centers)
                nearest_cluster = unique_clusters[distances.argmax()]
                cluster_data = df[df['Cluster'] == nearest_cluster].copy()
            else:
                st.warning("No clusters found in the dataset.")
                return pd.DataFrame()
        
        required_columns = ['Calories', 'ProteinContent', 'FatContent', 
                            'CarbohydrateContent', 'SodiumContent', 
                            'CholesterolContent', 'SaturatedFatContent']
        cluster_features = cluster_data[required_columns]
        cluster_features_scaled = models['scaler'].transform(cluster_features)
        similarities = cosine_similarity(input_data_scaled, cluster_features_scaled).flatten()
        
        cluster_data['Similarity'] = similarities
        rf_predictions = models['rf_classifier'].predict(cluster_features_scaled)
        cluster_data['Classification'] = rf_predictions
        final_recommendations = cluster_data[cluster_data['Classification'] == 1].sort_values(
            by='Similarity', ascending=False
        )
        
        if final_recommendations.empty:
            st.warning("No items passed classification. Returning most similar items instead.")
            final_recommendations = cluster_data.sort_values(by='Similarity', ascending=False)
        
        return final_recommendations[['Name', 'Calories', 'ProteinContent', 'FatContent', 
                                      'CarbohydrateContent', 'SodiumContent', 'CholesterolContent', 
                                      'SaturatedFatContent', 'SugarContent', 'RecipeInstructions', 'Similarity']].iloc[start_idx:start_idx + num_items]
    except Exception as e:
        st.error(f"Error in recommendation process: {str(e)}")
        return pd.DataFrame()

# Streamlit UI
st.title('🍅🧀MyHealthMyFood🥑🥬')
df = load_data()
models = load_models()

if df is not None and models is not None:
    gender = st.selectbox("Select your gender", ["Female", "Male"])
    weight = st.number_input("Enter your weight (kg)", min_value=30, max_value=200, value=70)
    height = st.number_input("Enter your height (cm)", min_value=100, max_value=250, value=160)
    age = st.number_input("Enter your age (years)", min_value=1, max_value=100, value=30)
    health_condition = st.selectbox("Select your health condition", 
                                    ["No Non-Communicable Disease", "Diabetic", "High Blood Pressure", "High Cholesterol"])
    
    if health_condition == "No Non-Communicable Disease":
        wellness_goal = st.selectbox("Select your wellness goal", ["Maintain Weight", "Lose Weight", "Muscle Gain"])
    
if st.button("Get Recommendations"):
    daily_calories = calculate_caloric_needs(gender, weight, height, age)
    protein_grams = 0.8 * weight
    fat_calories = 0.25 * daily_calories
    carb_calories = 0.55 * daily_calories
    fat_grams = fat_calories / 9
    carb_grams = carb_calories / 4
    meal_fraction = 0.3
    input_features = np.array([
        daily_calories * meal_fraction, protein_grams * meal_fraction, fat_grams * meal_fraction,
        carb_grams * meal_fraction, 2000 * meal_fraction, 200 * meal_fraction,
        (fat_grams * 0.3) * meal_fraction
    ])
    
    st.session_state['recommendation_idx'] = 0
    recommendations = recommend_food(input_features, df, models, start_idx=st.session_state['recommendation_idx'])
    
    if not recommendations.empty:
        st.write("Recommended food items:")
        st.write(recommendations)
    else:
        st.warning("No recommendations found. Please try different inputs.")

if st.button("Next Recommendations"):
    st.session_state['recommendation_idx'] += 5
    recommendations = recommend_food(input_features, df, models, start_idx=st.session_state['recommendation_idx'])
    
    if not recommendations.empty:
        st.write("Next recommended food items:")
        st.write(recommendations)
    else:
        st.warning("No more recommendations available.")
