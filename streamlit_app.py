import streamlit as st
import numpy as np
import pandas as pd
import joblib
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity


st.title('🍅 Diet & Recipe Recommendation App')

st.info('This app provides recipe recommendations based on your health conditions, BMI and wellness goals')

# User inputs
weight = st.number_input('Enter your weight (kg):', min_value=0)
height = st.number_input('Enter your height (cm):', min_value=0)
bmi = weight / ((height / 100) ** 2) if height > 0 else 0
st.write(f'Your BMI is: {bmi:.2f}')

health_condition = st.selectbox('Select your health condition:', 
                                ["No Non-Communicable Disease", "Diabetic", "High Blood Pressure", "High Cholesterol"])

wellness_goal = st.selectbox('Select your wellness goal:', 
                             ["No goals", "Lose Fat", "Gain Muscle"])

calorie = st.number_input('Enter your daily calorie requirement:', min_value=0)

# Define filtering functions
def filter_for_diabetic(df, weight, calorie):
    protein_requirement = 0.8 * weight
    fat_limit = 0.25 * calorie
    carb_limit = 0.6 * calorie
    sodium_limit = 2000
    cholesterol_limit = 200
    
    return df[(df['ProteinContent'] <= protein_requirement) & 
              (df['FatContent'] <= fat_limit) &
              (df['CarbohydrateContent'] <= carb_limit) &
              (df['SodiumContent'] < sodium_limit) &
              (df['CholesterolContent'] < cholesterol_limit)]

def filter_for_high_blood_pressure(df, calorie):
    protein_limit = 0.5 * calorie
    fat_limit = 0.25 * calorie
    carb_limit = 0.6 * calorie
    sodium_limit = 1000
    cholesterol_limit = 300
    
    return df[(df['ProteinContent'] <= protein_limit) & 
              (df['FatContent'] <= fat_limit) &
              (df['CarbohydrateContent'] <= carb_limit) &
              (df['SodiumContent'] < sodium_limit) &
              (df['CholesterolContent'] < cholesterol_limit)]

def filter_for_high_cholesterol(df, calorie):
    fat_limit = 0.15 * calorie
    saturated_fat_limit = 0.1 * calorie
    
    return df[(df['FatContent'] <= fat_limit) & 
              (df['SaturatedFatContent'] < saturated_fat_limit)]

# Preprocess data for content-based filtering
# Create TF-IDF vectors for the combined text
tfidf_vectorizer = TfidfVectorizer(stop_words='english')
tfidf_matrix = tfidf_vectorizer.fit_transform(df['CombinedText'])

