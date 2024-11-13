import streamlit as st
import numpy as np
import pandas as pd
import joblib
from sklearn.feature_extraction.text import TfidfVectorizer
import pickle
from sklearn.metrics.pairwise import cosine_similarity

def load_data():
    try:
        # URL of the raw CSV file from GitHub
        url = 'https://raw.githubusercontent.com/JyLee98ImTrying/DietRecipeRecommendationSystem/master/df_DR.csv'
        
        # Load the data with proper encoding and handling of quotes
        df = pd.read_csv(url, delimiter=',', encoding='utf-8', on_bad_lines='skip', quotechar='"')

        # Check the column names and first few rows
        print("Columns in dataset:", df.columns)
        print(df.head())

        # Check if the columns of interest are loaded correctly
        required_columns = ['Calories', 'ProteinContent', 'FatContent', 'CarbohydrateContent', 
                            'SodiumContent', 'CholesterolContent', 'SaturatedFatContent']
        
        missing_columns = [col for col in required_columns if col not in df.columns]
        
        if missing_columns:
            raise ValueError(f"Missing columns: {', '.join(missing_columns)}")

        return df

    except Exception as e:
        print(f"Error loading data: {str(e)}")
        return None


def calculate_caloric_needs(gender, weight, height, age):
    if gender == "Female":
        BMR = 655 + (9.6 * weight) + (1.8 * height) - (4.7 * age)
    else:
        BMR = 66 + (13.7 * weight) + (5 * height) - (6.8 * age)
    return BMR
