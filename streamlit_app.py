import streamlit as st
import numpy as np
import pandas as pd
import pickle
import os
import boto3
from sklearn.metrics.pairwise import cosine_similarity
from dataclasses import dataclass
from typing import Dict, Optional, List, Any
from dotenv import load_dotenv
from s3_config import download_csv_from_s3

# Load environment variables from .env file
load_dotenv()

def download_csv_from_s3():
    s3 = boto3.client(
        's3',
        aws_access_key_id=os.getenv("AWS_ACCESS_KEY_ID"),
        aws_secret_access_key=os.getenv("AWS_SECRET_ACCESS_KEY"),
    )
    bucket_name = os.getenv("S3_BUCKET_NAME")
    s3_file_key = os.getenv("S3_FILE_KEY")
    local_file_path = "df_DR.csv"  # Local file path to save the downloaded CSV

    # Download the file from S3
    s3.download_file(bucket_name, s3_file_key, local_file_path)
    return local_file_path

# Configuration and Constants
@dataclass
class Config:
    MIN_WEIGHT: int = 30
    MAX_WEIGHT: int = 200
    MIN_HEIGHT: int = 100
    MAX_HEIGHT: int = 250
    MIN_AGE: int = 1
    MAX_AGE: int = 100
    MEAL_FRACTION: float = 0.3
    DEFAULT_SODIUM: int = 2000
    DEFAULT_CHOLESTEROL: int = 200
    SAT_FAT_RATIO: float = 0.3
    PROTEIN_PER_KG: float = 0.8
    FAT_CALORIES_RATIO: float = 0.25
    CARB_CALORIES_RATIO: float = 0.55
    FAT_CALORIES_PER_GRAM: int = 9
    CARB_CALORIES_PER_GRAM: int = 4

class DataLoader:
    @staticmethod
    @st.cache_data
    def load_dataset() -> Optional[pd.DataFrame]:
        """Load and cache the dataset from S3."""
        try:
            # Download the CSV file from S3 using the helper function
            local_csv_path = download_csv_from_s3()
            
            # Read the downloaded CSV file
            df = pd.read_csv(local_csv_path)
            
            # Debug information
            st.write("Dataset loaded successfully from S3")
            st.write(f"Original shape: {df.shape}")
            st.write("Columns:", list(df.columns))
            
            # Basic data validation
            required_columns = [
                'Name', 'Calories', 'ProteinContent', 'FatContent', 
                'CarbohydrateContent', 'SodiumContent', 'CholesterolContent', 
                'SaturatedFatContent', 'Cluster'
            ]
            
            missing_columns = [col for col in required_columns if col not in df.columns]
            if missing_columns:
                st.error(f"Missing required columns: {missing_columns}")
                return None
                
            return df
            
        except Exception as e:
            st.error(f"Error loading dataset from S3: {str(e)}")
            return None

    @staticmethod
    def load_models() -> Optional[Dict[str, Any]]:
        """Load all required models."""
        try:
            models = {}
            model_files = ['kmeans.pkl', 'rf_classifier.pkl', 'scaler.pkl']
            
            for file in model_files:
                with open(file, 'rb') as f:
                    model_name = file.replace('.pkl', '')
                    models[model_name] = pickle.load(f)
            
            return models
        except Exception as e:
            st.error(f"Error loading models: {str(e)}")
            st.write("Current directory:", os.getcwd())
            st.write("Files available:", os.listdir())
            return None

class NutritionCalculator:
    @staticmethod
    def calculate_bmr(gender: str, weight: float, height: float, age: int) -> float:
        """Calculate Basal Metabolic Rate."""
        if gender == "Female":
            return 655 + (9.6 * weight) + (1.8 * height) - (4.7 * age)
        return 66 + (13.7 * weight) + (5 * height) - (6.8 * age)

    @staticmethod
    def calculate_macronutrients(weight: float, daily_calories: float, config: Config) -> np.ndarray:
        """Calculate macronutrient requirements for a meal."""
        protein_grams = config.PROTEIN_PER_KG * weight
        fat_calories = config.FAT_CALORIES_RATIO * daily_calories
        carb_calories = config.CARB_CALORIES_RATIO * daily_calories
        
        fat_grams = fat_calories / config.FAT_CALORIES_PER_GRAM
        carb_grams = carb_calories / config.CARB_CALORIES_PER_GRAM
        
        return np.array([
            daily_calories * config.MEAL_FRACTION,
            protein_grams * config.MEAL_FRACTION,
            fat_grams * config.MEAL_FRACTION,
            carb_grams * config.MEAL_FRACTION,
            config.DEFAULT_SODIUM * config.MEAL_FRACTION,
            config.DEFAULT_CHOLESTEROL * config.MEAL_FRACTION,
            (fat_grams * config.SAT_FAT_RATIO) * config.MEAL_FRACTION
        ])

class FoodRecommender:
    def __init__(self, df: pd.DataFrame, models: Dict[str, Any]):
        self.df = self.preprocess_dataframe(df)
        self.models = models
        self.required_columns = [
            'Calories', 'ProteinContent', 'FatContent', 'CarbohydrateContent',
            'SodiumContent', 'CholesterolContent', 'SaturatedFatContent'
        ]

    def preprocess_dataframe(self, df: pd.DataFrame) -> pd.DataFrame:
        """Preprocess the dataframe to ensure correct data types and handling."""
        try:
            # Create a copy to avoid modifying the original
            processed_df = df.copy()
            
            # Convert Cluster column to numeric, handling any non-numeric values
            processed_df['Cluster'] = pd.to_numeric(processed_df['Cluster'], errors='coerce')
            
            # Drop rows with NaN in Cluster column
            processed_df = processed_df.dropna(subset=['Cluster'])
            
            # Convert Cluster to integer type
            processed_df['Cluster'] = processed_df['Cluster'].astype(int)
            
            # Ensure all required columns are numeric
            for col in self.required_columns:
                if col in processed_df.columns:
                    processed_df[col] = pd.to_numeric(processed_df[col], errors='coerce')
            
            # Drop rows with NaN in required columns
            processed_df = processed_df.dropna(subset=self.required_columns)
            
            # Debug information
            st.write("DataFrame preprocessing complete:")
            st.write(f"Number of rows after preprocessing: {len(processed_df)}")
            st.write(f"Unique clusters: {processed_df['Cluster'].unique()}")
            
            return processed_df
            
        except Exception as e:
            st.error(f"Error in preprocessing dataframe: {str(e)}")
            return df

    def get_recommendations(self, input_data: np.ndarray) -> pd.DataFrame:
        """Generate food recommendations based on input data."""
        try:
           # Debug information
            st.write("Starting recommendation process...")
            st.write(f"Input data shape: {input_data.shape}")
            
            # Reshape and scale input data
            input_data_reshaped = input_data.reshape(1, -1)
            input_data_scaled = self.models['scaler'].transform(input_data_reshaped)
            
            # Get cluster prediction
            cluster_label = self.models['kmeans'].predict(input_data_scaled)[0]
            st.write(f"Predicted cluster: {cluster_label}")
            
            # Debug: Show cluster distribution
            cluster_dist = self.df['Cluster'].value_counts()
            st.write("Cluster distribution:", cluster_dist)
            
            # Filter by cluster
            cluster_data = self.df[self.df['Cluster'] == cluster_label].copy()
            st.write(f"Number of items in cluster {cluster_label}: {len(cluster_data)}")
            
            if cluster_data.empty:
                st.warning(f"No items found in cluster {cluster_label}")
                return pd.DataFrame()

            # Scale features and calculate similarities
            cluster_features = cluster_data[self.required_columns]
            cluster_features_scaled = self.models['scaler'].transform(cluster_features)
            similarities = cosine_similarity(input_data_scaled, cluster_features_scaled).flatten()
            
            # Add similarity scores and predictions
            cluster_data['Similarity'] = similarities
            rf_predictions = self.models['rf_classifier'].predict(cluster_features_scaled)
            cluster_data['Classification'] = rf_predictions
            
            # Get final recommendations
            final_recommendations = cluster_data[cluster_data['Classification'] == 1].sort_values(
                by='Similarity', ascending=False
            )
            
            # If no items pass classification, return top similar items
            if final_recommendations.empty:
                st.warning("No items passed classification. Returning most similar items.")
                final_recommendations = cluster_data.sort_values(by='Similarity', ascending=False)

            # Select and format output columns
            output_columns = ['Name'] + self.required_columns + ['Similarity']
            result = final_recommendations[output_columns].head(5)
            
            # Round numeric columns for better display
            numeric_columns = self.required_columns + ['Similarity']
            result[numeric_columns] = result[numeric_columns].round(2)
            
            return result

        except Exception as e:
            st.error(f"Error in recommendation process: {str(e)}")
            st.write("Full error details:", e)
            st.write("DataFrame columns:", self.df.columns)
            return pd.DataFrame()

class StreamlitUI:
    def __init__(self):
        self.config = Config()
        self.data_loader = DataLoader()
        self.df = self.data_loader.load_dataset()
        self.models = self.data_loader.load_models()

    def render(self):
        """Render the Streamlit UI."""
        st.title('🍅🧀MyHealthMyFood🥑🥬')
        
        if self.df is None or self.models is None:
            st.error("Failed to load required data or models")
            return

        # User inputs
        gender = st.selectbox("Select your gender", ["Female", "Male"])
        weight = st.number_input("Enter your weight (kg)", 
                               min_value=self.config.MIN_WEIGHT,
                               max_value=self.config.MAX_WEIGHT, 
                               value=30)
        height = st.number_input("Enter your height (cm)", 
                               min_value=self.config.MIN_HEIGHT,
                               max_value=self.config.MAX_HEIGHT, 
                               value=120)
        age = st.number_input("Enter your age (years)", 
                            min_value=self.config.MIN_AGE,
                            max_value=self.config.MAX_AGE, 
                            value=18)
        
        health_condition = st.selectbox(
            "Select your health condition",
            ["No Non-Communicable Disease", "Diabetic", "High Blood Pressure", "High Cholesterol"]
        )
        
        if health_condition == "No Non-Communicable Disease":
            st.selectbox(
                "Select your wellness goal",
                ["Maintain Weight", "Lose Weight", "Muscle Gain"]
            )

        if st.button("Get Recommendations"):
            self.process_recommendations(gender, weight, height, age)

    def process_recommendations(self, gender: str, weight: float, height: float, age: int):
        """Process and display food recommendations."""
        calculator = NutritionCalculator()
        daily_calories = calculator.calculate_bmr(gender, weight, height, age)
        input_features = calculator.calculate_macronutrients(weight, daily_calories, self.config)
        
        recommender = FoodRecommender(self.df, self.models)
        recommendations = recommender.get_recommendations(input_features)
        
        if not recommendations.empty:
            st.write("Recommended food items:")
            st.write(recommendations)
        else:
            st.warning("No recommendations found. Please try different inputs.")

def main():
    ui = StreamlitUI()
    ui.render()

if __name__ == "__main__":
    main()
