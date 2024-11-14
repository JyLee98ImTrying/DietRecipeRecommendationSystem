import streamlit as st
import numpy as np
import pandas as pd
import pickle
from sklearn.metrics.pairwise import cosine_similarity

# Configuration and Constants
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
    # Added new constants for wellness goals
    PROTEIN_MUSCLE_GAIN_MIN: float = 2.5
    PROTEIN_MUSCLE_GAIN_MAX: float = 3.0
    MAX_SATURATED_FAT_WEIGHT_LOSS: float = 5.0
    MAX_SUGAR_WEIGHT_LOSS: float = 10.0

class DataLoader:
    @staticmethod
    @st.cache_data
    def load_dataset() -> pd.DataFrame:
        try:
            df = pd.read_csv('df_DR.csv')
            return df
        except Exception as e:
            st.error(f"Error loading dataset: {str(e)}")
            return pd.DataFrame()

    @staticmethod
    def load_models() -> Dict[str, Any]:
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
            return {}

class NutritionCalculator:
    @staticmethod
    def calculate_bmr(gender: str, weight: float, height: float, age: int) -> float:
        if gender == "Female":
            return 655 + (9.6 * weight) + (1.8 * height) - (4.7 * age)
        return 66 + (13.7 * weight) + (5 * height) - (6.8 * age)

    @staticmethod
    def calculate_macronutrients(weight: float, daily_calories: float, wellness_goal: str, config: Config) -> np.ndarray:
        if wellness_goal == "Muscle Gain":
            protein_grams = config.PROTEIN_MUSCLE_GAIN_MIN * weight
            fat_calories = config.FAT_CALORIES_RATIO * daily_calories
            remaining_calories = daily_calories - (protein_grams * 4) - fat_calories
            carb_calories = remaining_calories
        else:
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
            'SodiumContent', 'CholesterolContent', 'SaturatedFatContent', 'SugarContent'
        ]
        self.config = Config()

    def preprocess_dataframe(self, df: pd.DataFrame) -> pd.DataFrame:
        try:
            processed_df = df.copy()
            processed_df['Cluster'] = pd.to_numeric(processed_df['Cluster'], errors='coerce')
            processed_df = processed_df.dropna(subset=['Cluster'])
            processed_df['Cluster'] = processed_df['Cluster'].astype(int)

            for col in self.required_columns:
                if col in processed_df.columns:
                    processed_df[col] = pd.to_numeric(processed_df[col], errors='coerce')
            processed_df = processed_df.dropna(subset=self.required_columns)

            st.write("DataFrame preprocessing complete:")
            st.write(f"Number of rows after preprocessing: {len(processed_df)}")
            st.write(f"Unique clusters: {processed_df['Cluster'].unique()}")

            return processed_df
        except Exception as e:
            st.error(f"Error in preprocessing dataframe: {str(e)}")
            return df

    def apply_wellness_goal_filters(self, recommendations: pd.DataFrame, wellness_goal: str, weight: float) -> pd.DataFrame:
        if recommendations.empty:
            return recommendations

        if wellness_goal == "Muscle Gain":
            min_protein_per_meal = self.config.PROTEIN_MUSCLE_GAIN_MIN * weight * self.config.MEAL_FRACTION
            high_protein_recommendations = recommendations[recommendations['ProteinContent'] >= min_protein_per_meal]
            if high_protein_recommendations.empty:
                st.warning("No meals meet the optimal protein requirement for muscle gain. Showing highest protein options available.")
                return recommendations.sort_values('ProteinContent', ascending=False)
            return high_protein_recommendations.sort_values('ProteinContent', ascending=False)

        elif wellness_goal == "Lose Weight":
            weight_loss_recommendations = recommendations[
                (recommendations['SaturatedFatContent'] <= self.config.MAX_SATURATED_FAT_WEIGHT_LOSS) &
                (recommendations['SugarContent'] <= self.config.MAX_SUGAR_WEIGHT_LOSS)
            ]
            if weight_loss_recommendations.empty:
                st.warning("No meals meet the optimal criteria for weight loss. Showing options with lowest saturated fat and sugar content.")
                return recommendations.sort_values(['SaturatedFatContent', 'SugarContent'], ascending=[True, True])
            return weight_loss_recommendations.sort_values(['SaturatedFatContent', 'SugarContent'], ascending=[True, True])

        return recommendations

    def get_recommendations(self, input_data: np.ndarray, wellness_goal: str = "Maintain Weight", weight: float = 70.0) -> pd.DataFrame:
        try:
            input_data_reshaped = input_data.reshape(1, -1)
            input_data_scaled = self.models['scaler'].transform(input_data_reshaped)
            cluster_label = self.models['kmeans'].predict(input_data_scaled)[0]

            cluster_data = self.df[self.df['Cluster'] == cluster_label].copy()

            if cluster_data.empty:
                return pd.DataFrame()

            cluster_features = cluster_data[self.required_columns]
            cluster_features_scaled = self.models['scaler'].transform(cluster_features)
            similarities = cosine_similarity(input_data_scaled, cluster_features_scaled).flatten()
            cluster_data['Similarity'] = similarities
            rf_predictions = self.models['rf_classifier'].predict(cluster_features_scaled)
            cluster_data['Classification'] = rf_predictions

            final_recommendations = cluster_data[cluster_data['Classification'] == 1].sort_values(by='Similarity', ascending=False)

            if final_recommendations.empty:
                final_recommendations = cluster_data.sort_values(by='Similarity', ascending=False)

            final_recommendations = self.apply_wellness_goal_filters(final_recommendations, wellness_goal, weight)

            output_columns = ['Name'] + self.required_columns + ['Similarity']
            result = final_recommendations[output_columns].head(5)
            numeric_columns = self.required_columns + ['Similarity']
            result[numeric_columns] = result[numeric_columns].round(2)

            if wellness_goal == "Muscle Gain":
                st.info(f"Recommended daily protein intake for muscle gain: "
                       f"{(self.config.PROTEIN_MUSCLE_GAIN_MIN * weight):.1f} - "
                       f"{(self.config.PROTEIN_MUSCLE_GAIN_MAX * weight):.1f} g")
            elif wellness_goal == "Lose Weight":
                st.info("These recommendations prioritize meals lower in saturated fat and sugar content")

            return result

        except Exception as e:
            st.error(f"Error in recommendation process: {str(e)}")
            return pd.DataFrame()

class StreamlitUI:
    def __init__(self):
        self.config = Config()
        self.data_loader = DataLoader()
        self.df = self.data_loader.load_dataset()
        self.models = self.data_loader.load_models()
        self.wellness_goal = "Maintain Weight"

    def render(self):
        st.title('🍅🧀MyHealthMyFood🥑🥬')

        if self.df is None or self.models is None:
            st.error("Failed to load required data or models")
            return

        gender = st.selectbox("Select your gender", ["Female", "Male"])
        weight = st.number_input("Enter your weight (kg)", 
                               min_value=self.config.MIN_WEIGHT,
                               max_value=self.config.MAX_WEIGHT, 
                               value=70)
        height = st.number_input("Enter your height (cm)", 
                               min_value=self.config.MIN_HEIGHT,
                               max_value=self.config.MAX_HEIGHT, 
                               value=160)
        age = st.number_input("Enter your age (years)", 
                            min_value=self.config.MIN_AGE,
                            max_value=self.config.MAX_AGE, 
                            value=30)
        
        health_condition = st.selectbox(
            "Select your health condition",
            ["No Non-Communicable Disease", "Diabetic", "High Blood Pressure", "High Cholesterol"]
        )
        
        if health_condition == "No Non-Communicable Disease":
            self.wellness_goal = st.selectbox(
                "Select your wellness goal",
                ["Maintain Weight", "Lose Weight", "Muscle Gain"]
            )

        if st.button("Get Recommendations"):
            self.process_recommendations(gender, weight, height, age)

    def process_recommendations(self, gender: str, weight: float, height: float, age: int):
        calculator = NutritionCalculator()
        daily_calories = calculator.calculate_bmr(gender, weight, height, age)
        input_features = calculator.calculate_macronutrients(
            weight, 
            daily_calories, 
            self.wellness_goal,
            self.config
        )
        
        recommender = FoodRecommender(self.df, self.models)
        recommendations = recommender.get_recommendations(
            input_features,
            self.wellness_goal,
            weight
        )
        
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
