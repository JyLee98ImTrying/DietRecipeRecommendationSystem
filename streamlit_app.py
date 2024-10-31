import streamlit as st
import numpy as np
import pandas as pd
import joblib
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

# Load data
df = pd.read_csv('RecipeData.csv')  # Update with your actual file path

# Streamlit UI
st.title('Diet and Recipe Recommendation System')

# User inputs
weight = st.number_input('Enter your weight (kg):', min_value=0.0, step=0.1)
height = st.number_input('Enter your height (cm):', min_value=0.0, step=0.1)
bmi = weight / ((height / 100) ** 2) if height > 0 else 0
st.write(f'Your BMI is: {bmi:.2f}')

health_condition = st.selectbox('Select your health condition:',
                                ["No Non-Communicable Disease", "Diabetic", "High Blood Pressure", "High Cholesterol"])

wellness_goal = st.selectbox('Select your wellness goal:',
                             ["No goals", "Lose Fat", "Gain Muscle"])

calorie = st.number_input('Enter your daily calorie requirement (kcal):', min_value=0.0, step=10.0)

# Preprocess data for content-based filtering
# Combine relevant text features into a single string
df['CombinedText'] = df['RecipeIngredientParts'] + ' ' + df['RecipeInstructions'] + ' ' + df['RecipeCategory']

# Create TF-IDF vectors for the combined text
tfidf_vectorizer = TfidfVectorizer(stop_words='english')
tfidf_matrix = tfidf_vectorizer.fit_transform(df['CombinedText'])

# Define a function to create a user profile vector
def create_user_profile(health_condition, wellness_goal):
    user_preferences = ''

    # Add health condition keywords
    if health_condition == "Diabetic":
        user_preferences += 'low sugar low carb high fiber '
    elif health_condition == "High Blood Pressure":
        user_preferences += 'low sodium low fat '
    elif health_condition == "High Cholesterol":
        user_preferences += 'low cholesterol low saturated fat '

    # Add wellness goal keywords
    if wellness_goal == "Lose Fat":
        user_preferences += 'low calorie low fat high protein '
    elif wellness_goal == "Gain Muscle":
        user_preferences += 'high protein high calorie '

    return user_preferences.strip()

# Model predictions
if st.button('Recommend Recipes'):
    # Predict with each model
    features = df.drop(columns=['Name', 'AuthorName', 'RecipeInstructions', 'RecipeCategory',
                                'RecipeIngredientParts', 'RecipeIngredientQuantities', 'CombinedText'])
    rf_prediction = rf_model.predict(features)
    dt_prediction = dt_model.predict(features)
    nb_prediction = nb_model.predict(features)
    xgb_prediction = xgb_model.predict(features)

    # Aggregate the results (simple majority vote)
    final_prediction = np.array([rf_prediction, dt_prediction, nb_prediction, xgb_prediction])
    final_prediction = np.mean(final_prediction, axis=0).round()

    # Filter recipes based on the selected condition and goal
    if health_condition == "No Non-Communicable Disease" and wellness_goal == "No goals":
        # No specific filtering; recommend based on model predictions
        recommended_indices = np.where(final_prediction == final_prediction[0])[0]
        recommended_recipes = df.iloc[recommended_indices].head(10)
    else:
        # Content-based filtering
        user_profile_text = create_user_profile(health_condition, wellness_goal)
        user_tfidf_vector = tfidf_vectorizer.transform([user_profile_text])

        # Compute cosine similarity between user profile and recipes
        cosine_similarities = cosine_similarity(user_tfidf_vector, tfidf_matrix).flatten()

        # Incorporate numeric features with weights
        nutrient_features = ['ProteinContent', 'FatContent', 'CarbohydrateContent', 'SugarContent',
                             'SodiumContent', 'CholesterolContent', 'CalorieContent']

        # Normalize numeric features
        df_norm = df[nutrient_features].copy()
        df_norm = (df_norm - df_norm.min()) / (df_norm.max() - df_norm.min())

        # Define weights for each nutrient based on health conditions and wellness goals
        weights = np.ones(len(nutrient_features))
        if health_condition == "Diabetic":
            weights[df_norm.columns.get_loc('SugarContent')] = 2.0  # Give higher weight to sugar
        elif health_condition == "High Blood Pressure":
            weights[df_norm.columns.get_loc('SodiumContent')] = 2.0  # Higher weight to sodium
        elif health_condition == "High Cholesterol":
            weights[df_norm.columns.get_loc('CholesterolContent')] = 2.0  # Higher weight to cholesterol

        if wellness_goal == "Lose Fat":
            weights[df_norm.columns.get_loc('FatContent')] = 2.0  # Higher weight to fat
            weights[df_norm.columns.get_loc('CalorieContent')] = 2.0  # Higher weight to calories
        elif wellness_goal == "Gain Muscle":
            weights[df_norm.columns.get_loc('ProteinContent')] = 2.0  # Higher weight to protein

        # Compute weighted sum of nutrient features
        nutrient_scores = df_norm.dot(weights)

        # Combine cosine similarity and nutrient scores
        combined_scores = 0.5 * cosine_similarities + 0.5 * (1 - nutrient_scores)  # Lower nutrient_scores are better

        # Get top recommendations
        recommended_indices = combined_scores.argsort()[:10]
        recommended_recipes = df.iloc[recommended_indices]

    # Display recommended recipes
    st.write('Recommended Recipes:')
    for idx, row in recommended_recipes.iterrows():
        st.subheader(row['Name'])
        st.write(f"Category: {row['RecipeCategory']}")
        st.write(f"Total Time: {row['TotalTime']} minutes")
        st.write(f"Ingredients: {row['RecipeIngredientParts']}")
        st.write(f"Instructions: {row['RecipeInstructions']}")
        st.write(f"Nutritional Information:")
        st.write(f"- Calories: {row['CalorieContent']} kcal")
        st.write(f"- Protein: {row['ProteinContent']} g")
        st.write(f"- Fat: {row['FatContent']} g")
        st.write(f"- Carbohydrates: {row['CarbohydrateContent']} g")
        st.write(f"- Sugar: {row['SugarContent']} g")
        st.write(f"- Sodium: {row['SodiumContent']} mg")
        st.write(f"- Cholesterol: {row['CholesterolContent']} mg")
        st.write("---")
