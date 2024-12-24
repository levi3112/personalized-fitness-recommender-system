import joblib
import pandas as pd
import json

from config import get

# model_path = get('model', 'model')
model_path = 'data/model.pkl'

# label_encoder_path = get('model', 'label_encoder')
label_encoder_path = 'data/label_encoder.pkl'

# json_path
json_path = 'data/exercise_info.json'

# Define the categorical and numerical features
cat_features = ['Gender']
num_features = ['Dream Weight', 'Actual Weight', 'Age', 'BMI']

# Load the model
loaded_model = joblib.load(model_path)

# Load the LabelEncoder
loaded_le = joblib.load(label_encoder_path)

# Load Exercise Json file
with open(json_path, 'r') as file:
    exercise_data = json.load(file)


def predict_workout_plan(
    gender: str,    # 'Male' / 'Female'
    age: int,
    actual_weight: int,
    dream_weight: int,
    bmi: float
) -> dict:
    # Define a new observation as a DataFrame
    # Replace 'Male' with the actual encoded value for the gender
    # data = ['Male', 25, 70, 45, 29]
    new_observation = pd.DataFrame([[gender, age, actual_weight, dream_weight, bmi]],
                                   # Gender, Age, Actual Weight, Dream Weight, BMI
                                   columns=cat_features + num_features)

    # Use the model to predict the exercise and intensity
    exercise_encoded, intensity, duration,calories_burned = loaded_model.predict(new_observation)[0]
    print(f"predict v1: { loaded_model.predict(new_observation)[0]}", )
    # Decode 'Exercise' back to its original form
    exercise = loaded_le.inverse_transform([int(exercise_encoded)])[0]

    # Get exercise details
    exercise_details = exercise_data.get(exercise, {})
    image = exercise_details.get("Image", "No image available")
    video = exercise_details.get("Video", "No video available")
    description = exercise_details.get("Description", "No description available")

    return {
            "exercise": exercise,
            "intensity": intensity,
            "duration": duration,
            "calories_burned":calories_burned,
            "image": image,
            "video": video,
            "description": description
        }

def predict_workout_plan_v2(
    gender: str,    # 'Male' / 'Female'
    age: int,
    actual_weight: int,
    dream_weight: int,
    bmi: float
) -> str:
    # Define a new observation as a DataFrame
    # Replace 'Male' with the actual encoded value for the gender
    # data = ['Male', 25, 70, 45, 29]
    new_observation = pd.DataFrame([[gender, age, actual_weight, dream_weight, bmi]],
                                   # Gender, Age, Actual Weight, Dream Weight, BMI
                                   columns=cat_features + num_features)

    # Use the model to predict the exercise and intensity
    exercise_encoded, intensity, duration, calories_burned = loaded_model.predict(new_observation)[0]
    print(f"predict v2: { loaded_model.predict(new_observation)[0]}", )

    # Decode 'Exercise' back to its original form
    exercise = loaded_le.inverse_transform([int(exercise_encoded)])[0]
 
    return f"Predicted Exercise: {exercise}, Intensity: {intensity}, Duration: {duration}, Calories: {calories_burned}"
