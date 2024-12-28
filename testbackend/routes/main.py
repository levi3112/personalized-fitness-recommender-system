import cv2
import numpy as np
from fastapi import APIRouter, Form, UploadFile, File
from collections import defaultdict
from typing import Optional
import random

from blood_report_analyzer.main import analyze_blood_sugar_report
from diet_plan_recommender.main import get_meal_plan
from nutrition_need_calculator.diseases import get_diseases
from nutrition_need_calculator.need_calculator import get_dietary_need
from medic.main import calculate_bmi, calculate_dream_weight
from workout_routine_recommender.recommender import predict_workout_plan, predict_workout_plan_v2

router = APIRouter(
    prefix="/api",
    tags=["core"],
    responses={404: {"description": "The requested URL was not found"}},
)


@router.post("")
async def root(
        height: int = Form(175),
        weight: int = Form(85),
        dream_weight: int = Form(0),
        age: int = Form(21),
        gender: str = Form("Male"),
        image: Optional[UploadFile] = File(None),
        diseases_info: str = Form(""),
        num_of_exercises: int = Form(15)
):
    print("data: ", )
    bmi = calculate_bmi(weight, height)
    dream_weight_cal = dream_weight if dream_weight != 0 else calculate_dream_weight(weight, bmi)
    diseases = get_diseases(None, bmi)

    print(f"data: ${dream_weight_cal}")

    if (image is None )or (image and image.content_type != "image/jpeg"):
        # return {300: {"description": "Only jpeg images are supported"}} # TODO fix this
        diseases = get_diseases(None, bmi)
    else:
        contents = await image.read()
        if contents:  # Check if the image file is not empty
            nparray = np.fromstring(contents, np.uint8)
            img = cv2.imdecode(nparray, cv2.IMREAD_COLOR)

            blood_sugar_level = analyze_blood_sugar_report(img)
            diseases = get_diseases(blood_sugar_level, bmi)
        else:
            diseases = get_diseases(None, bmi)

    nutrition_need = get_dietary_need(weight, height, age, gender.lower())  # 'male' 'female'
    workout_plan_1 = predict_workout_plan_v2(gender, age, weight, dream_weight_cal, bmi)  # TODO gender - 'Male' 'Female'

    # Generate workout_plan 10 times and store in a list
    # workout_plan_list = []
    # for _ in range(num_of_exercises):
    #    # Add small random variations to weight and dream_weight for each iteration
    #     randomized_weight = weight + random.uniform(-2, 2)  # Adding variation between -2 to 2 kg
    #     randomized_dream_weight = dream_weight + random.uniform(-1, 1)  # Variation in dream weight

    #     workout_plan = predict_workout_plan(gender, age, randomized_weight, randomized_dream_weight, bmi)
    #     workout_plan_list.append(workout_plan)

    # Dictionary to count the occurrences of each exercise
    exercise_counts = defaultdict(int)
    max_occurrences = 8  # Maximum occurrences for each exercise

    workout_plan_list = []
    while len(workout_plan_list) < num_of_exercises:
        # Add small random variations to weight and dream_weight
        randomized_weight = weight + random.uniform(-5, 5)  # Adding variation between -2 to 2 kg
        randomized_dream_weight = dream_weight_cal + random.uniform(-4, 4)  # Variation in dream weight

        # Predict a workout plan
        workout_plan = predict_workout_plan(gender, age, randomized_weight, randomized_dream_weight, bmi)

        # Check if the exercise has exceeded the max allowed occurrences
        exercise_name = workout_plan["exercise"]
        if exercise_counts[exercise_name] < max_occurrences:
            workout_plan_list.append(workout_plan)
            exercise_counts[exercise_name] += 1


    # meal_plan = get_meal_plan(['low_sodium_diet', 'low_fat_diet'], diseases, ['calcium', 'vitamin_c'], ['non-veg'],
    #                           'i love indian')
    # ['low_sodium_diet','low_fat_diet'], ['diabeties'], ['calcium','vitamin_c'], ['non-veg'],'i love indian'

    if diseases_info is not None:
        workout_plan_1 = "Not recommended until be validated by a doctor! " + workout_plan_1

    return {
        "need": nutrition_need,
        "workout_plan": workout_plan_list
        # "meal_plan": meal_plan,
    }
