# -*- coding: utf-8 -*-
"""CalculateCaloPerDayToAchiveTheWeighTarget

Automatically generated by Colab.

Original file is located at
    https://colab.research.google.com/drive/1yupSsUs9mwH_L0nW_4wL9PV9nfSY5YnH
"""

def calculate_bmr(weight, height, age, gender):
    if gender == 'Male':
        return 10 * weight + 6.25 * height - 5 * age + 5
    else:
        return 10 * weight + 6.25 * height - 5 * age - 161

def calculate_tdee(bmr, activity_level):
    activity_multipliers = {
        'sedentary': 1.2,
        'light': 1.375,
        'moderate': 1.55,
        'active': 1.725,
        'very_active': 1.9
    }
    return bmr * activity_multipliers.get(activity_level, 1.2)
#ước lượng rằng 1 kg cân nặng tương đương với khoảng 7700 kcal (calories).
def calculate_time_to_goal(current_weight, desired_weight, calorie_change_per_day):
    weight_change = desired_weight - current_weight  # Dương nếu tăng cân, âm nếu giảm cân
    total_calories_needed = weight_change * 7700  # Tổng calo cần để đạt mục tiêu
    time_in_days = total_calories_needed / calorie_change_per_day
    return abs(time_in_days)  # Trả về giá trị tuyệt đối để có số ngày dương

def calculate_calorie_range(tdee, calorie_change_per_day):
    # Tính lượng calo tối thiểu và tối đa cần tiêu thụ mỗi ngày
    min_calories = tdee - calorie_change_per_day  # Thâm hụt calo (giảm cân)
    max_calories = tdee + calorie_change_per_day  # Thặng dư calo (tăng cân)
    return min_calories, max_calories

# Thông tin người dùng
current_weight = 70  # kg
desired_weight = 75  # kg (thay đổi giá trị này nếu muốn tính tăng cân)
height = 175  # cm
age = 25  # năm
gender = 'Male'  # Giới tính
activity_level = 'moderate'  # Mức độ hoạt động

# Tính BMR và TDEE dựa trên cân nặng hiện tại
bmr = calculate_bmr(current_weight, height, age, gender)
tdee = calculate_tdee(bmr, activity_level)

# Lượng calo thặng dư hoặc thiếu hụt mỗi ngày
calorie_change_per_day = 1500  # Nếu muốn giảm cân, đặt giá trị dương cho thiếu hụt calo; nếu tăng cân thì đặt giá trị thặng dư calo

# Tính thời gian để đạt mục tiêu
days_to_goal = calculate_time_to_goal(current_weight, desired_weight, calorie_change_per_day)

# Tính lượng calo tối thiểu và tối đa cần tiêu thụ mỗi ngày
min_calories, max_calories = calculate_calorie_range(tdee, calorie_change_per_day)

# Hiển thị kết quả
print(f"BMR: {bmr:.2f} kcal/ngày")
print(f"TDEE: {tdee:.2f} kcal/ngày")
print(f"Lượng calo tối thiểu cần tiêu thụ để đạt mục tiêu: {min_calories:.2f} kcal/ngày")
print(f"Lượng calo tối đa cần tiêu thụ để đạt mục tiêu: {max_calories:.2f} kcal/ngày")

if desired_weight > current_weight:
    print(f"Thời gian để tăng cân từ {current_weight} kg lên {desired_weight} kg: {days_to_goal:.2f} ngày")
else:
    print(f"Thời gian để giảm cân từ {current_weight} kg xuống {desired_weight} kg: {days_to_goal:.2f} ngày")