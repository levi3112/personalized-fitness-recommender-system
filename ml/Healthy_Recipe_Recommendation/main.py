from fastapi import FastAPI, HTTPException, Query
from pydantic import BaseModel
from typing import List, Dict, Optional
from uuid import UUID, uuid4

import pandas as pd
import pulp as pl
import os

from CalculateCaloPerDayToAchiveTheWeighTarget import calculate_calo
app = FastAPI()

# Get the path to the directory containing this script
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
# Construct the path to the CSV file
CSV_PATH = os.path.join(BASE_DIR,'data','df_p2.csv')

class NutrientConfig(BaseModel):
    cal_lo: int = 2000  # Default value
    cal_up: int = 2500  # Default value
    pro_lo: int = 50    # Default value
    pro_up: int = 150   # Default value
    fat_lo: int = 20    # Default value
    fat_up: int = 70    # Default value
    sod_lo: int = 1000  # Default value
    sod_up: int = 2300  # Default value
class RecipeRequest(BaseModel):
    number_of_dishes: int = 5
    number_of_candidates: int =10
    nut_conf: NutrientConfig = NutrientConfig()    

@app.get("/")
def home():
    calTargetWeight.calculate_calo()
    return "Hello"

def recipe_recommend(df, number_of_dishes, number_of_candidates, nut_conf):
    tmp = df.copy()
    candidates_list = []

    for i in range(0, number_of_candidates):
        m = pl.LpProblem(sense=pl.LpMaximize)
        tmp['v'] = [pl.LpVariable(f'x{i}', cat=pl.LpBinary) for i in range(len(tmp))]

        m += pl.lpDot(tmp["rating"], tmp["v"])
        m += pl.lpSum(tmp["v"]) <= number_of_dishes
        m += pl.lpDot(tmp["calories"], tmp["v"]) >= nut_conf["cal_lo"]
        m += pl.lpDot(tmp["calories"], tmp["v"]) <= nut_conf["cal_up"]
        m += pl.lpDot(tmp["protein"], tmp["v"]) >= nut_conf["pro_lo"]
        m += pl.lpDot(tmp["protein"], tmp["v"]) <= nut_conf["pro_up"]
        m += pl.lpDot(tmp["fat"], tmp["v"]) >= nut_conf["fat_lo"]
        m += pl.lpDot(tmp["fat"], tmp["v"]) <= nut_conf["fat_up"]
        m += pl.lpDot(tmp["sodium"], tmp["v"]) >= nut_conf["sod_lo"]
        m += pl.lpDot(tmp["sodium"], tmp["v"]) <= nut_conf["sod_up"]

        m.solve(pl.PULP_CBC_CMD(msg=0, options=['maxsol 1']))

        if m.status == 1:
            tmp['val'] = tmp["v"].apply(lambda x: pl.value(x))
            ret = tmp.query('val==1')["title"].values
            candidates_list.append(ret)
            tmp = tmp.query('val==0')

    return candidates_list

# Load the DataFrame (df_p2)
df_p2 = pd.read_csv(CSV_PATH)  # Adjust the path to your actual CSV file

@app.post('/recommend')
def recommend(
    request: RecipeRequest = RecipeRequest(),
    season: str = Query("summer", description="Filter by season (summer, winter)"),
    meal_type: str = Query("breakfast", description="Filter by meal type (breakfast, low_cal)"),
    quick_recipe: bool = Query(False, description="Filter by quick recipe (few ingredients and directions)")
):
    try:
        df_filtered = df_p2.copy()

        # Apply filters based on query parameters
        if season:
            if season.lower() == "summer":
                df_filtered = df_filtered.query("summer==1")
            elif season.lower() == "winter":
                df_filtered = df_filtered.query("winter==1")

        if meal_type:
            if meal_type.lower() == "breakfast":
                df_filtered = df_filtered.query("breakfast==1")
            elif meal_type.lower() == "low_cal":
                df_filtered = df_filtered[df_filtered["low cal"] == 1]

        if quick_recipe:
            df_filtered = df_filtered.query("len_ingredients <= 9 and len_directions <= 3")

        # Call the recipe_recommend function with the filtered DataFrame
        recommendations = recipe_recommend(df_filtered, request.number_of_dishes, request.number_of_candidates, request.nut_conf.dict())
        return {"recommendations": [list(rec) for rec in recommendations]}

    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

if __name__ == "__main__":
    import uvicorn

    uvicorn.run(app, host="0.0.0.0", port=8000)