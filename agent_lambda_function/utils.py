from langchain_core.tools import tool
from typing import Literal
import pandas as pd


#The default values on the pydantic object were extract on  "notebooks/feature_engineering.ipynb". 
#These values are the most frequent ones, so if the person do not give this information we use
#the most frequent as default


# Mapping dictionaries based on the feature engineering logic
MAPPINGS = {
    'sex': {'Female': 0, 'Male': 1},
    "smoker": {"No": 0, "Yes": 1},
    "diagnosis_code": {"D1": 1.0, "D2": 2.0, "D3": 3.0, "D4": 4.0, "D5": 5.0},
    "exercise_frequency": {"Low": 0.0, "Moderate": 1.0, "High": 2.0},
    "diet_quality": {"Poor": 0.0, "Average": 1.0, "Good": 2.0},
    "income_bracket": {"Low": 0.0, "Middle": 1.0, "High": 2.0},
    "education_level": {"Primary": 0.0, "Secondary": 1.0, "Tertiary": 2.0}
}

@tool
def predict_alanine_aminotransferase(
    age: float = 53.0,
    sex: Literal["Male", "Female"] = "Female",
    bmi: float = 26.9,
    smoker: Literal["Yes", "No"] = "No",
    diagnosis_code: Literal["D1", "D2", "D3", "D4", "D5"] = "D5",
    medication_count: int = 3,
    days_hospitalized: int = 5,
    readmitted: Literal[0, 1] = 0,
    last_lab_glucose: float = 100.1,
    exercise_frequency: Literal["Low", "Moderate", "High"] = "Moderate",
    diet_quality: Literal["Poor", "Average", "Good"] = "Average",
    income_bracket: Literal["Low", "Middle", "High"] = "Middle",
    education_level: Literal["Primary", "Secondary", "Tertiary"] = "Secondary",
    urban: Literal[0, 1] = 1,
    albumin_globulin_ratio: float = 0.5037,
) -> float:
    """
    Predicts the Alanine Aminotransferase (ALT) levels for a patient using an XGBoost model.
    Use this tool when users ask for predictions, forecasts, or expected liver enzyme values.
    """
    global xgboost_model # Loaded in lifespan
    
    if xgboost_model is None:
        return "Error: Prediction model is not currently loaded in the system."

    try:
        # 1. Transform literals to numeric values used in training
        data_dict = {
            'age': age,
            'sex': MAPPINGS["sex"][sex],
            'bmi': bmi,
            'smoker': MAPPINGS["smoker"][smoker],
            'diagnosis_code': MAPPINGS["diagnosis_code"][diagnosis_code],
            'medication_count': medication_count,
            'days_hospitalized': days_hospitalized,
            'readmitted': readmitted,
            'last_lab_glucose': last_lab_glucose,
            'exercise_frequency': MAPPINGS["exercise_frequency"][exercise_frequency],
            'diet_quality': MAPPINGS["diet_quality"][diet_quality],
            'income_bracket': MAPPINGS["income_bracket"][income_bracket],
            'education_level': MAPPINGS["education_level"][education_level],
            'urban': urban,
            'albumin_globulin_ratio': albumin_globulin_ratio
        }

        # 2. Convert to DataFrame
        # IMPORTANT: This order MUST match X = data.drop(...) from your training script
        df = pd.DataFrame([data_dict])
        
        # Ensure the columns are in the exact order the model expects
        column_order = [
            'age', 'sex', 'bmi', 'smoker', 'diagnosis_code', 'medication_count',
            'days_hospitalized', 'readmitted', 'last_lab_glucose', 'exercise_frequency',
            'diet_quality', 'income_bracket', 'education_level', 'urban', 'albumin_globulin_ratio'
        ]
        df = df[column_order]

        # 3. Predict
        prediction = xgboost_model.predict(df)[0]

        return (
            f"Based on the clinical parameters provided, the predicted Alanine Aminotransferase (ALT) "
            f"level is {float(prediction):.2f} U/L."
        )

    except Exception as e:
        logger.error(f"XGBoost Prediction Error: {e}")
        return f"I encountered an error while trying to calculate the prediction: {str(e)}"