"""
ALT Prediction Tool
XGBoost-based prediction for Alanine Aminotransferase levels.
"""
import logging
from typing import Literal
import pandas as pd
from langchain_core.tools import tool

logger = logging.getLogger("rag-agent.tools.prediction")

# Mapping dictionaries based on feature engineering logic
MAPPINGS = {
    'sex': {'Female': 0, 'Male': 1},
    'smoker': {'No': 0, 'Yes': 1},
    'diagnosis_code': {'D1': 1.0, 'D2': 2.0, 'D3': 3.0, 'D4': 4.0, 'D5': 5.0},
    'exercise_frequency': {'Low': 0.0, 'Moderate': 1.0, 'High': 2.0},
    'diet_quality': {'Poor': 0.0, 'Average': 1.0, 'Good': 2.0},
    'income_bracket': {'Low': 0.0, 'Middle': 1.0, 'High': 2.0},
    'education_level': {'Primary': 0.0, 'Secondary': 1.0, 'Tertiary': 2.0},
    'urban': {'No': 0.0, 'Yes': 1.0},
    'readmitted': {'No': 0.0, 'Yes': 1.0}
}

# Column order must match training data
COLUMN_ORDER = [
    'age', 'sex', 'bmi', 'smoker', 'diagnosis_code', 'medication_count',
    'days_hospitalized', 'readmitted', 'last_lab_glucose', 'exercise_frequency',
    'diet_quality', 'income_bracket', 'education_level', 'urban', 'albumin_globulin_ratio'
]

@tool
def predict_alanine_aminotransferase(
    age: float = 53.0,
    sex: Literal["Male", "Female"] = "Female",
    bmi: float = 26.9,
    smoker: Literal["Yes", "No"] = "No",
    diagnosis_code: Literal["D1", "D2", "D3", "D4", "D5"] = "D5",
    medication_count: int = 3,
    days_hospitalized: int = 5,
    readmitted: Literal["No", "Yes"] = "No",
    last_lab_glucose: float = 100.1,
    exercise_frequency: Literal["Low", "Moderate", "High"] = "Moderate",
    diet_quality: Literal["Poor", "Average", "Good"] = "Average",
    income_bracket: Literal["Low", "Middle", "High"] = "Middle",
    education_level: Literal["Primary", "Secondary", "Tertiary"] = "Secondary",
    urban: Literal["No", "Yes"] = "Yes",
    albumin_globulin_ratio: float = 0.5037,
) -> str:
    """
    Predicts the Alanine Aminotransferase (ALT) levels for a patient using an XGBoost model.
    Use this tool when users ask for predictions, forecasts, or expected liver enzyme values.
    
    Args:
        age: Patient age in years
        sex: Patient biological sex
        bmi: Body Mass Index
        smoker: Smoking status
        diagnosis_code: Primary diagnosis code (D1-D5)
        medication_count: Number of current medications
        days_hospitalized: Days spent in hospital
        readmitted: Whether patient was readmitted
        last_lab_glucose: Most recent glucose reading
        exercise_frequency: Level of physical activity
        diet_quality: Overall diet quality assessment
        income_bracket: Socioeconomic status
        education_level: Highest education completed
        urban: Urban vs rural residence
        albumin_globulin_ratio: Lab test ratio
        
    Returns:
        Prediction result with ALT level in U/L
    """
    from tools.dependencies import get_xgboost_model
    
    xgboost_model = get_xgboost_model()
    
    if xgboost_model is None:
        return "Error: Prediction model is not currently loaded in the system."
    
    try:
        # 1. Transform literals to numeric values
        data_dict = {
            'age': age,
            'sex': MAPPINGS['sex'][sex],
            'bmi': bmi,
            'smoker': MAPPINGS['smoker'][smoker],
            'diagnosis_code': MAPPINGS['diagnosis_code'][diagnosis_code],
            'medication_count': medication_count,
            'days_hospitalized': days_hospitalized,
            'readmitted': MAPPINGS['readmitted'][readmitted],
            'last_lab_glucose': last_lab_glucose,
            'exercise_frequency': MAPPINGS['exercise_frequency'][exercise_frequency],
            'diet_quality': MAPPINGS['diet_quality'][diet_quality],
            'income_bracket': MAPPINGS['income_bracket'][income_bracket],
            'education_level': MAPPINGS['education_level'][education_level],
            'urban': MAPPINGS['urban'][urban],
            'albumin_globulin_ratio': albumin_globulin_ratio
        }
        
        # 2. Convert to DataFrame with correct column order
        df = pd.DataFrame([data_dict])
        df = df[COLUMN_ORDER]
        
        # 3. Make prediction
        prediction = xgboost_model.predict(df)[0]
        
        return (
            f"Based on the clinical parameters provided, the predicted Alanine Aminotransferase (ALT) "
            f"level is {float(prediction):.2f} U/L."
        )
        
    except Exception as e:
        logger.error(f"XGBoost Prediction Error: {e}", exc_info=True)
        return f"I encountered an error while trying to calculate the prediction: {str(e)}"