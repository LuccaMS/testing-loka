"""
Patient Data Analytics Tool
Queries the patient dataset CSV to answer analytical questions.
"""
import pandas as pd
import logging
from typing import Optional, Literal
from langchain_core.tools import tool

logger = logging.getLogger("rag-agent.tools.analytics")

# Global dataset (loaded once during initialization)
_patient_data: Optional[pd.DataFrame] = None

# Reverse mappings (for display purposes)
REVERSE_MAPPINGS = {
    'sex': {0: 'Female', 1: 'Male'},
    'smoker': {0: 'No', 1: 'Yes'},
    'diagnosis_code': {1.0: 'D1', 2.0: 'D2', 3.0: 'D3', 4.0: 'D4', 5.0: 'D5'},
    'exercise_frequency': {0.0: 'Low', 1.0: 'Moderate', 2.0: 'High'},
    'diet_quality': {0.0: 'Poor', 1.0: 'Average', 2.0: 'Good'},
    'income_bracket': {0.0: 'Low', 1.0: 'Middle', 2.0: 'High'},
    'education_level': {0.0: 'Primary', 1.0: 'Secondary', 2.0: 'Tertiary'},
    'urban': {0.0: 'No', 1.0: 'Yes'},
    'readmitted': {0.0: 'No', 1.0: 'Yes'}
}

# Forward mappings (same as prediction tool)
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


def init_patient_data(csv_path: str):
    """
    Load the patient dataset CSV into memory.
    Called during application startup.
    """
    global _patient_data
    try:
        _patient_data = pd.read_csv(csv_path)
        logger.info(f"✓ Loaded patient dataset: {len(_patient_data)} records, {len(_patient_data.columns)} columns")
    except Exception as e:
        logger.error(f"✗ Failed to load patient dataset: {e}")
        raise


@tool
def query_patient_data(
    # Categorical filters (using Literal for type safety)
    sex: Optional[Literal["Male", "Female"]] = None,
    smoker: Optional[Literal["Yes", "No"]] = None,
    diagnosis_code: Optional[Literal["D1", "D2", "D3", "D4", "D5"]] = None,
    readmitted: Optional[Literal["Yes", "No"]] = None,
    exercise_frequency: Optional[Literal["Low", "Moderate", "High"]] = None,
    diet_quality: Optional[Literal["Poor", "Average", "Good"]] = None,
    income_bracket: Optional[Literal["Low", "Middle", "High"]] = None,
    education_level: Optional[Literal["Primary", "Secondary", "Tertiary"]] = None,
    urban: Optional[Literal["Yes", "No"]] = None,
    
    # Numeric filters
    age_min: Optional[float] = None,
    age_max: Optional[float] = None,
    bmi_min: Optional[float] = None,
    bmi_max: Optional[float] = None,
    medication_count_min: Optional[int] = None,
    medication_count_max: Optional[int] = None,
    days_hospitalized_min: Optional[int] = None,
    days_hospitalized_max: Optional[int] = None,
    
    # Aggregation
    aggregation: Literal["count", "mean_age", "mean_bmi", "mean_alt", "mean_glucose", 
                        "sum_medications", "sum_days_hospitalized",
                        "max_age", "max_bmi", "max_alt",
                        "min_age", "min_bmi", "min_alt"] = "count",
    
    # Grouping
    group_by: Optional[Literal["sex", "diagnosis_code", "smoker", "readmitted", 
                               "exercise_frequency", "diet_quality", "urban"]] = None
) -> str:
    """
    Query the patient dataset to answer analytical questions.
    
    This tool answers questions like:
    - "How many males older than 40 are readmitted?"
    - "What's the average BMI for female smokers?"
    - "How many patients take more than 5 medications?"
    
    Args:
        sex: Filter by gender (Male/Female)
        smoker: Filter by smoking status (Yes/No)
        diagnosis_code: Filter by diagnosis (D1/D2/D3/D4/D5)
        readmitted: Filter by readmission status (Yes/No)
        exercise_frequency: Filter by exercise level (Low/Moderate/High)
        diet_quality: Filter by diet quality (Poor/Average/Good)
        income_bracket: Filter by income (Low/Middle/High)
        education_level: Filter by education (Primary/Secondary/Tertiary)
        urban: Filter by location (Yes=Urban/No=Rural)
        
        age_min: Minimum age (e.g., 40 for "older than 40")
        age_max: Maximum age (e.g., 65 for "younger than 65")
        bmi_min: Minimum BMI (e.g., 30 for "BMI over 30")
        bmi_max: Maximum BMI
        medication_count_min: Minimum medications (e.g., 5 for "more than 5 meds")
        medication_count_max: Maximum medications
        days_hospitalized_min: Minimum hospital days
        days_hospitalized_max: Maximum hospital days
        
        aggregation: What to calculate:
            - count: Count matching patients
            - mean_age: Average age
            - mean_bmi: Average BMI
            - mean_alt: Average ALT level
            - mean_glucose: Average glucose level
            - sum_medications: Total medications
            - sum_days_hospitalized: Total hospital days
            - max_age, min_age: Age range
            - max_bmi, min_bmi: BMI range
            - max_alt, min_alt: ALT range
        
        group_by: Group results by field (sex, diagnosis_code, smoker, etc.)
    
    Returns:
        String with the query result
    
    Examples:
        # How many males over 40 are readmitted?
        query_patient_data(sex="Male", age_min=40, readmitted="Yes", aggregation="count")
        
        # Average BMI for female smokers?
        query_patient_data(sex="Female", smoker="Yes", aggregation="mean_bmi")
        
        # Count by gender?
        query_patient_data(aggregation="count", group_by="sex")
    """
    if _patient_data is None:
        return "Error: Patient dataset not loaded."
    
    try:
        df = _patient_data.copy()
        
        # Apply categorical filters
        if sex is not None:
            df = df[df['sex'] == MAPPINGS['sex'][sex]]
        if smoker is not None:
            df = df[df['smoker'] == MAPPINGS['smoker'][smoker]]
        if diagnosis_code is not None:
            df = df[df['diagnosis_code'] == MAPPINGS['diagnosis_code'][diagnosis_code]]
        if readmitted is not None:
            df = df[df['readmitted'] == MAPPINGS['readmitted'][readmitted]]
        if exercise_frequency is not None:
            df = df[df['exercise_frequency'] == MAPPINGS['exercise_frequency'][exercise_frequency]]
        if diet_quality is not None:
            df = df[df['diet_quality'] == MAPPINGS['diet_quality'][diet_quality]]
        if income_bracket is not None:
            df = df[df['income_bracket'] == MAPPINGS['income_bracket'][income_bracket]]
        if education_level is not None:
            df = df[df['education_level'] == MAPPINGS['education_level'][education_level]]
        if urban is not None:
            df = df[df['urban'] == MAPPINGS['urban'][urban]]
        
        # Apply numeric filters
        if age_min is not None:
            df = df[df['age'] >= age_min]
        if age_max is not None:
            df = df[df['age'] <= age_max]
        if bmi_min is not None:
            df = df[df['bmi'] >= bmi_min]
        if bmi_max is not None:
            df = df[df['bmi'] <= bmi_max]
        if medication_count_min is not None:
            df = df[df['medication_count'] >= medication_count_min]
        if medication_count_max is not None:
            df = df[df['medication_count'] <= medication_count_max]
        if days_hospitalized_min is not None:
            df = df[df['days_hospitalized'] >= days_hospitalized_min]
        if days_hospitalized_max is not None:
            df = df[df['days_hospitalized'] <= days_hospitalized_max]
        
        logger.info(f"🔍 Filtered to {len(df)} records")
        
        # If no records match
        if len(df) == 0:
            return "No patients found matching the specified criteria."
        
        # Perform aggregation
        result = _perform_aggregation(df, aggregation, group_by)
        return result
    
    except Exception as e:
        logger.error(f"Query error: {e}", exc_info=True)
        return f"Error processing query: {str(e)}"


def _perform_aggregation(df: pd.DataFrame, aggregation: str, group_by: Optional[str] = None) -> str:
    """Perform aggregation on the dataframe."""
    
    # Group by if specified
    if group_by:
        if aggregation == 'count':
            grouped = df.groupby(group_by).size()
        elif aggregation == 'mean_age':
            grouped = df.groupby(group_by)['age'].mean()
        elif aggregation == 'mean_bmi':
            grouped = df.groupby(group_by)['bmi'].mean()
        elif aggregation == 'mean_alt':
            grouped = df.groupby(group_by)['alanine_aminotransferase'].mean()
        elif aggregation == 'mean_glucose':
            grouped = df.groupby(group_by)['last_lab_glucose'].mean()
        elif aggregation == 'sum_medications':
            grouped = df.groupby(group_by)['medication_count'].sum()
        elif aggregation == 'sum_days_hospitalized':
            grouped = df.groupby(group_by)['days_hospitalized'].sum()
        elif aggregation == 'max_age':
            grouped = df.groupby(group_by)['age'].max()
        elif aggregation == 'min_age':
            grouped = df.groupby(group_by)['age'].min()
        elif aggregation == 'max_bmi':
            grouped = df.groupby(group_by)['bmi'].max()
        elif aggregation == 'min_bmi':
            grouped = df.groupby(group_by)['bmi'].min()
        elif aggregation == 'max_alt':
            grouped = df.groupby(group_by)['alanine_aminotransferase'].max()
        elif aggregation == 'min_alt':
            grouped = df.groupby(group_by)['alanine_aminotransferase'].min()
        else:
            return f"Error: Unknown aggregation '{aggregation}'."
        
        # Format results with readable labels
        results = []
        for value, result in grouped.items():
            # Convert encoded value back to label
            if group_by in REVERSE_MAPPINGS and value in REVERSE_MAPPINGS[group_by]:
                label = REVERSE_MAPPINGS[group_by][value]
            else:
                label = str(value)
            
            if aggregation == 'count':
                results.append(f"{label}: {result} patients")
            else:
                results.append(f"{label}: {result:.2f}")
        
        return ", ".join(results)
    
    # No grouping - single aggregation
    else:
        if aggregation == 'count':
            return f"Found {len(df)} patients matching the criteria."
        elif aggregation == 'mean_age':
            return f"Average age: {df['age'].mean():.2f} years"
        elif aggregation == 'mean_bmi':
            return f"Average BMI: {df['bmi'].mean():.2f}"
        elif aggregation == 'mean_alt':
            return f"Average ALT: {df['alanine_aminotransferase'].mean():.2f} U/L"
        elif aggregation == 'mean_glucose':
            return f"Average glucose: {df['last_lab_glucose'].mean():.2f}"
        elif aggregation == 'sum_medications':
            return f"Total medications: {df['medication_count'].sum()}"
        elif aggregation == 'sum_days_hospitalized':
            return f"Total hospital days: {df['days_hospitalized'].sum()}"
        elif aggregation == 'max_age':
            return f"Maximum age: {df['age'].max():.0f} years"
        elif aggregation == 'min_age':
            return f"Minimum age: {df['age'].min():.0f} years"
        elif aggregation == 'max_bmi':
            return f"Maximum BMI: {df['bmi'].max():.2f}"
        elif aggregation == 'min_bmi':
            return f"Minimum BMI: {df['bmi'].min():.2f}"
        elif aggregation == 'max_alt':
            return f"Maximum ALT: {df['alanine_aminotransferase'].max():.2f} U/L"
        elif aggregation == 'min_alt':
            return f"Minimum ALT: {df['alanine_aminotransferase'].min():.2f} U/L"
        else:
            return f"Error: Unknown aggregation '{aggregation}'."