"""
Data preparation and feature engineering for nepotism modeling.

This script loads generated synthetic data, validates quality, applies transformations,
engineers features at multiple levels (candidate, employee, manager, department),
and saves processed datasets for downstream modeling.
"""

import pandas as pd
import numpy as np
from pathlib import Path


COLLEGE_EDUCATION_LEVELS = {"Bachelor", "Master", "Doctorate"}


def get_paths():
    """Define and return all input/output paths."""
    project_root = Path(__file__).parent.parent
    input_file = project_root / "data" / "generated" / "nepotism_synthetic_data.xlsx"
    output_dir = project_root / "data" / "processed"
    
    return {
        "input": input_file,
        "output_dir": output_dir,
        "candidates": output_dir / "candidates_model.csv",
        "employees": output_dir / "employees_model.csv",
        "managers": output_dir / "manager_features.csv",
        "departments": output_dir / "department_features.csv",
    }


def load_data(input_file):
    """
    Load Excel sheets and return dataframes.
    
    Parameters
    ----------
    input_file : Path
        Path to Excel file
        
    Returns
    -------
    dict
        Dictionary with 'candidates' and 'employees' DataFrames
    """
    candidates = pd.read_excel(input_file, sheet_name="candidates")
    employees = pd.read_excel(input_file, sheet_name="employees")
    
    print(f"✓ Loaded {len(candidates)} candidates")
    print(f"✓ Loaded {len(employees)} employees")
    
    return {"candidates": candidates, "employees": employees}


def validate_candidates(df):
    """
    Validate candidates DataFrame.
    
    Parameters
    ----------
    df : pd.DataFrame
        Candidates data
        
    Raises
    ------
    ValueError
        If validation fails
    """
    required_cols = {"candidate_id", "connection_strength", "family_link_flag", "referral_flag",
                     "high_school_gpa", "college_gpa", "education_level"}
    
    missing_cols = required_cols - set(df.columns)
    if missing_cols:
        raise ValueError(f"Missing required columns: {missing_cols}")
    
    # No missing IDs
    if df["candidate_id"].isna().any():
        raise ValueError("Found missing candidate_id values")
    
    if df["candidate_id"].duplicated().any():
        raise ValueError("Found duplicate candidate_id values")
    
    # Binary flags must be 0/1
    binary_cols = {"family_link_flag", "referral_flag"}
    for col in binary_cols:
        if not df[col].isin([0, 1]).all():
            raise ValueError(f"Column {col} contains values other than 0/1")
    
    # GPA values must be valid (0-100 scale)
    if df["high_school_gpa"].notna().any():
        invalid_hs = df["high_school_gpa"][(df["high_school_gpa"] < 0) | (df["high_school_gpa"] > 100)]
        if len(invalid_hs) > 0:
            raise ValueError(f"Found invalid HS GPA values (not in 0-100 range): {len(invalid_hs)} rows")
    
    if df["college_gpa"].notna().any():
        invalid_coll = df["college_gpa"][(df["college_gpa"] < 0) | (df["college_gpa"] > 100)]
        if len(invalid_coll) > 0:
            raise ValueError(f"Found invalid college GPA values (not in 0-100 range): {len(invalid_coll)} rows")
    
    print("✓ Candidates validation passed")


def validate_employees(df):
    """
    Validate employees DataFrame.
    
    Parameters
    ----------
    df : pd.DataFrame
        Employees data
        
    Raises
    ------
    ValueError
        If validation fails
    """
    required_cols = {"employee_id", "company_id", "manager_id", "department_id", "age",
                     "years_experience", "promoted_flag", "months_to_promotion", "salary",
                     "performance_score", "connection_strength", "family_link_flag", "referral_flag"}
    
    missing_cols = required_cols - set(df.columns)
    if missing_cols:
        raise ValueError(f"Missing required columns: {missing_cols}")
    
    # No missing IDs
    if df["employee_id"].isna().any():
        raise ValueError("Found missing employee_id values")
    
    if df["employee_id"].duplicated().any():
        raise ValueError("Found duplicate employee_id values")
    
    # Binary flags must be 0/1
    binary_cols = {"promoted_flag", "family_link_flag", "referral_flag"}
    for col in binary_cols:
        if not df[col].isin([0, 1]).all():
            raise ValueError(f"Column {col} contains values other than 0/1")
    
    # Age and years_experience must be reasonable
    if (df["age"] < 15).any() or (df["age"] > 100).any():
        raise ValueError("Found unreasonable age values (outside 15-100)")
    
    if (df["years_experience"] < 0).any() or (df["years_experience"] > 80).any():
        raise ValueError("Found unreasonable years_experience values")
    
    # months_to_promotion must be null when promoted_flag == 0
    invalid_promotion = df[(df["promoted_flag"] == 0) & (df["months_to_promotion"].notna())]
    if len(invalid_promotion) > 0:
        raise ValueError(f"Found {len(invalid_promotion)} rows with promoted_flag=0 but non-null months_to_promotion")
    
    print("✓ Employees validation passed")


def clean_dtypes(candidates, employees):
    """
    Clean and enforce data types.
    
    Parameters
    ----------
    candidates : pd.DataFrame
        Candidates data
    employees : pd.DataFrame
        Employees data
        
    Returns
    -------
    tuple
        Cleaned (candidates, employees) dataframes
    """
    # Candidates - numeric integers (non-ID columns)
    int_cols_cand = {"family_link_flag", "referral_flag", "hired_flag", "age", "years_experience"}
    for col in int_cols_cand:
        if col in candidates.columns:
            candidates[col] = candidates[col].astype("int64")
    
    # Candidates - floats
    float_cols_cand = {"high_school_gpa", "college_gpa", "interview_score", "test_score", 
                       "merit_score", "connection_strength"}
    for col in float_cols_cand:
        if col in candidates.columns:
            candidates[col] = candidates[col].astype("float64")
    
    # Employees - numeric integers (non-ID columns)
    int_cols_emp = {"age", "years_experience", "promoted_flag", "family_link_flag", 
                    "referral_flag", "role_level", "tenure_months"}
    for col in int_cols_emp:
        if col in employees.columns:
            employees[col] = employees[col].astype("int64")
    
    # Employees - floats
    float_cols_emp = {"months_to_promotion", "salary", "performance_score", "connection_strength",
                      "high_school_gpa", "college_gpa", "merit_score"}
    for col in float_cols_emp:
        if col in employees.columns:
            employees[col] = employees[col].astype("float64")
    
    # Ensure ID columns and categorical strings stay as object (str)
    str_cols_cand = {"candidate_id", "company_id", "scenario", "education_level", 
                     "discretionary_channel", "gender"}
    for col in str_cols_cand:
        if col in candidates.columns:
            candidates[col] = candidates[col].astype("object")
    
    str_cols_emp = {"employee_id", "company_id", "manager_id", "department_id", "scenario", 
                    "education_level", "discretionary_channel", "gender"}
    for col in str_cols_emp:
        if col in employees.columns:
            employees[col] = employees[col].astype("object")
    
    print("✓ Data types cleaned")
    
    return candidates, employees


def engineer_candidate_features(df):
    """
    Create engineered features for candidates.
    
    Parameters
    ----------
    df : pd.DataFrame
        Candidates data
        
    Returns
    -------
    pd.DataFrame
        Dataframe with engineered features
    """
    df = df.copy()
    
    # Treat any post-secondary degree as college-level education.
    df["has_college"] = df["education_level"].isin(COLLEGE_EDUCATION_LEVELS).astype("int64")
    
    # HS GPA z-score
    hs_gpa_mean = df["high_school_gpa"].mean()
    hs_gpa_std = df["high_school_gpa"].std() + 1e-8
    df["hs_gpa_zscore"] = (df["high_school_gpa"] - hs_gpa_mean) / hs_gpa_std
    df["hs_gpa_zscore"] = df["hs_gpa_zscore"].fillna(0)
    
    # College GPA z-score
    college_gpa_mean = df["college_gpa"].mean()
    college_gpa_std = df["college_gpa"].std() + 1e-8
    df["college_gpa_zscore"] = (df["college_gpa"] - college_gpa_mean) / college_gpa_std
    df["college_gpa_zscore"] = df["college_gpa_zscore"].fillna(0)
    
    # Use college GPA when available, otherwise fall back to the high school signal.
    df["combined_academic_score"] = np.where(
        df["has_college"] == 1,
        0.3 * df["hs_gpa_zscore"] + 0.7 * df["college_gpa_zscore"],
        df["hs_gpa_zscore"],
    )
    
    # Experience age ratio
    df["experience_age_ratio"] = df["years_experience"] / (df["age"] + 1)
    
    # is_high_connection (connection_strength >= some threshold, or any connection type)
    df["is_high_connection"] = ((df["connection_strength"] > 0) | 
                                (df["family_link_flag"] == 1) | 
                                (df["referral_flag"] == 1)).astype("int64")
    
    # is_family_or_referral
    df["is_family_or_referral"] = ((df["family_link_flag"] == 1) | 
                                   (df["referral_flag"] == 1)).astype("int64")
    
    return df


def engineer_employee_features(df):
    """
    Create engineered features for employees.
    
    Parameters
    ----------
    df : pd.DataFrame
        Employees data
        
    Returns
    -------
    pd.DataFrame
        Dataframe with engineered features
    """
    df = df.copy()
    
    # tenure_years (from tenure_months)
    df["tenure_years"] = df["tenure_months"] / 12.0
    
    # high_performer_flag (top 25% performance rating)
    perf_75 = df["performance_score"].quantile(0.75)
    df["high_performer_flag"] = (df["performance_score"] >= perf_75).astype("int64")
    
    # salary_band (quartiles). Use category codes so the feature stays valid
    # even if duplicate cut points reduce the number of distinct bins.
    salary_band = pd.qcut(df["salary"], q=4, duplicates="drop")
    df["salary_band"] = salary_band.cat.codes.add(1).astype("int64")
    
    # performance_salary_gap (correlation residual: actual salary - expected salary by performance)
    salary_by_perf = df.groupby("performance_score")["salary"].transform("mean")
    df["performance_salary_gap"] = df["salary"] - salary_by_perf
    
    # is_high_connection (any connection type)
    df["is_high_connection"] = ((df["connection_strength"] > 0) | 
                                (df["family_link_flag"] == 1) | 
                                (df["referral_flag"] == 1)).astype("int64")
    
    # promotion_eligible_flag (promoted or long tenure without promotion)
    df["promotion_eligible_flag"] = ((df["promoted_flag"] == 1) | 
                                     (df["tenure_years"] >= 5)).astype("int64")
    
    return df


def create_manager_features(df):
    """
    Create manager-level aggregated features.
    
    Parameters
    ----------
    df : pd.DataFrame
        Employees data (with engineered features)
        
    Returns
    -------
    pd.DataFrame
        Manager-level aggregated features
    """
    manager_df = df[df["manager_id"].notna()].copy()
    manager_groups = manager_df.groupby(["company_id", "scenario", "manager_id"], dropna=False)

    manager_features = manager_groups.agg(
        headcount=("employee_id", "size"),
        connected_employee_share=("is_high_connection", "mean"),
        family_link_share=("family_link_flag", "mean"),
        referral_share=("referral_flag", "mean"),
        promoted_share=("promoted_flag", "mean"),
        avg_salary=("salary", "mean"),
        avg_performance=("performance_score", "mean"),
        avg_merit=("merit_score", "mean"),
    ).reset_index()

    return manager_features


def create_department_features(df):
    """
    Create department-level aggregated features.
    
    Parameters
    ----------
    df : pd.DataFrame
        Employees data (with engineered features)
        
    Returns
    -------
    pd.DataFrame
        Department-level aggregated features
    """
    dept_groups = df.groupby(["company_id", "scenario", "department_id"], dropna=False)

    dept_features = dept_groups.agg(
        headcount=("employee_id", "size"),
        connected_employee_share=("is_high_connection", "mean"),
        family_link_share=("family_link_flag", "mean"),
        referral_share=("referral_flag", "mean"),
        promoted_share=("promoted_flag", "mean"),
        avg_salary=("salary", "mean"),
        avg_performance=("performance_score", "mean"),
        avg_merit=("merit_score", "mean"),
        manager_count=("manager_id", "nunique"),
    ).reset_index()

    return dept_features


def save_datasets(paths, candidates, employees, managers, departments):
    """
    Save processed datasets to CSV files.
    
    Parameters
    ----------
    paths : dict
        Dictionary with output paths
    candidates : pd.DataFrame
        Processed candidates
    employees : pd.DataFrame
        Processed employees
    managers : pd.DataFrame
        Manager aggregated features
    departments : pd.DataFrame
        Department aggregated features
    """
    # Create output directory
    paths["output_dir"].mkdir(parents=True, exist_ok=True)
    
    # Save datasets
    candidates.to_csv(paths["candidates"], index=False)
    employees.to_csv(paths["employees"], index=False)
    managers.to_csv(paths["managers"], index=False)
    departments.to_csv(paths["departments"], index=False)
    
    print("✓ Datasets saved successfully")


def print_summary(paths, candidates, employees, managers, departments):
    """
    Print summary of saved datasets.
    
    Parameters
    ----------
    paths : dict
        Dictionary with output paths
    candidates : pd.DataFrame
        Processed candidates
    employees : pd.DataFrame
        Processed employees
    managers : pd.DataFrame
        Manager aggregated features
    departments : pd.DataFrame
        Department aggregated features
    """
    print("\n" + "="*70)
    print("DATA PREPARATION SUMMARY")
    print("="*70)
    print(f"\nInput: {paths['input']}")
    print("\nOutput Files:")
    print(f"  - {paths['candidates']} ({len(candidates)} rows)")
    print(f"  - {paths['employees']} ({len(employees)} rows)")
    print(f"  - {paths['managers']} ({len(managers)} rows)")
    print(f"  - {paths['departments']} ({len(departments)} rows)")
    print("="*70 + "\n")


def main():
    """Main execution function."""
    # Setup
    paths = get_paths()
    
    print("\n" + "="*70)
    print("STARTING DATA PREPARATION")
    print("="*70 + "\n")
    
    # Load
    data = load_data(paths["input"])
    candidates = data["candidates"]
    employees = data["employees"]
    
    # Validate
    validate_candidates(candidates)
    validate_employees(employees)
    
    # Clean dtypes
    candidates, employees = clean_dtypes(candidates, employees)
    
    # Engineer features
    print("Engineered candidate features...")
    candidates = engineer_candidate_features(candidates)
    
    print("Engineered employee features...")
    employees = engineer_employee_features(employees)
    
    print("Aggregated manager features...")
    managers = create_manager_features(employees)
    
    print("Aggregated department features...")
    departments = create_department_features(employees)
    
    # Save
    save_datasets(paths, candidates, employees, managers, departments)
    
    # Summary
    print_summary(paths, candidates, employees, managers, departments)


if __name__ == "__main__":
    main()
