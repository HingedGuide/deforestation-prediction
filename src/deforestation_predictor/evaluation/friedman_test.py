"""
Script Name: Parse Logs and Calculate Statistical Significance (Friedman Test)

Description:
    This script reads all SLURM log files in a specified directory, extracts
    the test F0.5 scores, pairs them by their random seed, and performs 
    a Friedman test. It explicitly handles older log files that are missing 
    the 'GT' print statement by defaulting their horizon to 6 months.
"""

import os
import glob
import re
import pandas as pd
from scipy.stats import friedmanchisquare

def main():
    # Directory where SLURM logs are stored
    log_dir = "logs_horizon"  # Adjust this to your log folder if necessary
    
    # Find all .log files in the directory
    log_files = glob.glob(os.path.join(log_dir, "*.log"))
    
    if not log_files:
        print(f"Geen logbestanden gevonden in de map '{log_dir}'.")
        return

    print(f"Start met het scannen van {len(log_files)} logbestanden...\n")

    results = []

    # Regex patterns to extract the required metrics
    seed_pattern = re.compile(r"Using fixed random seed:\s+(\d+)")
    model_pattern = re.compile(r"FINAL TEST RESULTS \(([^)]+)\):")
    horizon_pattern = re.compile(r"GT:\s*(\d+)m")
    f05_pattern = re.compile(r"F0\.5 Score:\s+([\d\.]+)")

    for file_path in log_files:
        seed = None
        model = None
        horizon = None
        f05 = None
        
        with open(file_path, 'r') as f:
            lines = f.readlines()
            
            for line in lines:
                # Extract the random seed
                seed_match = seed_pattern.search(line)
                if seed_match:
                    seed = int(seed_match.group(1))
                
                # Extract the model name
                model_match = model_pattern.search(line)
                if model_match:
                    model = model_match.group(1)

                # Extract the prediction horizon (GT)
                horizon_match = horizon_pattern.search(line)
                if horizon_match:
                    horizon = int(horizon_match.group(1))
                
                # Extract the final F0.5 score
                f05_match = f05_pattern.search(line)
                if f05_match:
                    f05 = float(f05_match.group(1))
        
        # Fallback for the baseline 6-month logs that do not have the 'GT: 6m' print statement
        if horizon is None and seed is not None and model is not None and f05 is not None:
            horizon = 6

        # Append to results if all four values were successfully found or inferred
        if seed is not None and model is not None and horizon is not None and f05 is not None:
            # Create a combined condition name (e.g., "resunet3d_Horizon6m")
            condition = f"{model}_Horizon{horizon}m"
            results.append({'Seed': seed, 'Condition': condition, 'F0.5': f05})

    if not results:
        print("Er konden geen complete resultaten uit de logs gehaald worden.")
        print("Zijn alle runs al 100% afgerond?")
        return

    # Convert results list to a Pandas DataFrame
    df = pd.DataFrame(results)
    
    # Check for duplicates to help debugging log generation
    duplicates = df[df.duplicated(subset=['Seed', 'Condition'], keep=False)]
    if not duplicates.empty:
        print("Waarschuwing: Dubbele entries gevonden! De laatste run per unieke combinatie wordt behouden:")
        print(duplicates.sort_values(by=['Seed', 'Condition']).to_string())
        print("\n")

    # Remove duplicates, keeping only the most recent run for each Seed/Condition pair
    df_clean = df.drop_duplicates(subset=['Seed', 'Condition'], keep='last')
    
    # Create a pivot table where rows are Seeds, columns are Conditions, and values are F0.5 scores
    pivot_df = df_clean.pivot(index='Seed', columns='Condition', values='F0.5')
    
    # Drop rows (seeds) where one or more conditions crashed or are incomplete
    pivot_df_clean = pivot_df.dropna()

    print("=== Gevonden F0.5 Scores (Gepaard per Seed) ===")
    print(pivot_df_clean.to_string())
    print("===============================================\n")

    # Retrieve a list of conditions to evaluate
    conditions = pivot_df_clean.columns.tolist()
    
    if len(pivot_df_clean) < 3:
        print(f"Waarschuwing: Niet genoeg complete runs ({len(pivot_df_clean)}) om een betrouwbare Friedman test te doen.")
        print("Wacht tot er per model/horizon-combinatie minimaal 3 runs met dezelfde seed klaar zijn.")
        return
        
    if len(conditions) < 2:
        print("Er zijn minder dan 2 experimentele condities gevonden om te vergelijken.")
        return

    # Extract the scores per condition as separate arrays
    scores_per_condition = [pivot_df_clean[cond].values for cond in conditions]
    
    # Calculate Friedman statistics
    stat, p_value = friedmanchisquare(*scores_per_condition)

    print("=== FRIEDMAN TEST RESULTATEN ===")
    print(f"Aantal gepaarde iteraties: {len(pivot_df_clean)}")
    print(f"Vergeleken condities:      {', '.join(conditions)}")
    print(f"Test Statistiek:           {stat:.4f}")
    print(f"P-Waarde:                  {p_value:.4f}\n")

    if p_value < 0.05:
        print("CONCLUSIE: Er is een STATISTISCH SIGNIFICANT verschil tussen de geteste condities (p < 0.05).")
        print("Omdat de test significant is, zou je nu een Nemenyi post-hoc test kunnen doen")
        print("om te zien *welke* specifieke condities van elkaar verschillen.")
    else:
        print("CONCLUSIE: Er is GEEN statistisch significant verschil tussen de geteste condities (p >= 0.05).")

if __name__ == "__main__":
    main()