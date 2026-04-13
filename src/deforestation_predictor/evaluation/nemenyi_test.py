"""
Script Name: Parse Logs, Friedman Test, and Nemenyi Post-Hoc for Horizon Analysis

Description:
    This script reads SLURM log files, extracts F0.5 scores, pairs them by seed, 
    and performs a Friedman test based on Prediction Horizon (GT). It explicitly 
    handles older baseline logs missing the 'GT' print statement by defaulting 
    their horizon to 6 months. If the result is significant, it automatically 
    runs a Nemenyi post-hoc test to identify exactly which conditions differ.
"""

import os
import glob
import re
import pandas as pd
from scipy.stats import friedmanchisquare

# Import the post-hoc library
try:
    import scikit_posthocs as sp
except ImportError:
    print("De library 'scikit-posthocs' is niet gevonden. Installeer deze via:")
    print("pip install scikit-posthocs")
    exit()

def main():
    # Directory where SLURM logs are stored
    log_dir = "logs_horizon"
    
    # Find all .log files in the directory
    log_files = glob.glob(os.path.join(log_dir, "*.log"))
    
    if not log_files:
        print(f"Geen logbestanden gevonden in '{log_dir}'.")
        return

    results = []

    # Regex patterns to extract the required metrics
    seed_pattern = re.compile(r"Using fixed random seed:\s+(\d+)")
    model_pattern = re.compile(r"FINAL TEST RESULTS \(([^)]+)\):")
    horizon_pattern = re.compile(r"GT:\s*(\d+)m")
    f05_pattern = re.compile(r"F0\.5 Score:\s+([\d\.]+)")

    for file_path in log_files:
        seed, model, horizon, f05 = None, None, None, None
        
        with open(file_path, 'r') as f:
            for line in f:
                # Extract random seed
                seed_match = seed_pattern.search(line)
                if seed_match: seed = int(seed_match.group(1))
                
                # Extract model name
                model_match = model_pattern.search(line)
                if model_match: model = model_match.group(1)

                # Extract prediction horizon (GT)
                horizon_match = horizon_pattern.search(line)
                if horizon_match: horizon = int(horizon_match.group(1))
                
                # Extract final F0.5 score
                f05_match = f05_pattern.search(line)
                if f05_match: f05 = float(f05_match.group(1))
        
        # Fallback for the baseline 6-month logs that do not have the 'GT: 6m' print statement
        if horizon is None and seed is not None and model is not None and f05 is not None:
            horizon = 6

        # Append to results if all four values were successfully found or inferred
        if seed is not None and model is not None and horizon is not None and f05 is not None:
            # Combine model and horizon into a single unique condition identifier
            condition = f"{model}_Horizon{horizon}m"
            results.append({'Seed': seed, 'Condition': condition, 'F0.5': f05})

    if not results:
        print("Geen complete resultaten gevonden in de logs.")
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

    conditions = pivot_df_clean.columns.tolist()
    
    if len(pivot_df_clean) < 3:
        print("Niet genoeg complete gepaarde runs (minimaal 3 nodig) voor een betrouwbare test.")
        return

    print("=== Gevonden F0.5 Scores (Gepaard per Seed) ===")
    print(pivot_df_clean.to_string())
    print("===============================================\n")

    # 1. Friedman Test
    # Extract the scores per condition as separate arrays
    scores_per_condition = [pivot_df_clean[cond].values for cond in conditions]
    stat, p_value = friedmanchisquare(*scores_per_condition)

    print("=== FRIEDMAN TEST ===")
    print(f"Test Statistiek: {stat:.4f}")
    print(f"P-Waarde:        {p_value:.4f}\n")

    # 2. Nemenyi Post-Hoc Test (Only run if Friedman p < 0.05)
    if p_value < 0.05:
        print("CONCLUSIE: Er is een significant verschil gevonden. Start Nemenyi Post-Hoc Test...\n")
        
        # Run the Nemenyi test (expects a DataFrame or 2D array)
        nemenyi_results = sp.posthoc_nemenyi_friedman(pivot_df_clean)
        
        print("=== NEMENYI P-WAARDE MATRIX ===")
        # Print the p-values cleanly rounded to 4 decimals
        print(nemenyi_results.round(4).to_string())
        print("\nInterpretatie: Lees de tabel af als een kruistabel. Als de waarde tussen conditie A en")
        print("conditie B kleiner is dan 0.05, dan presteren deze twee statistisch gezien anders.")
    else:
        print("CONCLUSIE: Er is GEEN significant verschil (p >= 0.05).")
        print("Een post-hoc test is daarom niet nodig/verantwoord.")

if __name__ == "__main__":
    main()