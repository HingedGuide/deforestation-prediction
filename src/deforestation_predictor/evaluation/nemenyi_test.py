"""
Script Name: Parse Logs, Friedman Test, and Nemenyi Post-Hoc

Description:
    This script reads SLURM log files, extracts F0.5 scores, pairs them by seed, 
    and performs a Friedman test. If the result is significant, it automatically 
    runs a Nemenyi post-hoc test to identify exactly which models differ.
"""

import os
import glob
import re
import pandas as pd
from scipy.stats import friedmanchisquare

# Importeer de post-hoc library
try:
    import scikit_posthocs as sp
except ImportError:
    print("De library 'scikit-posthocs' is niet gevonden. Installeer deze via:")
    print("pip install scikit-posthocs")
    exit()

def main():
    log_dir = "logs_many_runs"
    log_files = glob.glob(os.path.join(log_dir, "*.log"))
    
    if not log_files:
        print(f"Geen logbestanden gevonden in '{log_dir}'.")
        return

    results = []

    # Regex patronen
    seed_pattern = re.compile(r"Using fixed random seed:\s+(\d+)")
    model_pattern = re.compile(r"FINAL TEST RESULTS \(([^)]+)\):")
    f05_pattern = re.compile(r"F0\.5 Score:\s+([\d\.]+)")

    for file_path in log_files:
        seed, model, f05 = None, None, None
        
        with open(file_path, 'r') as f:
            for line in f:
                seed_match = seed_pattern.search(line)
                if seed_match: seed = int(seed_match.group(1))
                
                model_match = model_pattern.search(line)
                if model_match: model = model_match.group(1)
                
                f05_match = f05_pattern.search(line)
                if f05_match: f05 = float(f05_match.group(1))
        
        if seed is not None and model is not None and f05 is not None:
            results.append({'Seed': seed, 'Model': model, 'F0.5': f05})

    if not results:
        print("Geen complete resultaten gevonden in de logs.")
        return

    df = pd.DataFrame(results)
    pivot_df = df.pivot(index='Seed', columns='Model', values='F0.5')
    pivot_df_clean = pivot_df.dropna()

    models = pivot_df_clean.columns.tolist()
    
    if len(pivot_df_clean) < 3:
        print("Niet genoeg complete gepaarde runs (minimaal 3 nodig) voor een betrouwbare test.")
        return

    print("=== Gevonden F0.5 Scores (Gepaard per Seed) ===")
    print(pivot_df_clean.to_string())
    print("===============================================\n")

    # 1. Friedman Test
    scores_per_model = [pivot_df_clean[model].values for model in models]
    stat, p_value = friedmanchisquare(*scores_per_model)

    print("=== FRIEDMAN TEST ===")
    print(f"Test Statistiek: {stat:.4f}")
    print(f"P-Waarde:        {p_value:.4f}\n")

    # 2. Nemenyi Post-Hoc Test (Alleen als Friedman < 0.05)
    if p_value < 0.05:
        print("CONCLUSIE: Er is een significant verschil gevonden. Start Nemenyi Post-Hoc Test...\n")
        
        # Voer de Nemenyi test uit (verwacht een DataFrame of 2D array)
        nemenyi_results = sp.posthoc_nemenyi_friedman(pivot_df_clean)
        
        print("=== NEMENYI P-WAARDE MATRIX ===")
        # Print de p-waarden overzichtelijk afgerond op 4 decimalen
        print(nemenyi_results.round(4).to_string())
        print("\nInterpretatie: Lees de tabel af als een kruistabel. Als de waarde tussen model A en")
        print("model B kleiner is dan 0.05, dan presteren deze twee modellen statistisch gezien anders.")
    else:
        print("CONCLUSIE: Er is GEEN significant verschil (p >= 0.05).")
        print("Een post-hoc test is daarom niet nodig/verantwoord.")

if __name__ == "__main__":
    main()