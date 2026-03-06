"""
Script Name: Parse Logs and Calculate Statistical Significance (Friedman Test)

Description:
    This script reads all SLURM log files in a specified directory, extracts
    the test F0.5 scores, pairs them by their random seed, and performs 
    a Friedman test to check for statistically significant differences 
    between the models.
"""

import os
import glob
import re
import pandas as pd
from scipy.stats import friedmanchisquare

def main():
    # Map waar je SLURM logs zijn opgeslagen
    log_dir = "logs_many_runs"
    
    # Zoek alle .log bestanden in de map
    log_files = glob.glob(os.path.join(log_dir, "*.log"))
    
    if not log_files:
        print(f"Geen logbestanden gevonden in de map '{log_dir}'.")
        return

    print(f"Start met het scannen van {len(log_files)} logbestanden...\n")

    results = []

    # Regex patronen om de juiste regels in de logs te vinden
    seed_pattern = re.compile(r"Using fixed random seed:\s+(\d+)")
    model_pattern = re.compile(r"FINAL TEST RESULTS \(([^)]+)\):")
    f05_pattern = re.compile(r"F0\.5 Score:\s+([\d\.]+)")

    for file_path in log_files:
        seed = None
        model = None
        f05 = None
        
        with open(file_path, 'r') as f:
            lines = f.readlines()
            
            for line in lines:
                # Zoek naar de seed
                seed_match = seed_pattern.search(line)
                if seed_match:
                    seed = int(seed_match.group(1))
                
                # Zoek naar het model type
                model_match = model_pattern.search(line)
                if model_match:
                    model = model_match.group(1)
                
                # Zoek naar de uiteindelijke F0.5 score
                f05_match = f05_pattern.search(line)
                if f05_match:
                    f05 = float(f05_match.group(1))
        
        # Als we alle drie de waarden in één bestand hebben gevonden, sla het op
        if seed is not None and model is not None and f05 is not None:
            results.append({'Seed': seed, 'Model': model, 'F0.5': f05})

    if not results:
        print("Er konden geen complete resultaten (Seed + Model + F0.5) uit de logs gehaald worden.")
        print("Zijn alle runs al 100% afgerond?")
        return

    # Maak een Pandas DataFrame van de verzamelde resultaten
    df = pd.DataFrame(results)
    
    # Maak een "Pivot Table" (draaitabel) waarbij de rijen de Seeds zijn, 
    # de kolommen de Modellen, en de waarden de F0.5 scores.
    # Dit zet de data perfect klaar voor een gepaarde test!
    pivot_df = df.pivot(index='Seed', columns='Model', values='F0.5')
    
    # Verwijder rijen (seeds) waarbij 1 of meer modellen gecrasht zijn of nog niet klaar zijn
    pivot_df_clean = pivot_df.dropna()

    print("=== Gevonden F0.5 Scores (Gepaard per Seed) ===")
    print(pivot_df_clean.to_string())
    print("===============================================\n")

    # Friedman Test Uitvoeren
    models = pivot_df_clean.columns.tolist()
    
    if len(pivot_df_clean) < 3:
        print(f"Waarschuwing: Niet genoeg complete runs ({len(pivot_df_clean)}) om een betrouwbare Friedman test te doen.")
        print("Wacht tot er per model minimaal 3 runs met dezelfde seed klaar zijn.")
        return
        
    if len(models) < 2:
        print("Er zijn minder dan 2 modellen gevonden om te vergelijken.")
        return

    # Haal de scores per model op als aparte lijsten
    scores_per_model = [pivot_df_clean[model].values for model in models]
    
    # Bereken de Friedman statistieken
    stat, p_value = friedmanchisquare(*scores_per_model)

    print("=== FRIEDMAN TEST RESULTATEN ===")
    print(f"Aantal gepaarde iteraties: {len(pivot_df_clean)}")
    print(f"Vergeleken modellen:       {', '.join(models)}")
    print(f"Test Statistiek:           {stat:.4f}")
    print(f"P-Waarde:                  {p_value:.4f}\n")

    if p_value < 0.05:
        print("CONCLUSIE: Er is een STATISTISCH SIGNIFICANT verschil tussen de modellen (p < 0.05).")
        print("Omdat de test significant is, zou je nu een Nemenyi post-hoc test kunnen doen")
        print("om te zien *welke* specifieke modellen van elkaar verschillen.")
    else:
        print("CONCLUSIE: Er is GEEN statistisch significant verschil tussen de modellen (p >= 0.05).")

if __name__ == "__main__":
    main()