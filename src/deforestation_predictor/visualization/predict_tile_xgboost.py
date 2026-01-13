import argparse
import numpy as np
import rasterio
import json
import pandas as pd
import joblib
import torch  # Alleen nodig als we functies hergebruiken die torch verwachten
from pathlib import Path
from tqdm import tqdm

# Importeer de data-loading logica uit je bestaande script
# Zorg dat predict_tile.py in je python path staat (staat in src/deforestation_predictor/visualization/)
from deforestation_predictor.visualization.predict_tile import build_inference_cube
from deforestation_predictor.preprocessing.catalog import build_raster_catalog

# Hardcoded variable lists (MOETEN EXACT MATCHEN MET TRAINING!)
MONTHLY_VARS = [
    'precipitation', 'temperature', 'confidence', 'lastmonth', 
    'timesinceloss', 'totallossalerts', 'previoussameseason', 'patchdensity'
]
STATIC_VARS = [
    "elevation", "slope", "peatlands", "initialforestcover", "historicloss",
    "forestheight", "forestedgedensity", "landpercentage", "populationcurrent",
    "populationincrease", "closenesstoroads", "closenesstowaterways",
    "closenesstocropland", "closenesstococoa", "closenesstocoffee",
    "closenesstofiber", "closenesstosoybean", "closenesstocattleabove2000",
    "closenesstocattleabove10000", "closenesstomining", "palmoilmills",
    "croplandcapacity100p", "croplandcapacitybelow50p", "croplandcapacityover50p",
    "catexcap", "aridityannual", "ariditydriestquarter", "closenesstoforestedge",
    "wetlands", "wdpa",
]

def predict_xgboost_chunked(model, X, context_months, chunk_size=1024):
    """
    Voert inference uit met XGBoost in blokken om geheugen te besparen.
    
    Args:
        model: Het geladen XGBoost model (via joblib).
        X: De input cube met shape [Channels, Time, Height, Width].
        context_months: Aantal maanden context dat het model verwacht.
        chunk_size: Grootte van het blok (bijv. 1024x1024 pixels).
    
    Returns:
        prob_map: 2D numpy array met waarschijnlijkheden (0-1).
    """
    C, T, H, W = X.shape
    
    # 1. Temporal Slicing (RQ2) - Exact zoals in train_baseline_xgboost.py
    # Pak alleen de laatste N maanden
    if T >= context_months:
        X = X[:, -context_months:, :, :]
        print(f"Sliced time dimension to last {context_months} months. New shape: {X.shape}")
    else:
        raise ValueError(f"Data has {T} months, but model expects {context_months} months.")
        
    # Update shape na slicing
    C, T, H, W = X.shape
    
    output_map = np.zeros((H, W), dtype=np.float32)
    
    print(f"Running XGBoost inference on {H}x{W} image...")
    print(f"Features per pixel: {C * T} (Channels={C} * Time={T})")

    # We itereren in blokken (chunks) om RAM te sparen
    # Geen overlap nodig bij XGBoost omdat het pixel-per-pixel werkt!
    for r in tqdm(range(0, H, chunk_size)):
        for c in range(0, W, chunk_size):
            # Bepaal grenzen van het huidige blok
            r_end = min(r + chunk_size, H)
            c_end = min(c + chunk_size, W)
            
            # Pak het blok: [C, T, h_chunk, w_chunk]
            chunk = X[:, :, r:r_end, c:c_end]
            hc, wc = chunk.shape[2], chunk.shape[3]
            
            # Flatten exact zoals in training script:
            # [C, T, h, w] -> transpose(2, 3, 0, 1) -> [h, w, C, T] -> reshape -> [N_pixels, Features]
            chunk_flat = chunk.transpose(2, 3, 0, 1).reshape(hc * wc, -1)
            
            # Predict
            # predict_proba geeft [N, 2], we willen de kans op klasse 1 (index 1)
            probs = model.predict_proba(chunk_flat)[:, 1]
            
            # Reshape terug naar 2D blok en plaats in output map
            output_map[r:r_end, c:c_end] = probs.reshape(hc, wc)
            
    return output_map

def main():
    print("Starting XGBoost tile prediction...")
    parser = argparse.ArgumentParser(description="Predict Tile with XGBoost")
    parser.add_argument("--data_root", type=str, required=True, help="Path to processed input data")
    parser.add_argument("--maxima_file", type=str, required=True, help="Path to maxima.json")
    parser.add_argument("--checkpoint", type=str, required=True, help="Path to .pkl model file")
    parser.add_argument("--tile_id", type=str, default="GABON")
    parser.add_argument("--date", type=str, default="2024-01-01", help="Target date YYYY-MM-DD")
    parser.add_argument("--context_months", type=int, default=12)
    parser.add_argument("--output_path", type=str, default="prediction_xgboost.tif")
    
    args = parser.parse_args()
    
    # 1. Load Catalogs
    print("Loading catalogs...")
    full_cat = build_raster_catalog(args.data_root)
    
    # Filter catalogs (zorg dat deze variabelen overeenkomen met training!)
    dyn_cat = full_cat[full_cat["variable"].isin(MONTHLY_VARS)].reset_index(drop=True)
    stat_cat = full_cat[full_cat["variable"].isin(STATIC_VARS)].reset_index(drop=True)
    
    # Load Maxima
    with open(args.maxima_file, "r") as f:
        maxima = json.load(f)

    # 2. Build Input Cube
    # We hergebruiken de functie uit je DL script, die doet precies het zware werk
    target_date = pd.to_datetime(args.date)
    
    # Let op: build_inference_cube verwacht 'gap=1' standaard, check of dit klopt met je training!
    X, profile = build_inference_cube(
        args.tile_id, target_date, dyn_cat, stat_cat, 
        context=args.context_months, gap=1, maxima=maxima
    )
    
    # 3. Load XGBoost Model
    print(f"Loading XGBoost model from {args.checkpoint}...")
    model = joblib.load(args.checkpoint)
    
    # 4. Predict
    prob_map = predict_xgboost_chunked(model, X, context_months=args.context_months)
    
    # 5. Save as GeoTIFF
    profile.update(dtype=rasterio.float32, count=1, compress='lzw')
    
    print(f"Saving prediction to {args.output_path}...")
    with rasterio.open(args.output_path, "w", **profile) as dst:
        dst.write(prob_map, 1)

if __name__ == "__main__":
    main()