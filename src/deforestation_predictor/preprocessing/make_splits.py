from pathlib import Path
from datetime import datetime

import numpy as np
import pandas as pd

from deforestation_predictor.preprocessing.catalog import (
    build_raster_catalog,
    build_gt_catalog,
)
from deforestation_predictor.preprocessing.builder import build_sample
from deforestation_predictor.preprocessing.windows import CONTEXT_MONTHS, GAP_MONTHS


