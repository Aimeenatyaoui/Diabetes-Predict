"""
Bobot MultiSURF final (rata-rata dari 10-Fold Cross Validation).
Konfigurasi terbaik: CBR + MultiSURF, K=9
Accuracy: 76.03%, Precision: 65.87%, Recall: 64.54%, F1-Score: 64.95%
"""

import numpy as np

# Rata-rata bobot dari 10 fold (sebelum normalisasi min-max internal)
MULTISURF_RAW_WEIGHTS = {
    "Pregnancies": 0.0417,
    "Glucose": 1.0000,
    "BloodPressure": 0.0155,
    "SkinThickness": 0.0354,
    "Insulin": 0.0983,
    "BMI": 0.2342,
    "DiabetesPedigreeFunction": 0.1447,
    "Age": 0.2594,
}

FEATURES_ORDER = [
    "Pregnancies",
    "Glucose",
    "BloodPressure",
    "SkinThickness",
    "Insulin",
    "BMI",
    "DiabetesPedigreeFunction",
    "Age",
]

# Array bobot sudah dalam urutan FEATURES_ORDER
MULTISURF_WEIGHTS_ARRAY = np.array(
    [MULTISURF_RAW_WEIGHTS[f] for f in FEATURES_ORDER], dtype=float
)

# Konfigurasi optimal
OPTIMAL_K = 9
