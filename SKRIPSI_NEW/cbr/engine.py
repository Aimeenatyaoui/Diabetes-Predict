from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
from sklearn.preprocessing import MinMaxScaler


FEATURES = [
    "Pregnancies",
    "Glucose",
    "BloodPressure",
    "SkinThickness",
    "Insulin",
    "BMI",
    "DiabetesPedigreeFunction",
    "Age",
]

# Di eksperimen, kolom ini dianggap "0 = missing" saat cleaning.
ZERO_AS_MISSING = {"Glucose", "BloodPressure", "SkinThickness", "Insulin", "BMI"}


@dataclass(frozen=True)
class CBRArtifacts:
    case_base_scaled: np.ndarray  # shape: (n_cases, n_features)
    case_base_labels: np.ndarray  # shape: (n_cases,)
    scaler: MinMaxScaler


def _pick_label_column(df: pd.DataFrame) -> str | None:
    for c in ("Outcome", "validated_outcome", "predicted_outcome"):
        if c in df.columns:
            return c
    return None


def load_case_base_df(path: str | Path) -> pd.DataFrame:
    """
    Load basis kasus dari file lokal atau URL (raw GitHub).

    - Jika `path` adalah URL http(s), akan dicoba dibaca via pandas.
    - Jika file lokal tidak ada / gagal dibaca, mengembalikan DataFrame kosong.
    """
    p = str(path)
    is_url = p.startswith(("http://", "https://"))

    try:
        if not is_url:
            if not isinstance(path, Path):
                path = Path(path)
            if not path.exists():
                return pd.DataFrame()
            if path.suffix.lower() in {".xlsx", ".xls"}:
                return pd.read_excel(path)
            return pd.read_csv(path)

        # URL: unduh dulu (lebih stabil untuk SSL di beberapa environment macOS)
        import io

        try:
            import certifi  # type: ignore
            import requests  # type: ignore

            resp = requests.get(p, timeout=30, verify=certifi.where())
            resp.raise_for_status()
            content = resp.content
        except Exception:
            # Fallback terakhir: coba pandas baca langsung dari URL.
            # (Bisa gagal pada environment tertentu karena SSL cert.)
            lower = p.lower()
            if lower.endswith((".xlsx", ".xls")):
                return pd.read_excel(p)
            return pd.read_csv(p)

        lower = p.lower()
        if lower.endswith((".xlsx", ".xls")):
            return pd.read_excel(io.BytesIO(content))
        return pd.read_csv(io.BytesIO(content))
    except Exception:
        return pd.DataFrame()


def build_artifacts_from_case_base(case_base_df: pd.DataFrame) -> CBRArtifacts:
    missing_cols = [c for c in FEATURES if c not in case_base_df.columns]
    if missing_cols:
        raise ValueError(f"Basis kasus belum punya kolom fitur: {missing_cols}")

    label_col = _pick_label_column(case_base_df)
    if not label_col:
        raise ValueError("Basis kasus belum punya label (Outcome / validated_outcome).")

    df = case_base_df.dropna(subset=FEATURES).copy()
    if len(df) < 3:
        raise ValueError("Basis kasus minimal 3 kasus agar retrieve stabil.")

    X = df[FEATURES].apply(pd.to_numeric, errors="coerce")
    y = pd.to_numeric(df[label_col], errors="coerce").astype("Int64")
    ok = X.notna().all(axis=1) & y.notna()
    X = X.loc[ok]
    y = y.loc[ok].astype(int)

    if len(X) < 3:
        raise ValueError("Basis kasus valid kurang dari 3 baris setelah pembersihan.")

    scaler = MinMaxScaler()
    X_scaled = scaler.fit_transform(X.values)

    return CBRArtifacts(
        case_base_scaled=np.asarray(X_scaled, dtype=float),
        case_base_labels=y.values.astype(int),
        scaler=scaler,
    )


def normalize_weights(raw_importances: np.ndarray | None) -> np.ndarray:
    if raw_importances is None:
        return np.ones(len(FEATURES), dtype=float)

    w = np.asarray(raw_importances, dtype=float).reshape(-1)
    if w.shape[0] != len(FEATURES):
        raise ValueError(f"Panjang bobot ({w.shape[0]}) harus {len(FEATURES)}.")

    denom = float(np.max(w) - np.min(w))
    if denom == 0.0:
        return np.ones_like(w, dtype=float)
    return (w - float(np.min(w))) / denom


def weighted_euclidean_distances(
    case_base_scaled: np.ndarray,
    u_scaled: np.ndarray,
    weights: np.ndarray,
) -> np.ndarray:
    # distances[i] = sqrt(sum_j weights[j] * (X[i,j] - u[j])^2)
    diff = case_base_scaled - u_scaled.reshape(1, -1)
    return np.sqrt(np.sum(weights.reshape(1, -1) * (diff**2), axis=1))


def predict_cbr(
    artifacts: CBRArtifacts,
    u_raw: dict[str, Any],
    *,
    k: int,
    weights: np.ndarray | None,
) -> dict[str, Any]:
    if k <= 0:
        raise ValueError("k harus > 0.")

    x = np.array([float(u_raw[f]) for f in FEATURES], dtype=float).reshape(1, -1)
    u_scaled = artifacts.scaler.transform(x).reshape(-1)

    w = normalize_weights(weights)
    distances = weighted_euclidean_distances(artifacts.case_base_scaled, u_scaled, w)

    k_eff = int(min(k, distances.shape[0]))
    k_idx = np.argsort(distances)[:k_eff]
    k_labels = artifacts.case_base_labels[k_idx]
    k_dist = distances[k_idx]

    # Voting mayoritas (tie-break: pilih label dengan total jarak lebih kecil)
    counts = {0: int(np.sum(k_labels == 0)), 1: int(np.sum(k_labels == 1))}
    if counts[0] == counts[1]:
        sum_d0 = float(np.sum(k_dist[k_labels == 0])) if counts[0] else float("inf")
        sum_d1 = float(np.sum(k_dist[k_labels == 1])) if counts[1] else float("inf")
        pred = 0 if sum_d0 <= sum_d1 else 1
    else:
        pred = 1 if counts[1] > counts[0] else 0

    # Skor risiko sederhana: proporsi Outcome=1 di tetangga terdekat
    risk_score = float(np.mean(k_labels == 1))

    preview = pd.DataFrame(
        {
            "rank": np.arange(1, k_eff + 1),
            "distance": np.round(k_dist, 6),
            "Outcome": k_labels,
        }
    )

    return {
        "predicted_outcome": int(pred),
        "risk_score": risk_score,
        "k": k_eff,
        "neighbor_counts": counts,
        "nearest_cases_preview": preview,
    }

