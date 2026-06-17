# Sistem Estimasi Risiko Diabetes Melitus Tipe 2
**Case-Based Reasoning (CBR) + MultiSURF Feature Weighting**

## Tentang
Aplikasi pendukung klinis berbasis Streamlit untuk estimasi risiko DM Tipe 2 pada pasien oleh tenaga kesehatan.

- **Metode:** Case-Based Reasoning (CBR)
- **Pembobotan fitur:** MultiSURF (rata-rata 10-Fold CV)
- **Konfigurasi terbaik:** K = 9 tetangga
- **Akurasi (10-Fold CV):** 76,03% | Precision: 65,87% | Recall: 64,54% | F1: 64,95%

## Instalasi

```bash
pip install -r requirements.txt
streamlit run app.py
```

## Struktur Proyek

```
├── app.py                  # Halaman utama
├── pages/
│   ├── 1_🧪_Deteksi.py     # Form input & hasil estimasi (Retrieve, Reuse, Revise, Retain)
│   ├── 2_📚_Basis_Kasus.py # Manajemen basis kasus
│   └── 3_ℹ️_Tentang.py     # Informasi sistem
├── cbr/
│   ├── engine.py           # Algoritma CBR (normalisasi, WED, voting)
│   └── weights_config.py   # Bobot MultiSURF final (K=9)
├── ui/
│   └── theme.py            # CSS & komponen UI
├── data/
│   └── case_base.csv       # Basis kasus (dibuat otomatis)
└── requirements.txt
```

## Alur CBR

1. **Retrieve** — Hitung Weighted Euclidean Distance ke seluruh basis kasus
2. **Reuse** — Voting mayoritas dari K=9 tetangga terdekat
3. **Revise** — Validasi klinis oleh tenaga kesehatan
4. **Retain** — Simpan kasus baru ke basis kasus
