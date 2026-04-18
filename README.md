# Eksperimen_SML_Lufthi-Arief-Syabana

Repositori ini berisi eksperimen machine learning untuk submission kelas **Sistem Machine Learning** — Dicoding.

## 👤 Identitas
- **Nama:** Lufthi Arief Syabana
- **Dataset:** Smartphone Usage and Addiction Analysis
- **Sumber:** [Kaggle - nalisha](https://www.kaggle.com/datasets/nalisha/smartphone-usage-and-addiction-analysis-dataset)
- **Task:** Binary Classification — Prediksi `addicted_label` (0 = Tidak Adiksi, 1 = Adiksi)

## 📁 Struktur Repository

```
Eksperimen_SML_Lufthi-Arief-Syabana/
├── .github/
│   └── workflows/
│       └── preprocessing.yml          # GitHub Actions workflow (Advance)
├── smartphone_usage_raw.csv           # Dataset mentah
└── preprocessing/
    ├── Eksperimen_Lufthi-Arief-Syabana.ipynb       # Notebook eksperimen lengkap
    ├── automate_Lufthi-Arief-Syabana.py            # Script otomatisasi preprocessing
    └── smartphone_usage_preprocessing/             # Output preprocessing
        ├── smartphone_usage_preprocessing.csv      # Dataset lengkap setelah preprocessing
        ├── train.csv                               # Data training (80%)
        └── test.csv                                # Data testing (20%)
```

## 🔄 Langkah-langkah Preprocessing

| No | Langkah | Detail |
|---|---|---|
| 1 | Drop kolom tidak relevan | `transaction_id`, `user_id` |
| 2 | Penanganan missing values | `addiction_level` → imputasi dengan modus |
| 3 | Label Encoding (ordinal) | `stress_level`, `addiction_level` |
| 4 | Label Encoding (binary) | `academic_work_impact` |
| 5 | One-Hot Encoding | `gender` → 3 kolom |
| 6 | Normalisasi | `StandardScaler` pada 9 kolom numerik |
| 7 | Train-Test Split | 80:20 stratified |

## ▶️ Cara Menjalankan

### Notebook Eksperimen
Buka dan jalankan `preprocessing/Eksperimen_Lufthi-Arief-Syabana.ipynb` di Jupyter/Colab.

### Script Otomatisasi
```bash
cd preprocessing
python automate_Lufthi-Arief-Syabana.py

# Dengan argumen custom
python automate_Lufthi-Arief-Syabana.py \
  --input ../smartphone_usage_raw.csv \
  --output smartphone_usage_preprocessing
```

## ⚙️ GitHub Actions (Advance)
Workflow otomatis akan terpantik ketika:
- Push ke branch `main` yang mengubah file dataset atau script preprocessing
- Jadwal mingguan setiap Senin pukul 07:00 UTC
- Trigger manual via GitHub UI (`workflow_dispatch`)

## 📦 Requirements
```
pandas==2.2.2
numpy==1.26.4
scikit-learn==1.5.1
matplotlib
seaborn
```
