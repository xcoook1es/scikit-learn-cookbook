# scikit-learn Cookbook

> Reproduksi kode + ringkasan + penjelasan teori dari buku **_scikit-learn Cookbook, Third Edition_** — John Sukup (Packt Publishing, 2025)

Repositori ini berisi satu Jupyter Notebook untuk setiap bab buku. Setiap notebook memuat:

1. **Ringkasan bab** — gambaran topik dan resep (*recipes*) yang dibahas.
2. **Reproduksi kode** — kode dari setiap resep dan latihan (*exercise*) di buku, dijalankan ulang lengkap dengan output.
3. **Penjelasan teori** — konsep matematis di balik setiap metode (rumus, intuisi, asumsi, kelebihan/kekurangan).
4. **Interpretasi hasil** — pembahasan output yang benar-benar dihasilkan saat notebook dijalankan, ditambah sel verifikasi kecil (misalnya membuktikan rumus OLS atau PCA secara numerik).

Bahasa penjelasan: **Bahasa Indonesia**. Kode mengikuti repositori resmi buku: [PacktPublishing/Scikit-Learn-Cookbook-Third-Edition](https://github.com/PacktPublishing/Scikit-Learn-Cookbook-Third-Edition).

---

## Daftar Isi

| Bab | Notebook | Topik Utama |
|----|----------|-------------|
| 01 | [Common Conventions and API Elements](Chapter_01_Common_Conventions_and_API_Elements/Chapter_01_Common_Conventions_and_API_Elements.ipynb) | Estimator, predictor, transformer, `fit/predict/transform`, hyperparameter vs parameter terlatih, Pipeline, cross-validation |
| 02 | [Pre-Model Workflow and Data Preprocessing](Chapter_02_Pre-Model_Workflow_and_Data_Preprocessing/Chapter_02_Pre-Model_Workflow_and_Data_Preprocessing.ipynb) | Imputasi (Simple/KNN/Iterative), scaling, encoding kategorikal, `ColumnTransformer`, Pipeline, `PolynomialFeatures`, `KBinsDiscretizer`, RFE, `SelectFromModel` |
| 03 | [Dimensionality Reduction Techniques](Chapter_03_Dimensionality_Reduction_Techniques/Chapter_03_Dimensionality_Reduction_Techniques.ipynb) | PCA, LDA, t-SNE |
| 04 | [Distance Metrics and Nearest Neighbors](Chapter_04_Distance_Metrics_and_Nearest_Neighbors/Chapter_04_Distance_Metrics_and_Nearest_Neighbors.ipynb) | KNN, metrik jarak (Euclidean/Manhattan/Minkowski), `GridSearchCV`, learning curve, confusion matrix |
| 05 | [Linear Models and Regularization](Chapter_05_Linear_Models_and_Regularization/Chapter_05_Linear_Models_and_Regularization.ipynb) | OLS, multikolinearitas, Ridge, Lasso, ElasticNet, regresi polinomial, spline |
| 06 | [Advanced Logistic Regression and Extensions](Chapter_06_Advanced_Logistic_Regression_and_Extensions/Chapter_06_Advanced_Logistic_Regression_and_Extensions.ipynb) | Regresi logistik, OvR vs multinomial (softmax), regularisasi L1/L2, multilabel, ROC–AUC |
| 07 | [Support Vector Machines and Kernel Methods](Chapter_07_Support_Vector_Machines_and_Kernel_Methods/Chapter_07_Support_Vector_Machines_and_Kernel_Methods.ipynb) | SVC/SVR, margin & support vector, kernel trick, tuning `C`/`gamma`, data berdimensi tinggi |
| 08 | [Tree-Based Algorithms and Ensemble Methods](Chapter_08_Tree-Based_Algorithms_and_Ensemble_Methods/Chapter_08_Tree-Based_Algorithms_and_Ensemble_Methods.ipynb) | Decision tree, Random Forest (bagging), Gradient Boosting, stacking, feature importance |

---

## Ringkasan Setiap Bab

### Bab 1 — Common Conventions and API Elements of scikit-learn
Fondasi seluruh buku: semua objek scikit-learn mengikuti kontrak yang sama. **Estimator** belajar lewat `fit()`, **predictor** menghasilkan `predict()`/`predict_proba()`/`score()`, dan **transformer** mengubah data lewat `transform()`/`fit_transform()`. Hyperparameter diatur lewat konstruktor atau `set_params()`, sedangkan hasil belajar disimpan di atribut berakhiran `_` (misalnya `coef_`). Bab ini juga memperkenalkan format data `X (n_samples, n_features)`, dataset bawaan, `Pipeline`, dan cross-validation, serta aturan penting: `fit` hanya pada data latih untuk mencegah *data leakage*.

### Bab 2 — Pre-Model Workflow and Data Preprocessing
Langkah-langkah sebelum pemodelan. **Data hilang** ditangani dengan `SimpleImputer` (mean/median/modus), `KNNImputer` (rata-rata tetangga terdekat), atau `IterativeImputer` (regresi berulang ala MICE). **Scaling** dengan `StandardScaler` (z-score), `MinMaxScaler` ([0,1]), dan `Normalizer` (norma per sampel). **Variabel kategorikal** diubah dengan `OneHotEncoder`/`LabelEncoder`, dan data campuran ditangani `ColumnTransformer`. Bab ini ditutup dengan **Pipeline** serta *feature engineering* (`PolynomialFeatures`, `KBinsDiscretizer`) dan seleksi fitur (`RFE`, `SelectFromModel`), termasuk satu contoh nyata *target leakage*.

### Bab 3 — Dimensionality Reduction Techniques
Mengatasi *curse of dimensionality*. **PCA** (tanpa label) mencari arah varians maksimum melalui vektor eigen matriks kovarians; **LDA** (dengan label) memaksimalkan rasio sebaran antar-kelas terhadap dalam-kelas (kriteria Fisher) dan menghasilkan maksimal C − 1 komponen; **t-SNE** adalah metode non-linear yang mempertahankan ketetanggaan lokal untuk visualisasi. Latihan menunjukkan PCA memangkas 64 fitur Digits menjadi 39 dengan penurunan akurasi kecil.

### Bab 4 — Building Models with Distance Metrics and Nearest Neighbors
**k-Nearest Neighbors**: model non-parametrik "lazy" yang memprediksi lewat voting/rata-rata k tetangga terdekat. Dibahas peran **k** dalam bias–variance trade-off, perbedaan **metrik jarak** (Euclidean, Manhattan, Minkowski), tuning dengan `GridSearchCV`, serta evaluasi dengan cross-validation, learning curve, confusion matrix, dan precision/recall/F1. Latihan pada Breast Cancer menunjukkan pentingnya scaling — dan bahwa satu split tunggal bisa memberi kesimpulan yang menyesatkan.

### Bab 5 — Linear Models and Regularization
Regresi linear (OLS) dan masalah **multikolinearitas** yang membuat koefisien tidak stabil. **Ridge** (L2) mengecilkan koefisien, **Lasso** (L1) menolkan sebagian koefisien (seleksi fitur), dan **ElasticNet** menggabungkan keduanya; *coefficient path* memperlihatkan efek kekuatan regularisasi. Bagian akhir memperluas model linear dengan fungsi basis non-linear: **polinomial** (global) dan **spline** (lokal, jauh lebih stabil).

### Bab 6 — Advanced Logistic Regression and Extensions
**Regresi logistik**: sigmoid, log-odds, dan pelatihan dengan log-loss (MLE). Perluasan ke **multikelas** dengan One-vs-Rest dan multinomial (softmax), **regularisasi** L1/L2 dengan parameter `C` (kebalikan kekuatan penalti), **klasifikasi multilabel** (satu model per label, subset accuracy vs Hamming loss, ko-okurensi label), dan **metrik evaluasi** lengkap termasuk kurva ROC dan AUC.

### Bab 7 — Support Vector Machines and Kernel Methods
**SVM** mencari hyperplane dengan margin terlebar; *soft margin* dengan parameter `C` mengizinkan pelanggaran. Model hanya bergantung pada **support vectors**. **Kernel trick** (linear, polinomial, RBF) menghasilkan batas keputusan non-linear lewat formulasi dual. Dibahas juga SVR (ε-insensitive loss), tuning `C`/`kernel`/`degree`, SVM pada data berdimensi tinggi (p > n), Platt scaling untuk probabilitas, dan evaluasi ROC.

### Bab 8 — Tree-Based Algorithms and Ensemble Methods
**Pohon keputusan** (CART, Gini/entropi) mudah diinterpretasikan tetapi rawan overfitting. **Random Forest** (bagging + subset fitur acak) menurunkan varians; disertai OOB score dan feature importance (MDI vs permutation). **Gradient Boosting** membangun pohon dangkal secara berurutan mengikuti gradien loss untuk menurunkan bias. Bab ditutup dengan tuning hyperparameter dan perbandingan bagging vs boosting vs **stacking**.

---

## Struktur Repositori

```
scikit-learn-Cookbook/
├── README.md
├── requirements.txt
├── Chapter_01_Common_Conventions_and_API_Elements/
├── Chapter_02_Pre-Model_Workflow_and_Data_Preprocessing/
├── Chapter_03_Dimensionality_Reduction_Techniques/
├── Chapter_04_Distance_Metrics_and_Nearest_Neighbors/
├── Chapter_05_Linear_Models_and_Regularization/
├── Chapter_06_Advanced_Logistic_Regression_and_Extensions/
├── Chapter_07_Support_Vector_Machines_and_Kernel_Methods/
└── Chapter_08_Tree-Based_Algorithms_and_Ensemble_Methods/
```

## Cara Menjalankan

```bash
git clone https://github.com/<username>/scikit-learn-Cookbook.git
cd scikit-learn-Cookbook
pip install -r requirements.txt
jupyter notebook
```

Atau buka notebook langsung di Google Colab (*File → Open notebook → GitHub*).

## Catatan Reproduksi

- **Bab 1:** repositori resmi buku tidak menyediakan notebook untuk Bab 1 (bab konseptual). Kode di notebook ini disusun mengikuti topik-topik Bab 1.
- **Bab 2 (latihan):** `fetch_california_housing()` mengunduh dataset dari internet saat pertama kali dijalankan, jadi butuh koneksi internet.
- **Bab 6:** di scikit-learn versi terbaru, solver `liblinear` tidak lagi mendukung multikelas secara langsung. Model One-vs-Rest dibungkus dengan `OneVsRestClassifier` (perilakunya setara dengan kode buku pada scikit-learn 1.5).
- **Bab 7:** pada resep evaluasi SVM, kode buku mencetak variabel `accuracy` dari resep sebelumnya; akurasi yang benar dihitung ulang di sel tambahan.
- Beberapa kode buku memakai bilangan acak tanpa seed (`make_circles` di Bab 4 dan noise di Bab 5). Agar hasil konsisten dengan interpretasi yang ditulis, seed global NumPy diset di sel tambahan sebelum kode tersebut (kode buku tidak diubah).
- Penjelasan teori disusun dengan bantuan LLM (diperbolehkan oleh ketentuan tugas), lalu dicek terhadap output notebook.

## Lingkungan

Notebook dijalankan dengan Python 3.13, scikit-learn 1.9, NumPy 2.5, pandas 3.0, matplotlib, dan seaborn. Kode kompatibel dengan scikit-learn ≥ 1.5.

## Referensi

- Sukup, J. (2025). *scikit-learn Cookbook* (3rd ed.). Packt Publishing.
- Dokumentasi scikit-learn: https://scikit-learn.org/stable/
- Kode resmi buku: https://github.com/PacktPublishing/Scikit-Learn-Cookbook-Third-Edition
