# scikit-learn Cookbook

> Reproduksi kode + ringkasan + penjelasan teori dari buku **_scikit-learn Cookbook, Third Edition_** — John Sukup (Packt Publishing, 2025)

Repositori ini berisi satu Jupyter Notebook untuk setiap bab buku. Setiap notebook memuat:

1. **Ringkasan bab** — gambaran topik dan resep (*recipes*) yang dibahas.
2. **Reproduksi kode** — kode dari setiap resep dan latihan (*exercise*) di buku, dijalankan ulang lengkap dengan output.
3. **Penjelasan teori** — konsep matematis di balik setiap metode (rumus, intuisi, asumsi, kelebihan/kekurangan).
4. **Interpretasi hasil** — pembahasan output yang benar-benar dihasilkan saat notebook dijalankan, ditambah sel verifikasi kecil (misalnya membuktikan rumus OLS atau PCA secara numerik).

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
| 09 | [Text Processing and Multiclass Classification](Chapter_09_Text_Processing_and_Multiclass_Classification/Chapter_09_Text_Processing_and_Multiclass_Classification.ipynb) | Tokenisasi, Bag of Words, TF-IDF, n-gram, POS tagging, Naive Bayes, OvR vs OvO, evaluasi model teks |
| 10 | [Clustering Techniques](Chapter_10_Clustering_Techniques/Chapter_10_Clustering_Techniques.ipynb) | K-Means, elbow method, agglomerative & dendrogram, DBSCAN, silhouette/Davies–Bouldin/ARI, Spectral Clustering, GMM |
| 11 | [Novelty and Outlier Detection](Chapter_11_Novelty_and_Outlier_Detection/Chapter_11_Novelty_and_Outlier_Detection.ipynb) | Isolation Forest, Local Outlier Factor, One-Class SVM, evaluasi deteksi anomali, penanganan outlier |
| 12 | [Cross-Validation and Model Evaluation Techniques](Chapter_12_Cross-Validation_and_Model_Evaluation_Techniques/Chapter_12_Cross-Validation_and_Model_Evaluation_Techniques.ipynb) | K-fold, stratified, LOOCV, nested CV, `cross_validate`, Grid vs Randomized search, learning & validation curve |
| 13 | [Deploying scikit-learn Models in Production](Chapter_13_Deploying_scikit-learn_Models_in_Production/Chapter_13_Deploying_scikit-learn_Models_in_Production.ipynb) | Serialisasi (pickle/joblib), batch prediction, monitoring & drift, `partial_fit`, versi model & metadata, gerbang CI/CD |

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

### Bab 9 — Text Processing and Multiclass Classification
Mengubah teks menjadi angka: tokenisasi, penghapusan stopword, dan **Bag of Words** (`CountVectorizer`) yang menghasilkan matriks dokumen–kata berdimensi tinggi dan sangat sparse. **TF-IDF** menurunkan bobot kata yang umum di banyak dokumen dan menormalkan panjang dokumen; *word embedding* dibahas sebagai pendekatan yang menangkap makna. Fitur tambahan berupa **n-gram**, **POS tagging**, dan NER. Klasifikasi teks dengan **Multinomial Naive Bayes**, SVM, dan regresi logistik (korpus NLTK Brown, Reuters, movie_reviews), strategi multikelas **One-vs-Rest vs One-vs-One**, serta evaluasi dengan precision/recall/F1 — termasuk eksperimen yang menunjukkan bahwa SVM linear jauh lebih cocok untuk teks daripada SVM RBF.

### Bab 10 — Clustering Techniques
*Unsupervised learning* untuk mengelompokkan data tanpa label. **K-Means** (algoritma Lloyd, k-means++, elbow method), **hierarchical/agglomerative clustering** (linkage Ward/single/complete/average, dendrogram), dan **DBSCAN** (cluster berbasis kepadatan, deteksi noise, k-distance plot). Evaluasi dengan **silhouette**, **Davies–Bouldin**, **Calinski–Harabasz**, dan **Adjusted Rand Index**. Bab ini juga membandingkan algoritma pada data berbentuk gumpalan, bulan sabit, dan lingkaran, lalu membahas **Spectral Clustering** (berbasis graf) dan **Gaussian Mixture Model** (EM, *soft clustering*, pemilihan komponen dengan BIC).

### Bab 11 — Novelty and Outlier Detection
Membedakan **outlier detection** (data latih sudah tercampur anomali) dan **novelty detection** (dilatih pada data normal saja). **Isolation Forest** mengisolasi anomali dengan pemisahan acak (jalur pendek = anomali), **Local Outlier Factor** membandingkan kepadatan lokal titik dengan tetangganya, dan **One-Class SVM** mempelajari batas wilayah normal dengan kernel. Bab ini juga membahas evaluasi deteksi anomali (precision/recall kelas outlier, ROC–AUC, PR-AUC), strategi menangani outlier (hapus, ganti median, winsorization, flagging), serta panduan memilih metode.

### Bab 12 — Cross-Validation and Model Evaluation Techniques
Estimasi performa yang jujur: **k-fold** dan **stratified k-fold**, **LOOCV**, dan **nested cross-validation** untuk tuning tanpa bias seleksi. API `cross_val_score`, `cross_validate` (banyak metrik sekaligus), **GridSearchCV** vs **RandomizedSearchCV** (dengan distribusi log-uniform), serta diagnosis bias–varians dengan **learning curve** dan **validation curve**.

### Bab 13 — Deploying scikit-learn Models in Production
Siklus hidup model setelah dilatih: **serialisasi** dengan `pickle`/`joblib` (beserta isu keamanan dan kompatibilitas versi), skalabilitas dengan `n_jobs` dan **prediksi batch**, **monitoring** penurunan performa akibat data/concept drift dan pembaruan inkremental dengan `partial_fit`, **versi model dan metadata**, serta **pipeline CI/CD** dengan gerbang kualitas otomatis. Notebook ini juga menandai beberapa kelemahan pada kode contoh buku (data validasi yang bocor dari data latih, metadata versi yang ditulis manual, dan data uji CI berupa angka acak) beserta versi perbaikannya.

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
├── Chapter_08_Tree-Based_Algorithms_and_Ensemble_Methods/
├── Chapter_09_Text_Processing_and_Multiclass_Classification/
├── Chapter_10_Clustering_Techniques/
├── Chapter_11_Novelty_and_Outlier_Detection/
├── Chapter_12_Cross-Validation_and_Model_Evaluation_Techniques/
└── Chapter_13_Deploying_scikit-learn_Models_in_Production/
```

## Cara Menjalankan

```bash
git clone https://github.com/xcoook1es/scikit-learn-cookbook.git
cd scikit-learn-cookbook
pip install -r requirements.txt
jupyter notebook
```

**Bab 9** memakai korpus NLTK. Unduh sekali sebelum menjalankan notebook:

```python
import nltk
for p in ['movie_reviews', 'reuters', 'brown', 'punkt', 'punkt_tab',
          'stopwords', 'averaged_perceptron_tagger_eng']:
    nltk.download(p)
```

Atau buka notebook langsung di Google Colab (*File → Open notebook → GitHub*).


## Lingkungan

Notebook dijalankan dengan Python 3.13, scikit-learn 1.9, NumPy 2.5, pandas 3.0, matplotlib, dan seaborn. Kode kompatibel dengan scikit-learn ≥ 1.5.

## Referensi

- Sukup, J. (2025). *scikit-learn Cookbook* (3rd ed.). Packt Publishing.
- Dokumentasi scikit-learn: https://scikit-learn.org/stable/
- Kode resmi buku: https://github.com/PacktPublishing/Scikit-Learn-Cookbook-Third-Edition
