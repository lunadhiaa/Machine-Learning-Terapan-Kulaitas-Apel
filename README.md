# Laporan Proyek Machine Learning - Lulu Nadhiatun Anisa

## Domain Proyek

Domain yang dipilih untuk proyek *machine learning* ini adalah **Pertanian** yang meninjau **Predictive Analytics: Kualitas Apel**  

### Latar Belakang

![Apple](https://github.com/user-attachments/assets/37ce2fed-7d9b-4604-aed0-c6caebb27c34)

Indonesia merupakan negara tropis yang memiliki potensi besar dalam bidang pertanian termasuk buah-buahan. Posisi Indonesia di garis khatulistiwa memberi keutungan geografis karena memungkinkan tumbuhnya beraneka ragam buah.Salah satu buah yang banyak dikonsumsi di Indonesia adalah apel[[1](https://sirisma.unisri.ac.id/berkas/4LAPORAN%20kemajuan_PENELITIAN_GANJAR_HASNA_FISIP2022.pdf)]. Produksi apel di Indonesia tercatat sebanyak 509.544 ton pada 2021. Jumlah tersebut turun 1,35% dibandingkan pada tahun sebelumnya yang sebesar 516.531 ton. Namun, produksi apel dalam negeri masih menghadapi berbagai tantangan, terutama terkait kualitas apel yang dihasilkan. Kualitas apel dapat menurun akibat berbagai faktor, seperti ukuran yang kecil, tingkat kematangan, dan kerenyahan buah. Penurunan kualitas apel dapat menyebabkan kerugian ekonomi bagi petani dan distributor[[2](https://dataindonesia.id/agribisnis-kehutanan/detail/produksi-apel-indonesia-sebanyak-509544-ton-pada-2021)]. Penerapan _predictive analytics_ dalam industri apel dapat memberikan manfaat bagi petani, distributor, dan konsumen. Petani dapat meningkatkan keuntungan dengan meningkatkan kualitas dan hasil panen apel. Distributor dapat mengurangi kerugian dan meningkatkan efisiensi rantai pasokan. Konsumen mendapatkan apel dengan kualitas yang lebih baik dan harga yang lebih stabi[[3](https://doi.org/10.47065/bulletincsr.v3i3.251)]

## Business Understanding
Pengembangan model prediksi untuk menilai kualitas apel memiliki banyak potensi manfaat bagi berbagai pihak, seperti petani, distributor, dan konsumen. Dengan adanya model ini, petani bisa lebih mudah dalam meningkatkan hasil panen mereka karena bisa memantau dan memprediksi kualitas apel secara lebih akurat. Hal ini tentu saja akan membantu petani dalam memilah apel berdasarkan kualitas, sehingga mereka dapat memastikan hanya apel berkualitas tinggi yang dipasarkan. Contoh potensi manfaat hasil prediksi kualitas apel yang akurat dapat membantu petani dalam melakukan pemilahan dan dapat menentukan harga jual buah kedepan untuk kedepannya.

### Problem Statements
Berdasarkan latar belakang yang telah dijelaskan, berikut adalah rincian masalah yang dapat diatasi melalui proyek ini:
- Bagaimana membuat model *machine learning* yang dapat memprediksi kualitas apel berdasarkan data visual dan sensorik?
- Model yang seperti apa yang memiliki akurasi paling baik?
- Bagaimana model dapat membantu petani dan distributor dalam meningkatkan kualitas dan nilai jual apel?

### Goals
Tujuan dari proyek ini adalah:
- Membuat model machine learning yang dapat memprediksi kualitas apel dengan baik melalui data visualisasi dan sensorik.
- Membandingkan algoritma model untuk menemukan akurasi terbaik dalam memprediksi kualitas apel.
- Mengembangkan aplikasi yang mudah digunakan untuk membantu petani dan distributor dalam menggunakan model machine learning untuk memprediksi kualitas apel.

### Solution Statements
- Menganalisis data dengan melakukan univariate analysis dan multivariate analysis untuk mengetahui kolerasi matrix antar fitur dan mendeteksi outlier. Serta Memahami data dengan visualisasi. 
- Melakukan proses data cleaning dan normalisai data agar mendapat prediksi yang baik.
- Membuat beberapa Model prediksi machine learning untuk mendapatkan model yang paling baik dari beberapa model yang telah dibuat untuk prediksi kualitas apel. Diantaranya adalah menggunakan:
    * Support Vector Machine (SVM) adalah algoritma yang digunakan untuk menemukan hyperplane dalam ruang N-dimensi (N - jumlah fitur) yang secara jelas mengklasifikasikan titik data. SVM dapat digunakan untuk menyelesaikan permasalahan klasifikasi, regresi, dan pendeteksian outlier.[[4](https://towardsdatascience.com/support-vector-machine-introduction-to-machine-learning-algorithms-934a444fca47)]
    * Random Forest adalah algoritma machine learning yang kuat yang dapat digunakan untuk berbagai tugas termasuk regresi dan klasifikasi. Ini adalah metode ensemble, yang berarti bahwa model random forest terdiri dari banyak decision tree kecil, yang disebut estimator, yang masing-masing menghasilkan prediksi mereka sendiri. Random forest menggabungkan prediksi estimator untuk menghasilkan prediksi yang lebih akurat .[[5](https://deepai.org/machine-learning-glossary-and-terms/random-forest)]

## Data Understanding
### EDA - Deskripsi Variabel
**Informasi Datasets**

| Jenis | Keterangan |
| ------ | ------ |
| Title | _Apple Quality_ |
| Source | [Kaggle](https://www.kaggle.com/datasets/nelgiriyewithana/apple-quality/data) |
| Maintainer | [Nidula Elgiriyewithana ⚡](https://www.kaggle.com/nelgiriyewithana) |
| License | Other (specified in description) |
| Visibility | Publik |
| Tags | _Computer Science, Education, Food, Data Visualization, Classification, Exploratory Data Analysis_ |
| Usability | 10.00 |

Berikut informasi pada dataset: 
Data yang digunakan dalam pembuatan model merupakan data primer, data ini didapat dari sebuah perusahaan pertanian Amerika, yang disediakan secara publik di kaggle dengan nama datasets yaitu: _Apple Quality_

| A_id | Size | Weight | Sweetness | Crunchiness | Juiciness | Ripeness | Acidity | Quality |
| ------ | ------ |------ | ------ | ------ | ------ |------ | ------ |------ |
| 0.0 | -3.970049 |-2.512336 | 5.346330 |-1.012009 | 1.844900 |0.329840	| -0.491590483  |good |
| 1.0 | -1.195217 |-2.839257 | 3.664059 |1.588232 | 0.853286 | 0.867530 | -0.722809367  |good |
| 2.0 | -0.292024 |	-1.351282 | -1.738429 | -0.342616 | 2.838636 |-0.038033	| 2.621636473  |bad |
| 3.0 | -0.657196 |-2.271627 | 1.324874 |-0.097875 | 3.637970 |-3.413761	| 0.790723217  |good |
| 4.0 | 1.364217 |-1.296612 | -0.384658 | -0.553006 | 3.030874 | -1.303849	| 0.501984036  |good |

Tabel 1. EDA Deskripsi Variabel_

Dilihat dari _Tabel 2. EDA Deskripsi Variabel_ dataset ini telah di *bersihkan* dan *normalisasi* terlebih dahulu oleh pembuat, sehingga mudah digunakan dan ramah bagi pemula. 
- Dataset berupa CSV (Comma-Seperated Values).
- Dataset memiliki 4001 sample dengan 9 fitur.
- Dataset memiliki 7 fitur bertipe float64 dan 2 fitur bertipe object.
- Terdapat 1 missing value dalam dataset.

### Variabel-variabel dataset adalah sebagai berikut:
- `A_id` : Identifikasi unik untuk setiap buah.
- `Size` : Ukuran buah.
- `Weight` : Berat buah.
- `Sweetness` : Tingkat kemanisan buah.
- `Crunchiness` : Tekstur yang menunjukkan kerenyahan buah.
- `Juiciness` : Tingkat kesegaran buah.
- `Ripeness` : Tahap kematangan buah.
- `Acidity` : Tingkat keasaman buah.
- `Quality` : Kualitas buah secara keseluruhan, baik atau buruk.

Dari ke 9 fitur dapat dilihat bahwa fitur `A_id` tidak mempengaruhi kualitas buah hingga akan di hapus.

### EDA - Univariate Analysis

![univariate analysis 1a](https://github.com/user-attachments/assets/e9330311-aac9-46f8-8080-a69ea2f4872b)

Gambar 1a. Analisis Univariat (Data Kategori Apel) 

![univariate analysis 1b](https://github.com/user-attachments/assets/e9543576-081f-447e-ab7d-d4d4970b3045)

Gambar 1b. Analisis Univariat (Data Numerik) 

 Berdasarkan _Gambar 1a_ , dapat dilihat bahwa distribusi data katagorik _Quality_ yang terdiri dari _good_ dan _bad_ kualitas apel, yang mana nilai data **bad** terdiri dari `1928` dan **good** terdiri dari `1862`, yang mana menunjukan perbandingan data yang tidak terlalu jauh. Pada _Gambar 1b,_ untuk data numerik memiliki karakteristik, yaitu:
  - Dilihat dari distribusi data numerik _Size_, ukuran rata-rata buah berkisar dari -2 sampai 2, dan memiliki nilai rata-rata _Mean_ adalah -0.51.
  - Rata-rata berat apel bernilai -0.99 dan nilai _max_ berat apel adalah 3.08.
  - Rata-rata tingkat kemanisan apel -0.48.
  - Tekstur kerenyahan apel berkisar dari 0 hingga 2 yang mana nilai ini menunjukan rata-rata apel itu renyah.
  - Tingkat kesegaran buah dan Kematangan buat berada pada nilai 0.50 dan 0.53.
  - Rata-rata tingkat keasaman buah bernilai 0.06.

 Nilai-nilai ini menunjukkan bahwa data  telah dinormalisasi dengan cara _z-score normalization_ . _z-score normalization_  mengubah data dengan cara:
 - Mengurangi rata-rata (mean) dari setiap data point.
 - Membagi hasil pengurangan tersebut dengan standar deviasi data.
 
Pada kasus ini, rata-rata (mean) data "Size" adalah -0.51 dan standar deviasi data "Size" tidak diketahui. Namun, dengan nilai minimum -2 dan maksimum 2, dapat diasumsikan bahwa data "Size" telah diubah skalanya sehingga memiliki mean 0 dan standar deviasi 1. Data numerik lainnya, seperti _"Weight", "Sweetness", "Crunchiness", "Juiciness", "Ripeness", dan "Acidity"_, juga telah dinormalisasi dengan cara yang sama.


### EDA - Multivariate Analysis

![multi1](https://github.com/user-attachments/assets/d926d5c4-8dfd-4a11-acf8-5082ebdf9340)

Gambar 2a. Analisis Multivariat

![multi2](https://github.com/user-attachments/assets/f615c6c9-3e5d-4f09-9be9-42bb4c1184a4)

Gambar 2b. Analisis Karakteristik apel

![multi3](https://github.com/user-attachments/assets/b3bd07ae-426a-4984-8776-42718fc7c2f3)

Gambar 2c. Analisis Matriks Korelasi

Pada _Gambar 2a. Analisis Multivariat_, dengan menggunakan fungsi _pairplot_ dari _library seaborn_, tampak terlihat relasi pasangan dalam dataset menunjukan pola acak. Pada pola sebaran data grafik pairplot, terterlihat bahwa _Size_ dan _Sweetness_ memiliki korelasi negatif menurun, yang mana semakin kecil ukuran buah rasa nya akan semakin manis.

Pada _Gambar 2b. Analisis Karakteristik_ menggunakan box plot dengan membandingkan distribusi berbagai karakteristik apel antara apel yang diklasifikasikan dalam kualitas *Good* dan *Bad*. Sn Dengan sebaran data untuk berbagai fitur seperti *size, weight, sweetness, crunchiness, juiciness, ripeness, and acidity*.

Pada _Gambar 2c. Analisis Matriks Korelasi_, merupakan _Correlation Matrix_ menunjukkan hubungan antar fitur dalam nilai korelasi. Jika diamati, fitur _Juiciness_ memiliki skor korelasi yang cukup besar `0.24` dengan fitur target _Acidity_ .

## Data Preparation
Pada proses _Data Preparation_ dilakukan kegiatan seperti _Data Gathering_, _Data Assessing_, dan _Data Cleaning_. Pada proses Data Gathering, data diimpor sedemikian rupa agar bisa dibaca dengan baik menggunakan dataframe Pandas. Untuk proses Data Assessing, berikut adalah beberapa pengecekan yang dilakukan:

- Duplicate data (data yang serupa dengan data lainnya).
- Missing value (data atau informasi yang "hilang" atau tidak tersedia)
- Outlier (data yang menyimpang dari rata-rata sekumpulan data yang ada).

Pada proses _Data Cleaning_ yang dilakukan adalah seperti:
- Converting Column Type (Mengubah tipe suatu kolom).
- Train Test Split (membagi data menjadi data latih dan data uji).
- Normalization (mentransformasi data ke dalam skala yang seragam sehingga semua fitur atau atribut memiliki rentang nilai yang sebanding).

| A_id | Size | Weight | Sweetness | Crunchiness | Juiciness | Ripeness | Acidity | Quality |
| ------ | ------ |------ | ------ | ------ | ------ |------ | ------ |------ |
| NaN | NaN | NaN | NaN |NaN | NaN| NaN	| Created_by_Nidula_Elgiriyewithana  | NaN |

Tabel 2. Melihat data missing value

Pada proyek kasus ini tidak ditemukannya data duplikat, tetapi ditemukannya _missing value_. Adapaun metode yang digunakan untuk mengatasi hal ini adalah dengan menerapkan _Dropping_ yaitu menghapus data yang _missing_ digunakannya metode ini dikarenakan jumlah missing value hanya berjumlah `1`. Lihat _Tabel 2. Melihat data missing value_. Adapun untuk _outlier_ juga dilakukan dengan metode _dropping_ menggunakan metode IQR.  IQR dihitung dengan mengurangkan kuartil ketiga (Q3) dari kuartil pertama (Q1) sebagaimana rumus berikut.

$$IQR = Q_3 - Q_1$$

- Q1 adalah kuartil pertama 
- Q3 adalah kuartil ketiga.

Setelah menggunakan metode IQR untuk menghilangkan _outlier_ pada dataset jumlah dataset menjadi `3790` yang awalnya adalah `4000`.
Pada proyek ini digunakan _Train Test Split_ pada library  *sklearn.model_selection* untuk membagi dataset menjadi data latih dan data uji dengan pembagian sebesar 20:80 dan random state sebesar 60. Pada proyek kasus ini digunakan _Normalization_ pada library _sklearn.preprocessing.MinMaxScaler_ untuk menormalisasi dataset. Semua proses ini diperlukan dalam rangka membuat model yang baik.

## Modeling
Algoritma pada proyek ini melakukan pemodelan dengan 2 algoritma, yaitu:

_Support Vector Machine (SVM)_ adalah algoritma machine learning yang digunakan untuk klasifikasi dan regresi. Algoritma ini bekerja dengan mencari hyperplane yang memisahkan data menjadi dua kelas dengan margin terbesar. Parameter yang digunakan pada SVM kali ini adalah parameter bawaan.
 
 Keuntungan  _Support Vector Machine (SVM)_ :
- Memiliki akurasi prediksi yang tinggi.
- Mampu menangani dataset dengan dimensi tinggi.
- Tidak sensitif terhadap outlier.
- Dapat digunakan untuk klasifikasi dan regresi.

Kerugian  _Support Vector Machine (SVM)_ :
- Sulit untuk memilih kernel dan parameter lainnya. 
- Sensitif terhadap outlier. 
- Membutuhkan banyak waktu komputasi untuk pelatihan.

_Random Forest_ adalah algoritma machine learning ensemble yang menggabungkan beberapa decision tree untuk meningkatkan akurasi prediksi. Algoritma ini bekerja dengan membuat banyak decision tree secara acak dan kemudian menggunakan voting untuk memprediksi kategori atau nilai data baru. Adapun parameter yang digunakan pada proyek ini adalah:
- `max_depth` kedalaman maksimum.

Keunggulan _Random Forest_ :
- Memiliki akurasi prediksi yang tinggi.
- Mampu menangani dataset dengan dimensi tinggi.
- Tidak sensitif terhadap outlier.

Kerugian _Random Forest_ :
- Cenderung overfit pada dataset kecil. 
- Membutuhkan banyak waktu komputasi untuk pelatihan. 
- Sulit untuk diinterpretasikan.

Parameter yang digunakan adalah:
- `kernel` memetakan data input ke ruang dimensi yang lebih tinggi sehingga memungkinkan pemisahan data yang lebih baik.
- `n_estimators` Jumlah pohon keputusan yang akan dibuat dalam ensemble.
- `random_stat` pengambilan sampel secara acak.

## Evaluation
Dalam tahap evaluasi, metrik yang digunakan adalah `accuracy`
Accuracy didapatkan dengan menghitung persentase dari jumlah prediksi yang benar dibagi dengan jumlah seluruh prediksi. Rumus:

$$\text{Accuracy} = \frac{\text{TP + TN}}{\text{TN + TP + FN + FP}} \times 100\%$$

*Penjelasan*
- TP (True Positive): Jumlah data positif yang diprediksi dengan benar sebagai positif.
- TN (True Negative): Jumlah data negatif yang diprediksi dengan benar sebagai negatif.
- FP (False Positive): Jumlah data negatif yang diprediksi secara tidak benar sebagai positif (Kesalahan Tipe I).
- FN (False Negative): Jumlah data positif yang diprediksi secara tidak benar sebagai negatif (Kesalahan Tipe II).

Rumus ini memecah akurasi menjadi rasio antara data yang diklasifikasikan dengan benar (TP dan TN) dengan jumlah total data. Mengalikan dengan 100% mengubah rasio menjadi persentase.

Berikut hasil accuracy model yang latih:

| Model | Accuracy |
| ------ | ------ |
| SVM | 0.77 |
| RandomForest  | 0.89 |

Tabel 4. Hasil Accuracy

![eval1](https://github.com/user-attachments/assets/fa012554-1ec6-43fd-b6f9-8f878eb66dad)

3a. Model Evaluation SVM

![eval2](https://github.com/user-attachments/assets/ef4dcf05-6331-4bb7-8b0a-101b07d7754b)

3b. Model Evaluation Random Forest

Dilihat dari _Tabel 4. Hasil Accuracy_ dan _Gambar 3a. Model Evaluatin SVM_ dapat diketahui bahwa hasil pemodelan menggunakan algoritma SVM menghasilkan akurasi 77% dengan hasil Confusion Matrix  model yang memiliki total 385 prediksi benar (290 untuk "Bad" dan 295 untuk "Good"), dan 173 prediksi salah (83 False Positive dan 90 False Negative). Juga pada hasil ROC Curve menunjukkan bahwa model memiliki performa yang cukup baik dengan nilai AUC 0.85, yang berarti model memiliki kemampuan yang baik dalam membedakan antara kelas "Good" dan "Bad". Secara keseluruhan, model SVM ini menunjukkan performa yang cukup baik, meskipun masih terdapat beberapa kesalahan dalam prediksi kedua kelas.

Sedangkan, dilihat dari  _Tabel 4. Hasil Accuracy_ dan _Gambar 3b. Model Evaluatin Random Forest_ menghasilkan model dengan akurasi lebih tinggi yaitu 89%. Dari Confusion Matrix, model berhasil membuat 677 prediksi yang benar (341 untuk kelas "Bad" dan 336 untuk kelas "Good"), dan 81 prediksi salah (32 False Positive dan 49 False Negative). Pada ROC Curve menunjukkan performa yang sangat baik dengan nilai AUC sebesar 0.96, yang berarti model ini memiliki kemampuan yang sangat kuat dalam membedakan kelas "Good" dan "Bad". Secara keseluruhan, model Random Forest menunjukkan performa yang sangat baik dengan sedikit kesalahan prediksi, dan kemampuannya untuk memisahkan kelas ditunjukkan oleh AUC yang tinggi. Model ini lebih baik dibandingkan dengan SVM berdasarkan evaluasi ROC dan Confusion Matrix.

Maka dari itu, algoritma Random Forest memiliki Accuracy yang lebih tinggi dengan accuracy 89%. Untuk itu model tersebut yang akan dipilih untuk digunakan. Diharapkan dengan model yang telah dikembangan dapat memprediksi kualitas apel dengan baik menggunakan Random Forest. Alasan mengapa metode Random Forest yang dipilih karena lebih tahan terhadap overfitting, lebih stabil pada data yang kompleks, lebih robust terhadap outliers dan missing data, serta menawarkan interpretasi yang jelas melalui feature importance. Model ini juga cenderung lebih mudah digunakan dan memberikan hasil yang baik tanpa perlu tuning yang ekstensif.

## Referensi
1. Sarnita Sadya.(2022). Produksi Apel Indonesia Sebanyak 509.544 Ton pada 2021.
2. Lomo, Christine P., et al. "Daya Terima Panelist terhadap Kualitas Cider Apel dalam Meningkatkan Nilai Gizi Pangan sebagai Imunitas Tubuh di Pandemi Covid-19." Agrista: Jurnal Ilmiah Mahasiswa Agribisnis UNS, vol. 4, no. 1, 2020, pp. 550-556
3. Wood, T. -.What is a Random Forest?. DeepAI. https://deepai.org/machine-learning-glossary-and-terms/random-forest
4. Gandhi, R. (2018). Support Vector Machine — Introduction to Machine Learning Algorithms: SVM model from scratch. Towards Data Science. https://towardsdatascience.com/support-vector-machine-introduction-to-machine-learning-algorithms-934a444fca47
