# Customer Review Sentiment Classification with Transformers

Project ini berisi pipeline lengkap untuk melakukan klasifikasi sentimen (positif/negatif) pada ulasan pelanggan menggunakan model transformer pre-trained DistilBERT yang sudah di-finetune.

Project sederhana ini hanya untuk latihan

---

## Struktur Repo

- `setup.py`  
  Script untuk mendownload dan menyimpan model tokenizer dan model pretrained `distilbert-base-uncased-finetuned-sst-2-english` secara lokal.

- `train.py`  
  Script untuk melatih ulang (fine-tune) model menggunakan dataset custom (`data.csv`) dengan label sentimen 0 (negatif) dan 1 (positif).

- `test.py`  
  Script untuk menguji model hasil training pada contoh kalimat dan menampilkan prediksi sentimen.

- `app.py`  
  Aplikasi web sederhana menggunakan Streamlit untuk melakukan prediksi sentimen secara interaktif.

- `data.csv`  
  Dataset berisi kolom `text` (ulasan pelanggan) dan `label` (sentimen: 0 atau 1).

---

## Instalasi

1. Clone repo ini  
   git clone <repo-url>

2. Install dependensi  
   `pip install -r requirements.txt`

Isi `requirements.txt` minimal:

```
transformers
datasets
torch
streamlit
```

---

## Cara Pakai

### 1. Download model pretrained dan tokenizer

```
python setup.py
```

### 2. Latih model dengan dataset sendiri

Pastikan file `data.csv` sudah ada di folder yang sama, lalu jalankan:

```
python train.py
```

### 3. Uji model dengan contoh kalimat

```
python test.py
```

### 4. Jalankan aplikasi web untuk prediksi interaktif

```
streamlit run app.py
```

---

## Catatan

- Model dasar yang dipakai adalah `distilbert-base-uncased-finetuned-sst-2-english` yang sudah dilatih untuk analisis sentimen.
- `train.py` menggunakan model lokal hasil download untuk melatih ulang dengan data baru.
- Dataset harus berformat CSV dengan dua kolom: `text` dan `label` (0 = negatif, 1 = positif).

---

## Contoh Output

```
Sentiment: positive
```

---

## Lisensi

MIT License

---
