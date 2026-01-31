# Workflow CI – MLflow Project (Basic)

Repository ini dibuat untuk memenuhi **Kriteria 3** pada submission Machine Learning menggunakan **MLflow Project** dan **GitHub Actions (CI)**.

## 🎯 Tujuan

Workflow ini memungkinkan proses **training model machine learning berjalan otomatis** setiap kali terjadi *push* ke repository GitHub.

## 📁 Struktur Repository

```
Workflow-CI
├── .github
│   └── workflows
│       └── ci.yml
├── MLProject
│   ├── modelling.py
│   ├── conda.yaml
│   ├── MLProject
│   └── titanic_preprocessed.csv
└── README.md
```

## ⚙️ Penjelasan Komponen

### 1. MLProject/

Folder ini berisi konfigurasi **MLflow Project**:

* **modelling.py**
  Script Python untuk melatih model machine learning.

* **conda.yaml**
  File environment untuk menentukan dependency yang dibutuhkan MLflow saat menjalankan project.

* **MLProject**
  File konfigurasi MLflow Project yang mendefinisikan:

  * Nama project
  * Environment (conda)
  * Entry point untuk menjalankan training model

* **titanic_preprocessed.csv**
  Dataset hasil preprocessing yang digunakan untuk training model.

### 2. .github/workflows/ci.yml

File workflow GitHub Actions yang berfungsi untuk:

* Menjalankan MLflow Project secara otomatis
* Terpicu saat terjadi **push** ke branch `main`
* Menjalankan proses training model menggunakan MLflow

## 🚀 Cara Kerja Workflow

1. User melakukan **push** ke repository
2. GitHub Actions otomatis berjalan
3. Workflow menjalankan:

   ```bash
   mlflow run MLProject
   ```
4. Script `modelling.py` dijalankan
5. Proses training selesai

## ✅ Status Kriteria

* [x] Folder MLProject dibuat
* [x] File MLProject tersedia
* [x] Workflow CI berjalan otomatis
* [x] Workflow berhasil dieksekusi (status success)

---

📌 *Dibuat sebagai bagian dari pembelajaran dan submission Machine Learning Workflow menggunakan MLflow.*
