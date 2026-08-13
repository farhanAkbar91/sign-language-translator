# Sistem Penerjemah Bahasa Isyarat BISINDO (Dukcapil)

Proyek klasifikasi dan deteksi real-time bahasa isyarat BISINDO menggunakan MediaPipe Holistic dan model LSTM (TensorFlow/Keras).

## Persyaratan Sistem
- Windows OS
- Python 3.11 (`py -3.11`)
- Webcam

## Cara Menggunakan Proyek

### 1. Setup Lingkungan (Hanya Sekali)
Double click file `setup_env.bat` atau jalankan di terminal:
```cmd
setup_env.bat
```
Skrip ini akan otomatis membuat Virtual Environment `.venv` berbasis Python 3.11 dan menginstall seluruh dependensi yang diperlukan.

### 2. Pengumpulan Data (`collect_data.py`)
Double click file `run_collect.bat` atau jalankan:
```cmd
.venv\Scripts\python collect_data.py
```
- Anda dapat memilih untuk merekam 1 kata tertentu atau seluruh kata secara berurutan.
- Tekan **q** di jendela video jika ingin membatalkan perekaman di tengah jalan.

### 3. Melatih Model (`train_model.py`)
Setelah merekam minimal 2 kata, jalankan:
```cmd
.venv\Scripts\python train_model.py
```
- Skrip akan melatih model LSTM berdasarkan data yang sudah direkam dan menyimpan hasilnya ke file `action_bisindo.h5` serta `actions.npy`.

### 4. Deteksi Real-Time Kamera (`predict_realtime.py`)
Jalankan deteksi real-time:
```cmd
.venv\Scripts\python predict_realtime.py
```
- Jendela webcam akan terbuka, mengenali gerakan isyarat secara real-time, dan menyusun kalimat di bagian bawah layar.
- Tekan **q** untuk keluar.
