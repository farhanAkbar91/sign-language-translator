import cv2
import numpy as np
import os
import sys
import time
import mediapipe as mp

try:
    mp_holistic = mp.solutions.holistic
    mp_drawing = mp.solutions.drawing_utils
    HAS_SOLUTIONS = True
except AttributeError:
    HAS_SOLUTIONS = False
    mp_holistic = None
    mp_drawing = None

# ==========================================
# 1. KONFIGURASI DATASET DUKCAPIL (BISINDO)
# ==========================================
DATA_PATH = os.path.join('Data_BISINDO') 

# 20 Kosakata Utama Layanan Dukcapil (BISINDO)
ACTIONS = np.array([
    # Kelompok Dokumen & Identitas
    'KTP',               # 1. KTP (Kartu Tanda Penduduk)
    'KK',                # 2. KK (Kartu Keluarga)
    'AKTA_KELAHIRAN',    # 3. Akta Kelahiran
    'AKTA_KEMATIAN',     # 4. Akta Kematian
    'AKTA_PERKAWINAN',   # 5. Akta Perkawinan / Buku Nikah
    'KIA',               # 6. KIA (Kartu Identitas Anak)
    'PASPOR',            # 7. Paspor
    'SURAT_PINDAH',      # 8. Surat Pindah

    # Kelompok Istilah & Status Diri
    'NIK',               # 9. NIK (Nomor Induk Kependudukan)
    'ALAMAT',            # 10. Alamat
    'NIKAH',             # 11. NIKAH / Kawin (Status Perkawinan)
    'BELUM_KAWIN',       # 12. Belum Kawin
    'CERAI',             # 13. Cerai
    'KEPALA_KELUARGA',   # 14. Kepala Keluarga
    'ANAK',              # 15. Anak

    # Kelompok Layanan & Proses
    'UBAH',              # 16. Ubah / Perbarui (Update data)
    'HILANG',            # 17. Hilang (Laporan kehilangan dokumen)
    'DAFTAR',            # 18. Daftar / Buat Baru
    'CETAK',             # 19. Cetak
    'PETUGAS'            # 20. Petugas (Pegawai Dukcapil)
])

# Jumlah sampel klip video per kata
NO_SEQUENCES = 30

# Jumlah frame per klip video
SEQUENCE_LENGTH = 30

# Buat folder penyimpanan secara otomatis
def create_data_folders():
    for action in ACTIONS:
        for sequence in range(NO_SEQUENCES):
            os.makedirs(os.path.join(DATA_PATH, action, str(sequence)), exist_ok=True)

# ==========================================
# 2. HELPER FUNCTIONS
# ==========================================

def mediapipe_detection(image, model):
    """Mengubah warna image ke RGB dan memproses dengan MediaPipe."""
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    image.flags.writeable = False
    results = model.process(image)
    image.flags.writeable = True
    image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)
    return image, results

def draw_styled_landmarks(image, results):
    """Menduga & menggambar landmark tangan dan pose di layar."""
    # Pose
    mp_drawing.draw_landmarks(
        image, results.pose_landmarks, mp_holistic.POSE_CONNECTIONS,
        mp_drawing.DrawingSpec(color=(80, 22, 10), thickness=2, circle_radius=2),
        mp_drawing.DrawingSpec(color=(80, 44, 121), thickness=2, circle_radius=1)
    )
    # Tangan Kiri
    mp_drawing.draw_landmarks(
        image, results.left_hand_landmarks, mp_holistic.HAND_CONNECTIONS,
        mp_drawing.DrawingSpec(color=(121, 22, 76), thickness=2, circle_radius=2),
        mp_drawing.DrawingSpec(color=(121, 44, 250), thickness=2, circle_radius=1)
    )
    # Tangan Kanan
    mp_drawing.draw_landmarks(
        image, results.right_hand_landmarks, mp_holistic.HAND_CONNECTIONS,
        mp_drawing.DrawingSpec(color=(245, 117, 66), thickness=2, circle_radius=2),
        mp_drawing.DrawingSpec(color=(245, 66, 230), thickness=2, circle_radius=1)
    )

def extract_keypoints(results):
    """Ekstraksi koordinat pose dan kedua tangan ke dalam 1 array flat (258 fitur)."""
    pose = np.array([[res.x, res.y, res.z, res.visibility] for res in results.pose_landmarks.landmark]).flatten() if results.pose_landmarks else np.zeros(33 * 4)
    lh = np.array([[res.x, res.y, res.z] for res in results.left_hand_landmarks.landmark]).flatten() if results.left_hand_landmarks else np.zeros(21 * 3)
    rh = np.array([[res.x, res.y, res.z] for res in results.right_hand_landmarks.landmark]).flatten() if results.right_hand_landmarks else np.zeros(21 * 3)
    return np.concatenate([pose, lh, rh])

def collect_data():
    if not HAS_SOLUTIONS:
        print("\n[ERROR] Modul 'mediapipe.solutions' tidak ditemukan pada Python versi ini.")
        print(f"Versi Python aktif: {sys.version.split()[0]}")
        print("MediaPipe Holistic memerlukan Python 3.11 atau 3.10.\n")
        print("Langkah untuk menjalankannya:")
        print("  1. Pastikan Python 3.11 terinstall (jalankan: py -3.11 --version)")
        print("  2. Install library pada Python 3.11: py -3.11 -m pip install opencv-python numpy mediapipe")
        print("  3. Jalankan skrip dengan: py -3.11 collect_data.py")
        return

    create_data_folders()
    
    cap = cv2.VideoCapture(0)
    if not cap.isOpened():
        print("Gagal membuka kamera. Pastikan webcam terhubung dan tidak sedang digunakan oleh aplikasi lain.")
        return

    cancelled = False

    # Inisialisasi model MediaPipe Holistic
    with mp_holistic.Holistic(min_detection_confidence=0.5, min_tracking_confidence=0.5) as holistic:
        for action in ACTIONS:
            if cancelled:
                break
            for sequence in range(NO_SEQUENCES):
                if cancelled:
                    break
                for frame_num in range(SEQUENCE_LENGTH):

                    ret, frame = cap.read()
                    if not ret:
                        print("Gagal membaca frame dari kamera.")
                        cancelled = True
                        break

                    # Deteksi
                    image, results = mediapipe_detection(frame, holistic)
                    draw_styled_landmarks(image, results)

                    # Logika Jeda/Countdown sebelum merekam klip baru
                    if frame_num == 0:
                        start_time = time.time()
                        countdown_sec = 2
                        while time.time() - start_time < countdown_sec:
                            ret, frame = cap.read()
                            if not ret:
                                break
                            image, results = mediapipe_detection(frame, holistic)
                            draw_styled_landmarks(image, results)
                            
                            time_left = int(countdown_sec - (time.time() - start_time)) + 1
                            cv2.putText(image, f'BERSIAP Dalam {time_left}s...', (120, 200),
                                        cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 255), 4, cv2.LINE_AA)
                            cv2.putText(image, f'Merekam "{action}" - Klip #{sequence + 1}', (15, 30),
                                        cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2, cv2.LINE_AA)
                            cv2.imshow('BISINDO Data Collection', image)
                            
                            if cv2.waitKey(10) & 0xFF == ord('q'):
                                cancelled = True
                                break
                        
                        if cancelled:
                            break
                    else:
                        cv2.putText(image, f'Merekam "{action}" - Klip #{sequence + 1} (Frame {frame_num})', (15, 30),
                                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2, cv2.LINE_AA)
                        cv2.imshow('BISINDO Data Collection', image)

                    # Ekstraksi dan Simpan Fitur ke file .npy
                    keypoints = extract_keypoints(results)
                    npy_path = os.path.join(DATA_PATH, action, str(sequence), f'{frame_num}.npy')
                    np.save(npy_path, keypoints)

                    # Tekan 'q' untuk keluar di tengah jalan
                    if cv2.waitKey(10) & 0xFF == ord('q'):
                        cancelled = True
                        break

    cap.release()
    cv2.destroyAllWindows()
    if cancelled:
        print("\nPengumpulan data dihentikan oleh pengguna.")
    else:
        print("\nPengumpulan data 20 kosakata Dukcapil selesai!")

if __name__ == '__main__':
    collect_data()