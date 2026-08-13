import os
import sys
import numpy as np
from sklearn.model_selection import train_test_split
from tensorflow.keras.utils import to_categorical
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense, Dropout
from tensorflow.keras.callbacks import TensorBoard, EarlyStopping, ModelCheckpoint
from sklearn.metrics import classification_report, confusion_matrix

# ==========================================
# 1. KONFIGURASI DATASET & LABEL
# ==========================================
DATA_PATH = os.path.join('Data_BISINDO')

# 20 Kosakata Utama Layanan Dukcapil (Harus persis sama dengan collect_data.py)
ACTIONS = np.array([
    'KTP', 'KK', 'AKTA_KELAHIRAN', 'AKTA_KEMATIAN', 'AKTA_PERKAWINAN',
    'KIA', 'PASPOR', 'SURAT_PINDAH', 'NIK', 'ALAMAT',
    'NIKAH', 'BELUM_KAWIN', 'CERAI', 'KEPALA_KELUARGA', 'ANAK',
    'UBAH', 'HILANG', 'DAFTAR', 'CETAK', 'PETUGAS'
])

NO_SEQUENCES = 30
SEQUENCE_LENGTH = 30
FEATURE_DIM = 258  # 33*4 (pose) + 21*3 (lh) + 21*3 (rh)

MODEL_NAME = 'action_bisindo.h5'
LOG_DIR = os.path.join('Logs')

# ==========================================
# 2. PROSES MEMUAT DATASET
# ==========================================
# ==========================================
# 2. PROSES MEMUAT DATASET
# ==========================================
def load_dataset():
    print("Memuat dataset dari folder Data_BISINDO...")
    
    # Deteksi kelas/kata mana saja yang memiliki data
    active_actions = []
    sequences, labels = [], []
    action_counts = {}

    for action in ACTIONS:
        action_path = os.path.join(DATA_PATH, action)
        if not os.path.exists(action_path):
            continue

        sample_count = 0
        for sequence in range(NO_SEQUENCES):
            window = []
            for frame_num in range(SEQUENCE_LENGTH):
                res_path = os.path.join(DATA_PATH, action, str(sequence), f"{frame_num}.npy")
                if os.path.exists(res_path):
                    res = np.load(res_path)
                    window.append(res)
                else:
                    break

            if len(window) == SEQUENCE_LENGTH:
                sample_count += 1

        if sample_count > 0:
            active_actions.append(action)
            action_counts[action] = sample_count

    if len(active_actions) == 0:
        print("\n[ERROR] Tidak ada data .npy yang ditemukan!")
        print("Silakan rekam dataset terlebih dahulu dengan menjalankan:")
        print("  .venv\\Scripts\\python collect_data.py")
        sys.exit(1)

    print("\nRingkasan Sampel Terdeteksi:")
    for act in ACTIONS:
        cnt = action_counts.get(act, 0)
        print(f" - [{act}]: {cnt}/{NO_SEQUENCES} sampel dimuat.")

    if len(active_actions) < 2:
        print(f"\n[PERINGATAN] Baru {len(active_actions)} kata ({active_actions}) yang memiliki sampel data.")
        print("Minimal diperlukan 2 kata yang memiliki data untuk melatih model klasifikasi.")
        print("Silakan jalankan `collect_data.py` untuk merekam kata lainnya.")
        sys.exit(1)

    # Buat label map khusus untuk kelas yang aktif saja
    active_actions = np.array(active_actions)
    label_map = {label: num for num, label in enumerate(active_actions)}

    for action in active_actions:
        for sequence in range(NO_SEQUENCES):
            window = []
            for frame_num in range(SEQUENCE_LENGTH):
                res_path = os.path.join(DATA_PATH, action, str(sequence), f"{frame_num}.npy")
                if os.path.exists(res_path):
                    res = np.load(res_path)
                    window.append(res)
                else:
                    break

            if len(window) == SEQUENCE_LENGTH:
                sequences.append(window)
                labels.append(label_map[action])

    X = np.array(sequences)
    y = to_categorical(labels, num_classes=len(active_actions))

    print(f"\nTotal Sampel Dimuat: {X.shape[0]}")
    print(f"Shape Fitur (X): {X.shape}")
    print(f"Shape Label (y): {y.shape}")

    # Simpan daftar kelas aktif agar dipakai oleh predict_realtime.py
    np.save('actions.npy', active_actions)

    return X, y, labels, active_actions

# ==========================================
# 3. MEMBANGUN MODEL LSTM
# ==========================================
def build_lstm_model(input_shape, num_classes):
    model = Sequential([
        LSTM(64, return_sequences=True, activation='relu', input_shape=input_shape),
        Dropout(0.2),
        LSTM(128, return_sequences=True, activation='relu'),
        Dropout(0.2),
        LSTM(64, return_sequences=False, activation='relu'),
        Dropout(0.2),
        Dense(64, activation='relu'),
        Dense(32, activation='relu'),
        Dense(num_classes, activation='softmax')
    ])

    model.compile(
        optimizer='adam',
        loss='categorical_crossentropy',
        metrics=['categorical_accuracy']
    )
    return model

# ==========================================
# 4. TRAINING LOOP & EVALUASI
# ==========================================
def train():
    X, y, raw_labels, active_actions = load_dataset()

    # Cek apakah stratify memungkinkan (setiap kelas minimal harus 2 sampel)
    from collections import Counter
    label_counts = Counter(raw_labels)
    can_stratify = all(count >= 2 for count in label_counts.values()) and (len(X) >= 5)

    if can_stratify:
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=0.1, random_state=42, stratify=raw_labels
        )
    else:
        print("\n[INFO] Jumlah sampel per kelas terlalu sedikit untuk stratifikasi. Menggunakan split biasa.")
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=0.1, random_state=42
        )

    print(f"\nUkuran Data Train: {X_train.shape[0]} sampel")
    print(f"Ukuran Data Test : {X_test.shape[0]} sampel")

    # Inisialisasi Model
    model = build_lstm_model((SEQUENCE_LENGTH, FEATURE_DIM), len(active_actions))
    model.summary()

    # Callbacks
    os.makedirs(LOG_DIR, exist_ok=True)
    tb_callback = TensorBoard(log_dir=LOG_DIR)
    checkpoint_callback = ModelCheckpoint(
        MODEL_NAME, monitor='val_categorical_accuracy', save_best_only=True, verbose=1
    )
    early_stop = EarlyStopping(
        monitor='val_loss', patience=25, restore_best_weights=True
    )

    print("\nMemulai Pelatihan Model LSTM...")
    history = model.fit(
        X_train, y_train,
        validation_data=(X_test, y_test),
        epochs=200,
        batch_size=16,
        callbacks=[tb_callback, checkpoint_callback, early_stop]
    )

    # Simpan Model Akhir
    model.save(MODEL_NAME)
    print(f"\nModel berhasil disimpan ke '{MODEL_NAME}'.")
    print("Daftar kelas tersimpan ke 'actions.npy'.")

    # Evaluasi Model
    if len(X_test) > 0:
        print("\n==========================================")
        print("EVALUASI MODEL DENGAN DATA TEST")
        print("==========================================")
        y_pred = model.predict(X_test)
        y_true_indices = np.argmax(y_test, axis=1)
        y_pred_indices = np.argmax(y_pred, axis=1)

        present_classes = sorted(list(set(y_true_indices) | set(y_pred_indices)))
        target_names = [active_actions[i] for i in present_classes]

        print("\nLaporan Klasifikasi (Classification Report):")
        print(classification_report(y_true_indices, y_pred_indices, labels=present_classes, target_names=target_names, zero_division=0))

        print("Confusion Matrix:")
        print(confusion_matrix(y_true_indices, y_pred_indices))

if __name__ == '__main__':
    train()
