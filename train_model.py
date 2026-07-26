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
def load_dataset():
    label_map = {label: num for num, label in enumerate(ACTIONS)}
    sequences, labels = [], []

    print("Memuat dataset dari folder Data_BISINDO...")
    missing_actions = []

    for action in ACTIONS:
        action_path = os.path.join(DATA_PATH, action)
        if not os.path.exists(action_path):
            missing_actions.append(action)
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
                sequences.append(window)
                labels.append(label_map[action])
                sample_count += 1

        print(f" - [{action}]: {sample_count}/{NO_SEQUENCES} sampel dimuat.")

    if len(sequences) == 0:
        print("\n[ERROR] Tidak ada data .npy yang ditemukan!")
        print("Silakan rekam dataset terlebih dahulu dengan menjalankan `py -3.11 collect_data.py`.")
        sys.exit(1)

    X = np.array(sequences)
    y = to_categorical(labels, num_classes=len(ACTIONS))

    print(f"\nTotal Sampel Dimuat: {X.shape[0]}")
    print(f"Shape Fitur (X): {X.shape}")
    print(f"Shape Label (y): {y.shape}")

    return X, y, labels

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
    X, y, raw_labels = load_dataset()

    # Split dataset 90% Train, 10% Test
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.1, random_state=42, stratify=raw_labels
    )

    print(f"\nUkuran Data Train: {X_train.shape[0]} sampel")
    print(f"Ukuran Data Test : {X_test.shape[0]} sampel")

    # Inisialisasi Model
    model = build_lstm_model((SEQUENCE_LENGTH, FEATURE_DIM), len(ACTIONS))
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

    # Evaluasi Model
    print("\n==========================================")
    print("EVALUASI MODEL DENGAN DATA TEST")
    print("==========================================")
    y_pred = model.predict(X_test)
    y_true_indices = np.argmax(y_test, axis=1)
    y_pred_indices = np.argmax(y_pred, axis=1)

    # Filter kelas yang ada pada data test
    present_classes = sorted(list(set(y_true_indices) | set(y_pred_indices)))
    target_names = [ACTIONS[i] for i in present_classes]

    print("\nLaporan Klasifikasi (Classification Report):")
    print(classification_report(y_true_indices, y_pred_indices, target_names=target_names))

    print("Confusion Matrix:")
    print(confusion_matrix(y_true_indices, y_pred_indices))

if __name__ == '__main__':
    train()
