import cv2
import numpy as np
import os
import sys
import mediapipe as mp

try:
    mp_holistic = mp.solutions.holistic
    mp_drawing = mp.solutions.drawing_utils
    HAS_SOLUTIONS = True
except AttributeError:
    HAS_SOLUTIONS = False

# 20 Kosakata Utama Layanan Dukcapil
ACTIONS = np.array([
    'KTP', 'KK', 'AKTA_KELAHIRAN', 'AKTA_KEMATIAN', 'AKTA_PERKAWINAN',
    'KIA', 'PASPOR', 'SURAT_PINDAH', 'NIK', 'ALAMAT',
    'NIKAH', 'BELUM_KAWIN', 'CERAI', 'KEPALA_KELUARGA', 'ANAK',
    'UBAH', 'HILANG', 'DAFTAR', 'CETAK', 'PETUGAS'
])

MODEL_PATH = 'action_bisindo.h5'

def mediapipe_detection(image, model):
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    image.flags.writeable = False
    results = model.process(image)
    image.flags.writeable = True
    image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)
    return image, results

def draw_styled_landmarks(image, results):
    mp_drawing.draw_landmarks(
        image, results.pose_landmarks, mp_holistic.POSE_CONNECTIONS,
        mp_drawing.DrawingSpec(color=(80, 22, 10), thickness=2, circle_radius=2),
        mp_drawing.DrawingSpec(color=(80, 44, 121), thickness=2, circle_radius=1)
    )
    mp_drawing.draw_landmarks(
        image, results.left_hand_landmarks, mp_holistic.HAND_CONNECTIONS,
        mp_drawing.DrawingSpec(color=(121, 22, 76), thickness=2, circle_radius=2),
        mp_drawing.DrawingSpec(color=(121, 44, 250), thickness=2, circle_radius=1)
    )
    mp_drawing.draw_landmarks(
        image, results.right_hand_landmarks, mp_holistic.HAND_CONNECTIONS,
        mp_drawing.DrawingSpec(color=(245, 117, 66), thickness=2, circle_radius=2),
        mp_drawing.DrawingSpec(color=(245, 66, 230), thickness=2, circle_radius=1)
    )

def extract_keypoints(results):
    pose = np.array([[res.x, res.y, res.z, res.visibility] for res in results.pose_landmarks.landmark]).flatten() if results.pose_landmarks else np.zeros(33 * 4)
    lh = np.array([[res.x, res.y, res.z] for res in results.left_hand_landmarks.landmark]).flatten() if results.left_hand_landmarks else np.zeros(21 * 3)
    rh = np.array([[res.x, res.y, res.z] for res in results.right_hand_landmarks.landmark]).flatten() if results.right_hand_landmarks else np.zeros(21 * 3)
    return np.concatenate([pose, lh, rh])

def main():
    if not HAS_SOLUTIONS:
        print("\n[ERROR] Modul 'mediapipe.solutions' tidak ditemukan.")
        print("Silakan jalankan via virtual environment Python 3.11:")
        print("  .venv\\Scripts\\python predict_realtime.py")
        return

    if not os.path.exists(MODEL_PATH):
        print(f"\n[ERROR] File model '{MODEL_PATH}' belum ada!")
        print("Silakan jalankan `.venv\\Scripts\\python train_model.py` terlebih dahulu untuk melatih model.")
        return

    # Muat kelas aktif dari actions.npy jika ada
    if os.path.exists('actions.npy'):
        actions = np.load('actions.npy')
    else:
        actions = ACTIONS

    from tensorflow.keras.models import load_model

    print(f"Memuat model '{MODEL_PATH}'...")
    model = load_model(MODEL_PATH)
    print(f"Model berhasil dimuat! ({len(actions)} kelas: {actions})")
    print("Membuka webcam...")

    sequence = []
    sentence = []
    predictions = []
    threshold = 0.7

    cap = cv2.VideoCapture(0)
    if not cap.isOpened():
        cap = cv2.VideoCapture(0, cv2.CAP_DSHOW)
    if not cap.isOpened():
        print("Gagal membuka webcam. Pastikan webcam terhubung dan tidak sedang digunakan aplikasi lain.")
        return

    with mp_holistic.Holistic(min_detection_confidence=0.5, min_tracking_confidence=0.5) as holistic:
        while cap.isOpened():
            ret, frame = cap.read()
            if not ret:
                break

            image, results = mediapipe_detection(frame, holistic)
            draw_styled_landmarks(image, results)

            # Ekstraksi keypoints
            keypoints = extract_keypoints(results)
            sequence.append(keypoints)
            sequence = sequence[-30:]  # Pertahankan 30 frame terakhir

            if len(sequence) == 30:
                input_data = np.expand_dims(sequence, axis=0)
                # Gunakan pemanggilan model langsung (Fast inference tanpa overhead predict)
                res = model(input_data, training=False).numpy()[0]
                best_class_idx = int(np.argmax(res))
                confidence = float(res[best_class_idx])
                predictions.append(best_class_idx)

                # Logika kestabilan prediksi (5 frame berturut-turut konsisten)
                if len(predictions) >= 5 and len(set(predictions[-5:])) == 1:
                    if confidence > threshold:
                        if best_class_idx < len(actions):
                            predicted_action = actions[best_class_idx]
                            if len(sentence) > 0:
                                if predicted_action != sentence[-1]:
                                    sentence.append(predicted_action)
                            else:
                                sentence.append(predicted_action)

                if len(sentence) > 5:
                    sentence = sentence[-5:]

                # Tampilkan hasil di bagian atas layar
                cv2.rectangle(image, (0, 0), (640, 40), (245, 117, 16), -1)
                predicted_name = actions[best_class_idx] if best_class_idx < len(actions) else "Unknown"
                text = f"Deteksi: {predicted_name} ({confidence*100:.1f}%)"
                cv2.putText(image, text, (10, 28), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 255), 2, cv2.LINE_AA)

                # Tampilkan riwayat kata/kalimat di bagian bawah layar
                cv2.rectangle(image, (0, 440), (640, 480), (245, 117, 16), -1)
                sentence_text = "Kalimat: " + " ".join(sentence)
                cv2.putText(image, sentence_text, (10, 470), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2, cv2.LINE_AA)

            cv2.imshow('BISINDO Dukcapil - Deteksi Real-Time', image)

            if cv2.waitKey(10) & 0xFF == ord('q'):
                break

    cap.release()
    cv2.destroyAllWindows()

if __name__ == '__main__':
    main()
