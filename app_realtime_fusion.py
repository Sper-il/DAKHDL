"""
Nhận diện cảm xúc thời gian thực (Mic + Cam) - Mô hình Late Fusion (FaceCNN + AudioCNN).
Tương thích với Model-face-audio-test (ResNet-18 xám 48x48 + AudioCNN Mel 128x128).

Cách chạy:
  python app_realtime_fusion.py
  hoặc chỉ định checkpoint cụ thể:
  python app_realtime_fusion.py --face-ckpt runs/face_resnet18_xxx/best.pt --audio-ckpt runs/audio_cnn_xxx/best.pt
"""
import os
import sys
import glob
import time
import queue
import argparse
import threading
from collections import deque
from pathlib import Path

import cv2
import numpy as np
import torch
import torch.nn.functional as F
import torchaudio.transforms as T

# Thêm đường dẫn thư mục hiện tại để nạp src.models & src.labels
CURRENT_DIR = Path(__file__).resolve().parent
sys.path.insert(0, str(CURRENT_DIR))

try:
    from src.models import FaceCNN, AudioCNN
    from src.labels import EMOTIONS, EMOTIONS_VI, FACE7_TO_4_IDX, class_names, class_names_vi
except ImportError:
    # Fallback dự phòng nếu chạy từ ngoài
    EMOTIONS = ["angry", "disgust", "fear", "happy", "sad", "surprise", "neutral", "contempt", "boredom", "excited"]
    EMOTIONS_VI = ["Gian du", "Ghe tom", "So hai", "Vui ve", "Buon", "Ngac nhien", "Trung tinh", "Khinh bi", "Chan nan", "Hao hung"]
    FACE7_TO_4_IDX = [0, 3, 4, 6]
    def class_names(n):
        return ["angry", "happy", "sad", "neutral"] if n == 4 else EMOTIONS[:n]
    def class_names_vi(n):
        return ["Gian du", "Vui ve", "Buon", "Trung tinh"] if n == 4 else EMOTIONS_VI[:n]

# ==========================================
# CẤU HÌNH THỜI GIAN THỰC
# ==========================================
SAMPLE_RATE = 16000
AUDIO_SECONDS = 3.0
AUDIO_SAMPLES = int(SAMPLE_RATE * AUDIO_SECONDS)
AUDIO_RMS_MIN = 0.005          # Ngưỡng phát hiện tiếng nói (trên ngưỡng coi như có nói)

FACE_SIZE = 48                 # ResNet-18 nhánh mặt nhận ảnh 48x48
MEL_SIZE = 128                 # AudioCNN nhánh tiếng nhận Mel 128x128

TARGET_FPS = 60
INFER_EVERY = 0.35             # Khoảng thời gian (giây) giữa 2 lần dự đoán
SMOOTH = 0.6                   # Hệ số làm mượt xác suất (0 = tức thời, 0.7 = rất đầm)
FACE_LOST_SEC = 1.0            # Thời gian mất mặt trước khi coi như không có cam


# ==========================================
# BIẾN ĐỔI DỮ LIỆU
# ==========================================
mel_transform = T.MelSpectrogram(sample_rate=SAMPLE_RATE, n_fft=1024, hop_length=512, n_mels=MEL_SIZE)
to_db = T.AmplitudeToDB(stype="power", top_db=80)

# Khởi tạo Haar Cascade phát hiện khuôn mặt
CASCADE_FILENAME = "haarcascade_frontalface_default.xml"
_cascade_candidates = [
    CURRENT_DIR / CASCADE_FILENAME,
    CURRENT_DIR.parent / "code lấy cam vs mic" / CASCADE_FILENAME,
    CURRENT_DIR.parent / CASCADE_FILENAME,
]
_cascade_path = None
for p in _cascade_candidates:
    if p.exists():
        _cascade_path = str(p)
        break
if _cascade_path is None and hasattr(cv2, "data") and hasattr(cv2.data, "haarcascades"):
    _cascade_path = cv2.data.haarcascades + CASCADE_FILENAME

face_cascade = cv2.CascadeClassifier(_cascade_path if _cascade_path else CASCADE_FILENAME)


def crop_face(gray, box):
    """Cắt khuôn mặt, thêm lề 10%, resize về 48x48 và cân bằng histogram."""
    h, w = gray.shape
    x, y, bw, bh = [int(v) for v in box]
    m = int(0.1 * max(bw, bh))
    x0, y0 = max(0, x - m), max(0, y - m)
    x1, y1 = min(w, x + bw + m), min(h, y + bh + m)
    crop = gray[y0:y1, x0:x1]
    if crop.size == 0:
        return None
    crop = cv2.resize(crop, (FACE_SIZE, FACE_SIZE), interpolation=cv2.INTER_AREA)
    return cv2.equalizeHist(crop)


def face_to_tensor(crop_gray, mean=0.5, std=0.5):
    """Ảnh uint8 (48, 48) -> tensor [1, 1, 48, 48]."""
    arr = crop_gray.astype(np.float32) / 255.0
    arr = (arr - mean) / (std + 1e-6)
    return torch.from_numpy(arr).unsqueeze(0).unsqueeze(0)


def audio_to_tensor(buf, mean=0.0, std=1.0):
    """Mảng âm thanh float32 16kHz -> Mel-spectrogram tensor [1, 1, 128, 128]."""
    wav = torch.from_numpy(buf).float().unsqueeze(0)
    mel = to_db(mel_transform(wav))
    if mel.shape[-1] != MEL_SIZE:
        mel = F.interpolate(mel.unsqueeze(0), size=(MEL_SIZE, MEL_SIZE), mode="bilinear", align_corners=False).squeeze(0)
    mel = (mel - mean) / (std + 1e-6)
    return mel.unsqueeze(0)


def find_latest_ckpt(pattern):
    matches = sorted(glob.glob(str(CURRENT_DIR / pattern)))
    return matches[-1] if matches else None


def load_model_branch(ckpt_path, branch, num_classes=7, device="cpu"):
    """Nạp nhánh mô hình từ file checkpoint."""
    if not ckpt_path or not os.path.exists(ckpt_path):
        return None, 0.5 if branch == "face" else 0.0, 0.5 if branch == "face" else 1.0, num_classes

    ck = torch.load(ckpt_path, map_location=device, weights_only=False)
    n = ck.get("num_classes", num_classes)
    mean = float(ck.get("mean", 0.5 if branch == "face" else 0.0))
    std = float(ck.get("std", 0.5 if branch == "face" else 1.0))

    if branch == "face":
        m_cfg = ck.get("cfg", {}).get("model", {})
        model = FaceCNN(
            num_classes=n,
            pretrained=False,
            remove_maxpool=bool(m_cfg.get("remove_maxpool", False)),
            input_size=int(m_cfg.get("input_size", FACE_SIZE)),
            dropout=float(m_cfg.get("dropout", 0.0))
        )
    else:
        m_cfg = ck.get("cfg", {}).get("model", {})
        model = AudioCNN(
            num_classes=n,
            dropout=float(m_cfg.get("dropout", 0.3))
        )

    state = ck["model"] if "model" in ck else ck
    model.load_state_dict(state)
    model.to(device)
    model.eval()
    print(f"--> Đã nạp thành công nhánh {branch.upper()} từ: {ckpt_path} (mean={mean:.3f}, std={std:.3f}, classes={n})")
    return model, mean, std, n


# ==========================================
# CHƯƠNG TRÌNH CHÍNH
# ==========================================
def main():
    parser = argparse.ArgumentParser(description="Real-time Late Fusion Emotion Recognition")
    parser.add_argument("--face-ckpt", type=str, default=None, help="Đường dẫn checkpoint best.pt của FaceCNN")
    parser.add_argument("--audio-ckpt", type=str, default=None, help="Đường dẫn checkpoint best.pt của AudioCNN")
    parser.add_argument("--w-face", type=float, default=0.5, help="Trọng số nhánh mặt (0.0 .. 1.0, mặc định 0.5)")
    parser.add_argument("--num-classes", type=int, default=7, help="Số lớp cảm xúc (7 hoặc 10)")
    args = parser.parse_args()

    import sounddevice as sd

    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Thiết bị tính toán: {device}")

    # Tự động tìm checkpoint mới nhất nếu người dùng không truyền tham số
    face_ckpt = args.face_ckpt or find_latest_ckpt("runs/face_*/best.pt") or find_latest_ckpt("runs/face_*/**/best.pt")
    audio_ckpt = args.audio_ckpt or find_latest_ckpt("runs/audio_*/best.pt") or find_latest_ckpt("runs/audio_*/**/best.pt")

    face_model, face_mean, face_std, n_face = load_model_branch(face_ckpt, "face", args.num_classes, device)
    audio_model, audio_mean, audio_std, n_audio = load_model_branch(audio_ckpt, "audio", args.num_classes, device)

    # Nếu audio là 4 lớp (DataViet) còn face là 7 lớp -> chiếu face 7->4 lớp để fusion đồng bộ
    num_classes = n_audio if (audio_model and n_audio == 4) else (n_face if face_model else (n_audio if audio_model else args.num_classes))
    labels_en = class_names(num_classes)
    labels_vi = class_names_vi(num_classes)

    if face_model is None and audio_model is None:
        print("\n[CHÚ Ý] Chưa tìm thấy checkpoint trong thư mục runs/!")
        print("  --> Ứng dụng sẽ chạy ở chế độ DEMO / TEST tín hiệu (mô phỏng xác suất).")
        print("  --> Khi train xong, bạn chỉ cần để best.pt trong runs/ hoặc dùng --face-ckpt / --audio-ckpt\n")

    w_face = float(args.w_face)

    # ---- Micro Background Thread ----
    audio_buffer = np.zeros(AUDIO_SAMPLES, dtype=np.float32)
    audio_lock = threading.Lock()
    audio_queue = queue.Queue()

    def audio_callback(indata, frames, time_info, status):
        audio_queue.put(indata[:, 0].copy())

    def update_audio_buffer():
        nonlocal audio_buffer
        while True:
            try:
                chunk = audio_queue.get(timeout=1)
            except queue.Empty:
                continue
            with audio_lock:
                audio_buffer = np.roll(audio_buffer, -len(chunk))
                audio_buffer[-len(chunk):] = chunk[-AUDIO_SAMPLES:]

    threading.Thread(target=update_audio_buffer, daemon=True).start()

    try:
        mic_stream = sd.InputStream(samplerate=SAMPLE_RATE, channels=1, callback=audio_callback)
        mic_stream.start()
        mic_connected = True
    except Exception as e:
        print(f"[Cảnh báo Mic] Không mở được microphone: {e}")
        mic_connected = False

    # ---- Camera Initialization ----
    cap = None
    cam_connected = False
    for cam_idx in [0, 1]:
        test_cap = cv2.VideoCapture(cam_idx)
        if test_cap.isOpened():
            test_cap.set(cv2.CAP_PROP_FPS, TARGET_FPS)
            test_cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1280)
            test_cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 720)
            ret, _ = test_cap.read()
            if ret:
                cap = test_cap
                cam_connected = True
                print(f"--> Đã kết nối Webcam (index {cam_idx}) - Mục tiêu: {TARGET_FPS} FPS")
                break
            test_cap.release()

    if not cam_connected:
        cap = cv2.VideoCapture(0)
        if cap.isOpened():
            cap.set(cv2.CAP_PROP_FPS, TARGET_FPS)
            cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1280)
            cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 720)
        cam_connected = cap.isOpened()
        if not cam_connected:
            print("[Cảnh báo Cam] Không mở được webcam. Chế độ Camera sẽ tắt.")

    user_cam_on, user_mic_on = True, True
    smooth_box, last_face_time = None, 0.0
    latest_face_crop = None
    frame_count = 0

    fps_history = deque(maxlen=30)
    current_fps = 0.0
    frame_interval = 1.0 / TARGET_FPS

    probs_smooth = None
    last_infer = 0.0
    status_text, mode_str, mode_col = "Dang cho tin hieu...", "", (0, 0, 255)
    p_f_display, p_a_display = None, None
    mic_level = 0.0

    print("\n--- PHÍM TẮT ---")
    print(" [C]: Bật/Tắt Webcam  |  [M]: Bật/Tắt Micro")
    print(" [W]: Đổi trọng số Fusion (0.7 -> 0.5 -> 0.3)  |  [Q]: Thoát\n")

    while True:
        now = time.time()
        fps_history.append(now)
        if len(fps_history) > 1:
            current_fps = (len(fps_history) - 1) / (fps_history[-1] - fps_history[0])

        ret, frame = cap.read() if (cam_connected and user_cam_on) else (False, None)
        cam_active = bool(ret) and user_cam_on
        if not cam_active:
            frame = np.zeros((720, 1280, 3), dtype=np.uint8)
            cv2.putText(frame, "CAMERA OFF (Privacy Mode)", (380, 360), cv2.FONT_HERSHEY_SIMPLEX, 1.0, (0, 0, 255), 2)

        # 1. Phát hiện & cắt khuôn mặt (tối ưu scale 0.5 để giữ 60 FPS)
        if cam_active and not face_cascade.empty():
            gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            if frame_count % 3 == 0:
                scale = 0.5
                small_gray = cv2.resize(gray, (0, 0), fx=scale, fy=scale)
                faces = face_cascade.detectMultiScale(small_gray, 1.2, 4, minSize=(int(50 * scale), int(50 * scale)))
                if len(faces) > 0:
                    box = np.array(max(faces, key=lambda b: b[2] * b[3]), dtype=np.float32) / scale
                    smooth_box = box if smooth_box is None else 0.7 * smooth_box + 0.3 * box
                    last_face_time = now

            if smooth_box is not None and (now - last_face_time) < FACE_LOST_SEC:
                crop = crop_face(gray, smooth_box)
                if crop is not None:
                    latest_face_crop = crop
                x, y, bw, bh = [int(v) for v in smooth_box]
                cv2.rectangle(frame, (x, y), (x + bw, y + bh), (0, 255, 0), 2)
        frame_count += 1

        # 2. Âm thanh từ Mic
        with audio_lock:
            buf = audio_buffer.copy()
        mic_active = user_mic_on and mic_connected
        mic_level = float(np.sqrt(np.mean(buf ** 2))) if mic_active else 0.0
        has_audio = mic_active and mic_level > AUDIO_RMS_MIN

        # 3. Trạng thái có mặt
        has_face = cam_active and (smooth_box is not None) and (now - last_face_time < FACE_LOST_SEC) and (latest_face_crop is not None)

        # 4. Xác định chế độ
        if has_audio and has_face:
            mode_str = f"MODE: MULTIMODAL LATE FUSION (W_face={w_face:.2f})"
            mode_col = (0, 255, 0)
        elif has_audio:
            mode_str = "MODE: AUDIO-ONLY (Chi dung giong noi)"
            mode_col = (0, 255, 255)
        elif has_face:
            mode_str = "MODE: FACE-ONLY (Chi dung khuon mat)"
            mode_col = (255, 255, 0)
        else:
            mode_str = "MODE: CHO TIN HIEU (Noi hoac dua mat vao khung hinh)"
            mode_col = (0, 0, 255)

        # 5. Suy luận định kỳ (Inference & Late Fusion)
        if now - last_infer >= INFER_EVERY:
            last_infer = now
            if not has_face and not has_audio:
                probs_smooth = None
                status_text = "Dang cho tin hieu..."
                p_f_display, p_a_display = None, None
            else:
                p_f, p_a = None, None

                # Nhánh mặt
                if has_face:
                    if face_model is not None:
                        f_in = face_to_tensor(latest_face_crop, face_mean, face_std).to(device)
                        with torch.no_grad():
                            logits_f = face_model(f_in)
                            p_f = F.softmax(logits_f, dim=-1).cpu().numpy()[0]
                        if num_classes == 4 and len(p_f) == 7:
                            p_f = p_f[FACE7_TO_4_IDX]
                            p_f = p_f / (p_f.sum() + 1e-9)
                    else:
                        # Demo mock khi chưa có file weights
                        p_f = np.ones(num_classes, dtype=np.float32) / num_classes

                # Nhánh tiếng
                if has_audio:
                    if audio_model is not None:
                        a_in = audio_to_tensor(buf, audio_mean, audio_std).to(device)
                        with torch.no_grad():
                            logits_a = audio_model(a_in)
                            p_a = F.softmax(logits_a, dim=-1).cpu().numpy()[0]
                    else:
                        # Demo mock khi chưa có file weights
                        p_a = np.ones(num_classes, dtype=np.float32) / num_classes

                p_f_display, p_a_display = p_f, p_a

                # Late Fusion: p = w * p_face + (1 - w) * p_audio
                if p_f is not None and p_a is not None:
                    p_final = w_face * p_f + (1.0 - w_face) * p_a
                elif p_f is not None:
                    p_final = p_f
                else:
                    p_final = p_a

                probs_smooth = p_final if probs_smooth is None else SMOOTH * probs_smooth + (1.0 - SMOOTH) * p_final
                k = int(np.argmax(probs_smooth))
                status_text = f"Cam xuc: {labels_vi[k]} ({labels_en[k].capitalize()}) - {probs_smooth[k] * 100:.1f}%"

        # 6. Vẽ giao diện người dùng
        # Nền banner trên cùng
        cv2.rectangle(frame, (0, 0), (frame.shape[1], 85), (20, 20, 20), -1)
        cv2.putText(frame, mode_str, (20, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.65, mode_col, 2)
        cv2.putText(frame, status_text, (20, 68), cv2.FONT_HERSHEY_SIMPLEX, 0.85, (255, 255, 255), 2)

        # Hiển thị FPS
        fps_col = (0, 255, 0) if current_fps >= 50 else ((0, 255, 255) if current_fps >= 30 else (0, 100, 255))
        cv2.putText(frame, f"FPS: {current_fps:.1f}", (frame.shape[1] - 150, 30),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.65, fps_col, 2)

        # Thanh xác suất từng cảm xúc
        if probs_smooth is not None:
            panel_x, panel_y = 20, 105
            cv2.rectangle(frame, (panel_x - 10, panel_y - 10), (panel_x + 360, panel_y + num_classes * 24 + 10), (30, 30, 30), -1)
            for i, (name_en, name_vi, pr) in enumerate(zip(labels_en, labels_vi, probs_smooth)):
                yy = panel_y + i * 24
                label_str = f"{name_vi[:10]:<10}"
                cv2.putText(frame, label_str, (panel_x, yy + 15), cv2.FONT_HERSHEY_SIMPLEX, 0.45, (220, 220, 220), 1)
                bar_len = int(200 * pr)
                bar_color = (0, 215, 255) if pr == probs_smooth.max() else (180, 130, 70)
                cv2.rectangle(frame, (panel_x + 95, yy + 2), (panel_x + 95 + bar_len, yy + 17), bar_color, -1)
                cv2.putText(frame, f"{pr * 100:.1f}%", (panel_x + 100 + 205, yy + 15), cv2.FONT_HERSHEY_SIMPLEX, 0.42, (200, 200, 200), 1)

        # Thông tin chi tiết 2 nhánh (Face vs Voice)
        sub_x = frame.shape[1] - 380
        cv2.rectangle(frame, (sub_x - 10, 95), (frame.shape[1] - 20, 175), (30, 30, 30), -1)
        if p_f_display is not None:
            kf = int(np.argmax(p_f_display))
            cv2.putText(frame, f"Khuon mat: {labels_vi[kf]} ({p_f_display[kf]*100:.1f}%)", (sub_x, 125),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.52, (255, 255, 0), 1)
        else:
            cv2.putText(frame, "Khuon mat: [Khong co mat]", (sub_x, 125), cv2.FONT_HERSHEY_SIMPLEX, 0.52, (150, 150, 150), 1)

        if p_a_display is not None:
            ka = int(np.argmax(p_a_display))
            cv2.putText(frame, f"Giong noi : {labels_vi[ka]} ({p_a_display[ka]*100:.1f}%)", (sub_x, 155),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.52, (0, 255, 255), 1)
        else:
            cv2.putText(frame, "Giong noi : [Khong co tieng]", (sub_x, 155), cv2.FONT_HERSHEY_SIMPLEX, 0.52, (150, 150, 150), 1)

        # Thanh âm lượng & phím tắt ở đáy
        bottom_y = frame.shape[0] - 20
        cv2.rectangle(frame, (0, frame.shape[0] - 60), (frame.shape[1], frame.shape[0]), (20, 20, 20), -1)
        # Meter mic
        cv2.putText(frame, f"Mic RMS: {mic_level:.4f}", (20, bottom_y - 15), cv2.FONT_HERSHEY_SIMPLEX, 0.45, (180, 180, 180), 1)
        mic_meter = min(150, int(mic_level / 0.05 * 150))
        cv2.rectangle(frame, (150, bottom_y - 25), (150 + 150, bottom_y - 13), (60, 60, 60), 1)
        cv2.rectangle(frame, (150, bottom_y - 25), (150 + mic_meter, bottom_y - 13), (0, 255, 0) if has_audio else (100, 100, 100), -1)

        cv2.putText(frame, "[C]: Cam  |  [M]: Mic  |  [W]: Do trong so (W_face)  |  [Q]: Thoat",
                    (20, bottom_y + 10), cv2.FONT_HERSHEY_SIMPLEX, 0.48, (200, 200, 200), 1)

        cv2.imshow("Nhan dien Cam xuc Da phuong thuc (Late Fusion)", frame)

        # Điều tiết nhịp khung hình 60 FPS
        elapsed = time.time() - now
        wait_ms = max(1, int((frame_interval - elapsed) * 1000))
        key = cv2.waitKey(wait_ms) & 0xFF
        if key == ord("q"):
            break
        elif key == ord("c"):
            user_cam_on = not user_cam_on
            latest_face_crop = None
        elif key == ord("m"):
            user_mic_on = not user_mic_on
        elif key == ord("w"):
            # Chuyển đổi trọng số giữa 0.7 (ưu tiên mặt), 0.5 (cân bằng), 0.3 (ưu tiên tiếng)
            if abs(w_face - 0.5) < 0.05:
                w_face = 0.7
            elif abs(w_face - 0.7) < 0.05:
                w_face = 0.3
            else:
                w_face = 0.5
            print(f"--> Da doi trong so Fusion: W_face={w_face:.2f}, W_audio={1.0 - w_face:.2f}")

    if cam_connected and cap is not None:
        cap.release()
    if mic_connected:
        try:
            mic_stream.stop()
            mic_stream.close()
        except Exception:
            pass
    cv2.destroyAllWindows()
    for _ in range(5):
        cv2.waitKey(1)


if __name__ == "__main__":
    main()
