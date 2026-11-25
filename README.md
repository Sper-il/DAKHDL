# Nhận diện cảm xúc từ khuôn mặt + giọng nói (late fusion)

> [!IMPORTANT]
> ## 🚀 Hướng dẫn chạy Demo Real-time (Camera + Micro) bằng model đã train sẵn
>
> Project đã có sẵn checkpoint `best.pt` cho cả nhánh **Face** (`runs/face_resnet18_20261117_062124/best.pt`) và **Audio** (`runs/audio_cnn_20261117_063812/best.pt`), tự động nạp khi chạy.
>
> ### 1. Chạy nhanh (khuyên dùng)
> Mở Terminal tại thư mục `Model-face-audio-test` và chạy:
> ```bash
> cd "/Users/minhhoa/Downloads/Project đồ án/Model-face-audio-test"
> ./run_realtime.sh
> ```
> *(Script tự kích hoạt môi trường `../.venv` và chạy `app_realtime_fusion.py`.)*
>
> ### 2. Tùy chọn dòng lệnh (nếu cần)
> ```bash
> # Đổi trọng số ưu tiên khuôn mặt (W_face = 0.7, W_audio = 0.3)
> ./run_realtime.sh --w-face 0.7
>
> # Chỉ định checkpoint cụ thể
> ./run_realtime.sh --face-ckpt runs/face_resnet18_20261117_062124/best.pt --audio-ckpt runs/audio_cnn_20261117_063812/best.pt
>
> # Đổi số lớp cảm xúc (7 hoặc 10)
> ./run_realtime.sh --num-classes 7
> ```
>
> ### 3. Cấp quyền & Thiết bị (macOS)
> - Lần đầu chạy, bấm **Allow (Cho phép)** khi macOS hỏi quyền truy cập **Camera** và **Microphone** cho Terminal/IDE.
> - **Camera:** tự động lấy webcam mặc định (MacBook).
> - **Micro:** lấy thiết bị Input mặc định của macOS (**System Settings → Sound → Input**). Kiểm tra thanh **Mic RMS** ở góc dưới màn hình khi nói.
>
> ### 4. Phím tắt khi đang chạy (bấm trên cửa sổ video)
> | Phím | Chức năng |
> |---|---|
> | **`C`** | Bật / tắt Camera |
> | **`M`** | Bật / tắt Micro |
> | **`W`** | Đổi trọng số Late Fusion (`0.5` → `0.7` → `0.3`) |
> | **`Q`** | Thoát chương trình |

---

Dự án gồm **2 phần**:

| Phần | Nội dung | Trạng thái |
|---|---|---|
| **1. Huấn luyện** | Train 2 nhánh (mặt, tiếng), log chi tiết, đánh giá, fusion | **Có trong gói này** |
| **2. Triển khai (deploy)** | API/webcam demo, đóng Docker | Làm sau (chưa có trong gói này) |

## 1. Ý tưởng mô hình (dễ giải thích)

```
Ảnh mặt 48x48 xám ──► ResNet-18 ──► xác suất p_mặt ─┐
                                                      ├─► p = w·p_mặt + (1−w)·p_tiếng ──► cảm xúc
Mel-spectrogram 128x128 ──► CNN 4 khối ──► p_tiếng ──┘
```

- **Nhánh mặt**: ResNet-18 (ImageNet pretrained), đổi lớp đầu sang ảnh xám.
- **Nhánh tiếng**: CNN 4 khối (Conv-BN-ReLU ×2 + MaxPool) trên Mel-spectrogram, có SpecAugment.
- **Fusion**: trung bình có trọng số hai phân bố xác suất; trọng số `w` chọn trên tập validation.

## 2. Cấu trúc thư mục

```
emotion-mm/
├── README.md
├── requirements.txt
├── train.py              # train 1 nhánh      (python train.py --config configs/face.yaml)
├── evaluate.py           # đánh giá lại checkpoint
├── fuse_eval.py          # đánh giá late fusion (chỉ cần numpy)
├── configs/              # face.yaml, audio.yaml
├── src/                  # code: dữ liệu, mô hình, vòng lặp train, log, chỉ số
├── scripts/
│   ├── check_data.py            # kiểm tra data trước khi train
│   ├── make_synthetic_data.py   # tạo data giả để chạy thử
│   ├── smoke_test.sh            # chạy thử toàn bộ pipeline bằng data giả (2 epoch)
│   └── plot_logs.py             # vẽ đồ thị từ log
└── notebooks/train_colab.ipynb  # notebook chạy trên Colab
```

## 3. Data mong đợi

Giải nén `data.zip` được thư mục `data/`; **`data_root` là thư mục chứa thư mục `data` đó**.

```
<data_root>/data/splits/face_{train,val,test}.csv         cột: label, pixels, (Usage)
<data_root>/data/splits/audio_{train,val,test}_proc.csv   cột: mel_path, label, (speaker, source)
<data_root>/data/processed/audio/...npy                   Mel 128x128 (dB, -80..0)
```

Nhãn 0–9 dùng chung cho hai nhánh: `angry, disgust, fear, happy, sad, surprise, neutral, contempt, boredom, excited`.
`num_classes: 7` chỉ lấy 7 lớp đầu (cảm xúc cơ bản); `10` lấy đủ bộ.

Kiểm tra data:
```bash
python scripts/check_data.py --data-root /content/data
```

## 4. Chạy

### Trên Colab (khuyên dùng)
Upload thư mục `emotion-mm` và `data.zip` lên Google Drive, mở `notebooks/train_colab.ipynb`, bật GPU T4, chạy lần lượt các ô.

### Trên máy cá nhân
```bash
pip install -r requirements.txt
python scripts/check_data.py --data-root D:/data_goc
python train.py --config configs/face.yaml  --set data_root=D:/data_goc train.num_workers=0
python train.py --config configs/audio.yaml --set data_root=D:/data_goc train.num_workers=0
python fuse_eval.py --face-run runs/face_resnet18_<thời gian> --audio-run runs/audio_cnn_<thời gian>
```
- **Windows**: đặt `train.num_workers=0`.
- **Máy không có GPU NVIDIA**: vẫn chạy được trên CPU nhưng chậm; nên train trên Colab, máy cá nhân dùng để chạy thử và demo.

### Chạy thử nhanh bằng data giả (nên làm trước)
```bash
bash scripts/smoke_test.sh
```

### Ghi đè cấu hình
Mọi tham số trong `configs/*.yaml` đều ghi đè được bằng `--set`:
```bash
python train.py --config configs/face.yaml --set num_classes=10 train.lr=0.0005 model.input_size=96
```
Tiếp tục run bị ngắt: `python train.py --config configs/face.yaml --run-dir runs/<tên> --resume`

## 5. Log chi tiết — tham số biến đổi ra sao

Mỗi lần train tạo một thư mục `runs/<tên>_<thời gian>/`:

| File | Nội dung |
|---|---|
| `train.log` | Nhật ký văn bản có dấu thời gian (cấu hình, số mẫu từng lớp, trọng số lớp, mỗi epoch) |
| `config.yaml`, `env.json` | Cấu hình đầy đủ đã dùng; phiên bản Python/PyTorch/CUDA, tên GPU |
| `data_summary.json`, `model.txt` | Số mẫu từng lớp mỗi split, mean/std chuẩn hóa; cấu trúc mô hình + số tham số |
| `steps.csv` | Mỗi N bước: `loss`, `batch_acc`, `lr`, `grad_norm` (tổng), `loss_scale` (AMP) |
| `epochs.csv` | Mỗi epoch: `train_loss/acc`, `val_loss/acc/macro_f1/bal_acc`, `lr`, `grad_norm_mean`, thời gian |
| `per_class.csv` | Mỗi epoch, mỗi lớp: precision / recall / F1 / support (trên val) |
| **`params.csv`** | **Mỗi epoch, mỗi tensor tham số**: xem bên dưới |
| `tb/` | TensorBoard: scalar + histogram trọng số và gradient |
| `results.json` | Kết quả cuối (val, test): acc, macro-F1, từng lớp, theo nguồn data (nhánh tiếng) |
| `confusion_{val,test}.{csv,png}` | Ma trận nhầm lẫn |
| `probs_*.npy`, `labels_*.npy` | Xác suất dự đoán, dùng cho `fuse_eval.py` |
| `best.pt`, `last.pt` | Checkpoint tốt nhất / mới nhất (để resume) |

**`params.csv` — cột nào nói lên điều gì**

| Cột | Ý nghĩa |
|---|---|
| `param`, `shape`, `numel` | Tên (vd `backbone.layer1.0.conv1.weight`), kích thước, số phần tử |
| `w_norm`, `w_mean`, `w_std`, `w_min`, `w_max` | Thống kê trọng số cuối epoch |
| `delta_norm` | ‖w_epoch − w_epoch_trước‖ : trọng số đã dịch chuyển bao nhiêu |
| `update_ratio` | `delta_norm / ‖w_trước‖` : **mức thay đổi tương đối**. Rất gần 0 → lớp gần như đứng yên; rất lớn (vài chục %) → không ổn định |
| `grad_norm_mean`, `grad_norm_max` | Norm gradient của tensor đó (lấy mẫu mỗi `param_every_steps` bước, đã unscale AMP, trước khi cắt gradient) |

Vẽ đồ thị (đường cong train/val, heatmap norm gradient / update_ratio / norm trọng số theo epoch × lớp):
```bash
python scripts/plot_logs.py --run-dir runs/face_resnet18_<thời gian>      # ghi vào <run>/plots
tensorboard --logdir runs                                                # xem histogram
```

## 6. Các điểm cần lưu ý về data

1. **Mất cân bằng lớp nặng** (vd disgust, boredom, excited rất ít). Code dùng loss có trọng số lớp (`train.class_weight: inv_sqrt`) và chọn mô hình theo **macro-F1**, không theo accuracy. Hãy báo cáo macro-F1 và recall từng lớp.
2. **Nguồn data gắn với nhãn (nhánh tiếng)**: nhãn 7, 8, 9 chỉ có ở bộ ASVP; disgust chỉ có ở CREMA/RAVDESS. Mô hình có thể học "bộ nào thu âm" thay vì cảm xúc. Vì vậy `results.json` có **kết quả tách theo nguồn**; nên lấy **7 lớp làm kết quả chính** và coi 10 lớp là phần mở rộng.
3. **Mel đã ở thang dB (-80..0)**: code chuẩn hóa theo mean/std của tập train (lưu trong checkpoint).
4. Chia tập tiếng theo **người nói** (không người nói nào nằm ở hai tập) → số liệu đáng tin hơn.

## 7. Late fusion và hạn chế (ghi vào báo cáo)

Ảnh mặt và âm thanh **không thuộc cùng một clip**. `fuse_eval.py` ghép cặp ngẫu nhiên một mẫu mặt và một mẫu tiếng **cùng nhãn**, lặp nhiều seed, chọn trọng số `w` trên val, báo cáo trên test (trung bình ± độ lệch chuẩn). Vì hai nhánh mắc lỗi độc lập (khác người), lợi ích của fusion có thể **cao hơn** so với data ghép cặp thật. Muốn kết luận chắc hơn cần data audio-video cùng clip (vd CREMA-D bản video).

## 8. Gợi ý thí nghiệm (ablation) cho báo cáo

| Thí nghiệm | Lệnh `--set` |
|---|---|
| 7 vs 10 lớp | `num_classes=7` / `num_classes=10` |
| Có / không trọng số lớp | `train.class_weight=inv_sqrt` / `none` |
| Pretrained hay không (mặt) | `model.pretrained=true` / `false` |
| Ảnh lớn hơn (mặt) | `model.input_size=96` hoặc `model.remove_maxpool=true` |
| Có / không SpecAugment (tiếng) | `augment.freq_mask=0 augment.time_mask=0` |
| Fusion số học / hình học | `fuse_eval.py --method arithmetic` / `geometric` |

## 9. Xử lý sự cố

- **Hết bộ nhớ GPU**: giảm `train.batch_size` (vd 64 → 32).
- **Không tải được trọng số ImageNet** (không có mạng): code cảnh báo và train từ khởi tạo ngẫu nhiên; hoặc đặt `model.pretrained=false`.
- **`Không tìm thấy ... dưới data_root`**: `data_root` phải là thư mục *chứa* thư mục `data`.
- **Thiếu TensorBoard**: log CSV vẫn được ghi; `pip install tensorboard` để có thêm histogram.

## 10. Trạng thái kiểm thử (trung thực)

- Đã chạy kiểm thử trên data giả: đọc data (mặt/tiếng, cache, đường dẫn), chuẩn hóa, trọng số lớp, **chỉ số (khớp scikit-learn)**, **fuse_eval**, vẽ đồ thị log, kiểm tra cú pháp toàn bộ.
- **Chưa chạy thử vòng lặp train PyTorch** (mô hình, DataLoader, AMP, ghi log tham số) vì môi trường tạo gói không có PyTorch. Hãy chạy `bash scripts/smoke_test.sh` trước tiên; nếu có lỗi, gửi nội dung lỗi để sửa.
