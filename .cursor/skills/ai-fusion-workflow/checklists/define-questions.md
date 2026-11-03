# Bộ câu hỏi bắt buộc trước khi code

Trả lời đầy đủ và ghi vào `docs/progress_status.md` mục "Project Context" trước khi viết bất kỳ dòng code nào.
Mapping: mục 0 → `project name`, mục 1 → `problem` + `target users` + `input / output`, mục 2 → `core features`, mục 3–4 → `technology criteria`.

## 0. Project Name — Tên project là gì?

- Tên đầy đủ của project là gì?
- Slug ngắn để đặt thư mục là gì? (ví dụ `lab1`, `face-seg`)
- Tagline một câu mô tả project làm gì?

## 1. What? — Bài toán, người dùng, input / output là gì?

### Problem
- Bài toán cụ thể cần giải là gì? (phân loại, phát hiện, phân đoạn, dự đoán…)
- Tại sao bài toán này quan trọng? (giá trị thực tế, ai đang đau)

### Target users
- Người dùng mục tiêu là ai? (vai trò, trình độ kỹ thuật)
- Họ dùng trong bối cảnh nào? (lab, bệnh viện, nhà máy, học tập…)
- Họ kỳ vọng gì ở output? (xem trên UI, nhận file, gọi API…)

### Input / Output
- Dữ liệu đầu vào là gì? (hình ảnh, văn bản, bảng số, âm thanh… kèm shape / định dạng)
- Output mong đợi là gì? (label, bounding box, mask, số, xác suất… kèm shape / định dạng)
- Dữ liệu lấy từ đâu? Có sẵn hay phải thu thập?
- Mô tả dữ liệu: bao nhiêu mẫu, bao nhiêu lớp, có cân bằng không?
- Chia train / validation / test thế nào? (ví dụ 70/15/15)

## 2. Core Features — Tính năng cốt lõi là gì?

Liệt kê rồi phân loại Must / Should / Could, mỗi tính năng kèm lý do một dòng.

- Must (bắt buộc có, không có thì project thất bại): <tính năng> — vì sao?
- Should (nên có, có thì đạt điểm cao / dùng tốt hơn): <tính năng> — vì sao?
- Could (có thì tốt, để tương lai): <tính năng> — vì sao?
- Tính năng nào nằm ngoài phạm vi (explicitly out of scope)?

## 3. Why? — Tại sao chọn cách này?

- Tại sao chọn kiến trúc model này? Có model nào khác đã thử và vì sao không chọn?
- Tại sao chọn feature / input này? Có lựa chọn nào đã cân nhắc?
- Tại sao chọn tech stack (framework, thư viện)?
- Luôn kèm lý do rõ ràng, không chọn vì "thấy người ta dùng".

## 4. Where / When / Which + Technology Criteria — Chạy ở đâu, khi nào, bằng gì, theo tiêu chí nào?

- **Where** — chạy ở đâu? (cloud, edge, local GPU, CPU-only, mobile…)
- **When** — khi nào chạy? (real-time, batch, mỗi giờ, mỗi ngày…)
- **Which** — dùng framework / thư viện / công cụ nào?
  - Deep learning: PyTorch, TensorFlow, JAX?
  - Vision: torchvision, Albumentations, OpenCV?
  - NLP: transformers, spaCy?
  - Tabular: scikit-learn, XGBoost, LightGBM?
  - Tracking experiment: MLflow, Weights & Biases, TensorBoard?
- **Technology criteria** — tiêu chí chọn công nghệ là gì?
  - Ràng buộc phần cứng? (VRAM, CPU-only, edge device…)
  - Ngân sách hiệu năng? (latency tối đa, throughput tối thiểu, RAM tối đa)
  - Giấy phép / chi phí? (open-source, bản quyền, phí cloud…)
  - Kỹ năng nhóm? (nhóm đã quen gì, có thời gian học cái mới không?)
  - Khả năng deploy / bảo trì? (export ONNX, Docker, API…)

## 5. How? — Làm thế nào?

- Dữ liệu chảy qua pipeline thế nào? (raw → processed → train → predict → output)
- Model sẽ train và deploy ra sao?
- Hyperparameter chính là gì? (learning rate, batch size, epoch, optimizer…)
- Làm sao đánh giá hiệu năng? Metric gì, threshold bao nhiêu?
- Có cần data augmentation không? Loại nào?
- Pre-processing input: chuẩn hóa, resize, normalize?
- Post-processing output: NMS, threshold, format lại file?

## 6. Evaluation — đo lường thế nào?

### Model level
- Accuracy, precision, recall, F1 (cho classification).
- mAP@0.5, mAP@0.5:0.95 (cho detection).
- IoU, Hausdorff distance (cho segmentation).
- Inference time per sample.
- Kích thước model, RAM cần thiết.

### Full pipeline
- Latency end-to-end (preprocess + inference + postprocess).
- Throughput (request/giây).
- Tài nguyên sử dụng (CPU, GPU, RAM, disk).
- Xử lý lỗi và phục hồi.
- Trải nghiệm người dùng (nếu có UI).

### Tiêu chí đạt / không đạt
- Ví dụ: Accuracy ≥ 0.90, Latency ≤ 50 ms, mAP ≥ 0.85.

## 7. Score / Điểm số mong đợi (nếu là bài tập / cuộc thi)

- Tổng điểm tối đa của bài?
- Mỗi tính năng được mấy điểm?
- Điểm cho solution (tech + AI)?
- Điểm cho implementation?
- Điểm cho evaluation (model + pipeline)?

## Lưu ý

- Trả lời càng cụ thể càng tốt — số liệu, tên thư viện, tên metric phải rõ ràng.
- Nếu chưa biết, ghi "TBD" và liệt kê trong mục "Open Questions" của `docs/progress_status.md`.
- Sau khi trả lời đủ mục 0–4 (project name, problem, target users, input / output, core features, technology criteria), mới chuyển sang giai đoạn Design.
