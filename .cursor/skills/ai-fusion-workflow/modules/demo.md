# Demo — Chạy thử và trình diễn

Chuẩn làm demo chạy được end-to-end: một lệnh là thấy input → output, có đo latency, có kịch bản trình bày.

## 1. Mục tiêu
- Người mới clone repo, làm theo README là chạy được demo trong dưới 5 phút (không tính tải model nặng).
- Demo chứng minh đúng `core features` mức Must đã chốt trong Define.

## 2. Nội dung bắt buộc
- **Kịch bản demo**: 2-3 case đại diện (case dễ, case khó, case biên), mỗi case có input mẫu trong `data/processed/demo/` và output kỳ vọng.
- **Cách chạy 1 lệnh**: script `python -m src.demo --input <path> --checkpoint <path>` hoặc notebook `notebooks/demo.ipynb` chạy từ đầu đến cuối không lỗi.
- **Giao diện (nếu có)**: Gradio / Streamlit / API FastAPI tối giản — ghi rõ cách chạy, port, ví dụ request/response.
- **Đo live**: hiển thị inference time / mẫu và latency end-to-end ngay trong demo output.
- **Example Input → Output**: khối JSON / ảnh before-after để dán vào README mục 7.
- **Giới hạn demo**: ghi rõ demo chưa xử lý gì (ví dụ chưa batch lớn, chưa video realtime).

## 3. Template file
- `src/demo.py` hoặc `src/demo/__main__.py` — load config + checkpoint, chạy preprocess → inference → postprocess, lưu output vào `results/demo/<ts>/`.
- `notebooks/demo.ipynb` — bản trình diễn có hình ảnh, chạy theo thứ tự từ trên xuống dưới không cần sửa code.
- `data/processed/demo/` — 3-5 file input mẫu nhẹ (< 5 MB) được commit kèm repo.
- `results/demo/<ts>/` — output demo + `metrics.json` mini (latency, confidence).

## 4. Checklist review
- [ ] Fresh clone + cài requirements + 1 lệnh là chạy được demo?
- [ ] Input mẫu có kèm trong repo, không phụ thuộc đường dẫn máy cá nhân?
- [ ] Output demo có khớp định dạng đã hứa trong Design (`Interfaces / APIs`)?
- [ ] Có hiển thị latency / confidence, không chỉ ảnh đẹp?
- [ ] Có case khó / case thất bại để chứng minh trung thực, không chỉ case đẹp?
- [ ] README mục Example Input → Output đã đồng bộ với output demo thật?
- [ ] Có ghi Limitations của demo (chưa realtime, chỉ CPU/GPU nào)?

## 5. Rủi ro thường gặp
- Demo chạy trên máy dev nhưng sang máy khác thiếu dependency / sai path → dùng relative path, ghim version trong `requirements.txt`, test trên môi trường sạch.
- Checkpoint quá nặng không gửi được → dùng LFS / link tải + script `download_model.py`, không commit file > 100 MB trực tiếp.
- Demo chỉ chạy case đẹp → người xem mất tin khi thử case thật → chuẩn bị sẵn case khó và giải thích giới hạn.
- Không đo latency live → không chứng minh được Technology criteria → luôn in `inference_time_ms` và `end_to_end_ms` trong output demo.

## 6. Prompt mẫu
```
Bạn là AI Engineer. Hãy viết script demo:
- Input: ảnh / file trong data/processed/demo/.
- Load checkpoint <path> + configs/eval_config.yaml.
- Chạy full pipeline preprocess → inference → postprocess.
- Lưu output vào results/demo/<timestamp>/ kèm metrics mini (latency, confidence).
- In 1 khối Example Input → Output để tôi dán vào README.
- Toàn bộ path là relative, 1 lệnh python -m src.demo là chạy được.
```
