# Kiểm tra yêu cầu giữa kỳ — 20/10/2026

Đề yêu cầu ít nhất 5 lớp, khoảng 5.000 train + 50 test/lớp, hai bộ 224×224 và
32×32, dung lượng tối đa 25 MB và 20 MB, mô hình không quá 6M tham số và dưới 1G
FLOPs.

## Kết quả kỹ thuật cuối

| Lớp | Train | Test |
|---|---:|---:|
| apple | 5.000 | 50 |
| banana | 5.000 | 50 |
| orange | 5.000 | 50 |
| strawberry | 5.000 | 50 |
| pineapple | 5.000 | 50 |
| **Tổng mỗi bộ** | **25.000** | **250** |

- Mid224: **25.250 file AVIF RGB 224×224**, 23.148.133 byte = 23,148133 MB,
  dưới 25.000.000 byte.
- Mid32: **25.250 file WebP RGB 32×32**, 14.948.274 byte = 14,948274 MB,
  dưới 20.000.000 byte.
- Audit đã giải mã toàn bộ 50.500 file, kiểm tra hash manifest, file dư/thiếu,
  cặp hai độ phân giải, trùng encoded/pixel và giao nhóm train/test.
- Model chạy chính FruitLite-L2: **1.250.879 learnable parameters**;
  **860.413.737 FLOPs ở 224** và **17.973.033 ở 32**. Đạt giới hạn mô hình theo
  profiler hiện có. Variant legacy 408.179 tham số vẫn đọc được checkpoint cũ.
- Chạy `train.py` ở thư mục gốc để audit, chia validation, train development,
  refit đủ 25.000 ảnh và test cả hai kích thước. `last.pt`/`best.pt` cập nhật tại
  chỗ; checkpoint đủ trạng thái resume; log batch/epoch, chỉ số và biểu đồ lưu
  trong `outputs/fruitlite_l2_seed2026/`.
- Bằng chứng hạ tầng, profile, loader và kiểm thử:
  [training_readiness.json](training_readiness.json). Chưa có accuracy chính thức.

## Nguồn bổ sung

Để bù phần thiếu cũ, bộ kỹ thuật thêm 2.947 ảnh train: 520 banana từ
`luischuquim/BananaRipeness`, 112 strawberry gốc từ Zenodo
`10.5281/zenodo.20415464`, và 2.315 pineapple từ PQC500 của `ncyuMARSLab`.
Các ảnh được resize trực tiếp từ ảnh gốc, không dùng augmentation để tăng quota.
Mỗi nguồn được giữ `source_identity`, URL, author, group ID và hash trong
[final_input_manifest.csv](final_input_manifest.csv).

## Các điều kiện còn phải hoàn tất trước khi nộp

Audit kỹ thuật không đồng nghĩa dữ liệu đã đạt chất lượng khoa học/pháp lý cuối:

- Banana và PQC500 chưa có SPDX license xác nhận trong metadata hiện có; manifest
  đặt `rights_verified=false`.
- Nguồn bổ sung chưa được duyệt thủ công toàn bộ nhãn và chất lượng; manifest đặt
  `photo_verified=false`.
- Cờ near-duplicate và tính độc lập vật thể của bộ legacy chưa được review hết.
- Chưa có training/test chính thức, log từng epoch, Top-1, confusion matrix và PDF.
- Chưa đối chiếu kiến trúc/dataset với các nhóm khác.

Trạng thái chính xác hiện tại là **technical_quota_complete_rights_and_full_visual_review_pending**.
Không dùng số liệu smoke, CLIP screening hoặc proxy score làm accuracy chính thức.
