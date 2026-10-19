# FruitLite-L — Deep Learning Midterm

Phân loại 5 lớp: **apple, banana, orange, strawberry, pineapple**.

Bộ dữ liệu kỹ thuật hiện đã đủ quota đầu vào của đề: **5.000 train + 50 test cho
mỗi lớp**, dùng chung cho Mid224 và Mid32. Audit cuối nằm trong
[final_input_audit.json](results/final_input_audit.json), manifest đầy đủ ở
[final_input_manifest.csv](results/final_input_manifest.csv), và biên bản nguồn ở
[input_source_review.json](results/input_source_review.json).

## Trạng thái dữ liệu

| Lớp | Train | Test |
|---|---:|---:|
| apple | 5.000 | 50 |
| banana | 5.000 | 50 |
| orange | 5.000 | 50 |
| strawberry | 5.000 | 50 |
| pineapple | 5.000 | 50 |
| **Tổng mỗi độ phân giải** | **25.000** | **250** |

- **Mid224:** 25.250 AVIF, RGB, 224×224; **23,148133 MB / 25 MB**.
- **Mid32:** 25.250 WebP, RGB, 32×32; **14,948274 MB / 20 MB**.
- Kiểm tra toàn bộ **50.500 file**: đúng cặp, đúng hình học/định dạng, đủ manifest,
  không trùng encoded/decode pixel trong bundle, group ID không giao giữa train và
  test.
- Bộ mới bổ sung 520 banana từ `luischuquim/BananaRipeness`, 112 strawberry gốc
  từ Zenodo `10.5281/zenodo.20415464`, và 2.315 ảnh pineapple từ PQC500 của
  `ncyuMARSLab`. Không dùng augmentation để lấp quota.

**Điều kiện kỹ thuật đã đạt; điều kiện nộp bài chưa hoàn tất.** Quyền sử dụng của
banana/PQC500 chưa có SPDX license xác nhận, và toàn bộ nhãn, nhóm vật thể, near
duplicate cần review thủ công. Manifest đánh dấu `rights_verified=false`,
`photo_verified=false` cho các nguồn bổ sung. Không coi bộ này là đã sẵn sàng nộp
cho đến khi hoàn tất các review đó.

Đề gốc: [Midterm exam.docx](Midterm%20exam.docx). Báo cáo đối chiếu chi tiết:
[requirements_audit.md](results/requirements_audit.md). Ảnh mẫu nguồn bổ sung và
screening chỉ là bằng chứng review, không phải ground truth.

## Cấu trúc và pipeline

| Đường dẫn | Nội dung |
|---|---|
| `DataNew/Mid224`, `DataNew/Mid32` | Bundle hiện hành, ngoài Git theo `.gitignore` |
| `data/metadata/splits.csv` | Manifest loader hiện dùng |
| `data/metadata/build_settings.json` | Layout `paired-compressed-v1`, giới hạn MB |
| `results/final_input_manifest.csv` | Bản manifest có thể audit/đối chiếu |
| `results/input_validation.json` | Kết quả audit quota, hình học, hash, dung lượng |
| `results/input_source_review.json` | Nguồn bổ sung và trạng thái rights/visual review |
| `src/input_dataset.py` | Audit AVIF/WebP và loader path validation |
| `src/promote_input_dataset.py` | Quy trình promote fail-closed cho nguồn đã screening |
| `src/`, `configs/`, `tests/` | Mã thu thập, mô hình, train/evaluate và kiểm thử |

`src/train.py` đọc trực tiếp `DataNew` qua `mid224_path`/`mid32_path`; không còn
phụ thuộc ảnh PNG giả định trong `data/Mid224`.

FruitLite-L có **408.179 learnable parameters**; profiling batch 1 cho **587.091.697
FLOPs ở 224** và **12.099.313 FLOPs ở 32**. Hai giá trị đạt giới hạn đề theo quy
ước `src/profile.py`.

## Kiểm tra

Chỉ dùng Microsoft Store Python 3.13 đã chọn:

```powershell
& 'C:\Users\anhna\AppData\Local\Microsoft\WindowsApps\python3.13.exe' -X utf8 -B -m src.input_dataset audit
```

Bộ test hiện có: **150 passed** với OMP/MKL và PyTorch intra/inter-op threads = 1.
Chưa chạy huấn luyện chính thức, chưa có Top-1 theo epoch và chưa có PDF cuối.
