# FruitLite-L — Deep Learning Midterm

Phân loại 5 lớp: **apple, banana, orange, strawberry, pineapple**.
Đã có hai bộ ảnh trong `DataNew`, nhưng **chưa đủ dữ liệu và chưa có kết quả
train/test chính thức để nộp bài**.

Đã đọc [đề giữa kỳ](Midterm%20exam.docx) và kiểm tra trực tiếp toàn bộ ảnh ngày
**19/10/2026**. Xem [đối chiếu yêu cầu](results/requirements_audit.md),
[số liệu kiểm tra](results/dataset_audit.json) và
[40 ảnh mẫu](results/dataset_preview.jpg).

## Dữ liệu hiện có

| Lớp | Train | Test | Thiếu so với 5.000 train |
|---|---:|---:|---:|
| apple | 5.000 | 50 | 0 |
| banana | 4.480 | 50 | 520 |
| orange | 5.000 | 50 | 0 |
| strawberry | 4.888 | 50 | 112 |
| pineapple | 2.685 | 50 | 2.315 |
| **Tổng mỗi độ phân giải** | **22.053** | **250** | **2.947** |

- **Mid224:** 22.303 ảnh AVIF, RGB, 224×224; **20,425389 MB / 25 MB**.
- **Mid32:** 22.303 ảnh WebP, RGB, 32×32; **13,071104 MB / 20 MB**.
- Cả hai bộ có cùng định danh và split, tổng **44.606 file ảnh**. Đã giải mã toàn
  bộ; đúng kích thước/định dạng, không thiếu/thừa theo manifest, không trùng hoàn
  toàn theo byte hoặc pixel trong từng bộ. Tính MB theo 1.000.000 byte.
- Cần duyệt tiếp nhãn, chất lượng nén và nhóm cùng vật thể/phiên chụp. Hồ sơ cũ có
  16.000 định danh mang cờ tương đồng chưa giải quyết; đây không phải số ảnh đã
  được xác nhận trùng. Quyền sử dụng và tính độc lập của nguồn cũ chưa xác minh hết.

Đề ghi **khoảng 5.000** train/lớp. Mốc 5.000 trong bảng dùng để đo phần thiếu;
không phải yêu cầu phải có chính xác 5.000 hoặc bắt buộc chia validation theo tỷ lệ
cụ thể. Không dùng augmentation hoặc các bản resize làm mẫu gốc mới để lấp số lượng.

## Cấu trúc

| Đường dẫn | Nội dung |
|---|---|
| `Midterm exam.docx` | Đề gốc |
| `DataNew/Mid224`, `DataNew/Mid32` | Hai bộ ảnh hiện tại, chưa hoàn tất |
| `DataNew/*.json` | Hồ sơ nguồn và kiểm duyệt lần xuất gần nhất |
| `data/fruit_sources_08/originals`, `paired` | Ảnh nguồn và PNG ứng viên phục vụ bổ sung/xuất lại |
| `data/metadata/fruit_acquisition_08` | Journal thu thập, quyết định duyệt, loại bỏ và nguồn |
| `data/metadata/sources` | Metadata nguồn, giấy phép và inventory đã chuyển khỏi thư mục kế hoạch cũ |
| `data/metadata/dataset_history` | Bằng chứng kiểm tra, cờ gần trùng và mẫu bổ sung chưa được chấp nhận |
| `data/metadata/compression_study_*` | Metadata thử nghiệm, ảnh nguồn mẫu và các development reservations cần giữ |
| `src/`, `configs/`, `tests/` | Mã thu thập/xử lý, mô hình, train/evaluate và kiểm thử |
| `results/` | Manifest, báo cáo kiểm tra và biên nhận dọn project |

Ảnh nguồn, ứng viên và hai bộ dữ liệu nằm ngoài Git theo `.gitignore`.
[Manifest](results/dataset_manifest.csv) dùng cột `output` dạng tên `.jpg` lịch sử;
khi đối chiếu với file thật, dùng cùng stem với `.avif` cho Mid224 và `.webp` cho Mid32.
Các ứng viên trong `data` và lịch sử kiểm tra chưa được tính thêm vào bảng số lượng.

## Mô hình và việc còn thiếu

FruitLite-L là CNN tự ghép từ convolution, depthwise 3×3/5×5, residual projection
và channel attention. **408.179 learnable parameters**; profiling batch 1 inference:
**587.091.697 FLOPs** ở 224 và **12.099.313 FLOPs** ở 32. Quy ước trong
`src/profile.py`: `2 × MACs + scalar operations`. Cần đối chiếu kiến trúc và dữ liệu
với các nhóm khác theo lưu ý trong đề.

**Loader hiện chưa dùng được DataNew:** `src/train.py` và audit chính thức đọc
`data/metadata/splits.csv` cùng ảnh PNG trong `data/Mid224` / `data/Mid32`;
các đường dẫn đó chưa tồn tại. Cần hoàn tất dữ liệu, duyệt nhóm nguồn, đóng băng
split rồi thống nhất loader với AVIF/WebP trước khi chạy huấn luyện thật.

Hai config là thiết lập khởi đầu: SGD momentum 0,9, LR 0,05, weight decay 0,0005,
warmup 5 epoch; Mid224 batch 64 / 150 epoch, Mid32 batch 256 / 200 epoch. Đây chưa
phải cấu hình đã được kiểm chứng trên dữ liệu cuối. Còn thiếu log từng epoch,
Top-1/test của cả hai bộ và báo cáo PDF. Kết quả smoke/pilot lịch sử chỉ kiểm tra
kỹ thuật, không phải độ chính xác chính thức.

## Chạy kiểm tra

Chỉ dùng Microsoft Store Python 3.13 đã chọn; không tạo môi trường Python khác.
Từ thư mục project, kiểm tra mô hình bằng:

```powershell
& 'C:\Users\anhna\AppData\Local\Microsoft\WindowsApps\python3.13.exe' -X utf8 -m src.profile
```

Chạy bộ test với giới hạn luồng trong riêng tiến trình để kiểm tra trên Windows:

```powershell
& 'C:\Users\anhna\AppData\Local\Microsoft\WindowsApps\python3.13.exe' -X utf8 -c "import os; os.environ['OMP_NUM_THREADS']='1'; os.environ['MKL_NUM_THREADS']='1'; import torch, pytest; torch.set_num_threads(1); torch.set_num_interop_threads(1); raise SystemExit(pytest.main(['-q', '-p', 'no:cacheprovider', 'tests']))"
```

Đã dọn thư mục kế hoạch cũ, bản sao backup/staging, checkpoint smoke, cache,
log worker và ảnh thử codec có thể tạo lại. Ảnh hiện tại, ảnh nguồn, PNG ứng viên,
journal/approval, development reservations và bằng chứng kiểm duyệt cần thiết được
giữ lại. Chi tiết trong [biên nhận dọn project](results/project_cleanup.json);
lịch sử đường dẫn cũ và mới được lưu trong
`data/metadata/dataset_history/relocations.json`. Các đường dẫn backup/preview cũ
trong hồ sơ lịch sử không còn là file đang tồn tại.
