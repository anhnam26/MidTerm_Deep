# FruitLite-L2 — Deep Learning Midterm

Phân loại 5 lớp: **apple, banana, orange, strawberry, pineapple**.

Bộ dữ liệu hiện có **5.000 train + 50 test cho mỗi lớp**, dùng chung cho Mid224 và
Mid32. Đề ghi **khoảng 5.000** ảnh train/lớp; đúng 5.000 là mục tiêu triển khai.
Audit cuối nằm trong
[final_input_audit.json](results/final_input_audit.json), manifest đầy đủ ở
[final_input_manifest.csv](results/final_input_manifest.csv), và biên bản nguồn ở
[input_source_review.json](results/input_source_review.json).

## Bắt đầu train

Chạy duy nhất **[train.py](train.py)** ở thư mục gốc, bằng Microsoft Store Python
3.13 đã chọn trong IDE. Hoặc chạy:

```powershell
& 'C:\Users\anhna\AppData\Local\Microsoft\WindowsApps\python3.13.exe' -X utf8 -B train.py
```

Không cần chuẩn bị thêm split hay chạy script khác. Chương trình dùng CUDA khi
có, kiểm tra toàn bộ ảnh và giới hạn model, sau đó:

1. Chia cố định 22.500 ảnh train / 2.500 validation, giữ nguyên 250 ảnh test.
2. Train development Mid224 (180 epoch), chọn `best` bằng validation Top-1,
   hòa Top-1 thì chọn loss thấp hơn; so sánh trọng số thường và EMA.
3. Train lại từ đầu trên đủ 25.000 ảnh với số epoch và loại trọng số đã chọn.
4. Thực hiện tương tự cho Mid32 (220 epoch development).
5. Chỉ sau khi cả hai lượt train xong mới đánh giá test và xuất kết quả.

Đây là recipe khởi đầu nhằm tối ưu accuracy, chưa phải cấu hình đã được xác nhận
bằng kết quả train. Không tải pretrained weights. Test không tham gia chọn epoch,
hyperparameter hoặc `best`. Không đổi cấu hình sau khi xem test để rồi diễn giải
lại test đó như một phép đánh giá độc lập.

**Dừng và tiếp tục:** nhấn Ctrl+C; chạy lại chính `train.py`. Chương trình phục hồi
model, optimizer, AMP scaler, EMA, RNG và lịch sử từ epoch hoàn tất gần nhất.
Màn hình ghi rõ `RESUME ... epoch=10 -> next_epoch=11`. Chỉ epoch đang dang dở
chạy lại; các epoch đã hoàn tất không train lại. Checkpoint đầu tiên có sau
epoch 1; nếu dừng trước đó thì chưa có trạng thái để resume. Nếu mất `last.pt`
nhưng còn lịch sử/checkpoint của lượt cũ, chương trình báo lỗi để tránh tự train
lại từ đầu.

Audit giải mã đầy đủ chỉ chạy khi chưa có cache hợp lệ. Những lần sau kiểm tra
inventory và SHA256 **tất cả file ảnh**, rồi dùng lại audit đã lưu nếu ảnh,
manifest và quy tắc kiểm tra giữ nguyên. Không chỉ dựa vào timestamp/kích thước,
không giải mã lại 50.500 ảnh. Lần đầu sau bản cập nhật này cần tạo cache một lần.
Muốn ép audit đầy đủ: thêm `--full-audit`.

Nếu đổi mã nguồn/cấu hình/data, dùng tên lượt mới:

```powershell
& 'C:\Users\anhna\AppData\Local\Microsoft\WindowsApps\python3.13.exe' -X utf8 -B train.py --run-name experiment_02
```

Chỉ kiểm tra đầu vào và loader, chưa train:

```powershell
& 'C:\Users\anhna\AppData\Local\Microsoft\WindowsApps\python3.13.exe' -X utf8 -B train.py --check --run-name input_check
```

Có thể chỉnh `configs/mid224.json`, `configs/mid32.json`. CLI có `--epochs`,
`--batch-size`, `--workers`, `--device`, `--sizes`; CLI ghi đè JSON. Mặc định
4 workers; dùng `--workers 0` khi cần xử lý lỗi multiprocessing Windows.
Worker validation được tái sử dụng giữa các lượt đánh giá để giảm thời gian
khởi động trên Windows; worker train khởi tạo theo seed của mỗi epoch.

## Checkpoint và log

Màn hình chỉ hiện **một dòng tổng kết sau mỗi epoch hoàn tất**: loss, train/
validation/EMA accuracy, best accuracy, LR và thời gian. Thông báo khởi động,
dừng/lỗi và kết quả test vẫn hiện; chi tiết batch và từng lớp chỉ ghi trong file.

Kết quả mặc định nằm trong `outputs/fruitlite_l2_seed2026/`, ngoài Git.
Mỗi `Mid224` / `Mid32` có hai thư mục `development` và `final`:

| File | Nội dung |
|---|---|
| `last.pt` | Ghi thay thế sau mỗi epoch train + validation hoàn tất; đủ trạng thái resume |
| `best.pt` | Một file cố định chứa trọng số được chọn, không tạo file checkpoint theo số epoch |
| `settings.json` | Toàn bộ hyperparameter, kiến trúc, profile, môi trường, seed, hash code/data/split |
| `console.log` | Log đầy đủ: batch, ETA, tốc độ, loss, accuracy, P/R/F1 từng lớp và các dòng tổng kết màn hình |
| `batches.jsonl` | Mỗi batch: loss, running accuracy, LR, gradient norm, AMP scale, mixup, timing, attempt |
| `epochs.csv`, `epochs.json` | Mọi epoch hoàn tất: train/val/EMA loss và Top-1, precision/recall/F1, AUROC, ECE, Brier, confusion matrix, GPU memory, timing |
| `learning_curves.png` | Đường loss và accuracy sau khi xong mỗi giai đoạn |
| `summary.json` | Trạng thái, epoch tốt nhất và cách chọn trọng số |
| `final/test/` | `metrics.json`, xác suất từng ảnh trong `predictions.csv`, confusion matrix và log test |

Trong `development`, `best` giữ validation tốt nhất từ đầu đến hiện tại. Trong
`final`, không có validation vì dùng đủ train pool: `best.pt` cập nhật trọng số
thường/EMA đã khóa, và dùng **epoch cuối đã chọn trước** để test; không chọn bằng
training accuracy. Checkpoint ghi tạm rồi thay thế; `last.pt` là nguồn phục hồi
CSV/JSON/`best.pt` nếu bị ngắt trong lúc xuất file. Chỉ có `last.pt` và `best.pt`
trong mỗi thư mục train, không tích lũy hàng trăm checkpoint.

`workflow.json`, `development_split.csv`, `preflight.json`, `input_validation.json`
và bản sao code trong `source/` ở gốc lượt chạy lưu bằng chứng cấu hình. Batch
thuộc epoch bị ngắt vẫn giữ trong JSONL kèm `attempt`; chỉ `epochs.*` và checkpoint
đại diện các epoch đã hoàn tất. Test hoàn tất được nhận diện khi chạy lại và
không suy luận lại.

## Kiến trúc và recipe

FruitLite-L2 dùng một đồ thị cho cả hai độ phân giải. Mỗi block mở rộng kênh,
chia nhánh depthwise 3×3/5×5, ghép kênh, attention ở các stage cuối, chiếu 1×1 và
residual shortcut. Phần cuối được tăng chiều rộng/độ sâu để dùng phần FLOPs còn
lại. Đây là một tổ hợp kiến trúc tự cấu hình; chưa xác nhận tính độc nhất so với
các nhóm sinh viên khác.

| Phần | Kênh đầu ra | Số block | Stride block đầu |
|---|---:|---:|---:|
| Stem 3×3 | 16 | 1 | 2 |
| Stage 1 | 32 | 1 | 1 |
| Stage 2 | 56 | 2 | 2 |
| Stage 3 | 96 | 3 | 2 |
| Stage 4 | 240 | 4 | 2 |
| Head 1×1 + GAP + dropout + linear | 384 → 5 | 1 | 1 |

**1.250.879 learnable parameters**, **860.413.737 FLOPs ở 224**, **17.973.033 ở 32**,
batch 1 theo `src/profile.py`: `2 × MACs` cộng BN/activation/pooling/gate/residual.
Giới hạn được kiểm tra trước từng lượt train: ≤6M tham số, <1G FLOPs.

- AdamW: LR 0,0015; betas (0,9; 0,999); weight decay 0,01 cho weight, không decay
  bias/BN; warmup 5 epoch, cosine về 1% LR ban đầu.
- Batch 64 ở Mid224, 256 ở Mid32; seed 2026; AMP float16 trên CUDA; clip gradient
  norm 5; EMA tối đa 0,999 có warmup.
- Cross-entropy với label smoothing 0,05; dropout 0,2; mixup alpha 0,2 với xác suất
  0,25; crop/flip/rotation/color jitter và random erasing nhẹ.
- Accuracy train đo trên ảnh đã augmentation, nhãn gốc khi có mixup; validation
  và test không augmentation. Loss train có smoothing/mixup, loss val/test là CE
  thường nên không so trực tiếp như hai loss cùng định nghĩa.
- Các ảnh cùng specimen PQC500 không bị chia qua train/validation. Có **22.685**
  ảnh train chưa xác minh nhóm vật thể; fallback chia theo source identity, vì thế
  validation chưa chứng minh độc lập vật thể/near duplicate cho dữ liệu legacy.

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
| `train.py` | Điểm chạy chính: audit → development → refit → test cả hai kích thước |
| `src/training_*.py` | Chia validation, lưu trạng thái, metrics, điều phối |
| `src/`, `configs/`, `tests/` | Mã thu thập, mô hình và kiểm thử |

`src/train.py` đọc trực tiếp `DataNew` qua `mid224_path`/`mid32_path`; không còn
phụ thuộc ảnh PNG giả định trong `data/Mid224`.

Variant FruitLite legacy (408.179 tham số) còn được giữ để đọc checkpoint cũ và
kiểm tra ablation. File chạy chính dùng **L2**.

## Kiểm tra

Chỉ dùng Microsoft Store Python 3.13 đã chọn:

```powershell
& 'C:\Users\anhna\AppData\Local\Microsoft\WindowsApps\python3.13.exe' -X utf8 -B -m src.input_dataset audit
```

Kiểm thử bao gồm giới hạn model, backward, split cố định, specimen grouping,
checkpoint overwrite, ngắt/resume, lỗi ghi file, AMP overflow và toàn chu trình
hai độ phân giải trên fixture nhỏ. Các kiểm tra hạ tầng không phải accuracy
chính thức. Chưa chạy huấn luyện đầy đủ và chưa có PDF báo cáo cuối.

Biên bản kiểm tra chuẩn bị train: [training_readiness.json](results/training_readiness.json).
