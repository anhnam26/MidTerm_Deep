# FruitLite-L — Deep Learning Midterm

Phân loại **apple, banana, orange, strawberry, pineapple**, cùng ảnh và split cho **Mid224** và **Mid32**, hai bộ trọng số học riêng. Đây là dự án đang triển khai, **chưa có dataset đủ quota hoặc accuracy chính thức**.

**Trạng thái sau reset phiên05:** theo yêu cầu người dùng, toàn bộ `D:\DeepLearning\MidTerm\data` (raw, crops, metadata, previews, khảo sát codec) đã bị xóa, acquisition đã dừng. Chưa có Mid224/Mid32 và không tự tải lại dữ liệu. Các lệnh thu thập dưới đây chỉ là hướng dẫn sử dụng công cụ, không phải tiến trình đang chạy. Kết quả pilot/khảo sát cũ là lịch sử, không phải dataset được bàn giao; artifact dữ liệu cũ không còn tồn tại.

## 1. Yêu cầu và trạng thái

| Yêu cầu | Cách triển khai / trạng thái |
|---|---|
| Khoảng 5000 train + 50 test / lớp | Build chính thức yêu cầu đúng 4500 train + 500 val + 50 test; final train dùng cả train+val. Chưa đủ dữ liệu. |
| Mid224, Mid32 | Mọi ảnh bộ cuối phải đúng **224×224 / 32×32 pixel**. Builder resize trực tiếp crop nguồn RGB; giữ tỷ lệ + pad xám + Lanczos + PNG. ID và split đồng nhất. Chưa có hai bộ sau reset. |
| Tổng dung lượng Mid224 / Mid32 | Xác nhận mới: **Mid224 ≤25.000.000 byte (25 MB)**, **Mid32 ≤20.000.000 byte (20 MB)**; **1 MB = 1.000.000 byte**, không phải MiB. Tính tổng file ảnh train+val+test của mỗi bộ, không phải từng ảnh hoặc ZIP; không bắt buộc bằng đúng giới hạn. Build/audit từ chối vượt ngưỡng. Chưa có bộ chính thức đạt giới hạn. |
| L ≤6M tham số, <1G FLOPs | FruitLite-L: **408179** learnable parameters; 224: **587091697 FLOPs**; 32: **12099313 FLOPs**, convention phía dưới. |
| Không sao chép nguyên kiến trúc có sẵn | CNN tự ghép lịch stage/channels, split depthwise 3x3/5x5, residual projection và selective attention. Dùng các building blocks phổ biến, **không tuyên bố sáng tạo nghiên cứu**; đăng ký kiến trúc/dữ liệu để tránh trùng nhóm. |
| Train/test cả hai độ phân giải, log từng epoch | Đã có pipeline và CUDA smoke/overfit; chưa có training/test thật. |
| Báo cáo PDF | Chưa xuất báo cáo cuối vì chưa có số liệu thật. Nội dung cần báo cáo được lưu trong kế hoạch. |

## 2. Môi trường

Workspace: `D:\DeepLearning\MidTerm`. Dùng Python hiện có của project/VS Code, **không tạo environment mới**. Đã xác nhận Python 3.13.14, PyTorch 2.7.1+cu118, torchvision 0.22.1, CUDA và RTX 4060 Laptop. Dependencies đang có: torch, torchvision, Pillow, numpy, requests, scikit-learn, pytest. Phiên này không cài thêm package.

Chạy PowerShell từ workspace:

```powershell
Set-Location 'D:\DeepLearning\MidTerm'
python -c "import sys,torch; print(sys.executable); print(torch.__version__); print(torch.cuda.is_available())"
python -m pytest -q 'D:\DeepLearning\MidTerm\tests'
python -m src.profile --cuda
```

## 3. Dữ liệu và provenance

Nguồn chính ban đầu: [Open Images V7](https://storage.googleapis.com/openimages/web/download_v7.html), bounding boxes train V6/validation+test V5 và metadata tương ứng. Ảnh tải chọn lọc từ [CVDF public mirror](https://github.com/cvdfoundation/open-images-dataset). Mirror đã rotate ảnh theo bounding box, không áp dụng Rotation thêm lần nữa.

`inventory` stream CSV chính thức, chỉ giữ 5 nhãn; tối đa một crop/ảnh nguồn, không lấy ảnh được gán nhiều lớp trái cây mục tiêu, depiction/group-of/inside/truncated, hoặc box area <4%. Chỉ đánh dấu eligible metadata CC BY 2.0 có author và landing URL. **Đây chỉ là ứng viên bounding-box subset, không phải tổng toàn bộ ảnh Open Images hoặc ảnh sạch cuối cùng.** Nhãn thiếu đối tượng vẫn có thể gây false negative; nhiều ảnh món ăn hoặc nhãn sai vẫn lọt qua.

Manifest lưu ID, lớp, split nguồn, parent/group, bounding box, original URL/landing URL, author/title/license, SHA256 ảnh nguồn/crop, dHash và kết quả tải. Metadata/annotations Open Images có license riêng (CC BY 4.0); ảnh được khai báo CC BY 2.0 nhưng vẫn phải xác minh từng nguồn. Giữ attribution, đánh dấu crop/resize là chỉnh sửa; không phân phối ảnh chưa kiểm tra quyền.

### Kiểm kê và pilot

```powershell
python -m src.data inventory --splits validation test train
python -m src.data download --splits validation test train --per-class 150 --workers 4
python -m src.data sheets
python -m src.data pilot-report
```

Kiểm kê train tải stream annotation lớn (~nhiều triệu dòng), cần vài phút hoặc lâu hơn. Không tải toàn bộ Open Images. CSV hoàn thành được tái sử dụng; `--refresh` đọc lại nguồn. Downloader resume **theo ảnh đã hoàn thành**, retry 3 lần/network timeout, giới hạn ảnh nguồn 25MB; chưa resume byte của ảnh đang tải. Lần chạy tiếp sẽ retry failed trong lựa chọn hiện tại. `--per-class` là **số ứng viên thử**, không phải cam kết số ảnh usable; ảnh nhỏ hơn 224 sau crop bị loại.

Kết quả trong `D:\DeepLearning\MidTerm\data\metadata\inventory_summary.json`, `downloads.csv`, `download_summary.json`, `contact_sheets\`. Không coi metadata eligibility hoặc download thành công là acceptance.

### Kết quả kiểm kê/pilot thực tế của phiên triển khai đầu tiên

| Lớp | Ứng viên bbox đủ metadata, cả 3 split | Đã thử tải | Crop ≥224, chờ duyệt | Crop quá nhỏ |
|---|---:|---:|---:|---:|
| apple | 529 | 191 | 179 | 12 |
| banana | 423 | 123 | 101 | 22 |
| orange | 601 | 226 | 197 | 29 |
| strawberry | 701 | 203 | 166 | 37 |
| pineapple | 188 | 154 | 128 | 26 |
| **Tổng** | **2442** | **897** | **771** | **126** |

Đã duyệt sơ bộ contact sheets của 109 crop đầu; chưa duyệt toàn bộ 771 crop/giấy phép. Đã ghi **9 reject**, chưa accept ảnh nào: 2 ảnh món ăn/bao bì rõ ràng và 7 nhãn orange mà tiêu đề nguồn xác nhận kumquat/lemon/tomato/grapefruit/apricot. 2 nhóm gần trùng được dHash gợi ý, cần xem ảnh để quyết định; không coi dHash là ground truth.

Ảnh nguồn mirror chiếm ~249 MiB (không gồm crop/sheets). Đây là 500 ứng viên train (100/class) + toàn bộ 397 ứng viên validation/test; không phải tập test chính thức. Có đủ cỡ pilot 100–200 crop/class, nhưng **nguồn bbox được lọc này không thể tự đáp ứng quota**. Cần bổ sung nguồn; không suy ra toàn bộ Open Images cũng chỉ có số này. Build đã được chạy thử và **từ chối đúng** vì chưa có 5050 ảnh accepted độc lập/class; chưa tạo Mid224/Mid32 chính thức hoặc splits.csv.

### Mở rộng acquisition ngoài bbox (phiên 02)

`D:\DeepLearning\MidTerm\src\acquire.py` kiểm kê Open Images V7 **human-positive image-level labels**, đối chiếu attribution/license metadata trên tập ~9M ảnh. Tái sử dụng cả ứng viên bbox chưa tải trong `candidates_*.csv` (giữ crop bbox qua CVDF). Ưu tiên nhãn người; machine labels chỉ là **gợi ý tìm ảnh, không ground truth**, phải review. Nhãn máy có human-negative/conflict bị loại. Commons duyệt categories/subcategories có giới hạn, chỉ JPEG đủ metadata CC BY/CC BY-SA/CC0/PD và kích thước khai báo; category không chứng minh đúng lớp. Giữ nghĩa vụ attribution/share-alike riêng từng nguồn.

```powershell
Set-Location 'D:\DeepLearning\MidTerm'
python -m src.acquire oi-inventory
python -m src.acquire commons-inventory --depth 3 --max-categories 200
# Tùy chọn tìm thêm weak-label candidates, KHÔNG tự duyệt:
python -m src.acquire machine-inventory --threshold 0.5
python -m src.acquire report
python -m src.acquire download --per-class 6500 --workers 4
# Thay cho download một lần: theo dõi CSV inventory mới tối đa 12h và tự tải tiếp.
python -m src.acquire watch --hours 12 --per-class 6500 --workers 4
```

6500 là mục tiêu **ảnh tải thành công chưa reject/lớp**, dư cho review/dedup, không phải 6500 accepted. Downloader không tải lại ID đã thử; retry failed bổ sung bằng `--retry-failed`. Không chạy đồng thời `watch`, `download`, `src.data download` hoặc `import-local` vì cùng ghi `downloads.csv`. Lock supplemental bảo vệ hai downloader supplemental; sau crash chỉ bỏ stale `expanded_download.lock` khi xác nhận PID trong đó không còn chạy. Inventory hoàn thành/cache được giữ; stream annotation lỗi giữa chừng phải đọc lại giai đoạn chưa hoàn thành (không byte-resume).

Ảnh toàn khung được giữ trong `data/raw`, bản RGB phục vụ review tối đa cạnh dài 1024 trong `data/crops` (không upscale); min side vẫn >=224. Download URL thumbnail có thể là ảnh đã resize ở nguồn; không tuyên bố đây là camera-original resolution. Giữ original source/landing URL và hash cho cross-source dedup, không ghi đè crops pilot hiện có. SHA256+dHash còn cần xem nhóm/cùng scene thủ công. Dừng nếu ổ còn dưới8GiB; lỗi429/503 chờ trước retry.

Checkpoint lịch sử phiên02 đã tải thêm **676 ảnh đạt kích thước** từ human-label validation/test: tổng **1447 downloaded**, **9 reject**, **0 accepted**, **1438 pending review**. Pending theo lớp: apple281,banana159,orange575,strawberry274,pineapple149. Đây là snapshot cũ, không phải inventory hoàn tất/quota đủ. Khi đó train human labels, machine labels và Commons đang kiểm kê; hiện acquisition đã dừng và dữ liệu đã xóa ở phiên05.

Khi chạy acquisition mới, công cụ sẽ ghi tiến độ tại `D:\DeepLearning\MidTerm\data\metadata\expanded_download_progress.json` (downloaded/rejected/pending/accepted), `expanded_inventory_summary.json`, `expanded_watcher_progress.json` và logs tương ứng. **Các file cũ đã bị xóa, không còn trạng thái live tại những đường dẫn này.** Nghiên cứu nguồn lịch sử: Mendeley PineappleLeaf-Fruit có2514 ảnh **tất cả3 lớp**, chưa biết quota fruit độc lập; Pineapple-Slice1437 gốc là lát cắt,14370 bản augment không phải mẫu độc lập. Chưa nhập các nguồn này; API Mendeley đã gặp403 challenge khi khảo sát.

### Bổ sung từ tự chụp/nguồn hợp pháp

Chuẩn bị CSV riêng gồm các cột `id,class,path,group_id,source_url,landing_url,license,author,title`. ID bắt đầu `local_`; path là ảnh nguồn thực, tối thiểu mỗi cạnh 224; path tương đối được tính từ vị trí CSV. Author/license/nguồn phải có thật. Với ảnh tự chụp, ghi tên tác giả, quyết định cấp phép và URI nguồn nội bộ; group_id phải gom cùng quả/phiên chụp/video, không tự coi mỗi frame là mẫu độc lập.

```powershell
python -m src.data import-local --manifest 'D:\DeepLearning\MidTerm\data\metadata\local_sources.csv'
python -m src.data sheets
```

Không nhập ảnh augmentation, ảnh resize của cùng nguồn, hoặc nhiều crop để lấp quota. `import-local` không tự thu thập nguồn; Commons dùng công cụ bổ sung phía trên. Fruits360 cần kiểm kê riêng/tuân thủ license và group các frame cùng quả, không dùng bản100x100 cho Mid224.

### Review và split

1. Xem contact sheets **và crop gốc**: đúng lớp, đối tượng rõ, không cartoon/bao bì, không đa lớp, tránh lỗi/blur/watermark nghiêm trọng. Xem thêm bản 32x32 trước acceptance.
2. Kiểm tra trang nguồn, author/license. Sửa `group_id` trong manifest để gom cùng quả/scene/video/series khi biết; dHash không phát hiện mọi liên hệ.
3. Ghi một ID/line trong file danh sách thật. Ví dụ lệnh ghi quyết định (không auto-accept toàn bộ):

```powershell
python -m src.data review --ids 'D:\DeepLearning\MidTerm\data\metadata\accepted_ids.txt' --decision accept --note 'Visually checked class/crop/32px and original source author/license'
python -m src.data review --ids 'D:\DeepLearning\MidTerm\data\metadata\rejected_ids.txt' --decision reject --note 'Ambiguous class or unsuitable crop'
python -m src.data build
python -m src.data audit
```

Build từ chối nếu không đủ **5050 nhóm độc lập được duyệt/lớp**. Group theo parent/group ID, SHA256 và dHash ≤6 bit (bảo thủ; có false positive và false negative). Nhóm conflict labels bị loại; mỗi nhóm chỉ lấy một đại diện lớn nhất. Split seed 2026 trên tất cả nguồn vì đây là benchmark classification mới, không phải benchmark Open Images. Duyệt thủ công nhóm gần trùng trong `duplicate_groups.json` và review group ID trước khi chốt.

Build tạo ảnh trong thư mục staging và kiểm tra cả hai bộ **trước khi** xuất Mid224/Mid32 và đóng băng `splits.csv`. Vượt tổng dung lượng hoặc validation thất bại thì xóa staging, không tạo split chính thức. Build từ chối overwrite manifest hoặc thư mục Mid224/Mid32 hiện có; nếu lần xuất bị gián đoạn cần kiểm tra thủ công, không tự xóa dataset cũ.

Audit kiểm tra quota/ID/group leakage, số lượng, mode RGB/size (đúng224×224 hoặc32×32), file dư/thiếu (kể cả file không phải PNG), SHA256 ảnh đã resize và **tổng dung lượng từng bộ**. Khi có dataset mới, `D:\DeepLearning\MidTerm\data\metadata\audit.json` báo `total_bytes`, `total_mb`, `limit_bytes`, `limit_mb`, `remaining_bytes` và counts theo split/lớp; giữ `total_mib`/`limit_mib` để quy đổi tương thích, không dùng MiB làm ngưỡng. Đúng ngưỡng được chấp nhận. Tính kích thước file thực, không phải allocated disk space; raw/crops, metadata, model và outputs nằm ngoài hai thư mục dataset. Audit không thay thế kiểm duyệt bằng mắt. Không chỉnh split khi đã thấy kết quả test.

**Rủi ro dung lượng:** 25.250 ảnh/bộ chỉ có ngân sách trung bình khoảng **990 byte/ảnh Mid224** và **792 byte/ảnh Mid32**. PNG hiện tại có thể không đáp ứng. Cần đo phương án nén và kiểm tra chất lượng trước khi đổi định dạng; không tự giảm quota, đổi kích thước hoặc dùng ảnh trùng để vượt kiểm tra. Sửa audit không đồng nghĩa đã tạo được dataset phù hợp giới hạn.

### Khảo sát codec trước khi build

Công cụ chỉ đọc manifest/crops, lấy mẫu seeded theo lớp và tỷ lệ provider/label_source, không ghi quyết định accept hoặc đổi codec production. Hai kích thước được resize trực tiếp từ cùng crop nguồn, không tạo Mid32 từ ảnh Mid224 đã nén. Dùng thư mục output mới theo mẫu `compression_study_*` bên trong metadata để builder/audit tự đọc development reservations:

```powershell
python -m src.compression --output 'D:\DeepLearning\MidTerm\data\metadata\compression_study_next' --per-class 100 --seed 2026 --preview-per-class 5
```

Kết quả lịch sử phiên04 trên **500 candidates chưa được accepted (100/lớp)**, không lỗi hash/decode. **Ảnh và artifact khảo sát đã bị xóa trong reset phiên05.** Bảng vẫn dùng MiB để ghi đúng đơn vị phép đo cũ, không phải giới hạn MB hiện hành:

| Codec khảo sát | Mid224 ngoại suy MiB | Mid32 ngoại suy MiB |
|---|---:|---:|
| PNG optimize | 1627,87 | 51,77 |
| JPEG q0/q1 optimize | 20,68 | 7,44 |
| WebP q0 method4 | 21,77 | 2,36 |
| WebP q1 method4 | 29,30 | 3,14 |
| WebP q5 method4 | 38,30 | 3,86 |
| JPEG q30 optimize | 98,38 | 11,28 |
| WebP q30 method4 | 72,49 | 6,18 |

**Không phải nghiệm thu:** tổng thực tế của25.250ảnh có thể khác ngoại suy, nguồn/mức độ chi tiết chưa đại diện đầy đủ và nhãn/license chưa được xác minh. JPEGq1 bị khối màu mạnh; WebPq0 giữ hình dạng tốt hơn trong preview nhưng mất chi tiết. q0/q1 chỉ là khảo sát biên, chưa được phê duyệt. PNG vẫn là codec builder/loader hiện hành.

`report.json` lưu cấu hình, counts, bytes, RMSE và projections; RMSE không chứng minh chất lượng nhận dạng. `sample.csv`, `measurements.csv`, comparisonPNG lossless và `review_template.csv` hỗ trợ review. Template không phải reviews.csv và không được import như accept tự động. `development_reservations.json` giữ IDs/group/sourcehash/crophash/dHash; builder loại cả related duplicate group khỏi test, vẫn giữ đúng quota và từ chối thiếu unseen testgroups. Audit cũng từ chối test đã reserve. Giữ các file reservations khi di chuyển/copy metadata; output ngoài mẫu đường dẫn trên không tự được discover. Mẫu chỉ được dùng train/development, không dùng test cuối.

## 4. FruitLite-L

| Phần | Channels / repetitions | Stride đầu |
|---|---|---|
| Stem conv 3x3 + BN/ReLU | 3 →16 | 2 |
| Stage 1 | 32 / 1 | 1 |
| Stage 2 | 56 / 2 | 2 |
| Stage 3 | 96 / 3 | 2 |
| Stage 4 | 160 / 2 | 2 |
| Head 1x1 | 224 | 1 |
| Classifier | GAP → dropout .2 → linear 5 | — |

Block: expansion ×2 → chia channels thành depthwise 3x3 và 5x5 → concatenate → channel attention ở stages 3/4 → linear projection + BN → residual (projection khi cần) → ReLU. Cùng graph, stride và số tham số cho cả 224/32, không dùng pretrained weights.

Profiling batch1 inference: 224 = **278270880 MACs**, 32 = **5735328 MACs**. FLOPs dùng **2×MACs + scalar operations** của bias, eval BN, ReLU/sigmoid, GAP, residual add, gate multiply. Sigmoid tính 4 scalar ops; exp không phải đơn vị hardware latency. Concat/reshape/memory traffic không tính arithmetic FLOPs. `src/profile.py` kiểm tra operator leaf chưa được hỗ trợ, không âm thầm chỉ báo MACs thành FLOPs. Kết quả lưu `D:\DeepLearning\MidTerm\outputs\profile.json`; kiểm chứng với convention giảng viên nếu dùng tool khác.

## 5. Huấn luyện và đánh giá

Chỉ chạy training thật sau dataset audit. Defaults khởi đầu, **chưa được tối ưu**:

- SGD Nesterov momentum .9; LR .05; weight decay 5e-4; warmup 5 epoch rồi cosine đến .01×LR.
- 224: batch64, 150 epoch; 32: batch256, 200 epoch. Giảm batch khi thiếu VRAM; khi đổi batch cần tune LR trên validation.
- Cross entropy label smoothing .05; dropout .2; horizontal flip, rotation ±10°, light color jitter, random resized crop thận trọng (32: scale ≥.85; 224: ≥.75).
- Normalization cố định ImageNet mean/std chỉ là scale đầu vào, **không dùng pretrained model**.
- AMP CUDA; seed 2026; worker0 mặc định phù hợp Windows. Không bảo đảm bit-identical giữa hardware/PyTorch versions.

```powershell
python -m src.train train --config 'D:\DeepLearning\MidTerm\configs\mid224.json'
python -m src.train train --config 'D:\DeepLearning\MidTerm\configs\mid32.json'
python -m src.train evaluate --checkpoint 'D:\DeepLearning\MidTerm\outputs\dev224_seed2026\best.pt' --split val --output 'D:\DeepLearning\MidTerm\outputs\val224_seed2026'
```

**Config overrides CLI/default values cùng tên.** Để đổi cấu hình hãy sửa/copy JSON, hoặc bỏ `--config` và dùng flags. Mỗi run dùng output directory mới, không overwrite kết quả. Lưu `settings.json`, `epochs.csv`, `best.pt`, `last.pt`, `summary.json`. Chưa hỗ trợ resume training từ checkpoint; checkpoint có optimizer/scaler để hỗ trợ sau nếu cần. Chỉ load checkpoint tin cậy, dùng `weights_only=True`.

Kế hoạch thử nghiệm validation: baseline `--no-multiscale --no-attention`, thêm multiscale, thêm attention, LR/regularization/augmentation, 3 seed. Không nhất thiết chạy toàn bộ tổ hợp; ưu tiên ablation nhỏ có lý do. Khóa settings và số epoch sau validation. Final train trên 25000 ảnh và test 250 ảnh:

```powershell
python -m src.train train --size 224 --epochs 150 --batch-size 64 --final --output 'D:\DeepLearning\MidTerm\outputs\final224'
python -m src.train evaluate --checkpoint 'D:\DeepLearning\MidTerm\outputs\final224\last.pt' --split test --output 'D:\DeepLearning\MidTerm\outputs\test224'
python -m src.train train --size 32 --epochs 200 --batch-size 256 --final --output 'D:\DeepLearning\MidTerm\outputs\final32'
python -m src.train evaluate --checkpoint 'D:\DeepLearning\MidTerm\outputs\final32\last.pt' --split test --output 'D:\DeepLearning\MidTerm\outputs\test32'
```

Số epoch trên chỉ minh họa cấu hình ban đầu; thay bằng giá trị đã chọn trên validation trước final. Final không dùng validation chọn checkpoint; evaluate test từ chối development/smoke checkpoint. Đây là guardrail, không ngăn người dùng cố tình tune sau test; giữ nguyên quy trình khoa học.

Evaluation xuất Top1 đúng/tổng, loss, precision/recall/F1/support từng lớp, confusion matrix (hàng=true, cột=pred), ID/pred/confidence. Dataset split fingerprint phải khớp training.

## 6. Kiểm tra đã chạy

```powershell
python -m src.train train --smoke --size 32 --epochs 2 --batch-size 10 --output 'D:\DeepLearning\MidTerm\outputs\smoke32'
python -m src.train train --smoke --size 224 --epochs 1 --batch-size 4 --output 'D:\DeepLearning\MidTerm\outputs\smoke224'
python -m src.smoke --steps 100
```

FakeData smoke kiểm tra end-to-end train/log/save; **không có ý nghĩa accuracy midterm**. Pilot overfit một ảnh/lớp kiểm tra learnability trên chính 5 ảnh; accuracy 100% ở cả hai size sau 100 steps **không phải validation/test accuracy**, ảnh còn chưa duyệt license đầy đủ. Không nộp các smoke results như kết quả chính thức. Tests kiểm tra forward/backward/limits, crop/hash, thiếu quota, group split/conflicts, CSV attribution/multiline metadata, local import, build hai độ phân giải/freeze, audit và xuất evaluation/guardrail test.

GPU step benchmark đã chạy (10 steps sau warmup, random tensors, AMP, không tính đọc ảnh/augmentation): 224 batch64 ~0.099 giây/step, peak allocated ~1676 MiB; 32 batch256 ~0.027 giây/step, ~166 MiB. Đây **không phải thời gian epoch dataset thật**, và không bảo đảm tốc độ dài hạn trên laptop. Có thể đo lại:

```powershell
python -m src.benchmark --size 224 --batch-size 64 --steps 10
python -m src.benchmark --size 32 --batch-size 256 --steps 10
```

## 7. Việc còn lại

1. Theo dõi inventory/download bổ sung đang chạy; xác định source coverage và nguồn bổ sung khi thiếu; quota chưa được chứng minh.
2. Tiếp tục đánh giá crop gốc và32px, kiểm duyệt license và source group, loại gần trùng; không auto-accept weak labels/category.
3. Chứng minh phương án lưu/nén đáp ứng cả quota, kích thước, chất lượng và giới hạn tổng 25/20 MiB; đủ ảnh được duyệt rồi build/audit và đóng băng Mid224/Mid32. Benchmark batch/VRAM/thời gian epoch trên dữ liệu thật.
4. Ablation/tuning chỉ validation, khóa cấu hình, final train/test, biểu đồ log/confusion và PDF đúng yêu cầu.

Không có Git repository ở workspace khi bắt đầu: không git init/remote/commit/push. `plan_action/`, data và outputs được ignore để không commit dữ liệu, working notes hoặc checkpoint ngoài ý muốn.