# Kiểm tra yêu cầu giữa kỳ — 19/10/2026

**Có dữ liệu, nhưng chưa đủ điều kiện để chốt bài nộp.** Đối chiếu trực tiếp với
[đề gốc](../Midterm%20exam.docx); số liệu kiểm tra máy nằm trong
[dataset_audit.json](dataset_audit.json), danh sách ảnh trong
[dataset_manifest.csv](dataset_manifest.csv).

## Số lượng thực tế

Hai bộ nằm ở `DataNew/Mid224` và `DataNew/Mid32`, dùng chung **22.303 ảnh định danh**
mỗi bộ: **22.053 train + 250 test**, tổng cộng **44.606 file ảnh**.

| Lớp | Train hiện có | Test hiện có | Thiếu so với 5.000 train |
|---|---:|---:|---:|
| apple | 5.000 | 50 | 0 |
| banana | 4.480 | 50 | 520 |
| orange | 5.000 | 50 | 0 |
| strawberry | 4.888 | 50 | 112 |
| pineapple | 2.685 | 50 | 2.315 |
| **Tổng** | **22.053** | **250** | **2.947** |

Đề ghi **“about 5000”**, không bắt buộc chính xác 5.000 hoặc một tỷ lệ validation
cụ thể. Mốc 5.000 được dùng để lượng hóa phần thiếu; pineapple chỉ đạt 53,7% mốc
này, nên chưa thể kết luận đạt yêu cầu số lượng. Hai bộ có cùng định danh và split;
không cộng số ảnh của hai độ phân giải thành số mẫu độc lập.

## Đối chiếu từng yêu cầu

| Yêu cầu trong đề | Kết quả kiểm tra | Đánh giá |
|---|---|---|
| Ít nhất 5 lớp | apple, banana, orange, strawberry, pineapple | Đạt số lớp |
| Khoảng 5.000 train/lớp | Bảng trên; thiếu nhiều nhất ở pineapple | Chưa đạt mục tiêu |
| 50 test/lớp | Đủ 50 × 5 ở cả hai bộ | Đạt số lượng |
| Mid224 đúng 224×224 | Giải mã toàn bộ 22.303 ảnh AVIF, RGB, một frame | Đạt kỹ thuật |
| Mid32 đúng 32×32 | Giải mã toàn bộ 22.303 ảnh WebP, RGB, một frame | Đạt kỹ thuật |
| Mid224 không quá 25M | 20.425.389 byte = 20,425389 MB | Đạt dung lượng |
| Mid32 không quá 20M | 13.071.104 byte = 13,071104 MB | Đạt dung lượng |
| L không quá 6M learnable parameters | FruitLite-L: 408.179 | Đạt theo mã hiện có |
| L dưới 1G FLOPs | 587.091.697 ở 224; 12.099.313 ở 32 | Đạt theo quy ước profiler |
| Train và test L trên cả hai bộ | Chưa có kết quả chính thức; chỉ có bằng chứng smoke/pilot cũ | Chưa đạt |
| Báo cáo settings, Top-1 và từng epoch | Có cấu hình khởi đầu, chưa có số liệu huấn luyện thật | Chưa đạt |
| Báo cáo cuối dạng PDF | Chưa có PDF cuối | Chưa đạt |
| Kiến trúc không giống hệt mạng có sẵn; không trùng nhóm khác | Có CNN tự ghép; chưa có đối chiếu với nhóm khác | Chưa xác minh đầy đủ |

Dung lượng tính theo file ảnh bên trong từng thư mục Mid224/Mid32, không phải ZIP,
không cộng dữ liệu nguồn hoặc metadata nằm ngoài hai bộ. Dùng ngưỡng bảo thủ
**1 MB = 1.000.000 byte**; đề không chỉ rõ định dạng PNG/JPEG/AVIF/WebP.
FLOPs dùng `2 × MACs + scalar operations` theo `src/profile.py`, batch 1,
inference; không phải thời gian chạy hoặc phép đo phần cứng.

## Chất lượng và tính độc lập

- Kiểm tra toàn bộ file: không thiếu/thừa so với manifest, đúng cặp hai độ phân
  giải, không có nhóm trùng byte hoặc trùng pixel giải mã trong từng bộ.
- Tất cả ảnh giữ nguyên hash so với inventory sau lần bổ sung gần nhất. Manifest
  hiện chỉ có `type=real`; đây là nhãn metadata, không tự chứng minh mọi ảnh đều là
  mẫu chụp độc lập.
- Hồ sơ trước đó ghi **16.000 định danh có cờ tương đồng dHash chưa giải quyết**.
  Đây là nghi vấn cần duyệt nội dung và nhóm nguồn, **không phải 16.000 ảnh đã được
  chứng minh trùng**. Kiểm tra trùng byte/pixel không loại trừ crop, đổi góc hoặc
  cùng quả xuất hiện ở cả train và test.
- [Ảnh mẫu kiểm tra](dataset_preview.jpg) lấy seed 42: 6 train + 2 test/lớp,
  hiển thị Mid224 cạnh Mid32. Việc xem 40 mẫu không thay thế duyệt toàn bộ nhãn,
  quyền sử dụng, nhóm cùng vật thể và chất lượng nén. Mid224 được nén rất mạnh;
  dung lượng đạt không chứng minh giữ đủ thông tin để phân loại.
- Mẫu banana có đối tượng đỏ khó xác định, ảnh món ăn và bối cảnh bàn/đồ uống
  (`banana_train_real_03457`, `00261`, `00768`). Cần thống nhất tiêu chí nhìn thấy
  trái cây và duyệt lại; chưa tự gán lại nhãn hoặc xóa những ảnh này.
- Hồ sơ nguồn hiện vẫn đánh dấu kiểm duyệt ngữ nghĩa/quyền sử dụng và nhóm vật thể
  chưa hoàn tất. Giữ nguyên các trạng thái đó, không suy diễn thành đã đạt.

## Trở ngại trước khi huấn luyện

`src/train.py` hiện đọc `data/metadata/splits.csv` và ảnh PNG trong
`data/Mid224` / `data/Mid32`. Các đường dẫn chính thức này chưa có. Dữ liệu thực tế
lại dùng `results/dataset_manifest.csv`, AVIF/WebP và nằm trong `DataNew`.
Vì vậy chạy config hiện tại chưa thể train trực tiếp trên bộ đang có.

Để hoàn tất bài: bổ sung ảnh thật phù hợp cho ba lớp còn thiếu, duyệt nhãn và nhóm
nguồn để tránh rò rỉ train/test, xác nhận bộ dữ liệu/kiến trúc không trùng nhóm
khác, rồi thống nhất loader và split. Sau đó mới train/test hai độ phân giải,
lưu log từng epoch và làm PDF. Việc kiểm tra và dọn project này chưa thực hiện
thu thập bổ sung hoặc huấn luyện.
