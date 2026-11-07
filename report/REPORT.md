# Báo cáo bài thực hành Ngày 1 – Đọc prediction YOLO11 để hiểu định dạng nhãn

**Ngày chạy:** 2026-11-07

**Runtime Colab:** CPU/GPU

**Python / PyTorch / Ultralytics:**  3.13.18 / 2.14.1+cpu / 8.4.145

**Checkpoint:** `yolo11n-cls.pt`, `yolo11n.pt`, `yolo11n-seg.pt`

**Ngưỡng detection/segmentation dùng khi tạo JSON:** `0.35` (chỉ đổi nếu GV cho phép; nếu đổi hãy ghi rõ)

**Thay đổi so với notebook nguồn:** Không / mô tả rõ thay đổi

> **Cách dùng mẫu này:** copy/điền trên máy, lưu thành `REPORT.md`, rồi **upload** vào panel Files
> của Colab (notebook không tự tạo file này). Ô cuối Colab sẽ đóng ZIP cùng `day1_lab_outputs/`.
> Giải nén và đưa vào `report/` trên GitHub, rồi nộp **link repo** trên LMS.
> Không ghi họ tên, MSSV, email, số điện thoại hoặc dữ liệu cá nhân khác vào báo cáo/output;
> họ tên/MSSV chỉ nằm trong tên repository.

### Từ cần nhớ (đọc nhanh trước khi viết)

| Từ            | Nghĩa ngắn                                                    |
| ------------- | ------------------------------------------------------------- |
| Prediction    | Kết quả model vừa đoán ra; chưa phải nhãn đúng                |
| Ground truth  | Nhãn do người gán theo quy tắc                                |
| Taxonomy      | Danh sách tên lớp có sẵn kèm model (vd. ImageNet-1K, COCO-80) |
| Checkpoint    | File model `.pt` đã huấn luyện sẵn                            |
| Guideline     | Quy tắc gán nhãn do người viết                                |
| Annotator     | Người gán nhãn                                                |
| Reviewer / QC | Người kiểm tra nhãn                                           |
| Escalation    | Ca khó → hỏi Lab Coach, không tự bịa                          |
| Model score   | Số “tin” của model; không phải điểm chất lượng nhãn người     |

## 1. Phân loại ảnh

**Mở file:** `day1_lab_outputs/classification_predictions.json` và `visuals/classification_top5.png`.  
Chỉ dùng ảnh `sample_id = traffic`, lấy dòng `rank = 1`.

1. Chép record hạng 1: `class_id`=468 ; `class_name`=cab ; `rank`=1 ; `score`=0.510915 ; `taxonomy_name`=ImageNet-1K
2. Record này nói về **cả bức ảnh** hay về **một vật thể riêng** trong ảnh? Vì sao bạn nghĩ vậy?
3. Danh sách tên lớp (class list) do đâu mà có: model tự nghĩ ra, hay đi kèm taxonomy/checkpoint?
4. Vì sao cần ghi cả `class_id`, `class_name` và `taxonomy_name`, không chỉ ghi mỗi tên lớp?
5. Nếu trong ảnh có nhiều vật / nhiều chủ đề, guideline cần nói rõ điều gì để người gán nhãn chọn **một** lớp?
6. Vì sao không được lấy `score` của model làm ground truth?

## 2. Phát hiện vật thể

**Mở file:** `day1_lab_outputs/detection_predictions.json` và `visuals/detection_predictions.png`.  
Chỉ dùng ảnh `sample_id = kitchen`. Chọn **một** record.  
Số lượng prediction theo ngưỡng: xem output ô so sánh `0.20` / `0.35` / `0.60` trong notebook.

1. Chép một record: `class_name`=oven ; `score`=0.686791 ; `bbox_xyxy`=[0.11, 187.54, 195.96 292.81] ; `bbox_width`=195.85 ; `bbox_height`=105.26
2. Lớp `oven`, score 0.686791; box từ góc trên-trái khoảng pixel (0.11, 187.54) đến góc dưới-phải (195.96 292.81); rộng khoảng 195.85 px, cao khoảng 105.26 px.
3. Ví dụ so hai ngưỡng trên ảnh `kitchen`: threshold 0.20 → nhiều prediction hơn (vd. 17); threshold 0.60 → ít hơn (vd. 6). 
4. Reviewer phải xem thêm nhiều hộp hơn, kể cả hộp nhiễu/score thấp, nên khối lượng kiểm tra tăng.
5. Quy tắc box: Box phải ôm sát vật thể, không cắt mất phần nhìn thấy của object; không vẽ một box chung cho hai object tách rời.
6. TTrên ảnh `kitchen`, vật bị cắt mép bàn/đồ dùng gần biên ảnh: tôi chọn **guideline** nếu rule đã có (ví dụ “còn ≥ 50% object trong khung thì vẫn gán”); nếu không chắc object là gì vì che khuất nặng thì **escalation** hỏi Lab Coach, không tự bịa lớp.

## 3. Phân đoạn theo từng đối tượng

**Mở file:** `day1_lab_outputs/segmentation_predictions.json` và `visuals/segmentation_prediction.png`.  
Chỉ dùng ảnh `sample_id = kitchen`. Chọn **một** record.

1. Chép một record: `instance_id`=kitchen-003 ; `class_name`=bowl ; `score`=0.698698 ; số điểm polygon=`polygon_point_count`=70 ; 3–5 điểm đầu của `polygon_xy`=[84.0, 289.0], [83.0, 290.0], [81.0, 290.0], [80.0, 291.0], [77.0, 291.0]]
2. Polygon bám theo biên thật của object (cong/mép), còn box chỉ là hình chữ nhật bao ngoài nên mất chi tiết hình dạng.
3. `instance_id` dùng để phân biệt từng object trong output lab (vd. hai `bowl` khác nhau là `kitchen-002` và `kitchen-006`). Nó không phải mã lớp (`class_id`) và không phải ID theo dõi theo thời gian (tracking).
4. Quy tắc biên mask: Mask phải đi sát biên nhìn thấy của instance; không tràn sang object bên cạnh; vùng bị che khuất thì chỉ vẽ phần còn nhìn thấy (trừ khi guideline nói khác).
5. Chỗ hai đồ dùng chạm nhau trên bàn `kitchen`: nếu guideline chưa nói rõ tách biên thế nào thì **escalation**; nếu guideline đã bảo “tách theo khe nhìn thấy giữa hai object” thì làm theo guideline.

## 4. Vòng đời và kiểm tra chất lượng

Lab hôm nay chỉ **xem prediction**. Chuỗi dưới đây là vòng đời production để bạn nối các khái niệm:

`ảnh thô → guideline → ground truth → huấn luyện → prediction → QC/rework`

Điền bảng (viết ngắn, có dẫn chứng từ JSON/PNG khi nói về lỗi):

| Tác vụ                | Ground truth trông như thế nào? (đơn vị nhãn) | Lỗi hoặc chỗ khó quyết định bạn thấy trên ảnh/JSON                  | Annotator làm gì?                           | Reviewer xem gì?                                                 |
| --------------------- | --------------------------------------------- | ------------------------------------------------------------------- | ------------------------------------------- | ---------------------------------------------------------------- |
| Phân loại ảnh         | Một lớp cho cả ảnh theo taxonomy ImageNet-1K  | Top-1 có thể không khớp trực giác cảnh giao thông (prediction ≠ GT) | Đọc guideline chọn đúng 1 lớp cho ảnh       | Đúng taxonomy, đúng phạm vi cấp ảnh, không chép score thành nhãn |
| Phát hiện vật thể     | Mỗi object: class + box xyxy                  | Hạ threshold ra thêm hộp nhiễu; object cắt mép                      | Vẽ box theo rule, đủ object trong guideline | Box có ôm sát không, thiếu/thừa object, ca che khuất             |
| Instance segmentation | Mỗi instance: class + polygon                 | Biên hai vật chạm nhau / vùng mờ                                    | Vẽ mask bám biên từng instance              | Mask có tràn object khác không, đủ instance cùng lớp             |

## 5. An toàn dữ liệu

1. Quy tắc bảo vệ dữ liệu của bạn cho bài lab này là gì? (viết 1 câu)
   Chỉ dùng ảnh COCO công khai có sẵn trong notebook; không tải ảnh cá nhân, khuôn mặt, biển số hay dữ liệu nội bộ lên Colab/GitHub.
2. Nếu gặp ảnh hoặc dữ liệu không đúng phạm vi, bạn sẽ dừng và báo cho ai?
   Dừng thao tác và báo Lab Coach / GV phụ trách lab.

## 6. Checklist trước khi nộp link repo

Kiểm tra trên GitHub trong thư mục `report/`:

- [x] `REPORT.md` (file này đã điền)
- [x] `day1_lab_outputs/classification_predictions.json`
- [x] `day1_lab_outputs/detection_predictions.json`
- [x] `day1_lab_outputs/segmentation_predictions.json`
- [x] `day1_lab_outputs/IMAGE_ATTRIBUTION.md`
- [x] `day1_lab_outputs/visuals/classification_top5.png`
- [x] `day1_lab_outputs/visuals/detection_predictions.png`
- [x] `day1_lab_outputs/visuals/segmentation_prediction.png`
- [x] Ô validation cuối notebook báo `PASS`
- [x] Không có họ tên, MSSV hoặc dữ liệu nhạy cảm trong báo cáo/output
- [ ] Đã nộp **link repository** trên LMS
