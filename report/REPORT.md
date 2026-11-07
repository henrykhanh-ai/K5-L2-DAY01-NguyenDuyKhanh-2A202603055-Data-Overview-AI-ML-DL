# Báo cáo bài thực hành Ngày 1 – Đọc prediction YOLO11 để hiểu định dạng nhãn

## Thông tin lần chạy

| Mục | Giá trị |
| --- | --- |
| Ngày chạy | 06/11/2026 |
| Runtime Colab | CPU (`Device: cpu`; PyTorch là bản `+cpu` nên không dùng GPU) |
| Python / PyTorch / Ultralytics | 3.13.18 / 2.11.0+cpu / 8.4.145 |
| Checkpoint | `yolo11n-cls.pt`, `yolo11n.pt`, `yolo11n-seg.pt`; cả ba báo `PASS` SHA-256 (cls `c62d41bf9625…`, det `0ebbc80d4a76…`, seg `55ed65c56c91…`) |
| Ảnh mẫu | `traffic` (COCO 210273, 640×428, sha256 `3ec23de63592…`), `kitchen` (COCO 397133, 640×427, sha256 `09e1d25c75f7…`), `dining` (COCO 166918, 480×640, sha256 `a7f8457580a2…`); cả ba báo `PASS` checksum; attribution trong `day1_lab_outputs/IMAGE_ATTRIBUTION.md` (CC BY 2.0) |
| Ngưỡng detection / segmentation khi tạo JSON | `0.35` |
| Thay đổi so với notebook nguồn | Không: tôi không đổi checkpoint, threshold, package hay mã nguồn |

### Từ cần nhớ

| Từ | Nghĩa ngắn |
| --- | --- |
| Prediction | Kết quả model vừa đoán ra; chưa phải nhãn đúng |
| Ground truth | Nhãn do người gán theo quy tắc |
| Taxonomy | Danh sách tên lớp có sẵn kèm model (vd. ImageNet-1K, COCO-80) |
| Checkpoint | File model `.pt` đã huấn luyện sẵn |
| Guideline | Quy tắc gán nhãn do người viết |
| Annotator | Người gán nhãn |
| Reviewer / QC | Người kiểm tra nhãn |
| Escalation | Ca khó → hỏi Lab Coach, không tự bịa |
| Model score | Số “tin” của model; không phải điểm chất lượng nhãn người |

---

## 1. Phân loại ảnh (ảnh `traffic`)

**File tôi đã mở:** `classification_predictions.json`, `visuals/classification_top5.png`.

### 1.1. Record hạng 1

`sample_id`=traffic ; `class_id`=468 ; `class_name`=cab ; `rank`=1 ; `score`=0.510915 ; `taxonomy_name`=ImageNet-1K

Tôi chép thêm cả top-5 để dễ so sánh:

| rank | class_id | class_name | Nghĩa | score |
| ---: | ---: | --- | --- | ---: |
| 1 | 468 | cab | xe taxi | 0.510915 |
| 2 | 654 | minibus | xe buýt nhỏ | 0.164284 |
| 3 | 734 | police_van | xe cảnh sát | 0.085848 |
| 4 | 757 | recreational_vehicle | xe nhà di động | 0.054110 |
| 5 | 829 | streetcar | tàu điện | 0.048193 |

Năm score cộng lại là 0.863; phần còn lại 0.137 chia cho 995 lớp khác.

### 1.2. Record này nói về cả bức ảnh hay một vật?

Record này nói về **cả bức ảnh**. Lúc đầu tôi tưởng `cab` là một chiếc taxi cụ thể trong ảnh, nhưng xem lại JSON
thì không có `bbox_xyxy` hay `polygon_xy` nào, nên không biết chiếc taxi nào, ở đâu. Cả 5 record top-5 đều chung
`sample_id` = traffic, nghĩa là model chỉ trả lời câu “nhìn tổng thể, ảnh này giống cái gì nhất”.

So với detection thì khác hẳn: cùng ảnh này, `yolo11n.pt` khoanh được **26 hộp** (xe, xe buýt, người), mỗi hộp
có tọa độ riêng. Còn phân loại chỉ cho đúng một nhãn `cab` cho cả ảnh.

### 1.3. Danh sách lớp từ đâu ra?

Danh sách lớp **đi kèm checkpoint**, model không tự nghĩ ra. Tôi hiểu là danh sách 1.000 tên lớp ImageNet-1K
được cất sẵn trong file `yolo11n-cls.pt`. Model chỉ có đúng 1.000 “ô” đầu ra, mỗi ô ứng với một tên; nó chấm
điểm cả 1.000 ô rồi lấy ô cao nhất, chứ không có ô nào để viết ra tên mới.

Tôi thấy điều này trong code notebook, ở ô tạo `classification_predictions.json`:
`"class_name": result.names[class_id]`. Tức là tên lớp được **tra từ bảng `result.names` có sẵn** theo số
`class_id`, model không tự sinh ra chữ.

Hệ quả nhìn thấy rõ ở ảnh `traffic`. Tôi tra bảng 1.000 tên thì thấy ImageNet-1K có một vài lớp nơi chốn như
`restaurant` (762), `bakery` (415), `library` (624), nhưng **không có** “đường phố” hay “cảnh giao thông”; gần nhất
chỉ là `street_sign` (919), `traffic_light` (920). Nên model đành chọn tên gần nhất mà nó có: cả top-5 đều là **tên
các loại xe**, và ngay cả `cab` cũng chỉ được 0.51.

### 1.4. Vì sao cần ghi cả `class_id`, `class_name` và `taxonomy_name`?

Vì một con số hay một cái tên đứng riêng thì dễ hiểu nhầm. Tôi kiểm tra trực tiếp trên hai checkpoint và thấy:

- Cùng số **2**: trong COCO-80 là `car`, còn trong ImageNet-1K là `great_white_shark`.
- Cùng loại xe taxi: ImageNet có lớp `cab` (468), còn COCO **không có** lớp `cab`, chỉ có `car`.

Nên tôi phải ghi đủ cả ba: `class_id` để máy dùng, `class_name` để người đọc hiểu, và `taxonomy_name` để biết đang
tra theo bảng nào. Thiếu một trường thì không chắc nhãn đó là gì, và gộp dữ liệu từ nhiều nguồn sẽ bị nhầm.

### 1.5. Nếu ảnh có nhiều vật / nhiều chủ đề, guideline cần nói rõ gì?

Tôi thử tự gán nhãn ảnh `traffic` và thấy ngay vấn đề: tùy cách nhìn, có ít nhất **3 đáp án đều hợp lý**.

| Nếu chọn theo | Đáp án | Căn cứ (lấy từ `detection_predictions.json`) |
| --- | --- | --- |
| Vật **lớn nhất** | xe buýt | Hộp `bus` 0.912557 rộng 129.84 × cao 132.96 ≈ 17.264 px², lớn nhất trong 26 hộp, gấp 2.6 lần hộp lớn thứ hai |
| Vật xuất hiện **nhiều nhất** | ô tô | 14/26 hộp là `car`, 8 là `bus`, 4 là `person` |
| **Model** tự chọn | taxi | `cab` 0.510915 |

Ba người gán theo ba cách sẽ ra ba nhãn khác nhau cho cùng một ảnh. Vì vậy guideline phải **chốt một tiêu chí**
và nói rõ khi nào được dừng lại hỏi. Tôi đề xuất:

> 1. Nhãn là **chủ thể chính**: vật chiếm diện tích lớn nhất trong ảnh.
> 2. Nếu hai vật lớn nhất chênh nhau **dưới 20%** diện tích, chọn vật **gần tâm ảnh** hơn.
> 3. Chỉ chọn trong danh sách lớp của taxonomy. Các lớp dễ nhầm (`cab` / `minibus` / `police_van` /
>    `school_bus` / `trolleybus`) phải có định nghĩa và ảnh mẫu.
> 4. Nếu không lớp nào mô tả đúng chủ thể chính, hoặc vẫn không phân định được → gắn cờ **escalation**, không
>    tự đoán.

Áp dụng thử cho ảnh `traffic`: theo luật 1, chủ thể chính là **xe buýt** (lớn gấp 2.6 lần vật thứ hai nên không cần
luật 2). Trên ảnh gốc, đó là chiếc **xe buýt thành phố màu xanh lá** (biển tuyến 581) ở góc dưới-trái. Nhưng tôi tra
bảng ImageNet-1K thì **không có lớp “bus” chung**, chỉ có `minibus` (654), `school_bus` (779), `trolleybus` (874).
Chiếc xe này to, không phải minibus; không phải xe buýt trường học; và tôi không thấy dây điện phía trên nên cũng
không chắc là trolleybus. Không lớp nào khớp, nên theo luật 4 tôi **gắn cờ hỏi Lab Coach**, không tự chọn bừa.

Qua ví dụ này tôi thấy: có guideline thì mọi người chọn ra **cùng một chủ thể chính** (xe buýt) thay vì ba đáp án
khác nhau, và chủ thể đó khác với đáp án của model (`cab`). Nếu dự án thật sự cần nhiều chủ đề cho một ảnh thì phải đổi sang multi-label, không ép chọn một.

### 1.6. Vì sao không được lấy `score` làm ground truth?

`score` chỉ cho biết model **tự tin** bao nhiêu, không cho biết model **đúng** hay sai. Ở ảnh này model chỉ tin
`cab` 0.51, gần như năm ăn năm thua. Trong cùng file `classification_predictions.json`, tôi còn thấy một ví dụ rõ hơn:
ảnh `kitchen` (nhà bếp) có top-1 là `gong` (cái cồng, 577) với score 0.420192. Có lẽ model nhìn mấy cái chảo đồng
treo trên tường thành cái cồng: score khá cao mà nhãn thì sai hẳn. Ground truth phải do người gán theo guideline và có reviewer kiểm tra. Nếu lấy prediction làm nhãn, lần huấn luyện sau model sẽ học lại chính lỗi của nó.

---

## 2. Phát hiện vật thể (ảnh `kitchen`)

**File tôi đã mở:** `detection_predictions.json`, `visuals/detection_predictions.png`. Ảnh `kitchen` rộng
640 px, cao 427 px.

### 2.1. Record tôi chọn

`class_id`=0 ; `class_name`=person ; `score`=0.912624 ; `bbox_xyxy`=[385.33, 69.24, 498.92, 348.92] ;
`bbox_width`=113.58 ; `bbox_height`=279.68 (`coordinate_unit`=pixel, `bbox_format`=xyxy, `score_threshold`=0.35)

### 2.2. Giải thích box bằng lời thường

Gốc (0, 0) là góc trên-trái của ảnh; sang phải thì x tăng, xuống dưới thì y tăng. `bbox_xyxy` là
`[x_min, y_min, x_max, y_max]`, nên:

- **Góc trên-trái** là (385, 69): cách mép trái 385 px, cách mép trên 69 px. Đối chiếu trên PNG, đây là ngang
  **đỉnh đầu** người đầu bếp đang đứng quay lưng.
- **Góc dưới-phải** là (499, 349): trên PNG là ngang **bàn chân** chạm sàn.
- **Rộng** 498.92 − 385.33 ≈ 113.6 px; **cao** 348.92 − 69.24 = 279.68 px. Tôi tính ra khớp với
  `bbox_width`/`bbox_height`; lệch 0.01 vì notebook tính từ tọa độ chưa làm tròn.
- Hộp cao gấp khoảng 2.5 lần bề ngang (279.68 ÷ 113.58), đúng dáng người đứng thẳng.

### 2.3. Số prediction theo từng ngưỡng

- Ngưỡng **0.20**: **17** prediction.
- Ngưỡng **0.60**: **6** prediction.

(Ngưỡng 0.35 dùng khi tạo JSON: 11 prediction.)

| Ngưỡng `conf` | Số prediction | Các lớp |
| ---: | ---: | --- |
| 0.20 | 17 | 2 person, 5 bowl, 2 oven, 2 cup, 3 spoon, 1 potted plant, 1 dining table, 1 bottle |
| 0.35 | 11 | 2 person, 5 bowl, 2 oven, 2 cup |
| 0.60 | 6 | 2 person, 2 bowl, 2 oven |

### 2.4. Khi hạ ngưỡng, reviewer phải làm thêm gì?

Khi hạ từ 0.35 xuống 0.20, số hộp tăng từ 11 lên 17. Trong 6 hộp thêm vào có 5 hộp là vật nhỏ, mảnh (3 spoon,
1 bottle, 1 potted plant) và 1 hộp `dining table` mà model chỉ tin dưới 0.35. Vậy reviewer phải:

1. Duyệt nhiều hộp hơn và **xóa hộp sai** (không có vật, hoặc sai lớp).
2. Tìm **hộp trùng**: hai hộp cho cùng một vật.
3. Soi kỹ các hộp điểm thấp xem có đúng là vật nhỏ hoặc bị che không.
4. Vẫn phải rà cả ảnh tìm **vật bị bỏ sót**, vì ngay cả ở 0.20 tôi vẫn thấy bồn rửa và dao treo tường không có
   hộp, dù COCO-80 có lớp `sink` (71) và `knife` (43).

Nói ngắn gọn: hạ ngưỡng thì ít bỏ sót hơn, nhưng reviewer tốn công kiểm tra hơn.

### 2.5. Quy tắc box cho người gán nhãn

> “Mỗi vật thuộc danh sách lớp được vẽ **đúng một** hộp, ôm sát **phần nhìn thấy** của vật: không chừa nền thừa,
> không cắt mất phần thấy được. Vật bị che một phần vẫn phải gán nếu nhận ra là gì; không đoán phần bị che. Vật bị
> cắt ở mép ảnh thì hộp dừng ở mép ảnh. Không bỏ vật vì nhỏ hay mờ; dùng cờ `occluded` (bị che) hoặc `truncated`
> (bị cắt mép) thay vì bỏ.”

### 2.6. Một vật bị che / cắt mép trên ảnh `kitchen`, và cách xử lý

**Vật tôi chọn:** `person` thứ hai, score 0.610931, `bbox_xyxy`=[0.12, 263.18, 61.55, 310.78].

Lúc đầu tôi tưởng có 2 `person` là model đếm trùng. Nhìn kỹ ảnh thì `person` này nằm ở mép trái và chỉ là **một
bàn tay đang chỉ vào cái bàn**; cả người nằm ngoài khung ảnh. JSON khớp với điều đó: `x_min` = 0.12 ≈ 0 tức là hộp **chạm mép
trái**, và hộp chỉ rộng 61 px, cao 48 px, cỡ một bàn tay. Đây là vật **bị cắt mép (truncated)**.

**Tôi chọn: viết thành quy tắc trong guideline.** Vì sao:

- Cắt mép là chuyện **lặp lại**: chỉ riêng ảnh này đã có 3 hộp chạm mép trái (`person` x_min=0.12, `oven`
  x_min=0.11, `bowl` x_min=0.32). Ảnh khác chắc chắn cũng gặp, nên cần luật chung để mọi người làm giống nhau.
- Trường hợp này **không có gì mơ hồ**: nhìn rõ là tay người, nên lớp `person` đã chắc. Không cần hỏi Lab Coach.
- Luật đã có sẵn ở 2.5: vẫn gán `person`, hộp ôm đúng phần tay nhìn thấy và dừng ở mép ảnh, gắn cờ `truncated`.

Tôi chỉ escalation nếu phần lộ ra quá ít, đến mức không nhận ra được đó là vật gì.

---

## 3. Phân đoạn theo từng đối tượng (ảnh `kitchen`)

**File tôi đã mở:** `segmentation_predictions.json`, `visuals/segmentation_prediction.png`.

### 3.1. Record tôi chọn

`instance_id`=kitchen-001 ; `class_id`=0 ; `class_name`=person ; `score`=0.899318 ; `polygon_point_count`=348 ;
5 điểm đầu của `polygon_xy`=[446.0, 70.0], [445.0, 71.0], [444.0, 71.0], [443.0, 72.0], [442.0, 72.0] ;
`bbox_xyxy`=[385.45, 66.44, 498.02, 348.58] (`model_file`=yolo11n-seg.pt, `score_threshold`=0.35)

### 3.2. So với box, polygon cho thêm thông tin gì?

Box chỉ có 4 số, là một hình chữ nhật, nên luôn dính cả nền phía sau. Polygon thì có 348 điểm đi vòng theo đường
viền người, bắt đầu từ đỉnh đầu (y ≈ 70). Ở chỗ viền cong như đầu, các điểm sát nhau (5 điểm đầu cách nhau 1 px);
ở chỗ viền thẳng, điểm thưa hơn (xa nhất 28 px).

Tôi tính thử diện tích từ JSON để xem chênh lệch bao nhiêu:

| | Diện tích |
| --- | ---: |
| Box (112.6 × 282.1 px) | ≈ 31.760 px² |
| Polygon (hình người thật) | ≈ 18.170 px² |
| Phần là người thật trong box | **≈ 57%** |

Kết quả làm tôi khá bất ngờ: gần một nửa cái box là tường, tủ, nồi phía sau chứ không phải người. Vậy polygon cho
biết **hình dáng thật** và **diện tích thật** của vật, còn box thì không. Với cái bàn (`kitchen-005`) chỉ 55% box
là bàn, vì mặt bàn nhìn nghiêng thành hình thang.

Tôi cũng để ý: cùng người này, detection cho 0.912624 tại [385.33, 69.24, 498.92, 348.92], còn segmentation cho
0.899318 với box lệch vài pixel. Hai model cho hai prediction riêng, không cái nào là đáp án chuẩn.

### 3.3. `instance_id` dùng để làm gì?

Tôi hiểu `instance_id` chỉ là **số thứ tự** để phân biệt từng vật trong file kết quả lần chạy này. Ví dụ ảnh
`kitchen` có 4 vật được gọi là `bowl`, đều `class_id`=45, nhưng mỗi vật một `instance_id` riêng (`kitchen-002`,
`-003`, `-006`, `-010`).

- **Không phải class ID:** số thứ tự không cho biết vật thuộc lớp nào; lớp nằm ở `class_id`/`class_name`.
- **Không phải tracking ID:** nó không theo một vật qua nhiều khung hình video. Chạy lại hoặc sang ảnh khác thì
  `kitchen-001` có thể là vật khác.

### 3.4. Quy tắc vẽ biên mask cho người gán nhãn

> “Mỗi vật một polygon riêng. Đặt điểm dọc theo **biên nhìn thấy** của vật, lệch với biên thật không quá khoảng
> 2 px; không tô sang nền hay vật khác. Phần bị che thì không vẽ; nếu vật bị che thành nhiều mảnh thì vẽ đủ các
> mảnh và gắn cùng một instance. Hai vật chạm nhau: đường biên chung thuộc về vật nằm **phía trước**; hai mask
> không được chồng lên nhau.”

### 3.5. Một chỗ hai vật chạm nhau trên ảnh `kitchen`, và cách xử lý

**Chỗ tôi chọn:** vật ở mép trái mà model gọi là `bowl` (`kitchen-006`, 0.528202, `bbox_xyxy`=[0.11, 302.85,
85.7, 370.15]). Trên ảnh gốc đây là **chiếc bánh pie trong khuôn**, đặt **chạm** mặt bàn (`kitchen-005`, dining
table 0.578416).

Trên PNG tôi thấy màu tô của khuôn bánh và của bàn đè lên nhau ở chỗ này. Để chắc chắn, tôi tô lại hai polygon từ JSON và
đếm số pixel bị **cả hai mask cùng nhận**:

| Vật model gọi là `bowl` | Diện tích mask | Pixel chung với mask bàn | Tỷ lệ |
| --- | ---: | ---: | ---: |
| `kitchen-006` (khuôn bánh pie, mép trái) | 4.003 px | 907 px | **22.7%** |
| `kitchen-003` | 2.117 px | 35 px | 1.7% |
| `kitchen-002` | 2.414 px | 13 px | 0.5% |

Với `kitchen-002` (khuôn sứ trắng) và `kitchen-003` (bát nhân màu xanh), mask bàn đã gần như chừa chỗ. Riêng
`kitchen-006` thì gần **1/4 diện tích** bị tính là “vừa là bát, vừa là bàn”. Vỏ bánh màu nâu nhạt gần giống màu mặt
bàn gỗ nên ranh giới mờ, và mask đã tô chồng lên nhau ở đó.

**Tôi chọn: viết thành quy tắc trong guideline.** Vì sao:

- Bát, cốc, đĩa đặt trên bàn có trong **hầu hết** ảnh bếp và ảnh bữa ăn (ảnh `dining` cũng vậy), nên phải có luật
  chung, không thể hỏi từng ca.
- Cách xử lý **đã rõ**: vật nằm phía trước (cái khuôn bánh) giữ đường viền chung; mask bàn phải chừa ra phần đó;
  hai mask **không được chồng nhau**. Áp luật này thì 907 px chung kia thuộc về khuôn bánh.

Tôi chỉ escalation nếu nhìn ảnh không xác định được vật nào nằm phía trước.

---

## 4. Vòng đời và kiểm tra chất lượng

  `ảnh thô → guideline → ground truth → huấn luyện → prediction → QC/rework`

  | Tác vụ | Ground truth trông như thế nào? (đơn vị nhãn) | Lỗi hoặc chỗ khó quyết định bạn thấy trên ảnh/JSON | Annotator làm gì? | Reviewer xem gì? |
  | --- | --- | --- | --- | --- |
  | Phân loại ảnh | **1 nhãn cho cả ảnh**: `class_id` + `class_name` + `taxonomy_name`, không tọa độ | `traffic`: top-1 `cab` chỉ 0.510915; cả top-5 là loại xe. Vật lớn nhất là xe buýt thành phố nhưng ImageNet không có lớp “bus” chung → khó chọn nhãn | Chọn 1 nhãn theo luật “chủ thể chính” (1.5); không lớp nào khớp thì gắn cờ escalation | Nhãn có đúng luật không, ID có khớp taxonomy không, các annotator có chọn giống nhau không; xử lý ảnh bị gắn cờ |
  | Phát hiện vật thể | **Mỗi vật 1 hộp**: lớp + `[x_min, y_min, x_max, y_max]` (pixel) | `kitchen`: `person` 0.610931 chỉ là bàn tay ở mép trái (x_min=0.12); hộp `oven` 0.6343 trùm cả bồn rửa; bàn gỗ, bồn rửa không có hộp ở 0.35 | Vẽ hộp sát phần nhìn thấy cho **mọi** vật trong danh sách lớp, kể cả vật nhỏ/bị cắt; gắn cờ `truncated`/`occluded` (2.5) | Vật bị sót, hộp thừa/trùng, hộp quá rộng/chật, sai lớp; hộp chạm mép có cờ `truncated` chưa |
  | Instance segmentation | **Mỗi vật 1 polygon** theo viền: `instance_id` + lớp + `[[x, y], …]` | `kitchen`: mask `kitchen-006` (khuôn bánh, model gọi `bowl`) và mask bàn `kitchen-005` cùng nhận 907 px (22.7%) ở chỗ chạm nhau | Vẽ theo viền nhìn thấy; mỗi vật tách riêng; vật phía trước giữ đường viền, không vẽ chồng (3.4) | Viền có bám sát vật không, mask có chồng nhau không, có gộp/tách sai vật không, `instance_id` có trùng không |

**Cách tôi nối các mắt xích:** các lỗi trong cột thứ 3 là lỗi của **prediction**, không phải của người gán nhãn. Tôi
dùng chúng để biết ca nào khó, rồi viết thành **guideline** (1.5, 2.5, 3.4). **Annotator** gán **ground truth** theo
guideline đó, không chép prediction. Ground truth dùng để **huấn luyện** model; model mới lại cho **prediction**.
Ở bước **QC**, reviewer soi đúng những ca khó này: annotator làm sai luật thì trả về **rework**; nhiều annotator cùng
làm khác nhau ở một ca thì nghĩa là guideline chưa rõ, phải **sửa guideline** (hoặc escalation nếu chưa có câu trả lời).

---

## 5. An toàn dữ liệu

1. **Quy tắc của tôi:** chỉ dùng 3 ảnh COCO công khai mà notebook đã cố định bằng checksum, giữ đủ ghi công
   CC BY 2.0, và không đưa ảnh cá nhân, khuôn mặt hay biển số ngoài phạm vi, dữ liệu khách hàng/nội bộ hay thông
   tin định danh của tôi vào Colab, báo cáo hoặc repository công khai.
2. **Nếu gặp dữ liệu sai phạm vi:** tôi dừng ngay, không xử lý tiếp và không commit; báo **Lab Coach / GV phụ
   trách lớp** qua kênh hỗ trợ của lớp. Nếu lỡ đã push lên GitHub thì xóa khỏi repo và báo GV/Lab Coach.

---

## 6. Checklist trước khi nộp link repo

Kiểm tra trên GitHub trong thư mục `report/`:

- [ ] `REPORT.md` (file này đã điền)
- [ ] `day1_lab_outputs/classification_predictions.json`
- [ ] `day1_lab_outputs/detection_predictions.json`
- [ ] `day1_lab_outputs/segmentation_predictions.json`
- [ ] `day1_lab_outputs/IMAGE_ATTRIBUTION.md`
- [ ] `day1_lab_outputs/visuals/classification_top5.png`
- [ ] `day1_lab_outputs/visuals/detection_predictions.png`
- [ ] `day1_lab_outputs/visuals/segmentation_prediction.png`
- [x] Ô validation cuối notebook báo `PASS`
- [x] Không có họ tên, MSSV hoặc dữ liệu nhạy cảm trong báo cáo/output
- [ ] Đã nộp **link repository** trên LMS
