# Mall Customer Clustering

Dự án phân tích và phân nhóm khách hàng trung tâm thương mại bằng thuật toán K-Means. Mục tiêu là tìm các nhóm khách hàng có đặc điểm gần nhau về tuổi, thu nhập và điểm chi tiêu, để việc mô tả hành vi khách hàng dễ hiểu hơn.

> README này mô tả cấu trúc và cấu hình hiện có trong repository. Cấu hình mô hình cuối được đọc từ `config.yaml`; kết quả huấn luyện được ghi trong `models/kmeans/model_metadata.json`.

## Mục lục

- [Bài toán và dữ liệu](#bài-toán-và-dữ-liệu)
- [Luồng hoạt động](#luồng-hoạt-động)
- [Cấu hình mô hình hiện tại](#cấu-hình-mô-hình-hiện-tại)
- [Cấu trúc thư mục](#cấu-trúc-thư-mục)
- [Cài đặt](#cài-đặt)
- [Cách sử dụng](#cách-sử-dụng)
- [Kết quả và báo cáo](#kết-quả-và-báo-cáo)
- [Kiểm thử](#kiểm-thử)
- [Lưu ý khi đọc kết quả](#lưu-ý-khi-đọc-kết-quả)

## Bài toán và dữ liệu

Dữ liệu gốc nằm tại `data/raw/Mall_Customers.csv`. Bản dữ liệu hiện có gồm 200 khách hàng và các cột:

| Cột | Ý nghĩa | Dùng làm đặc trưng của mô hình cuối? |
|---|---|---|
| `CustomerID` | Mã định danh khách hàng | Không |
| `Gender` | Giới tính | Không |
| `Age` | Tuổi | Có |
| `Annual Income (k$)` | Thu nhập hàng năm, đơn vị nghìn đô la | Có |
| `Spending Score (1-100)` | Điểm chi tiêu từ 1 đến 100 | Có |

`CustomerID` chỉ dùng để nhận diện, không phản ánh mức độ giống nhau giữa khách hàng. Cấu hình mô hình cuối cũng không đưa `Gender` vào tập đặc trưng.

## Luồng hoạt động

```text
config.yaml ── thiết lập đường dẫn, đặc trưng và tham số
      │
      ├── data/raw/Mall_Customers.csv
      │       ├── src/eda.py ─────────────── khảo sát và biểu đồ dữ liệu
      │       ├── src/preprocess.py ──────── làm sạch, mã hóa, chuẩn hóa
      │       └── train/evaluate/experiments/cluster_analysis
      │                                      huấn luyện và phân tích nhóm
      │
      └── src/pipeline.py ────────────────── điều phối toàn bộ các bước
                                               │
                 progress_results/, reports/, logs/, models/kmeans/
                                               │
                                    src/predict.py: nhóm khách mới
```

Pipeline tự động chạy lần lượt: khảo sát dữ liệu, tiền xử lý, huấn luyện K-Means, đánh giá, thực nghiệm cấu hình, phân tích cụm, minh họa hội tụ, quản lý biểu đồ và tạo báo cáo. Các module đọc tham số tập trung qua `src/config.py` từ `config.yaml`.

### Vai trò từng module và mối liên hệ

| Module | Chức năng | Liên hệ trong luồng |
|---|---|---|
| `config.py` | Đọc cấu hình và phân giải đường dẫn | Được các module khác dùng chung để tránh lặp tham số. |
| `feature_processing.py` | Làm sạch bản ghi, chọn và kiểm tra đặc trưng, mã hóa giới tính | Huấn luyện, đánh giá, phân tích cụm và dự đoán dùng cùng cách chuẩn bị feature. |
| `eda.py` | Kiểm tra dữ liệu, ngoại lệ và tạo biểu đồ khám phá | Đọc dữ liệu gốc; hình được đưa vào danh mục và báo cáo. |
| `preprocess.py` | Làm sạch, mã hóa giới tính khi cần, chuẩn hóa dữ liệu | Chuẩn bị các bản CSV cho huấn luyện và thực nghiệm. |
| `models/kmeans.py` | Cài đặt K-Means bằng NumPy | Được huấn luyện, đánh giá và dự đoán sử dụng. |
| `train.py` | Thử nhiều K và tạo kết quả Elbow | Kết quả được `evaluate.py` và báo cáo sử dụng. |
| `evaluate.py` | Tính Silhouette, Davies–Bouldin, Calinski–Harabasz | So sánh K để hỗ trợ chọn cấu hình. |
| `experiments.py` | So sánh scaler và tập đặc trưng | Bổ sung bằng chứng về lựa chọn tiền xử lý. |
| `train_final_model.py` | Huấn luyện cấu hình cuối, lưu model, scaler và metadata | Tạo đúng bộ artifacts mà `predict.py` nạp. |
| `cluster_analysis.py` | Tạo hồ sơ và mô tả từng cụm | Dùng nhãn phân cụm để diễn giải nhóm khách hàng. |
| `demo_convergence.py` | Minh họa quá trình tâm cụm ổn định | Tạo hình giải thích hoạt động thuật toán. |
| `figure_reviews.py` | Lưu nhận xét và đánh giá riêng cho từng biểu đồ | Được danh mục ảnh và báo cáo dùng chung. |
| `figure_manager.py` | Quét ảnh, ghi danh mục kèm nhận xét và đánh giá | Danh mục được `report.py` dùng để tạo báo cáo có hình và diễn giải. |
| `report.py` | Tổng hợp bảng, hình, nhận xét và đánh giá thành Markdown/HTML/PDF | Đầu ra cuối của pipeline. |
| `tracking.py` | Ghi thông tin lần chạy và metrics | Được pipeline dùng để lưu trạng thái và thời gian. |
| `pipeline.py` | Điều phối các bước theo đúng thứ tự | Kết nối toàn bộ module trên thành một lệnh chạy. |
| `predict.py` | Nhận thông tin khách mới và dự đoán cụm | Dùng model và scaler đã lưu từ `train_final_model.py`. |

Notebook hợp nhất trong `notebooks/` trình bày EDA đến dự đoán khách mới; nó không thay thế luồng chạy chuẩn `python -m src.pipeline`.

## Cấu hình mô hình hiện tại

Theo `config.yaml`, mô hình cuối dùng:

- Thuật toán: K-Means tự cài đặt bằng NumPy (`src/models/kmeans.py`).
- Số cụm: `K = 6`.
- Đặc trưng: `Age`, `Annual Income (k$)`, `Spending Score (1-100)`.
- Chuẩn hóa: `StandardScaler`.
- Khởi tạo tâm: `k-means++`.
- `random_state = 42`, tối đa 300 vòng lặp.

Mô hình đã lưu tại `models/kmeans/mall_customer_kmeans.pkl`; bộ chuẩn hóa đi kèm là `models/kmeans/standard_scaler.pkl`. `models/kmeans/model_metadata.json` ghi lại cấu hình và kết quả của lần huấn luyện đã lưu.

Ngoài mô hình cuối, dự án thử nghiệm K từ 2 đến 10, hai tập đặc trưng (có và không có giới tính) và hai cách chuẩn hóa (`StandardScaler`, `MinMaxScaler`). Mục đích là so sánh ảnh hưởng của các lựa chọn này. Cấu hình so sánh riêng trong `config.yaml` là có giới tính + MinMaxScaler + K=2; đây không phải cấu hình mô hình cuối.

## Cấu trúc thư mục

```text
Mall_Customer_Clustering/
├── agents/                 Ghi chú về bài toán, quy tắc và quy trình dự án
├── data/
│   ├── raw/                 Dữ liệu gốc Mall_Customers.csv
│   └── processed/           CSV sau chuẩn hóa
├── docs/                   Nhật ký thực nghiệm và danh mục hình
├── logs/                   Nhật ký chạy pipeline
├── models/
│   ├── kmeans/              Mô hình cuối, scaler và metadata đã lưu
│   └── preprocessing/      Scaler của bước tiền xử lý tổng quát
├── notebooks/               Một notebook EDA, phân cụm và dự đoán
├── progress_results/        Bảng kết quả, phân cụm, metrics và tracking runs
├── reports/
│   ├── figures/             Biểu đồ được tạo trong quá trình phân tích
│   └── report_*.md/html/pdf Báo cáo theo từng lần chạy
├── src/
│   ├── feature_processing.py Dùng chung bước làm sạch và chuẩn bị feature
│   ├── models/kmeans.py     Cài đặt thuật toán K-Means
│   ├── cluster_analysis.py  Mô tả đặc điểm cụm khách hàng
│   ├── config.py            Đọc config.yaml và phân giải đường dẫn
│   ├── demo_convergence.py  Minh họa tâm cụm hội tụ
│   ├── eda.py               Phân tích khám phá dữ liệu
│   ├── evaluate.py          Đánh giá nhiều giá trị K
│   ├── experiments.py       So sánh đặc trưng và scaler
│   ├── figure_reviews.py   Nhận xét và đánh giá từng biểu đồ
│   ├── figure_manager.py    Quản lý danh mục biểu đồ
│   ├── pipeline.py          Chạy tuần tự toàn bộ quy trình
│   ├── predict.py           Dự đoán cụm cho khách mới
│   ├── preprocess.py        Làm sạch, mã hóa và chuẩn hóa dữ liệu
│   ├── report.py            Tạo báo cáo
│   ├── tracking.py          Ghi thông tin lần chạy và metrics
│   ├── train.py             Huấn luyện và tạo kết quả Elbow
│   └── train_final_model.py Huấn luyện và lưu mô hình cuối
├── tests/                   Kiểm thử K-Means và tiền xử lý
├── config.yaml              Cấu hình dùng chung
├── requirements.txt         Danh sách thư viện Python
└── README.md
```

Các bản sao trước khi chạy pipeline được giữ trong `backups/` trên máy này; thư mục này được loại khỏi Git.

## Cài đặt

Yêu cầu Python 3.12. Các phiên bản thư viện trực tiếp đã được ghim trong `requirements.txt`. Trong PowerShell, chọn interpreter Python 3.12 rồi tạo môi trường riêng:

```powershell
$python312 = "$env:LOCALAPPDATA\Programs\Python\Python312\python.exe"
& $python312 --version
& $python312 -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install -r requirements.txt
```

Nếu Python 3.12 được cài ở vị trí khác, thay `$python312` bằng đường dẫn tới interpreter đó. Không dùng Python MSYS nếu thiếu wheel Windows cho NumPy/SciPy. Nếu PowerShell chặn kích hoạt môi trường, gọi trực tiếp `.venv\Scripts\python.exe`.

## Cách sử dụng

Chạy toàn bộ quy trình phân tích và tạo báo cáo:

```powershell
python -m src.pipeline
```

Huấn luyện riêng mô hình cuối theo cấu hình trong `config.yaml`:

```powershell
python -m src.train_final_model
```

Dự đoán cụm cho một khách hàng mới (chương trình hỏi tuổi, thu nhập và điểm chi tiêu):

```powershell
python -m src.predict
```

Có thể mở hai notebook trong Jupyter để xem các bước tương tác:

- `notebooks/mall_customer_clustering.ipynb`: notebook duy nhất, gồm EDA, tiền xử lý, scaling, chọn K, đánh giá, hồ sơ cụm, dự đoán mẫu và cấu hình so sánh.
- Các ảnh trong notebook được lưu thành file riêng dưới `notebooks/reports/figures/customer_segmentation/unified/`; kết quả bảng lưu dưới `notebooks/progress_results/unified_notebook/`.
- Cấu hình so sánh `with_gender + MinMaxScaler + K=2` được ghi tách biệt; mô hình cuối của pipeline chính vẫn là `without_gender + StandardScaler + K=6`.

## Kết quả và báo cáo

- `data/processed/`: dữ liệu đã chuẩn hóa theo StandardScaler và MinMaxScaler.
- `progress_results/`: kết quả Elbow, metrics, thực nghiệm, hồ sơ cụm và dữ liệu khách hàng được gán cụm.
- `models/kmeans/`: mô hình và scaler đã lưu, cùng metadata lần huấn luyện.
- `reports/`: báo cáo Markdown, HTML, PDF và hình minh họa.
- `logs/pipeline.log`: nhật ký các giai đoạn pipeline.
- `docs/experiment_logs.csv`: nhật ký tracking các lần chạy.

Các báo cáo có dấu thời gian; hãy mở báo cáo có thời gian mới nhất để xem kết quả gần nhất. Một lần chạy pipeline có thể tạo thêm hoặc cập nhật các file kết quả.

## Kiểm thử

Chạy các kiểm thử hiện có bằng:

```powershell
python -m pytest -v
```

## Lưu ý khi đọc kết quả

- Elbow/Inertia, Silhouette, Davies–Bouldin và Calinski–Harabasz là các góc nhìn khác nhau để so sánh cấu hình; không có một con số duy nhất tự động chứng minh nhóm là hoàn hảo.
- Mã cụm như `0` đến `5` chỉ là nhãn kỹ thuật. Cụm số 0 không tốt hơn cụm số 1; hãy xem hồ sơ tuổi, thu nhập và chi tiêu để hiểu ý nghĩa từng nhóm.
- `config.yaml` là nguồn cấu hình để huấn luyện/dự đoán mới. `model_metadata.json` mô tả mô hình đã lưu. Nếu thay cấu hình, hãy huấn luyện lại mô hình cuối trước khi dự đoán để mô hình, scaler và đặc trưng khớp nhau.
