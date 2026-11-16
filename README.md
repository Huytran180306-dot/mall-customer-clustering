# DỰ ÁN PHÂN CỤM KHÁCH HÀNG TRUNG TÂM THƯƠNG MẠI
## Mall Customer Clustering Using K-Means

---

## 1. Giới thiệu đề tài

Dự án **Mall Customer Clustering** được thực hiện nhằm phân tích và phân nhóm khách hàng của một trung tâm thương mại dựa trên các đặc điểm cá nhân và hành vi chi tiêu.

Dự án sử dụng thuật toán học máy không giám sát K-Means (K-Means Clustering) để tìm kiếm những nhóm khách hàng có đặc điểm tương đồng.

Thông qua việc phân cụm, doanh nghiệp có thể hiểu rõ hơn về đặc điểm của từng nhóm khách hàng, từ đó hỗ trợ xây dựng chiến lược tiếp thị, chăm sóc khách hàng và kinh doanh phù hợp.

## 2. Mục tiêu dự án

Các mục tiêu chính:

- Phân tích tổng quan dữ liệu khách hàng.
- Kiểm tra chất lượng và cấu trúc dữ liệu.
- Tiền xử lý và chuẩn hóa dữ liệu.
- Xây dựng thuật toán K-Means.
- Xác định số cụm khách hàng phù hợp.
- Đánh giá chất lượng kết quả phân cụm.
- Phân tích đặc điểm của từng nhóm khách hàng.
- Xây dựng mô hình K-Means hoàn chỉnh để dự đoán nhóm khách hàng mới.
- Xây dựng quy trình thực thi tự động (Pipeline).

## 3. Bộ dữ liệu

Dự án sử dụng bộ dữ liệu Mall Customers.

**Tên tệp dữ liệu:** `Mall_Customers.csv`

**Quy mô dữ liệu:**
- Tổng số bản ghi: 200 khách hàng.
- Tổng số thuộc tính: 5.
- Dữ liệu gồm cả thuộc tính dạng số và dạng phân loại.

### Các thuộc tính trong bộ dữ liệu

| Thuộc tính | Ý nghĩa |
|---|---|
| CustomerID | Mã định danh khách hàng |
| Gender | Giới tính |
| Age | Độ tuổi |
| Annual Income (k$) | Thu nhập hàng năm |
| Spending Score (1-100) | Điểm chi tiêu của khách hàng |

Trong quá trình phân cụm, CustomerID được sử dụng để nhận diện khách hàng, không được đưa vào tập đặc trưng huấn luyện.

Các đặc trưng sử dụng cho mô hình gồm:
- Gender
- Age
- Annual Income (k$)
- Spending Score (1-100)

## 4. Công nghệ và thư viện sử dụng

| Công nghệ | Chức năng |
|---|---|
| Python 3.12 | Ngôn ngữ lập trình chính |
| Pandas | Xử lý dữ liệu |
| NumPy | Tính toán số học |
| Matplotlib | Trực quan hóa dữ liệu |
| Seaborn | Xây dựng biểu đồ thống kê |
| Scikit-learn | Chuẩn hóa dữ liệu và đánh giá mô hình |
| Joblib | Lưu và tải mô hình, bộ chuẩn hóa |
| Jupyter Notebook | Phân tích và trình bày kết quả |
| Pytest | Kiểm thử chương trình |
| Git | Quản lý mã nguồn |

## 5. Cấu trúc thư mục dự án

```text
Mall_Customer_Clustering/
│
├── agents/
│   ├── problem_definition.md
│   ├── project_rules.md
│   └── workflow.md
│
├── data/
│   ├── raw/
│   │   └── Mall_Customers.csv
│   │
│   └── processed/
│       ├── Mall_Customers_StandardScaler.csv
│       └── Mall_Customers_MinMaxScaler.csv
│
├── docs/
│   └── images/
│
├── logs/
│
├── models/
│   ├── kmeans/
│   └── preprocessing/
│
├── notebooks/
│   ├── 01_exploratory_data_analysis.ipynb
│   └── 02_kmeans_clustering.ipynb
│
├── progress_results/
│
├── reports/
│   ├── archive/
│   └── figures/
│       └── customer_segmentation/
│
├── src/
│   ├── models/
│   │   └── kmeans.py
│   │
│   ├── cluster_analysis.py
│   ├── demo_convergence.py
│   ├── eda.py
│   ├── evaluate.py
│   ├── experiments.py
│   ├── figure_manager.py
│   ├── pipeline.py
│   ├── predict.py
│   ├── preprocess.py
│   ├── report.py
│   ├── tracking.py
│   ├── train.py
│   └── train_final_model.py
│
├── tests/
│   ├── test_kmeans.py
│   └── test_preprocessing.py
│
├── .gitignore
├── config.yaml
├── README.md
└── requirements.txt
```

## 6. Quy trình thực hiện dự án

Dự án được triển khai theo các giai đoạn sau:

### Giai đoạn 1: Phân tích khám phá dữ liệu (EDA)

Thực hiện:
- Đọc dữ liệu khách hàng.
- Kiểm tra số lượng bản ghi và thuộc tính.
- Kiểm tra dữ liệu thiếu.
- Kiểm tra dữ liệu trùng lặp.
- Thống kê mô tả.
- Phân tích phân phối các thuộc tính.
- Xây dựng biểu đồ trực quan.
- Phân tích mối tương quan giữa các thuộc tính.

### Giai đoạn 2: Tiền xử lý dữ liệu

Các công việc:
- Lựa chọn những thuộc tính phù hợp.
- Mã hóa thuộc tính Gender.
- Chuẩn hóa dữ liệu bằng StandardScaler.
- Chuẩn hóa dữ liệu bằng MinMaxScaler.
- Lưu dữ liệu sau tiền xử lý.
- Lưu bộ chuẩn hóa bằng Joblib.

Hai phương pháp chuẩn hóa được sử dụng để phục vụ quá trình thực nghiệm và so sánh.

### Giai đoạn 3: Xây dựng thuật toán K-Means

Dự án xây dựng lớp KMeans tại:

`src/models/kmeans.py`

Các thành phần chính:
- Khởi tạo các tâm cụm.
- Tính khoảng cách giữa dữ liệu và tâm cụm.
- Gán dữ liệu vào cụm gần nhất.
- Cập nhật vị trí tâm cụm.
- Kiểm tra điều kiện hội tụ.
- Tính toán Inertia.
- Dự đoán cụm cho dữ liệu mới.

Thuật toán được xây dựng theo hướng lập trình hướng đối tượng.

### Giai đoạn 4: Xác định số cụm tối ưu

Sử dụng phương pháp Elbow để khảo sát số lượng cụm K.

Thực nghiệm với các giá trị:

`K = 2 đến K = 10`

Dựa trên sự thay đổi của Inertia để quan sát điểm khuỷu và lựa chọn số lượng cụm phù hợp.

### Giai đoạn 5: Đánh giá chất lượng phân cụm

Sử dụng ba chỉ số đánh giá:

**Silhouette Score**

Đánh giá mức độ gần gũi của một điểm dữ liệu với cụm của nó so với các cụm khác. Giá trị càng cao thường thể hiện kết quả phân cụm càng tốt.

**Davies-Bouldin Index**

Đánh giá sự tương đồng giữa các cụm. Giá trị càng thấp thường thể hiện các cụm càng tách biệt.

**Calinski-Harabasz Score**

Đánh giá tỷ lệ giữa độ phân tán giữa các cụm và độ phân tán bên trong cụm. Giá trị càng cao thường thể hiện chất lượng phân cụm càng tốt.

### Giai đoạn 6: Thực nghiệm và so sánh

Dự án thực hiện các thử nghiệm với:
- Không sử dụng thuộc tính giới tính.
- Có sử dụng thuộc tính giới tính.
- StandardScaler.
- MinMaxScaler.
- Nhiều giá trị K khác nhau.

Mục tiêu là so sánh ảnh hưởng của lựa chọn đặc trưng và phương pháp chuẩn hóa đến chất lượng phân cụm.

### Giai đoạn 7: Phân tích đặc điểm từng cụm

Sau khi phân cụm, tiến hành:
- Thống kê số lượng khách hàng trong mỗi cụm.
- Tính độ tuổi trung bình.
- Tính thu nhập trung bình.
- Tính điểm chi tiêu trung bình.
- Phân tích phân bố giới tính.
- Trực quan hóa kết quả.

### Giai đoạn 8: Lưu mô hình và dự đoán

Mô hình cuối cùng được xây dựng tại:

`src/train_final_model.py`

Mô hình và bộ chuẩn hóa được lưu để sử dụng lại.

Chương trình dự đoán được xây dựng tại:

`src/predict.py`

Người dùng có thể nhập thông tin khách hàng:
- Giới tính.
- Độ tuổi.
- Thu nhập hàng năm.
- Điểm chi tiêu.

Hệ thống thực hiện tiền xử lý dữ liệu và dự đoán nhóm khách hàng tương ứng.

### Giai đoạn 9: Tự động hóa Pipeline

Tệp thực thi chính:

`src/pipeline.py`

Pipeline điều phối các giai đoạn phân tích, huấn luyện, đánh giá, thực nghiệm, quản lý biểu đồ và tạo báo cáo.

Quá trình thực thi được ghi nhận thông qua hệ thống Log và Tracking.

## 7. Kết quả thực nghiệm

### 7.1. Kết quả phân tích dữ liệu

- Bộ dữ liệu gồm 200 khách hàng.
- Có 5 thuộc tính ban đầu.
- Không phát hiện dữ liệu thiếu trong kết quả kiểm tra.
- Không phát hiện bản ghi trùng lặp trong kết quả kiểm tra.

### 7.2. Kết quả đánh giá mô hình

Các chỉ số được sử dụng để so sánh chất lượng phân cụm:

- Silhouette Score.
- Davies-Bouldin Index.
- Calinski-Harabasz Score.
- Inertia.

Kết quả thực nghiệm cho thấy việc lựa chọn đặc trưng và phương pháp chuẩn hóa có ảnh hưởng đến chất lượng phân cụm.

### 7.3. Mô hình cuối cùng

Cấu hình mô hình cuối cùng đã được chạy thử:

- Thuật toán: K-Means.
- Số cụm: K = 2.
- Phương pháp chuẩn hóa: MinMaxScaler.
- Số thuộc tính đầu vào: 4.
- Bộ dữ liệu: 200 khách hàng.

Mô hình và bộ chuẩn hóa được lưu bằng Joblib.

### 7.4. Kết quả dự đoán

Chương trình dự đoán đã được chạy thử thành công với thông tin khách hàng nhập từ bàn phím.

Kết quả trả về mã cụm khách hàng tương ứng.

## 8. Hướng dẫn cài đặt và sử dụng

### Bước 1: Tải dự án

```bash
git clone <LINK_GITHUB_CUA_BAN>
```

### Bước 2: Di chuyển vào thư mục dự án

```bash
cd Mall_Customer_Clustering
```

### Bước 3: Tạo môi trường ảo

```bash
python -m venv .venv
```

### Bước 4: Kích hoạt môi trường ảo

Đối với Windows PowerShell:

```powershell
.\.venv\Scripts\Activate.ps1
```

### Bước 5: Cài đặt thư viện

```bash
pip install -r requirements.txt
```

### Bước 6: Chạy toàn bộ Pipeline

```bash
python src/pipeline.py
```

### Bước 7: Chạy chương trình dự đoán

```bash
python src/predict.py
```

### Bước 8: Kiểm thử chương trình

```bash
python -m pytest -v
```

## 9. Các Notebook trong dự án

Dự án sử dụng hai Notebook chính:

**01_exploratory_data_analysis.ipynb**

Thực hiện phân tích khám phá dữ liệu, thống kê mô tả, trực quan hóa và đánh giá mối tương quan giữa các thuộc tính.

**02_kmeans_clustering.ipynb**

Trình bày quá trình tiền xử lý dữ liệu, xây dựng K-Means, xác định số cụm, đánh giá chất lượng và phân tích đặc điểm khách hàng sau phân cụm.

## 10. Hướng phát triển

Một số hướng phát triển trong tương lai:

- Thử nghiệm thêm các thuật toán phân cụm khác.
- Cải thiện phương pháp lựa chọn số cụm.
- Bổ sung thêm các chỉ số đánh giá.
- Xây dựng giao diện trực quan để người dùng nhập thông tin khách hàng.
- Phát triển hệ thống phân tích khách hàng tương tác.
- Tích hợp mô hình vào ứng dụng hỗ trợ quản lý và tiếp thị.

## 11. Kết luận

Dự án đã triển khai quy trình phân tích và phân cụm khách hàng bằng thuật toán K-Means.

Thông qua các bước phân tích dữ liệu, tiền xử lý, xây dựng mô hình, thực nghiệm và đánh giá, dự án cho thấy khả năng ứng dụng kỹ thuật học máy không giám sát trong bài toán phân nhóm khách hàng.

Ngoài việc xây dựng mô hình, dự án còn triển khai hệ thống Pipeline, kiểm thử chương trình, lưu trữ mô hình và hỗ trợ dự đoán nhóm khách hàng mới.

---

**Tên dự án:** Mall Customer Clustering

**Lĩnh vực:** Khai phá dữ liệu và Học máy

**Thuật toán chính:** K-Means Clustering

**Ngôn ngữ lập trình:** Python