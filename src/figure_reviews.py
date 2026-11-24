"""Short interpretations for images included in generated reports."""

FIGURE_REVIEWS = {
    "Age_boxplot.png": ("Phân bố tuổi và các điểm nằm ngoài khoảng tứ phân vị.", "Dùng để nhận diện độ phân tán và tuổi khác biệt; không loại khách hàng chỉ dựa trên biểu đồ."),
    "age_distribution.png": ("Tần suất khách hàng theo tuổi.", "Cho biết nhóm tuổi phổ biến và hình dạng phân bố; tuổi được giữ lại làm đặc trưng phân cụm."),
    "Annual_Income_k$_boxplot.png": ("Phân bố thu nhập năm và các giá trị ngoại lệ theo IQR.", "Giúp đánh giá độ phân tán thu nhập; ngoại lệ cần được xem xét, không tự động xóa."),
    "correlation_heatmap.png": ("Mức tương quan tuyến tính giữa các biến số.", "Cho biết quan hệ tuyến tính nhưng không chứng minh quan hệ nhân quả; các biến khác nhau vẫn có thể bổ sung thông tin cho K-Means."),
    "gender_distribution.png": ("Số lượng khách hàng theo giới tính.", "Kiểm tra mức cân bằng của biến giới tính; mô hình cuối không dùng biến này, còn cấu hình so sánh có thể dùng."),
    "income_distribution.png": ("Tần suất khách hàng theo thu nhập năm.", "Cho biết mức thu nhập phổ biến và độ lệch phân bố; nên đọc cùng biểu đồ scatter để xét hành vi chi tiêu."),
    "income_spending_scatter.png": ("Quan hệ giữa thu nhập năm và điểm chi tiêu.", "Các vùng điểm tập trung gợi ý khả năng phân khúc; tuổi và đánh giá chỉ số mới xác nhận chất lượng cụm."),
    "spending_distribution.png": ("Tần suất điểm chi tiêu của khách hàng.", "Cho thấy các mức chi tiêu phổ biến; phân bố không đều có thể ảnh hưởng đến tâm cụm."),
    "Spending_Score_1-100_boxplot.png": ("Phân bố điểm chi tiêu và các điểm ngoài khoảng tứ phân vị.", "Hữu ích để nhận diện độ phân tán; điểm xa không đồng nghĩa dữ liệu sai."),
    "elbow_method.png": ("Inertia giảm khi tăng số cụm K.", "Mức giảm bắt đầu dịu đáng kể khoảng K=5–6; Elbow là gợi ý, cần đối chiếu metrics và khả năng diễn giải."),
    "calinski_harabasz_score.png": ("So sánh độ tách cụm theo Calinski–Harabasz; giá trị cao hơn thường tốt hơn.", "Trong kết quả hiện tại K=6 cao nhất trong khoảng thử nghiệm; chỉ số này không đánh giá ý nghĩa kinh doanh."),
    "davies_bouldin_index.png": ("So sánh độ giống nhau giữa cụm theo Davies–Bouldin; giá trị thấp hơn thường tốt hơn.", "K=6 cho giá trị thấp nhất trong kết quả hiện tại; nên đọc cùng Silhouette và hồ sơ cụm."),
    "silhouette_score.png": ("So sánh độ gắn kết và tách biệt của cụm theo Silhouette; cao hơn thường tốt hơn.", "K=6 đạt điểm cao nhất trong kết quả hiện tại; điểm số không thay thế việc kiểm tra kích thước và đặc trưng từng cụm."),
    "kmeans_convergence.png": ("Diễn biến cập nhật tâm cụm qua các vòng lặp K-Means.", "Tâm ổn định cho thấy thuật toán hội tụ ở lần chạy minh họa; một lần chạy không đại diện mọi khởi tạo."),
    "ch_experiments.png": ("So sánh Calinski–Harabasz giữa các tổ hợp đặc trưng và scaler.", "Dùng để xem lựa chọn tiền xử lý ảnh hưởng độ tách cụm ra sao; chỉ so sánh các cấu hình trong cùng thử nghiệm."),
    "dbi_experiments.png": ("So sánh Davies–Bouldin giữa các tổ hợp đặc trưng và scaler.", "Cấu hình có chỉ số thấp hơn có cụm gọn/tách hơn theo metric này; cần cân nhắc cùng hai metric còn lại."),
    "silhouette_experiments.png": ("So sánh Silhouette giữa các tổ hợp đặc trưng và scaler.", "Cấu hình có điểm cao hơn thường có phân cụm tốt hơn theo metric này; kết quả còn phụ thuộc đặc trưng và K."),
}


def review_figure(filename, category=""):
    """Return an observation and assessment for a catalog image."""
    if filename in FIGURE_REVIEWS:
        return FIGURE_REVIEWS[filename]
    category = str(category).replace("_", " ").replace("/", " – ")
    return (
        f"Hình thuộc nhóm phân tích {category or 'khác'}.",
        "Đối chiếu hình với dữ liệu và chỉ số liên quan trước khi kết luận.",
    )
