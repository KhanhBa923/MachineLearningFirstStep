import numpy as np
import tensorflow as tf


# ============================================================
# Ví dụ Clustering thực tế: Phân nhóm quán cà phê trong thành phố
# ============================================================
# Mục tiêu:
# - Không dùng nhãn có sẵn (unsupervised learning)
# - Từ dữ liệu vận hành quán, tự tìm ra các nhóm quán tương đồng
#
# Feature (mỗi quán 6 chỉ số):
# 1) gia_trung_binh_kvnd      : Giá trung bình 1 đơn (nghìn VND)
# 2) diem_danh_gia            : Điểm đánh giá trung bình (1-5)
# 3) tg_phuc_vu_phut          : Thời gian phục vụ trung bình (phút)
# 4) do_day_cho_percent       : Mức sử dụng chỗ ngồi trung bình (%)
# 5) ty_le_khach_quay_lai     : % khách quay lại
# 6) don_moi_ngay             : Số đơn trung bình mỗi ngày
# ============================================================


FEATURE_NAMES = [
    "gia_trung_binh_kvnd",
    "diem_danh_gia",
    "tg_phuc_vu_phut",
    "do_day_cho_percent",
    "ty_le_khach_quay_lai",
    "don_moi_ngay",
]


def create_synthetic_coffee_shop_data(seed=42):
    # Tạo dữ liệu giả lập theo 3 nhóm quán phổ biến ngoài thực tế
    # (chỉ để học thuật, không phải dữ liệu khảo sát chính thức)
    rng = np.random.default_rng(seed)

    # Nhóm A: Giá mềm - phục vụ nhanh - lượng đơn cao
    a = rng.normal(
        loc=[28, 4.0, 4.5, 72, 45, 420],
        scale=[6, 0.25, 1.2, 9, 10, 80],
        size=(180, 6),
    )

    # Nhóm B: Premium - đánh giá cao - khách quay lại cao
    b = rng.normal(
        loc=[78, 4.6, 8.5, 58, 72, 210],
        scale=[12, 0.18, 1.8, 10, 9, 55],
        size=(160, 6),
    )

    # Nhóm C: Tầm trung - đông khách giờ cao điểm - phục vụ vừa
    c = rng.normal(
        loc=[46, 4.2, 6.5, 85, 57, 360],
        scale=[8, 0.2, 1.4, 7, 8, 65],
        size=(170, 6),
    )

    x = np.vstack([a, b, c]).astype(np.float32)

    # Ép dữ liệu về dải hợp lý để giống dữ liệu vận hành thực tế hơn
    x[:, 0] = np.clip(x[:, 0], 12, 130)   # giá trung bình (k VND)
    x[:, 1] = np.clip(x[:, 1], 3.0, 5.0)  # rating
    x[:, 2] = np.clip(x[:, 2], 2.0, 20.0) # thời gian phục vụ
    x[:, 3] = np.clip(x[:, 3], 25, 100)   # độ đầy chỗ
    x[:, 4] = np.clip(x[:, 4], 10, 95)    # khách quay lại
    x[:, 5] = np.clip(x[:, 5], 40, 900)   # đơn/ngày

    return x


def zscore_normalize(x):
    # Chuẩn hóa Z-score để các feature cùng thang đo,
    # tránh feature lớn (vd: đơn/ngày) lấn át feature nhỏ (vd: rating)
    mean = x.mean(axis=0)
    std = x.std(axis=0) + 1e-6
    x_norm = (x - mean) / std
    return x_norm, mean, std


def kmeans_tensorflow(x_norm, k=3, iterations=40, seed=42):
    # K-Means tự cài bằng TensorFlow:
    # 1) Khởi tạo centroid ngẫu nhiên từ dữ liệu
    # 2) Gán mỗi điểm vào centroid gần nhất
    # 3) Cập nhật centroid = trung bình các điểm trong cụm
    # Lặp lại nhiều lần để hội tụ

    tf.random.set_seed(seed)

    x_tf = tf.convert_to_tensor(x_norm, dtype=tf.float32)
    num_points = tf.shape(x_tf)[0]

    # Chọn ngẫu nhiên k điểm làm centroid ban đầu
    init_indices = tf.random.shuffle(tf.range(num_points))[:k]
    centroids = tf.gather(x_tf, init_indices)

    for _ in range(iterations):
        # Tính khoảng cách bình phương từ từng điểm đến từng centroid
        # distances shape: (N, K)
        distances = tf.reduce_sum(
            tf.square(tf.expand_dims(x_tf, axis=1) - tf.expand_dims(centroids, axis=0)),
            axis=2,
        )

        # Gán điểm vào cụm gần nhất
        assignments = tf.argmin(distances, axis=1, output_type=tf.int32)

        # Tính centroid mới bằng trung bình theo cụm
        new_centroids = tf.math.unsorted_segment_mean(x_tf, assignments, num_segments=k)

        # Xử lý cụm rỗng (nếu có): giữ centroid cũ
        counts = tf.math.unsorted_segment_sum(
            tf.ones_like(assignments, dtype=tf.float32), assignments, num_segments=k
        )
        counts = tf.expand_dims(counts, axis=1)  # shape (K, 1)
        centroids = tf.where(counts > 0, new_centroids, centroids)

    return assignments.numpy(), centroids.numpy()


def describe_cluster_profiles(centroids_norm, mean, std):
    # Đưa centroid từ không gian chuẩn hóa về thang đo gốc để dễ đọc
    centroids_real = centroids_norm * std + mean

    print("\n=== Hồ sơ các cụm (centroid) ===")
    for idx, c in enumerate(centroids_real):
        print(f"\nCụm {idx}:")
        for name, value in zip(FEATURE_NAMES, c):
            print(f"  - {name}: {value:.2f}")

    return centroids_real


def suggest_cluster_names(centroids_real):
    # Đặt tên gợi ý cho cụm dựa trên 2 chỉ số dễ hiểu: giá trung bình + số đơn/ngày
    # (đây chỉ là heuristic để đọc kết quả, không ảnh hưởng thuật toán)
    avg_price = centroids_real[:, 0]
    daily_orders = centroids_real[:, 5]

    names = {}
    high_price_cluster = int(np.argmax(avg_price))
    high_orders_cluster = int(np.argmax(daily_orders))

    names[high_price_cluster] = "Premium trải nghiệm"
    names[high_orders_cluster] = "Giá mềm - lưu lượng cao"

    for cluster_id in range(len(centroids_real)):
        if cluster_id not in names:
            names[cluster_id] = "Tầm trung - cân bằng"

    return names


def predict_new_shops(new_shops, centroids_norm, mean, std):
    # Dự đoán cụm cho quán mới bằng centroid gần nhất
    new_shops_norm = (new_shops - mean) / std

    distances = np.sum(
        (new_shops_norm[:, None, :] - centroids_norm[None, :, :]) ** 2,
        axis=2,
    )
    cluster_ids = np.argmin(distances, axis=1)
    return cluster_ids


def main():
    # 1) Tạo dữ liệu quán cà phê mô phỏng thực tế
    x = create_synthetic_coffee_shop_data(seed=42)

    # 2) Chuẩn hóa dữ liệu
    x_norm, mean, std = zscore_normalize(x)

    # 3) Chạy K-Means
    assignments, centroids_norm = kmeans_tensorflow(x_norm, k=3, iterations=45, seed=42)

    # 4) In hồ sơ cụm để hiểu đặc điểm từng nhóm quán
    centroids_real = describe_cluster_profiles(centroids_norm, mean, std)

    # 5) Đặt tên gợi ý cho các cụm
    cluster_names = suggest_cluster_names(centroids_real)

    print("\n=== Tên gợi ý cho cụm ===")
    for cluster_id, cluster_name in cluster_names.items():
        count = int(np.sum(assignments == cluster_id))
        print(f"Cụm {cluster_id}: {cluster_name} | số quán: {count}")

    # 6) Dự đoán thử vài quán mới
    # Format feature giữ đúng thứ tự trong FEATURE_NAMES
    new_shops = np.array(
        [
            [25, 4.1, 4.0, 78, 43, 500],   # kiểu giá mềm, đơn cao
            [88, 4.8, 9.5, 54, 76, 180],   # kiểu premium
            [48, 4.3, 6.8, 82, 59, 340],   # kiểu tầm trung
        ],
        dtype=np.float32,
    )

    new_cluster_ids = predict_new_shops(new_shops, centroids_norm, mean, std)

    print("\n=== Gán cụm cho quán mới ===")
    for i, (shop, cluster_id) in enumerate(zip(new_shops, new_cluster_ids), start=1):
        print(f"Quán mới {i}: {shop}")
        print(f" -> thuộc Cụm {cluster_id}: {cluster_names[int(cluster_id)]}")


if __name__ == "__main__":
    main()
