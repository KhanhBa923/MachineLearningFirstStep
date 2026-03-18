import numpy as np
import tensorflow as tf


# ============================================================
# Ví dụ: Phân loại mức độ phát triển thành phố (3 lớp)
# ============================================================
# Lớp 0: phát triển thấp
# Lớp 1: phát triển trung bình
# Lớp 2: phát triển cao
#
# Dữ liệu dưới đây là dữ liệu TỰ SINH nhưng theo dải giá trị gần thực tế:
# - Mật độ dân số (người/km2): khoảng 100 -> 25,000
# - PM2.5 (µg/m3): khoảng 5 -> 120
# - CO2 bình quân đầu người (tấn/năm): khoảng 0.5 -> 20
# - Không gian xanh (m2/người): khoảng 2 -> 80
# - Điểm giao thông công cộng: 0 -> 100
# - GDP bình quân đầu người (nghìn USD): khoảng 2 -> 100
#
# Lưu ý: Đây không phải dữ liệu điều tra chính thức, chỉ là mô phỏng
# để học cách làm classification với bối cảnh đô thị thực tế hơn.
# ============================================================


CLASS_NAMES = ["Thấp", "Trung bình", "Cao"]


def create_city_dataset(num_samples=3000, seed=42):
    # Dùng numpy để mô phỏng dữ liệu linh hoạt hơn
    rng = np.random.default_rng(seed)

    # 1) Sinh các chỉ số theo dải hợp lý
    # Log-normal cho mật độ dân số để phản ánh phân bố lệch phải (nhiều thành phố mật độ vừa,
    # một số ít thành phố rất dày đặc)
    density = rng.lognormal(mean=np.log(3500), sigma=0.8, size=num_samples)
    density = np.clip(density, 100, 25000)

    pm25 = rng.normal(loc=38, scale=18, size=num_samples)
    pm25 = np.clip(pm25, 5, 120)

    co2 = rng.normal(loc=6.5, scale=3.0, size=num_samples)
    co2 = np.clip(co2, 0.5, 20)

    green_space = rng.normal(loc=24, scale=12, size=num_samples)
    green_space = np.clip(green_space, 2, 80)

    transit_score = rng.normal(loc=58, scale=20, size=num_samples)
    transit_score = np.clip(transit_score, 0, 100)

    gdp_kusd = rng.lognormal(mean=np.log(25), sigma=0.6, size=num_samples)
    gdp_kusd = np.clip(gdp_kusd, 2, 100)

    # 2) Tạo điểm phát triển tổng hợp (development_score)
    # Chuẩn hóa về [0, 1] cho từng nhóm chỉ số trước khi trộn trọng số
    gdp_norm = (gdp_kusd - 2) / (100 - 2)
    transit_norm = transit_score / 100
    green_norm = (green_space - 2) / (80 - 2)

    pollution_index = 0.6 * ((pm25 - 5) / (120 - 5)) + 0.4 * ((co2 - 0.5) / (20 - 0.5))

    # Mật độ "đẹp" thường ở mức đô thị hóa vừa phải đến cao,
    # quá thấp hoặc quá cao đều gây áp lực hạ tầng.
    density_quality = np.exp(-((density - 6000) ** 2) / (2 * (3500**2)))

    development_score = (
        0.30 * gdp_norm
        + 0.22 * transit_norm
        + 0.18 * green_norm
        + 0.15 * density_quality
        + 0.15 * (1 - pollution_index)
    )

    # Thêm nhiễu nhỏ để dữ liệu bớt "quá sạch", giống dữ liệu thực tế hơn
    development_score = np.clip(
        development_score + rng.normal(0, 0.035, size=num_samples), 0, 1
    )

    # 3) Gán nhãn 3 mức phát triển từ score
    # < 0.40: thấp, 0.40 - <0.67: trung bình, >= 0.67: cao
    labels = np.digitize(development_score, bins=[0.40, 0.67]).astype(np.int32)

    # 4) Gom feature theo đúng thứ tự cột
    # [density, pm25, co2, green_space, transit_score, gdp_kusd]
    features = np.stack(
        [density, pm25, co2, green_space, transit_score, gdp_kusd], axis=1
    ).astype(np.float32)

    return features, labels


def normalize_features(x_train, x_test):
    # Chuẩn hóa theo train set để model học ổn định hơn
    mean = np.mean(x_train, axis=0)
    std = np.std(x_train, axis=0) + 1e-6
    x_train_norm = (x_train - mean) / std
    x_test_norm = (x_test - mean) / std
    return x_train_norm, x_test_norm, mean, std


def build_tf_dataset(features, labels, batch_size=64, shuffle=True):
    ds = tf.data.Dataset.from_tensor_slices((features, labels))
    if shuffle:
        ds = ds.shuffle(buffer_size=len(features))
    ds = ds.batch(batch_size).prefetch(tf.data.AUTOTUNE)
    return ds


def build_model(input_dim):
    # Mô hình nhiều lớp nhỏ cho multi-class classification
    model = tf.keras.Sequential(
        [
            tf.keras.layers.Input(shape=(input_dim,)),
            tf.keras.layers.Dense(32, activation="relu"),
            tf.keras.layers.Dense(16, activation="relu"),
            tf.keras.layers.Dense(3, activation="softmax"),
        ]
    )

    # sparse_categorical_crossentropy dùng cho nhãn số nguyên 0/1/2
    model.compile(
        optimizer=tf.keras.optimizers.Adam(learning_rate=0.003),
        loss="sparse_categorical_crossentropy",
        metrics=["accuracy"],
    )
    return model


def predict_examples(model, mean, std):
    # 3 thành phố giả định để dự đoán thử
    # Thứ tự feature: [density, pm25, co2, green_space, transit_score, gdp_kusd]
    new_cities = np.array(
        [
            [12000, 16, 4.0, 35, 82, 55],   # xu hướng phát triển cao
            [4500, 42, 7.5, 18, 55, 22],    # xu hướng trung bình
            [900, 78, 11.0, 8, 28, 7],      # xu hướng thấp
        ],
        dtype=np.float32,
    )

    # Chuẩn hóa theo mean/std của train set trước khi predict
    new_cities_norm = (new_cities - mean) / std

    probs = model.predict(new_cities_norm, verbose=0)
    preds = np.argmax(probs, axis=1)

    print("\n=== Dự đoán thử theo bối cảnh thành phố ===")
    for idx, (city, prob, pred) in enumerate(zip(new_cities, probs, preds), start=1):
        print(f"Thành phố {idx} | feature={city}")
        print(
            f" -> p(Thấp)={prob[0]:.3f}, p(Trung bình)={prob[1]:.3f}, p(Cao)={prob[2]:.3f}"
        )
        print(f" -> Kết luận: {CLASS_NAMES[int(pred)]}")


def train_and_evaluate():
    # 1) Tạo dữ liệu mô phỏng gần thực tế
    x, y = create_city_dataset(num_samples=3000, seed=42)

    # 2) Chia train/test
    split = int(0.8 * len(x))
    x_train, y_train = x[:split], y[:split]
    x_test, y_test = x[split:], y[split:]

    # 3) Chuẩn hóa đặc trưng
    x_train_norm, x_test_norm, mean, std = normalize_features(x_train, x_test)

    # 4) Build tf.data.Dataset
    train_ds = build_tf_dataset(x_train_norm, y_train, batch_size=64, shuffle=True)
    test_ds = build_tf_dataset(x_test_norm, y_test, batch_size=64, shuffle=False)

    # 5) Build + train model
    model = build_model(input_dim=x_train_norm.shape[1])
    model.fit(train_ds, epochs=20, verbose=1)

    # 6) Đánh giá test
    test_loss, test_acc = model.evaluate(test_ds, verbose=0)
    print("\n=== Kết quả test ===")
    print("Test loss:", float(test_loss))
    print("Test accuracy:", float(test_acc))
    print(train_ds.cardinality().numpy(), "samples trong train set")
    print(test_ds.cardinality().numpy(), "samples trong test set")
    

    # 7) Dự đoán thử cho vài thành phố giả lập
    predict_examples(model, mean, std)


if __name__ == "__main__":
    train_and_evaluate()
