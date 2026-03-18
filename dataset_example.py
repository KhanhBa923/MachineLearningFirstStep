import tensorflow as tf


# ============================================================
# Ví dụ: Tạo và dùng tf.data.Dataset với dữ liệu đơn giản
# ============================================================
# Mục tiêu:
# 1) Tạo dữ liệu đầu vào (x) và nhãn (y)
# 2) Đóng gói thành Dataset
# 3) Áp dụng shuffle, batch, prefetch
# 4) Dùng dataset để train mô hình nhỏ
# ============================================================


def create_dataset():
    # Dữ liệu mẫu theo công thức y = 2x + 1
    # Dùng tf.range để tạo x từ 1 đến 100
    x = tf.cast(tf.range(1, 101), tf.float32)
    y = 2.0 * x + 1.0

    # Chuẩn hoá dữ liệu về khoảng nhỏ để train ổn định hơn
    # x_norm: [0.01 .. 1.00]
    # y_norm: [~0.015 .. 1.00]
    x_norm = x / 100.0
    y_norm = y / 201.0

    # from_tensor_slices: cắt dữ liệu theo từng sample
    # Mỗi phần tử dataset sẽ là 1 cặp (x_i, y_i)
    dataset = tf.data.Dataset.from_tensor_slices((x_norm, y_norm))

    # shuffle(100): trộn thứ tự mẫu để train ổn định hơn
    # batch(16): gom 16 sample thành 1 batch
    # prefetch(AUTOTUNE): chuẩn bị batch tiếp theo song song khi model đang train
    dataset = dataset.shuffle(100).batch(16).prefetch(tf.data.AUTOTUNE)

    return dataset


def inspect_dataset(dataset):
    # Lấy thử 1 batch đầu tiên để xem dữ liệu trông như thế nào
    for batch_x, batch_y in dataset.take(1):
        print("=== 1 batch mẫu ===")
        print("batch_x shape:", batch_x.shape)
        print("batch_y shape:", batch_y.shape)
        print("batch_x (5 phần tử đầu):", batch_x[:5].numpy())
        print("batch_y (5 phần tử đầu):", batch_y[:5].numpy())

# giống linear, nhưng dùng Keras API để thấy cách dùng dataset trong thực tế
def train_with_dataset(dataset): 
    # Mô hình rất đơn giản: 1 lớp Dense(1)
    # Input shape là (1,) vì mỗi sample chỉ có 1 đặc trưng x
    model = tf.keras.Sequential(
        [
            tf.keras.layers.Input(shape=(1,)),
            tf.keras.layers.Dense(1),
        ]
    )

    # Dùng MSE cho bài toán hồi quy.
    # Adam với learning rate vừa phải giúp hội tụ nhanh và ổn định.
    optimizer = tf.keras.optimizers.Adam(learning_rate=0.01)
    model.compile(optimizer=optimizer, loss="mse")

    # Dataset đang trả ra x có shape (batch,) -> cần (batch, 1) cho Dense
    # map() để reshape trước khi đưa vào model
    dataset_for_model = dataset.map(
        lambda batch_x, batch_y: (tf.expand_dims(batch_x, axis=-1), batch_y)
    )

    # Train vài epoch để thấy mô hình học được quan hệ y = 2x + 1
    history = model.fit(dataset_for_model, epochs=300, verbose=0)

    # Dự đoán thử với x = [10, 20, 30]
    test_x = tf.constant([[10.0], [20.0], [30.0]])

    # Vì model học trên dữ liệu đã chuẩn hoá, nên cần chuẩn hoá input test
    test_x_norm = test_x / 100.0
    pred_y_norm = model.predict(test_x_norm, verbose=0)

    # Quy đổi ngược từ y_norm về y gốc
    pred_y = pred_y_norm * 201.0

    print("\n=== Dự đoán thử ===")
    print("x:", test_x.numpy().reshape(-1))
    print("y dự đoán:", pred_y.reshape(-1))

    # In loss cuối để theo dõi chất lượng train
    print("loss cuối:", history.history["loss"][-1])
    
    weights, biases = model.layers[0].get_weights()
    print(f"Trọng số w học được: {weights}")
    print(f"Hằng số b học được: {biases}")


if __name__ == "__main__":
    ds = create_dataset()
    inspect_dataset(ds)
    train_with_dataset(ds)
