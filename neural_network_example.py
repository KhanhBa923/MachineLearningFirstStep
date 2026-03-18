"""
Neural Network (Mạng Nơ-ron) - Xây dựng từ đầu

Neural Network mô phỏng cách não bộ hoạt động:
  - Neurons (nơ-ron): nhận tín hiệu đầu vào, xử lý, rồi truyền tín hiệu ra
  - Layers (tầng): các nơ-ron xếp thành từng tầng
  - Weights (trọng số): độ mạnh/yếu của mỗi kết nối giữa các nơ-ron
  - Bias: giá trị hiệu chỉnh thêm cho mỗi nơ-ron

Quá trình học:
  1. Forward pass: đưa dữ liệu qua mạng, tính kết quả dự đoán
  2. Tính Loss: so sánh dự đoán với kết quả thật, đo "sai bao nhiêu"
  3. Backward pass (Backpropagation): tính đạo hàm, biết mỗi trọng số
     ảnh hưởng đến sai số bao nhiêu
  4. Cập nhật trọng số: điều chỉnh weights theo hướng giảm sai số

Ví dụ thực tế trong file này:
  1. Phân loại hoa Iris (3 loài) - bài toán kinh điển
  2. Dự đoán bệnh tiểu đường từ chỉ số sức khỏe
  3. Nhận dạng chữ số viết tay (MNIST đơn giản)
"""

import numpy as np


# =============================================================================
# PHẦN 1: CÁC HÀM KÍCH HOẠT (ACTIVATION FUNCTIONS)
# =============================================================================
# Hàm kích hoạt quyết định nơ-ron có "kích hoạt" (truyền tín hiệu) hay không.
# Không có hàm kích hoạt → mạng chỉ là phép tuyến tính, không học được gì phức tạp.

class Activation:
    """Tập hợp các hàm kích hoạt phổ biến."""

    @staticmethod
    def sigmoid(x):
        """
        Sigmoid: nén giá trị về khoảng (0, 1)
        Giống như "mức độ tự tin": 0 = không chắc, 1 = rất chắc
        Dùng cho: bài toán phân loại nhị phân (có/không, bệnh/khỏe)
        """
        # Clip để tránh overflow khi x quá lớn/nhỏ
        x = np.clip(x, -500, 500)
        return 1 / (1 + np.exp(-x))

    @staticmethod
    def sigmoid_derivative(x):
        """Đạo hàm sigmoid: dùng trong backpropagation."""
        s = Activation.sigmoid(x)
        return s * (1 - s)

    @staticmethod
    def relu(x):
        """
        ReLU (Rectified Linear Unit): giữ nguyên nếu dương, = 0 nếu âm
        Đơn giản nhưng hiệu quả, phổ biến nhất trong deep learning
        Giống như: "chỉ truyền tín hiệu khi có tín hiệu tích cực"
        """
        return np.maximum(0, x)

    @staticmethod
    def relu_derivative(x):
        """Đạo hàm ReLU: 1 nếu x > 0, 0 nếu x <= 0."""
        return (x > 0).astype(float)

    @staticmethod
    def softmax(x):
        """
        Softmax: chuyển vector thành phân phối xác suất (tổng = 1)
        Dùng cho: phân loại nhiều lớp (loại A: 70%, loại B: 20%, loại C: 10%)
        """
        # Trừ max để tránh overflow (trick ổn định số học)
        exp_x = np.exp(x - np.max(x, axis=1, keepdims=True))
        return exp_x / np.sum(exp_x, axis=1, keepdims=True)


# =============================================================================
# PHẦN 2: CÁC HÀM MẤT MÁT (LOSS FUNCTIONS)
# =============================================================================
# Loss function đo "mô hình sai bao nhiêu". Mục tiêu: giảm loss xuống thấp nhất.

class Loss:
    @staticmethod
    def binary_cross_entropy(y_true, y_pred):
        """
        Binary Cross-Entropy: dùng cho phân loại 2 lớp
        Phạt nặng khi mô hình "tự tin sai" (dự đoán 0.99 nhưng thực tế là 0)
        """
        y_pred = np.clip(y_pred, 1e-8, 1 - 1e-8)  # Tránh log(0)
        return -np.mean(y_true * np.log(y_pred) + (1 - y_true) * np.log(1 - y_pred))

    @staticmethod
    def categorical_cross_entropy(y_true, y_pred):
        """
        Categorical Cross-Entropy: dùng cho phân loại nhiều lớp
        y_true: one-hot encoded (vd: [0, 1, 0] = lớp thứ 2)
        """
        y_pred = np.clip(y_pred, 1e-8, 1 - 1e-8)
        return -np.mean(np.sum(y_true * np.log(y_pred), axis=1))


# =============================================================================
# PHẦN 3: LỚP NEURAL NETWORK
# =============================================================================

class NeuralNetwork:
    """
    Mạng nơ-ron nhiều tầng, xây dựng hoàn toàn từ đầu.

    Kiến trúc:
        Input Layer → Hidden Layer(s) → Output Layer

    Ví dụ: NeuralNetwork([4, 8, 3])
        - Input: 4 features (đặc trưng đầu vào)
        - Hidden: 8 nơ-ron (tầng ẩn xử lý trung gian)
        - Output: 3 nơ-ron (3 lớp phân loại)
    """

    def __init__(self, layer_sizes, activation='relu', output_activation='softmax',
                 learning_rate=0.01):
        """
        Args:
            layer_sizes: danh sách số nơ-ron mỗi tầng, vd [4, 8, 3]
            activation: hàm kích hoạt tầng ẩn ('relu' hoặc 'sigmoid')
            output_activation: hàm kích hoạt tầng output ('softmax' hoặc 'sigmoid')
            learning_rate: tốc độ học (bước nhảy khi cập nhật trọng số)
        """
        self.layer_sizes = layer_sizes
        self.learning_rate = learning_rate
        self.n_layers = len(layer_sizes) - 1  # Số tầng kết nối (không tính input)

        # Chọn hàm kích hoạt
        if activation == 'relu':
            self.activation = Activation.relu
            self.activation_deriv = Activation.relu_derivative
        else:
            self.activation = Activation.sigmoid
            self.activation_deriv = Activation.sigmoid_derivative

        self.output_activation = output_activation

        # --- Khởi tạo trọng số (Weight Initialization) ---
        # Dùng He initialization cho ReLU, Xavier cho sigmoid
        # Trọng số ban đầu không nên quá lớn (gradient explode) hay quá nhỏ (vanishing)
        self.weights = []
        self.biases = []

        for i in range(self.n_layers):
            n_in = layer_sizes[i]      # Số nơ-ron tầng trước
            n_out = layer_sizes[i + 1]  # Số nơ-ron tầng sau

            if activation == 'relu':
                # He initialization: phù hợp với ReLU
                scale = np.sqrt(2.0 / n_in)
            else:
                # Xavier initialization: phù hợp với sigmoid/tanh
                scale = np.sqrt(1.0 / n_in)

            # Trọng số ngẫu nhiên theo phân phối chuẩn, nhân với scale
            self.weights.append(np.random.randn(n_in, n_out) * scale)
            # Bias khởi tạo = 0
            self.biases.append(np.zeros((1, n_out)))

        # Lưu lịch sử loss để theo dõi quá trình học
        self.loss_history = []

    def forward(self, X):
        """
        Forward Pass: đưa dữ liệu qua mạng từ input → output.

        Tại mỗi tầng:
          z = X @ W + b          (tổ hợp tuyến tính)
          a = activation(z)      (kích hoạt phi tuyến)

        Lưu lại z và a để dùng trong backward pass.
        """
        self.z_values = []   # Giá trị trước kích hoạt
        self.a_values = [X]  # Giá trị sau kích hoạt (a[0] = input)

        current = X

        for i in range(self.n_layers):
            # Phép nhân ma trận: mỗi nơ-ron tính tổng có trọng số của inputs
            z = current @ self.weights[i] + self.biases[i]
            self.z_values.append(z)

            # Áp dụng hàm kích hoạt
            if i == self.n_layers - 1:
                # Tầng cuối: dùng output activation
                if self.output_activation == 'softmax':
                    a = Activation.softmax(z)
                else:
                    a = Activation.sigmoid(z)
            else:
                # Tầng ẩn: dùng activation thường
                a = self.activation(z)

            self.a_values.append(a)
            current = a

        return current

    def backward(self, X, y_true, y_pred):
        """
        Backward Pass (Backpropagation): tính gradient và cập nhật trọng số.

        Ý tưởng: Chain Rule (quy tắc chuỗi đạo hàm)
          - Bắt đầu từ sai số ở output
          - Truyền ngược lại qua từng tầng
          - Mỗi tầng tính: "trọng số này ảnh hưởng bao nhiêu đến sai số?"
          - Cập nhật trọng số theo hướng GIẢM sai số

        Đây là phần QUAN TRỌNG NHẤT - là cách mạng nơ-ron "học".
        """
        m = X.shape[0]  # Số mẫu dữ liệu

        # --- Tính gradient tầng output ---
        # Với softmax + cross-entropy, gradient rất đẹp: (y_pred - y_true)
        delta = y_pred - y_true

        # --- Truyền gradient ngược qua từng tầng ---
        for i in range(self.n_layers - 1, -1, -1):
            # Gradient của weights: input_của_tầng.T @ delta
            dW = self.a_values[i].T @ delta / m

            # Gradient của bias: trung bình delta theo chiều batch
            db = np.mean(delta, axis=0, keepdims=True)

            # Nếu không phải tầng đầu tiên, truyền gradient cho tầng trước
            if i > 0:
                # delta mới = delta @ W.T * đạo hàm kích hoạt
                delta = (delta @ self.weights[i].T) * self.activation_deriv(self.z_values[i - 1])

            # --- Cập nhật trọng số (Gradient Descent) ---
            # W = W - learning_rate * gradient
            # Trọng số di chuyển theo hướng NGƯỢC gradient (giảm loss)
            self.weights[i] -= self.learning_rate * dW
            self.biases[i] -= self.learning_rate * db

    def train(self, X, y, epochs=100, batch_size=32, verbose=True):
        """
        Huấn luyện mạng với Mini-batch Gradient Descent.

        Mini-batch: thay vì dùng toàn bộ dữ liệu (chậm) hay 1 mẫu (nhiễu),
        dùng nhóm nhỏ (batch) để cân bằng tốc độ và ổn định.

        Args:
            X: dữ liệu đầu vào
            y: nhãn thật (one-hot encoded cho multi-class)
            epochs: số lần lặp qua toàn bộ dữ liệu
            batch_size: kích thước mỗi mini-batch
            verbose: in tiến trình hay không
        """
        n_samples = X.shape[0]

        for epoch in range(epochs):
            # Xáo trộn dữ liệu mỗi epoch để mô hình không "nhớ thứ tự"
            indices = np.random.permutation(n_samples)
            X_shuffled = X[indices]
            y_shuffled = y[indices]

            epoch_loss = 0

            # Chia dữ liệu thành các mini-batch
            for start in range(0, n_samples, batch_size):
                end = min(start + batch_size, n_samples)
                X_batch = X_shuffled[start:end]
                y_batch = y_shuffled[start:end]

                # Forward: tính dự đoán
                y_pred = self.forward(X_batch)

                # Tính loss
                if self.output_activation == 'softmax':
                    batch_loss = Loss.categorical_cross_entropy(y_batch, y_pred)
                else:
                    batch_loss = Loss.binary_cross_entropy(y_batch, y_pred)
                epoch_loss += batch_loss * (end - start)

                # Backward: cập nhật trọng số
                self.backward(X_batch, y_batch, y_pred)

            epoch_loss /= n_samples
            self.loss_history.append(epoch_loss)

            if verbose and (epoch + 1) % (epochs // 5) == 0:
                acc = self.accuracy(X, y)
                print(f"  Epoch {epoch + 1:4d}/{epochs} | Loss: {epoch_loss:.4f} | Accuracy: {acc:.1f}%")

    def predict(self, X):
        """Dự đoán: chạy forward pass và lấy lớp có xác suất cao nhất."""
        probs = self.forward(X)
        if self.output_activation == 'softmax':
            return np.argmax(probs, axis=1)
        else:
            return (probs >= 0.5).astype(int).flatten()

    def accuracy(self, X, y):
        """Tính độ chính xác (%)."""
        preds = self.predict(X)
        if y.ndim > 1:
            # One-hot → chuyển về index
            y_labels = np.argmax(y, axis=1)
        else:
            y_labels = y
        return np.mean(preds == y_labels) * 100


# =============================================================================
# HÀM TIỆN ÍCH
# =============================================================================

def normalize(X):
    """
    Chuẩn hóa dữ liệu về mean=0, std=1 (Z-score normalization).
    Rất quan trọng cho Neural Network vì:
      - Giúp gradient descent hội tụ nhanh hơn
      - Tránh features có giá trị lớn "lấn át" features nhỏ
    """
    mean = np.mean(X, axis=0)
    std = np.std(X, axis=0)
    std[std == 0] = 1  # Tránh chia cho 0
    return (X - mean) / std, mean, std


def one_hot_encode(y, n_classes):
    """
    One-hot encoding: chuyển nhãn số thành vector nhị phân.
    Ví dụ: nhãn 2 với 3 lớp → [0, 0, 1]
    Neural network cần dạng này để tính loss với softmax.
    """
    encoded = np.zeros((len(y), n_classes))
    for i, label in enumerate(y):
        encoded[i, label] = 1
    return encoded


def train_test_split(X, y, test_ratio=0.2, seed=42):
    """Chia dữ liệu thành tập train và test."""
    np.random.seed(seed)
    n = len(X)
    indices = np.random.permutation(n)
    test_size = int(n * test_ratio)
    test_idx = indices[:test_size]
    train_idx = indices[test_size:]
    return X[train_idx], X[test_idx], y[train_idx], y[test_idx]


# =============================================================================
# VÍ DỤ 1: PHÂN LOẠI HOA IRIS (3 LOÀI)
# =============================================================================
# Bối cảnh: Cho 4 đặc trưng của bông hoa (dài/rộng đài hoa, dài/rộng cánh hoa),
# phân loại thành 3 loài: Setosa, Versicolor, Virginica.
# Đây là bài toán kinh điển trong Machine Learning.

def vi_du_phan_loai_hoa():
    print("=" * 60)
    print("VÍ DỤ 1: PHÂN LOẠI HOA IRIS (3 LOÀI)")
    print("=" * 60)
    print("Features: chiều dài/rộng đài hoa, chiều dài/rộng cánh hoa")
    print("Labels: Setosa, Versicolor, Virginica\n")

    # Tạo dữ liệu mô phỏng hoa Iris (150 mẫu, 4 features)
    np.random.seed(42)

    # Setosa: đài hoa nhỏ, cánh hoa rất nhỏ (dễ phân biệt)
    setosa = np.random.randn(50, 4) * [0.4, 0.3, 0.2, 0.1] + [5.0, 3.4, 1.5, 0.2]

    # Versicolor: trung bình
    versicolor = np.random.randn(50, 4) * [0.5, 0.3, 0.5, 0.2] + [5.9, 2.8, 4.3, 1.3]

    # Virginica: đài hoa lớn, cánh hoa lớn
    virginica = np.random.randn(50, 4) * [0.6, 0.3, 0.6, 0.3] + [6.6, 3.0, 5.6, 2.0]

    # Gộp dữ liệu
    X = np.vstack([setosa, versicolor, virginica])
    y = np.array([0] * 50 + [1] * 50 + [2] * 50)  # 0=Setosa, 1=Versicolor, 2=Virginica

    # Chuẩn hóa dữ liệu
    X_norm, _, _ = normalize(X)

    # Chia train/test
    X_train, X_test, y_train, y_test = train_test_split(X_norm, y)

    # One-hot encode nhãn
    y_train_oh = one_hot_encode(y_train, 3)
    y_test_oh = one_hot_encode(y_test, 3)

    # Tạo mạng: 4 input → 16 hidden → 8 hidden → 3 output
    # 2 tầng ẩn giúp mạng học được ranh giới phân loại phức tạp hơn
    model = NeuralNetwork(
        layer_sizes=[4, 16, 8, 3],
        activation='relu',
        output_activation='softmax',
        learning_rate=0.05
    )

    print("Kiến trúc mạng: 4 → 16 → 8 → 3")
    print("Huấn luyện...\n")

    model.train(X_train, y_train_oh, epochs=200, batch_size=16)

    # Đánh giá
    train_acc = model.accuracy(X_train, y_train_oh)
    test_acc = model.accuracy(X_test, y_test_oh)
    print(f"\nKết quả: Train accuracy = {train_acc:.1f}% | Test accuracy = {test_acc:.1f}%")

    # Thử dự đoán vài mẫu
    ten_hoa = ["Setosa", "Versicolor", "Virginica"]
    print("\nThử dự đoán 5 mẫu từ tập test:")
    preds = model.predict(X_test[:5])
    for i in range(5):
        actual = ten_hoa[y_test[i]]
        predicted = ten_hoa[preds[i]]
        dung_sai = "DUNG" if y_test[i] == preds[i] else "SAI"
        print(f"  Mau {i + 1}: That = {actual:12s} | Du doan = {predicted:12s} | {dung_sai}")


# =============================================================================
# VÍ DỤ 2: DỰ ĐOÁN BỆNH TIỂU ĐƯỜNG
# =============================================================================
# Bối cảnh: Dựa trên các chỉ số sức khỏe (BMI, huyết áp, glucose, tuổi...),
# dự đoán bệnh nhân có mắc tiểu đường hay không (phân loại nhị phân).

def vi_du_tieu_duong():
    print("\n" + "=" * 60)
    print("VÍ DỤ 2: DỰ ĐOÁN BỆNH TIỂU ĐƯỜNG")
    print("=" * 60)
    print("Features: glucose, huyet_ap, BMI, tuoi, insulin, so_lan_mang_thai")
    print("Label: co benh (1) / khong benh (0)\n")

    np.random.seed(123)
    n_samples = 400

    # --- Tạo dữ liệu mô phỏng ---
    # Người KHÔNG bệnh: glucose thấp, BMI bình thường, huyết áp ổn
    n_healthy = 230
    healthy = np.column_stack([
        np.random.normal(95, 15, n_healthy),    # glucose (mg/dL) - bình thường
        np.random.normal(72, 10, n_healthy),    # huyết áp (mmHg)
        np.random.normal(25, 4, n_healthy),     # BMI
        np.random.normal(35, 12, n_healthy),    # tuổi
        np.random.normal(100, 40, n_healthy),   # insulin
        np.random.randint(0, 4, n_healthy),     # số lần mang thai
    ])

    # Người CÓ bệnh: glucose cao, BMI cao hơn, tuổi lớn hơn
    n_diabetic = n_samples - n_healthy
    diabetic = np.column_stack([
        np.random.normal(155, 30, n_diabetic),  # glucose cao
        np.random.normal(82, 12, n_diabetic),   # huyết áp hơi cao
        np.random.normal(33, 5, n_diabetic),    # BMI cao
        np.random.normal(45, 10, n_diabetic),   # tuổi lớn hơn
        np.random.normal(180, 60, n_diabetic),  # insulin cao
        np.random.randint(1, 8, n_diabetic),    # nhiều lần mang thai hơn
    ])

    X = np.vstack([healthy, diabetic]).astype(float)
    y = np.array([0] * n_healthy + [1] * n_diabetic).astype(float)

    # Chuẩn hóa
    X_norm, _, _ = normalize(X)

    # Chia train/test
    X_train, X_test, y_train, y_test = train_test_split(X_norm, y)

    # Reshape nhãn cho binary classification
    y_train_2d = y_train.reshape(-1, 1)
    y_test_2d = y_test.reshape(-1, 1)

    # Tạo mạng: 6 input → 12 hidden → 6 hidden → 1 output (sigmoid)
    model = NeuralNetwork(
        layer_sizes=[6, 12, 6, 1],
        activation='relu',
        output_activation='sigmoid',
        learning_rate=0.01
    )

    print("Kien truc mang: 6 → 12 → 6 → 1 (sigmoid)")
    print("Huan luyen...\n")

    model.train(X_train, y_train_2d, epochs=300, batch_size=32)

    # Đánh giá
    train_acc = model.accuracy(X_train, y_train)
    test_acc = model.accuracy(X_test, y_test)
    print(f"\nKet qua: Train accuracy = {train_acc:.1f}% | Test accuracy = {test_acc:.1f}%")

    # Confusion matrix thủ công
    preds = model.predict(X_test)
    tp = np.sum((preds == 1) & (y_test == 1))  # True Positive: dự đoán bệnh, thật sự bệnh
    tn = np.sum((preds == 0) & (y_test == 0))  # True Negative: dự đoán khỏe, thật sự khỏe
    fp = np.sum((preds == 1) & (y_test == 0))  # False Positive: dự đoán bệnh, nhưng khỏe
    fn = np.sum((preds == 0) & (y_test == 1))  # False Negative: dự đoán khỏe, nhưng bệnh

    print(f"\nConfusion Matrix:")
    print(f"  True Positive  (benh, doan dung): {tp}")
    print(f"  True Negative  (khoe, doan dung): {tn}")
    print(f"  False Positive (khoe, doan benh): {fp} - bao dong gia")
    print(f"  False Negative (benh, doan khoe): {fn} - NGUY HIEM! bo sot benh")

    if tp + fp > 0:
        precision = tp / (tp + fp)
        print(f"\n  Precision: {precision:.1%} (trong so ca doan 'benh', bao nhieu % dung)")
    if tp + fn > 0:
        recall = tp / (tp + fn)
        print(f"  Recall:    {recall:.1%} (trong so nguoi benh that, phat hien duoc bao nhieu %)")


# =============================================================================
# VÍ DỤ 3: NHẬN DẠNG CHỮ SỐ VIẾT TAY (MNIST ĐƠN GIẢN)
# =============================================================================
# Bối cảnh: Mỗi chữ số viết tay là ảnh 8x8 pixel (64 giá trị).
# Phân loại thành 10 chữ số: 0, 1, 2, ..., 9.
# Đây là "Hello World" của deep learning.

def tao_chu_so_don_gian():
    """
    Tạo dữ liệu mô phỏng chữ số viết tay đơn giản (ảnh 8x8).
    Mỗi chữ số có pattern riêng + nhiễu ngẫu nhiên.
    """
    np.random.seed(42)

    # Tạo pattern cơ bản cho mỗi chữ số (8x8 = 64 pixels)
    patterns = {}

    # Số 0: hình oval rỗng
    patterns[0] = np.array([
        [0, 0, 1, 1, 1, 1, 0, 0],
        [0, 1, 1, 0, 0, 1, 1, 0],
        [1, 1, 0, 0, 0, 0, 1, 1],
        [1, 1, 0, 0, 0, 0, 1, 1],
        [1, 1, 0, 0, 0, 0, 1, 1],
        [1, 1, 0, 0, 0, 0, 1, 1],
        [0, 1, 1, 0, 0, 1, 1, 0],
        [0, 0, 1, 1, 1, 1, 0, 0],
    ]).flatten().astype(float)

    # Số 1: đường thẳng đứng
    patterns[1] = np.array([
        [0, 0, 0, 1, 1, 0, 0, 0],
        [0, 0, 1, 1, 1, 0, 0, 0],
        [0, 0, 0, 1, 1, 0, 0, 0],
        [0, 0, 0, 1, 1, 0, 0, 0],
        [0, 0, 0, 1, 1, 0, 0, 0],
        [0, 0, 0, 1, 1, 0, 0, 0],
        [0, 0, 0, 1, 1, 0, 0, 0],
        [0, 0, 1, 1, 1, 1, 0, 0],
    ]).flatten().astype(float)

    # Số 2: nét cong trên, ngang dưới
    patterns[2] = np.array([
        [0, 0, 1, 1, 1, 1, 0, 0],
        [0, 1, 0, 0, 0, 0, 1, 0],
        [0, 0, 0, 0, 0, 1, 1, 0],
        [0, 0, 0, 0, 1, 1, 0, 0],
        [0, 0, 0, 1, 1, 0, 0, 0],
        [0, 0, 1, 1, 0, 0, 0, 0],
        [0, 1, 1, 0, 0, 0, 0, 0],
        [0, 1, 1, 1, 1, 1, 1, 0],
    ]).flatten().astype(float)

    # Số 3: hai nét cong
    patterns[3] = np.array([
        [0, 1, 1, 1, 1, 1, 0, 0],
        [0, 0, 0, 0, 0, 1, 1, 0],
        [0, 0, 0, 0, 0, 1, 1, 0],
        [0, 0, 1, 1, 1, 1, 0, 0],
        [0, 0, 0, 0, 0, 1, 1, 0],
        [0, 0, 0, 0, 0, 1, 1, 0],
        [0, 0, 0, 0, 0, 1, 1, 0],
        [0, 1, 1, 1, 1, 1, 0, 0],
    ]).flatten().astype(float)

    # Số 4: nét dọc + ngang
    patterns[4] = np.array([
        [0, 0, 0, 0, 1, 1, 0, 0],
        [0, 0, 0, 1, 1, 1, 0, 0],
        [0, 0, 1, 0, 1, 1, 0, 0],
        [0, 1, 0, 0, 1, 1, 0, 0],
        [1, 1, 1, 1, 1, 1, 1, 0],
        [0, 0, 0, 0, 1, 1, 0, 0],
        [0, 0, 0, 0, 1, 1, 0, 0],
        [0, 0, 0, 0, 1, 1, 0, 0],
    ]).flatten().astype(float)

    # Tạo nhiều mẫu với nhiễu cho mỗi chữ số
    samples_per_digit = 80
    X_list = []
    y_list = []

    for digit in range(5):  # Dùng 5 chữ số (0-4) để đơn giản
        for _ in range(samples_per_digit):
            # Thêm nhiễu Gaussian để mô phỏng chữ viết tay khác nhau
            noisy = patterns[digit] + np.random.normal(0, 0.3, 64)
            noisy = np.clip(noisy, 0, 1)  # Giữ giá trị trong [0, 1]
            X_list.append(noisy)
            y_list.append(digit)

    return np.array(X_list), np.array(y_list)


def vi_du_nhan_dang_chu_so():
    print("\n" + "=" * 60)
    print("VÍ DỤ 3: NHẬN DẠNG CHỮ SỐ VIẾT TAY")
    print("=" * 60)
    print("Anh 8x8 pixel (64 features) → Phan loai chu so 0-4\n")

    # Tạo dữ liệu
    X, y = tao_chu_so_don_gian()
    print(f"Du lieu: {X.shape[0]} mau, moi mau {X.shape[1]} pixels")

    # Chia train/test
    X_train, X_test, y_train, y_test = train_test_split(X, y)

    # One-hot encode
    y_train_oh = one_hot_encode(y_train, 5)
    y_test_oh = one_hot_encode(y_test, 5)

    # Tạo mạng: 64 input → 32 hidden → 16 hidden → 5 output
    # Input lớn (64 pixels) nên cần tầng ẩn đủ rộng
    model = NeuralNetwork(
        layer_sizes=[64, 32, 16, 5],
        activation='relu',
        output_activation='softmax',
        learning_rate=0.01
    )

    print(f"Kien truc mang: 64 → 32 → 16 → 5")
    print("Huan luyen...\n")

    model.train(X_train, y_train_oh, epochs=300, batch_size=32)

    # Đánh giá
    train_acc = model.accuracy(X_train, y_train_oh)
    test_acc = model.accuracy(X_test, y_test_oh)
    print(f"\nKet qua: Train accuracy = {train_acc:.1f}% | Test accuracy = {test_acc:.1f}%")

    # Hiển thị vài mẫu dạng text (8x8 grid)
    print("\n--- Hien thi mau thu va du doan ---")
    preds = model.predict(X_test[:3])
    for i in range(3):
        print(f"\nMau {i + 1} (That: {y_test[i]}, Du doan: {preds[i]}):")
        img = X_test[i].reshape(8, 8)
        for row in img:
            line = ""
            for pixel in row:
                # Chuyển giá trị pixel thành ký tự để "vẽ" chữ số
                if pixel > 0.7:
                    line += "##"
                elif pixel > 0.3:
                    line += ".."
                else:
                    line += "  "
            print(f"    {line}")


# =============================================================================
# VÍ DỤ 4: TRỰC QUAN HÓA QUÁ TRÌNH HỌC
# =============================================================================

def vi_du_qua_trinh_hoc():
    print("\n" + "=" * 60)
    print("VÍ DỤ 4: TRỰC QUAN QUÁ TRÌNH HỌC CỦA NEURAL NETWORK")
    print("=" * 60)
    print("So sanh learning rate khac nhau\n")

    # Tạo dữ liệu đơn giản: phân loại 2 cụm
    np.random.seed(42)
    n = 200

    # Cụm 1: phía trái-dưới
    X1 = np.random.randn(n, 2) * 0.8 + [-2, -2]
    # Cụm 2: phía phải-trên
    X2 = np.random.randn(n, 2) * 0.8 + [2, 2]

    X = np.vstack([X1, X2])
    y = np.array([0] * n + [1] * n).reshape(-1, 1).astype(float)

    X_norm, _, _ = normalize(X)

    # Thử 3 learning rate khác nhau
    learning_rates = [0.001, 0.01, 0.1]

    for lr in learning_rates:
        np.random.seed(42)  # Cùng khởi tạo để so sánh công bằng
        model = NeuralNetwork(
            layer_sizes=[2, 8, 1],
            activation='relu',
            output_activation='sigmoid',
            learning_rate=lr
        )

        model.train(X_norm, y, epochs=100, batch_size=32, verbose=False)

        acc = model.accuracy(X_norm, y.flatten())
        final_loss = model.loss_history[-1]

        # Vẽ đồ thị loss bằng text
        print(f"\nLearning rate = {lr}:")
        print(f"  Final loss = {final_loss:.4f} | Accuracy = {acc:.1f}%")

        # Mini chart: loss mỗi 10 epochs
        print("  Loss: ", end="")
        for i in range(0, 100, 10):
            bar_len = int(model.loss_history[i] * 30)
            bar = "|" * min(bar_len, 30)
            print(f"{bar:30s} {model.loss_history[i]:.3f}", end="")
            if i < 90:
                print("\n        ", end="")
        print()

    print("\nNhan xet:")
    print("  - LR qua nho (0.001): hoc cham, chua hoi tu")
    print("  - LR vua (0.01): hoi tu tot, on dinh")
    print("  - LR qua lon (0.1): co the hoi tu nhanh nhung khong on dinh")


# =============================================================================
# CHẠY TẤT CẢ VÍ DỤ
# =============================================================================

if __name__ == "__main__":
    vi_du_phan_loai_hoa()
    vi_du_tieu_duong()
    vi_du_nhan_dang_chu_so()
    vi_du_qua_trinh_hoc()

    print("\n" + "=" * 60)
    print("TOM TAT KIEN THUC NEURAL NETWORK:")
    print("=" * 60)
    print("""
    1. KIEN TRUC:
       - Input layer: nhan du lieu dau vao
       - Hidden layers: xu ly, hoc dac trung (cang nhieu tang → cang phuc tap)
       - Output layer: dua ra ket qua (sigmoid cho 2 lop, softmax cho nhieu lop)

    2. QUA TRINH HOC:
       - Forward pass: tinh du doan
       - Loss: do sai so
       - Backward pass: tinh gradient (dao ham)
       - Update weights: dieu chinh trong so (gradient descent)

    3. HYPERPARAMETERS QUAN TRONG:
       - Learning rate: toc do hoc (qua lon → khong on dinh, qua nho → cham)
       - Batch size: kich thuoc mini-batch
       - So tang va so neuron: do phuc tap cua mo hinh
       - So epochs: so lan lap qua du lieu

    4. TIPS THUC TE:
       - Luon chuan hoa du lieu truoc khi dua vao mang
       - Bat dau voi mo hinh don gian, tang do phuc tap dan
       - Theo doi loss: neu khong giam → kiem tra learning rate
       - Train accuracy cao nhung test thap → overfitting
    """)
