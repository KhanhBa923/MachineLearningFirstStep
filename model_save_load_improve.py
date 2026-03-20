"""
Lưu, Tải và Cải tiến Model đã Train

Trong thực tế, sau khi train model xong, ta cần:
  1. LƯU model: để không phải train lại (tốn thời gian, tiền bạc)
  2. TẢI model: dùng model đã train để dự đoán dữ liệu mới
  3. CẢI TIẾN model: nâng cấp model cũ cho kết quả tốt hơn

File này hướng dẫn:
  Phần 1: Train model cơ bản → Lưu → Tải → Dự đoán
  Phần 2: Cải tiến model bằng nhiều kỹ thuật khác nhau
  Phần 3: So sánh model cũ vs model mới

Ví dụ: Dự đoán nhân viên có nghỉ việc hay không
  - Features: luong, so_nam_lam_viec, khoang_cach_nha, hai_long, so_du_an
  - Label: nghi_viec (1) hay o_lai (0)
"""

import numpy as np
import json
import os
from services import ActivationService, LossService, DataService

# Đường dẫn thư mục hiện tại để lưu model
MODEL_DIR = os.path.dirname(os.path.abspath(__file__))


# =============================================================================
# PHẦN 1: NEURAL NETWORK ĐƠN GIẢN + LƯU/TẢI
# =============================================================================

class SimpleNN:
    """
    Neural Network đơn giản 1 hidden layer.
    Có khả năng lưu và tải trọng số.
    """

    def __init__(self, n_input, n_hidden, n_output, learning_rate=0.01):
        self.lr = learning_rate

        # Khởi tạo trọng số ngẫu nhiên
        self.W1 = np.random.randn(n_input, n_hidden) * np.sqrt(2.0 / n_input)
        self.b1 = np.zeros((1, n_hidden))
        self.W2 = np.random.randn(n_hidden, n_output) * np.sqrt(2.0 / n_hidden)
        self.b2 = np.zeros((1, n_output))

        # Lưu lại kiến trúc để khi load biết cấu trúc mạng
        self.architecture = {
            "n_input": n_input,
            "n_hidden": n_hidden,
            "n_output": n_output,
            "learning_rate": learning_rate,
        }

        self.loss_history = []

    def forward(self, X):
        """Forward pass: Input → Hidden (ReLU) → Output (Sigmoid)"""
        self.z1 = X @ self.W1 + self.b1
        self.a1 = ActivationService.relu(self.z1)
        self.z2 = self.a1 @ self.W2 + self.b2
        self.a2 = ActivationService.sigmoid(self.z2)
        return self.a2

    def backward(self, X, y, output):
        """Backward pass: tính gradient và cập nhật trọng số."""
        m = X.shape[0]

        # Gradient tầng output
        d2 = output - y
        dW2 = self.a1.T @ d2 / m
        db2 = np.mean(d2, axis=0, keepdims=True)

        # Gradient tầng hidden
        d1 = (d2 @ self.W2.T) * (self.z1 > 0).astype(float)
        dW1 = X.T @ d1 / m
        db1 = np.mean(d1, axis=0, keepdims=True)

        # Cập nhật trọng số
        self.W2 -= self.lr * dW2
        self.b2 -= self.lr * db2
        self.W1 -= self.lr * dW1
        self.b1 -= self.lr * db1

    def train(self, X, y, epochs=100, verbose=True):
        """Huấn luyện model."""
        for epoch in range(epochs):
            output = self.forward(X)

            # Binary cross-entropy loss
            loss = LossService.binary_cross_entropy(y, output)
            self.loss_history.append(loss)

            self.backward(X, y, output)

            if verbose and (epoch + 1) % (epochs // 5) == 0:
                acc = self.accuracy(X, y)
                print(f"  Epoch {epoch + 1:4d}/{epochs} | Loss: {loss:.4f} | Acc: {acc:.1f}%")

    def predict(self, X):
        """Dự đoán: 0 hoặc 1."""
        probs = self.forward(X)
        return (probs >= 0.5).astype(int).flatten()

    def predict_proba(self, X):
        """Trả về xác suất."""
        return self.forward(X).flatten()

    def accuracy(self, X, y):
        preds = self.predict(X)
        return np.mean(preds == y.flatten()) * 100

    # -----------------------------------------------------------------
    # LƯU MODEL: chuyển trọng số thành file JSON
    # -----------------------------------------------------------------
    def save(self, filepath):
        """
        Lưu model ra file JSON.

        Lưu những gì:
          - architecture: kiến trúc mạng (để tạo lại model khi load)
          - weights: tất cả trọng số (W1, b1, W2, b2)
          - loss_history: lịch sử loss (để biết model train tốt chưa)

        Tại sao dùng JSON:
          - Đọc được bằng mắt (human-readable)
          - Dễ debug, kiểm tra
          - Trong thực tế thường dùng pickle, HDF5, hoặc safetensors
        """
        model_data = {
            "architecture": self.architecture,
            "weights": {
                "W1": self.W1.tolist(),  # numpy array → list để JSON serialize
                "b1": self.b1.tolist(),
                "W2": self.W2.tolist(),
                "b2": self.b2.tolist(),
            },
            "loss_history": self.loss_history,
        }

        with open(filepath, "w") as f:
            json.dump(model_data, f, indent=2)

        # Tính kích thước file
        size_kb = os.path.getsize(filepath) / 1024
        print(f"  Model da luu tai: {filepath}")
        print(f"  Kich thuoc file:  {size_kb:.1f} KB")
        print(f"  So tham so:       {self.count_params()}")

    # -----------------------------------------------------------------
    # TẢI MODEL: đọc trọng số từ file JSON
    # -----------------------------------------------------------------
    @classmethod
    def load(cls, filepath):
        """
        Tải model từ file JSON.

        Quy trình:
          1. Đọc file JSON
          2. Tạo model mới với kiến trúc đã lưu
          3. Gán trọng số đã lưu vào model
          4. Model sẵn sàng dự đoán (KHÔNG cần train lại)

        @classmethod: gọi được mà không cần tạo instance trước
          Ví dụ: model = SimpleNN.load("model.json")
        """
        with open(filepath, "r") as f:
            model_data = json.load(f)

        arch = model_data["architecture"]

        # Tạo model mới với kiến trúc đã lưu
        model = cls(
            n_input=arch["n_input"],
            n_hidden=arch["n_hidden"],
            n_output=arch["n_output"],
            learning_rate=arch["learning_rate"],
        )

        # Gán trọng số đã lưu (list → numpy array)
        model.W1 = np.array(model_data["weights"]["W1"])
        model.b1 = np.array(model_data["weights"]["b1"])
        model.W2 = np.array(model_data["weights"]["W2"])
        model.b2 = np.array(model_data["weights"]["b2"])

        model.loss_history = model_data.get("loss_history", [])

        print(f"  Model da tai tu: {filepath}")
        print(f"  Kien truc: {arch['n_input']} → {arch['n_hidden']} → {arch['n_output']}")

        return model

    def count_params(self):
        """Đếm tổng số tham số (trọng số + bias)."""
        return (self.W1.size + self.b1.size + self.W2.size + self.b2.size)


# =============================================================================
# PHẦN 2: MODEL CẢI TIẾN
# =============================================================================

class ImprovedNN:
    """
    Neural Network cải tiến với nhiều kỹ thuật nâng cao.

    Cải tiến so với SimpleNN:
      1. Nhiều hidden layers (deep network)
      2. Dropout: tắt ngẫu nhiên một số nơ-ron khi train → chống overfitting
      3. Batch Normalization: chuẩn hóa output mỗi layer → train nhanh hơn
      4. Learning Rate Decay: giảm dần tốc độ học → hội tụ tốt hơn
      5. Early Stopping: dừng train khi model không cải thiện → tránh overfitting
      6. L2 Regularization: phạt trọng số lớn → model đơn giản hơn
    """

    def __init__(self, layer_sizes, learning_rate=0.01, dropout_rate=0.2,
                 l2_lambda=0.001, lr_decay=0.99):
        """
        Args:
            layer_sizes: danh sách kích thước mỗi tầng, vd [5, 32, 16, 1]
            dropout_rate: tỷ lệ nơ-ron bị tắt (0.2 = tắt 20%)
            l2_lambda: hệ số regularization (càng lớn → phạt trọng số càng mạnh)
            lr_decay: hệ số giảm learning rate mỗi epoch (0.99 = giảm 1%)
        """
        self.lr = learning_rate
        self.initial_lr = learning_rate
        self.dropout_rate = dropout_rate
        self.l2_lambda = l2_lambda
        self.lr_decay = lr_decay
        self.layer_sizes = layer_sizes
        self.n_layers = len(layer_sizes) - 1

        # Khởi tạo trọng số cho tất cả các tầng
        self.weights = []
        self.biases = []
        for i in range(self.n_layers):
            scale = np.sqrt(2.0 / layer_sizes[i])
            self.weights.append(np.random.randn(layer_sizes[i], layer_sizes[i + 1]) * scale)
            self.biases.append(np.zeros((1, layer_sizes[i + 1])))

        # Tham số Batch Normalization cho mỗi hidden layer
        # gamma (scale) và beta (shift) là learnable parameters
        self.bn_gamma = []  # Hệ số scale
        self.bn_beta = []   # Hệ số shift
        for i in range(self.n_layers - 1):  # Không BN ở output
            self.bn_gamma.append(np.ones((1, layer_sizes[i + 1])))
            self.bn_beta.append(np.zeros((1, layer_sizes[i + 1])))

        self.loss_history = []
        self.training = True  # Phân biệt train vs predict mode

        self.architecture = {
            "layer_sizes": layer_sizes,
            "learning_rate": learning_rate,
            "dropout_rate": dropout_rate,
            "l2_lambda": l2_lambda,
            "lr_decay": lr_decay,
        }

    def batch_norm(self, x, layer_idx):
        """
        Batch Normalization: chuẩn hóa output của mỗi layer.

        Vấn đề: khi train deep network, phân phối dữ liệu thay đổi qua mỗi layer
        (Internal Covariate Shift) → train chậm, không ổn định.

        Giải pháp: chuẩn hóa về mean=0, std=1 rồi scale/shift lại.
            x_norm = (x - mean) / sqrt(var + eps)
            output = gamma * x_norm + beta

        gamma và beta được HỌC → mạng tự quyết định mức chuẩn hóa phù hợp.
        """
        mean = np.mean(x, axis=0, keepdims=True)
        var = np.var(x, axis=0, keepdims=True)
        x_norm = (x - mean) / np.sqrt(var + 1e-8)

        return self.bn_gamma[layer_idx] * x_norm + self.bn_beta[layer_idx]

    def dropout(self, x):
        """
        Dropout: tắt ngẫu nhiên một số nơ-ron trong quá trình train.

        Tại sao hiệu quả:
          - Buộc mạng KHÔNG phụ thuộc vào bất kỳ nơ-ron đơn lẻ nào
          - Giống như "hỏi ý kiến nhiều chuyên gia khác nhau" (ensemble effect)
          - Chỉ áp dụng khi TRAIN, không áp dụng khi PREDICT

        Inverted Dropout:
          - Nhân output với 1/(1-dropout_rate) khi train
          - Để khi predict (không dropout), output vẫn có cùng scale
        """
        if not self.training or self.dropout_rate == 0:
            return x

        # Tạo mask ngẫu nhiên: mỗi nơ-ron có (1-dropout_rate) cơ hội sống sót
        mask = np.random.binomial(1, 1 - self.dropout_rate, size=x.shape)

        # Nhân với mask và scale lên
        return x * mask / (1 - self.dropout_rate)

    def forward(self, X):
        self.z_values = []
        self.a_values = [X]
        self.dropout_masks = []

        current = X

        for i in range(self.n_layers):
            z = current @ self.weights[i] + self.biases[i]
            self.z_values.append(z)

            if i == self.n_layers - 1:
                # Tầng cuối: sigmoid (không BN, không dropout)
                a = ActivationService.sigmoid(z)
            else:
                # Tầng ẩn: BN → ReLU → Dropout
                z_bn = self.batch_norm(z, i)
                a = ActivationService.relu(z_bn)
                a = self.dropout(a)

            self.a_values.append(a)
            current = a

        return current

    def backward(self, X, y, output):
        m = X.shape[0]

        delta = output - y

        for i in range(self.n_layers - 1, -1, -1):
            # Gradient + L2 regularization
            # L2 thêm l2_lambda * W vào gradient → phạt trọng số lớn
            dW = self.a_values[i].T @ delta / m + self.l2_lambda * self.weights[i]
            db = np.mean(delta, axis=0, keepdims=True)

            if i > 0:
                delta = (delta @ self.weights[i].T) * (self.z_values[i - 1] > 0).astype(float)

            self.weights[i] -= self.lr * dW
            self.biases[i] -= self.lr * db

    def train_model(self, X_train, y_train, X_val=None, y_val=None,
                    epochs=200, batch_size=32, patience=20, verbose=True):
        """
        Huấn luyện với Early Stopping.

        Early Stopping: dừng train khi validation loss không giảm sau 'patience' epochs.

        Tại sao cần:
          - Train quá lâu → model "nhớ" dữ liệu train (overfitting)
          - Loss trên train tiếp tục giảm, nhưng loss trên val tăng lên
          - Early stopping dừng đúng lúc model tốt nhất

        Args:
            X_val, y_val: dữ liệu validation (dùng để kiểm tra overfitting)
            patience: số epochs chờ trước khi dừng
        """
        self.training = True
        n_samples = X_train.shape[0]

        # Lưu trọng số tốt nhất
        best_val_loss = float("inf")
        best_weights = None
        best_biases = None
        epochs_no_improve = 0

        for epoch in range(epochs):
            # --- Learning Rate Decay ---
            # Giảm dần LR để "bước nhỏ hơn" khi gần tối ưu
            self.lr = self.initial_lr * (self.lr_decay ** epoch)

            # Shuffle dữ liệu
            idx = np.random.permutation(n_samples)
            X_shuffled = X_train[idx]
            y_shuffled = y_train[idx]

            epoch_loss = 0

            # Mini-batch training
            for start in range(0, n_samples, batch_size):
                end = min(start + batch_size, n_samples)
                X_batch = X_shuffled[start:end]
                y_batch = y_shuffled[start:end]

                output = self.forward(X_batch)

                loss = LossService.binary_cross_entropy(y_batch, output)

                # Thêm L2 loss
                l2_loss = 0
                for w in self.weights:
                    l2_loss += np.sum(w ** 2)
                loss += 0.5 * self.l2_lambda * l2_loss

                epoch_loss += loss * (end - start)

                self.backward(X_batch, y_batch, output)

            epoch_loss /= n_samples
            self.loss_history.append(epoch_loss)

            # --- Early Stopping ---
            if X_val is not None:
                self.training = False  # Tắt dropout khi đánh giá
                val_pred = self.forward(X_val)
                val_loss = LossService.binary_cross_entropy(y_val, val_pred)
                self.training = True

                if val_loss < best_val_loss:
                    # Model cải thiện → lưu lại trọng số tốt nhất
                    best_val_loss = val_loss
                    best_weights = [w.copy() for w in self.weights]
                    best_biases = [b.copy() for b in self.biases]
                    epochs_no_improve = 0
                else:
                    epochs_no_improve += 1

                if epochs_no_improve >= patience:
                    if verbose:
                        print(f"  Early stopping tai epoch {epoch + 1} (khong cai thien sau {patience} epochs)")
                    # Khôi phục trọng số tốt nhất
                    self.weights = best_weights
                    self.biases = best_biases
                    break

            if verbose and (epoch + 1) % max(1, epochs // 5) == 0:
                train_acc = self.accuracy(X_train, y_train)
                msg = f"  Epoch {epoch + 1:4d}/{epochs} | Loss: {epoch_loss:.4f} | Train Acc: {train_acc:.1f}%"
                if X_val is not None:
                    val_acc = self.accuracy(X_val, y_val)
                    msg += f" | Val Acc: {val_acc:.1f}% | LR: {self.lr:.5f}"
                print(msg)

        self.training = False  # Chuyển sang predict mode sau khi train xong

    def predict(self, X):
        self.training = False
        probs = self.forward(X)
        return (probs >= 0.5).astype(int).flatten()

    def predict_proba(self, X):
        self.training = False
        return self.forward(X).flatten()

    def accuracy(self, X, y):
        preds = self.predict(X)
        return np.mean(preds == y.flatten()) * 100

    def save(self, filepath):
        """Lưu model cải tiến."""
        model_data = {
            "architecture": self.architecture,
            "weights": [w.tolist() for w in self.weights],
            "biases": [b.tolist() for b in self.biases],
            "bn_gamma": [g.tolist() for g in self.bn_gamma],
            "bn_beta": [b.tolist() for b in self.bn_beta],
            "loss_history": self.loss_history,
        }

        with open(filepath, "w") as f:
            json.dump(model_data, f, indent=2)

        size_kb = os.path.getsize(filepath) / 1024
        n_params = sum(w.size for w in self.weights) + sum(b.size for b in self.biases)
        print(f"  Model da luu tai: {filepath}")
        print(f"  Kich thuoc: {size_kb:.1f} KB | So tham so: {n_params}")

    @classmethod
    def load(cls, filepath):
        """Tải model cải tiến."""
        with open(filepath, "r") as f:
            data = json.load(f)

        arch = data["architecture"]
        model = cls(
            layer_sizes=arch["layer_sizes"],
            learning_rate=arch["learning_rate"],
            dropout_rate=arch["dropout_rate"],
            l2_lambda=arch["l2_lambda"],
            lr_decay=arch["lr_decay"],
        )

        model.weights = [np.array(w) for w in data["weights"]]
        model.biases = [np.array(b) for b in data["biases"]]
        model.bn_gamma = [np.array(g) for g in data["bn_gamma"]]
        model.bn_beta = [np.array(b) for b in data["bn_beta"]]
        model.loss_history = data.get("loss_history", [])
        model.training = False

        print(f"  Model da tai tu: {filepath}")
        print(f"  Kien truc: {' → '.join(str(s) for s in arch['layer_sizes'])}")

        return model


# =============================================================================
# TẠO DỮ LIỆU: DỰ ĐOÁN NHÂN VIÊN NGHỈ VIỆC
# =============================================================================

def tao_du_lieu_nhan_vien(n_samples=500, seed=42):
    """
    Tạo dữ liệu mô phỏng nhân viên công ty.

    Features:
      0. luong: mức lương (triệu/tháng)
      1. so_nam: số năm làm việc
      2. khoang_cach: khoảng cách nhà → công ty (km)
      3. hai_long: điểm hài lòng (1-10)
      4. so_du_an: số dự án đang tham gia

    Label:
      0 = ở lại, 1 = nghỉ việc
    """
    np.random.seed(seed)

    # Nhân viên Ở LẠI: lương cao, hài lòng cao, khoảng cách gần
    n_stay = int(n_samples * 0.6)
    stay = np.column_stack([
        np.random.normal(25, 5, n_stay),     # Lương 25 triệu ± 5
        np.random.normal(5, 2, n_stay),      # 5 năm kinh nghiệm
        np.random.normal(8, 4, n_stay),      # 8km đến công ty
        np.random.normal(7.5, 1, n_stay),    # Hài lòng 7.5/10
        np.random.normal(3, 1, n_stay),      # 3 dự án
    ])

    # Nhân viên NGHỈ VIỆC: lương thấp, hài lòng thấp, khoảng cách xa
    n_quit = n_samples - n_stay
    quit_emp = np.column_stack([
        np.random.normal(15, 4, n_quit),     # Lương 15 triệu
        np.random.normal(2, 1.5, n_quit),    # 2 năm kinh nghiệm
        np.random.normal(20, 8, n_quit),     # 20km đến công ty
        np.random.normal(4, 1.5, n_quit),    # Hài lòng 4/10
        np.random.normal(5, 2, n_quit),      # 5 dự án (quá tải)
    ])

    X = np.vstack([stay, quit_emp])
    y = np.array([0] * n_stay + [1] * n_quit).reshape(-1, 1).astype(float)

    # Clip giá trị bất hợp lý
    X[:, 0] = np.clip(X[:, 0], 5, 60)     # Lương: 5-60 triệu
    X[:, 1] = np.clip(X[:, 1], 0, 20)     # Số năm: 0-20
    X[:, 2] = np.clip(X[:, 2], 0.5, 50)   # Khoảng cách: 0.5-50km
    X[:, 3] = np.clip(X[:, 3], 1, 10)     # Hài lòng: 1-10
    X[:, 4] = np.clip(X[:, 4], 1, 10)     # Dự án: 1-10

    return X, y


normalize = DataService.normalize
split_data = DataService.train_val_test_split


# =============================================================================
# VÍ DỤ 1: TRAIN → LƯU → TẢI → DỰ ĐOÁN
# =============================================================================

def vi_du_luu_tai_model():
    print("=" * 65)
    print("VI DU 1: TRAIN MODEL → LUU → TAI → DU DOAN")
    print("=" * 65)

    # Tạo dữ liệu
    X, y = tao_du_lieu_nhan_vien(500)
    X_norm, mean, std = normalize(X)
    X_train, y_train, X_val, y_val, X_test, y_test = split_data(X_norm, y)

    feature_names = ["luong", "so_nam", "khoang_cach", "hai_long", "so_du_an"]
    print(f"\nDu lieu: {X.shape[0]} nhan vien, {X.shape[1]} features")
    print(f"  Train: {len(X_train)} | Val: {len(X_val)} | Test: {len(X_test)}")

    # --- BƯỚC 1: Train model ---
    print("\n--- BUOC 1: Train model co ban ---")
    np.random.seed(42)
    model = SimpleNN(n_input=5, n_hidden=16, n_output=1, learning_rate=0.05)
    model.train(X_train, y_train, epochs=200)

    train_acc = model.accuracy(X_train, y_train)
    test_acc = model.accuracy(X_test, y_test)
    print(f"\n  Ket qua: Train = {train_acc:.1f}% | Test = {test_acc:.1f}%")

    # --- BƯỚC 2: Lưu model ---
    print("\n--- BUOC 2: Luu model ---")
    model_path = os.path.join(MODEL_DIR, "model_nhan_vien_v1.json")
    model.save(model_path)

    # Lưu luôn thông tin chuẩn hóa (RẤT QUAN TRỌNG!)
    # Khi predict dữ liệu mới, phải chuẩn hóa CÙNG mean/std như lúc train
    norm_path = os.path.join(MODEL_DIR, "normalization_params.json")
    norm_data = {"mean": mean.tolist(), "std": std.tolist(), "features": feature_names}
    with open(norm_path, "w") as f:
        json.dump(norm_data, f, indent=2)
    print(f"  Normalization params da luu tai: {norm_path}")

    # --- BƯỚC 3: Tải model (giả lập khởi động lại chương trình) ---
    print("\n--- BUOC 3: Tai lai model (khong can train lai) ---")
    loaded_model = SimpleNN.load(model_path)

    # Tải lại thông tin chuẩn hóa
    with open(norm_path, "r") as f:
        norm_loaded = json.load(f)
    loaded_mean = np.array(norm_loaded["mean"])
    loaded_std = np.array(norm_loaded["std"])

    # Kiểm tra model loaded cho kết quả giống model gốc
    loaded_acc = loaded_model.accuracy(X_test, y_test)
    print(f"  Accuracy model goc:   {test_acc:.1f}%")
    print(f"  Accuracy model loaded: {loaded_acc:.1f}%")
    print(f"  Ket qua giong nhau:    {'CO' if abs(test_acc - loaded_acc) < 0.01 else 'KHONG'}")

    # --- BƯỚC 4: Dự đoán nhân viên mới ---
    print("\n--- BUOC 4: Du doan nhan vien moi ---")
    nhan_vien_moi = np.array([
        [30, 6, 5, 8, 2],    # Lương cao, gần, hài lòng → ở lại?
        [12, 1, 25, 3, 7],   # Lương thấp, xa, không hài lòng → nghỉ?
        [20, 3, 15, 6, 4],   # Trung bình → ?
    ])

    # Chuẩn hóa CÙNG mean/std như lúc train
    nhan_vien_norm = (nhan_vien_moi - loaded_mean) / loaded_std

    probs = loaded_model.predict_proba(nhan_vien_norm)
    preds = loaded_model.predict(nhan_vien_norm)

    for i, nv in enumerate(nhan_vien_moi):
        ket_qua = "NGHI VIEC" if preds[i] == 1 else "O LAI"
        print(f"\n  Nhan vien {i + 1}:")
        print(f"    Luong: {nv[0]:.0f} trieu | So nam: {nv[1]:.0f} | "
              f"KC: {nv[2]:.0f}km | Hai long: {nv[3]:.0f}/10 | Du an: {nv[4]:.0f}")
        print(f"    → Du doan: {ket_qua} (xac suat nghi: {probs[i]:.1%})")

    return model_path, norm_path


# =============================================================================
# VÍ DỤ 2: CẢI TIẾN MODEL
# =============================================================================

def vi_du_cai_tien_model():
    print("\n" + "=" * 65)
    print("VI DU 2: CAI TIEN MODEL")
    print("=" * 65)

    X, y = tao_du_lieu_nhan_vien(500)
    X_norm, mean, std = normalize(X)
    X_train, y_train, X_val, y_val, X_test, y_test = split_data(X_norm, y)

    # --- Model V1: Cơ bản (như ví dụ 1) ---
    print("\n--- MODEL V1: Co ban (1 hidden layer, khong ky thuat gi) ---")
    np.random.seed(42)
    model_v1 = SimpleNN(n_input=5, n_hidden=16, n_output=1, learning_rate=0.05)
    model_v1.train(X_train, y_train, epochs=200, verbose=False)
    v1_train = model_v1.accuracy(X_train, y_train)
    v1_test = model_v1.accuracy(X_test, y_test)
    print(f"  Kien truc: 5 → 16 → 1")
    print(f"  Ket qua: Train = {v1_train:.1f}% | Test = {v1_test:.1f}%")

    # --- Model V2: Thêm tầng ẩn (deeper) ---
    print("\n--- MODEL V2: Them tang an (deeper network) ---")
    print("  Ly do: Mang sau hon hoc duoc quan he PHUC TAP hon")
    print("  Rui ro: De bi overfitting neu du lieu it")
    np.random.seed(42)
    model_v2 = ImprovedNN(
        layer_sizes=[5, 32, 16, 1],  # 2 hidden layers thay vì 1
        learning_rate=0.05,
        dropout_rate=0,        # Chưa dùng dropout
        l2_lambda=0,           # Chưa dùng regularization
        lr_decay=1.0,          # Chưa dùng LR decay
    )
    model_v2.train_model(X_train, y_train, epochs=200, verbose=False)
    v2_train = model_v2.accuracy(X_train, y_train)
    v2_test = model_v2.accuracy(X_test, y_test)
    print(f"  Kien truc: 5 → 32 → 16 → 1")
    print(f"  Ket qua: Train = {v2_train:.1f}% | Test = {v2_test:.1f}%")

    # --- Model V3: Thêm Dropout ---
    print("\n--- MODEL V3: Them Dropout (chong overfitting) ---")
    print("  Ly do: Tat ngau nhien 20% neuron → mang khong phu thuoc 1 neuron nao")
    np.random.seed(42)
    model_v3 = ImprovedNN(
        layer_sizes=[5, 32, 16, 1],
        learning_rate=0.05,
        dropout_rate=0.2,      # Tắt 20% nơ-ron
        l2_lambda=0,
        lr_decay=1.0,
    )
    model_v3.train_model(X_train, y_train, epochs=200, verbose=False)
    v3_train = model_v3.accuracy(X_train, y_train)
    v3_test = model_v3.accuracy(X_test, y_test)
    print(f"  Kien truc: 5 → 32(drop20%) → 16(drop20%) → 1")
    print(f"  Ket qua: Train = {v3_train:.1f}% | Test = {v3_test:.1f}%")

    # --- Model V4: Thêm L2 Regularization ---
    print("\n--- MODEL V4: Them L2 Regularization ---")
    print("  Ly do: Phat trong so lon → model don gian hon, tong quat hoa tot hon")
    np.random.seed(42)
    model_v4 = ImprovedNN(
        layer_sizes=[5, 32, 16, 1],
        learning_rate=0.05,
        dropout_rate=0.2,
        l2_lambda=0.01,        # Phạt trọng số lớn
        lr_decay=1.0,
    )
    model_v4.train_model(X_train, y_train, epochs=200, verbose=False)
    v4_train = model_v4.accuracy(X_train, y_train)
    v4_test = model_v4.accuracy(X_test, y_test)
    print(f"  Kien truc: 5 → 32 → 16 → 1 + L2(0.01)")
    print(f"  Ket qua: Train = {v4_train:.1f}% | Test = {v4_test:.1f}%")

    # --- Model V5: Full combo + Early Stopping ---
    print("\n--- MODEL V5: FULL COMBO (Dropout + L2 + LR Decay + Early Stopping) ---")
    print("  Tat ca ky thuat ket hop → model tot nhat")
    np.random.seed(42)
    model_v5 = ImprovedNN(
        layer_sizes=[5, 32, 16, 1],
        learning_rate=0.05,
        dropout_rate=0.2,
        l2_lambda=0.005,
        lr_decay=0.995,         # Giảm LR 0.5% mỗi epoch
    )
    model_v5.train_model(
        X_train, y_train,
        X_val=X_val, y_val=y_val,  # Dùng validation để early stop
        epochs=500,
        patience=30,               # Dừng nếu 30 epochs không cải thiện
        verbose=True,
    )
    v5_train = model_v5.accuracy(X_train, y_train)
    v5_test = model_v5.accuracy(X_test, y_test)
    print(f"\n  Ket qua: Train = {v5_train:.1f}% | Test = {v5_test:.1f}%")

    # --- Lưu model tốt nhất ---
    best_model_path = os.path.join(MODEL_DIR, "model_nhan_vien_v5_best.json")
    print(f"\n--- Luu model tot nhat ---")
    model_v5.save(best_model_path)

    # --- BẢNG SO SÁNH ---
    print("\n" + "=" * 65)
    print("BANG SO SANH TAT CA CAC PHIEN BAN:")
    print("=" * 65)
    print(f"{'Model':<12s} {'Kien truc':<28s} {'Train':>8s} {'Test':>8s} {'Gap':>8s}")
    print("-" * 65)

    models_info = [
        ("V1 Basic", "5→16→1", v1_train, v1_test),
        ("V2 Deep", "5→32→16→1", v2_train, v2_test),
        ("V3 +Drop", "5→32→16→1 +Dropout", v3_train, v3_test),
        ("V4 +L2", "5→32→16→1 +Drop+L2", v4_train, v4_test),
        ("V5 Full", "5→32→16→1 +All", v5_train, v5_test),
    ]

    for name, arch, train_a, test_a in models_info:
        gap = train_a - test_a  # Gap lớn = overfitting
        gap_warning = " ← overfit!" if gap > 10 else ""
        print(f"  {name:<10s} {arch:<28s} {train_a:>7.1f}% {test_a:>7.1f}% {gap:>+7.1f}%{gap_warning}")

    print("\nGhi chu:")
    print("  - Gap = Train - Test. Gap cang NHO cang tot (it overfit)")
    print("  - Dropout + L2 giup GIAM gap (model tong quat hoa tot hon)")
    print("  - Early Stopping dung train dung luc, tranh train qua lau")

    return best_model_path


# =============================================================================
# VÍ DỤ 3: TẢI MODEL CẢI TIẾN VÀ SỬ DỤNG
# =============================================================================

def vi_du_su_dung_model_cai_tien(model_path, norm_path):
    print("\n" + "=" * 65)
    print("VI DU 3: TAI VA SU DUNG MODEL CAI TIEN")
    print("=" * 65)

    # Tải model
    print("\n--- Tai model ---")
    model = ImprovedNN.load(model_path)

    # Tải thông tin chuẩn hóa
    with open(norm_path, "r") as f:
        norm_data = json.load(f)
    mean = np.array(norm_data["mean"])
    std = np.array(norm_data["std"])
    features = norm_data["features"]

    # Dự đoán hàng loạt nhân viên
    print("\n--- Du doan hang loat ---")
    danh_sach_nv = [
        {"ten": "Nguyen Van A", "data": [35, 8, 3, 9, 2]},    # Senior, lương cao, gần
        {"ten": "Tran Thi B",   "data": [10, 0.5, 30, 3, 6]},  # Fresher, xa, quá tải
        {"ten": "Le Van C",     "data": [22, 3, 10, 6, 3]},     # Junior, trung bình
        {"ten": "Pham Thi D",   "data": [18, 2, 18, 4, 5]},     # Lương thấp, không hài lòng
        {"ten": "Hoang Van E",  "data": [28, 5, 7, 8, 3]},      # Ổn định
    ]

    print(f"\n  {'Ten':<16s} {'Luong':>6s} {'Nam':>4s} {'KC':>4s} {'HL':>4s} {'DA':>4s} │ {'Ket qua':<12s} {'Xac suat':>8s}")
    print("  " + "-" * 70)

    for nv in danh_sach_nv:
        X_new = np.array([nv["data"]])
        X_new_norm = (X_new - mean) / std

        prob = model.predict_proba(X_new_norm)[0]
        pred = "NGHI VIEC" if prob >= 0.5 else "O LAI"

        # Mức rủi ro
        if prob >= 0.7:
            risk = "CAO"
        elif prob >= 0.4:
            risk = "TB"
        else:
            risk = "THAP"

        d = nv["data"]
        print(f"  {nv['ten']:<16s} {d[0]:>5.0f}M {d[1]:>3.0f}y {d[2]:>3.0f}km {d[3]:>3.0f} {d[4]:>3.0f}  │ "
              f"{pred:<12s} {prob:>7.1%} (rui ro: {risk})")

    # Phân tích feature quan trọng (simple weight analysis)
    print("\n--- Phan tich feature quan trong ---")
    print("  (Dua tren trong so tuyet doi cua tang dau)")
    w1 = np.abs(model.weights[0])
    importance = np.mean(w1, axis=1)  # Trung bình trọng số mỗi feature
    importance_norm = importance / np.sum(importance) * 100

    # Sắp xếp theo mức quan trọng
    sorted_idx = np.argsort(importance_norm)[::-1]
    for rank, idx in enumerate(sorted_idx, 1):
        bar = "|" * int(importance_norm[idx] * 2)
        print(f"  {rank}. {features[idx]:<15s} {importance_norm[idx]:>5.1f}% {bar}")


# =============================================================================
# CHẠY TẤT CẢ
# =============================================================================

if __name__ == "__main__":
    # Ví dụ 1: Train → Lưu → Tải → Dự đoán
    model_v1_path, norm_path = vi_du_luu_tai_model()

    # Ví dụ 2: Cải tiến model qua nhiều phiên bản
    best_model_path = vi_du_cai_tien_model()

    # Ví dụ 3: Tải model cải tiến và sử dụng thực tế
    vi_du_su_dung_model_cai_tien(best_model_path, norm_path)

    # Dọn dẹp file model tạm
    for f in [model_v1_path, best_model_path, norm_path]:
        if os.path.exists(f):
            os.remove(f)
    print("\n  (Da xoa cac file model tam)")

    print("\n" + "=" * 65)
    print("TOM TAT:")
    print("=" * 65)
    print("""
    1. LUU/TAI MODEL:
       - Luu trong so (weights, biases) ra file (JSON, pickle, HDF5...)
       - PHAI luu ca normalization params (mean, std)
       - Khi tai: tao lai kien truc → gan trong so → san sang predict

    2. CAC KY THUAT CAI TIEN:
       - Deeper network: them tang an → hoc quan he phuc tap hon
       - Dropout: tat ngau nhien neuron → chong overfitting
       - L2 Regularization: phat trong so lon → model don gian hon
       - Batch Normalization: chuan hoa moi layer → train nhanh hon
       - Learning Rate Decay: giam LR dan → hoi tu tot hon
       - Early Stopping: dung khi val loss khong giam → tranh overfit

    3. QUY TRINH THUC TE:
       a) Train model co ban → danh gia
       b) Cai tien tung buoc, so sanh ket qua
       c) Chon model tot nhat (test accuracy cao, gap train-test nho)
       d) Luu model + normalization params
       e) Deploy: tai model → chuan hoa du lieu moi → predict
    """)
