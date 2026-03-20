"""
Services - Các class tiện ích dùng chung cho tất cả ví dụ ML/AI

Tập hợp các hàm được sử dụng lặp lại ở nhiều file:
  - ActivationService: các hàm kích hoạt (sigmoid, relu, softmax, tanh)
  - LossService: các hàm mất mát (binary/categorical cross-entropy, MSE)
  - DataService: xử lý dữ liệu (normalize, train_test_split, one_hot_encode)
"""

import numpy as np


# =============================================================================
# ACTIVATION SERVICE - Các hàm kích hoạt
# =============================================================================

class ActivationService:
    """
    Tập hợp các hàm kích hoạt phổ biến trong Neural Network.

    Hàm kích hoạt quyết định nơ-ron có "kích hoạt" (truyền tín hiệu) hay không.
    Không có hàm kích hoạt → mạng chỉ là phép tuyến tính, không học được gì phức tạp.
    """

    @staticmethod
    def sigmoid(x):
        """
        Sigmoid: nén giá trị về khoảng (0, 1).
        Dùng cho: phân loại nhị phân (có/không, bệnh/khỏe).
        """
        x = np.clip(x, -500, 500)
        return 1 / (1 + np.exp(-x))

    @staticmethod
    def sigmoid_derivative(x):
        """Đạo hàm sigmoid: dùng trong backpropagation."""
        s = ActivationService.sigmoid(x)
        return s * (1 - s)

    @staticmethod
    def relu(x):
        """
        ReLU (Rectified Linear Unit): giữ nguyên nếu dương, = 0 nếu âm.
        Phổ biến nhất trong deep learning.
        """
        return np.maximum(0, x)

    @staticmethod
    def relu_derivative(x):
        """Đạo hàm ReLU: 1 nếu x > 0, 0 nếu x <= 0."""
        return (x > 0).astype(float)

    @staticmethod
    def softmax(x):
        """
        Softmax: chuyển vector thành phân phối xác suất (tổng = 1).
        Dùng cho: phân loại nhiều lớp.
        """
        exp_x = np.exp(x - np.max(x, axis=1, keepdims=True))
        return exp_x / np.sum(exp_x, axis=1, keepdims=True)

    @staticmethod
    def tanh(x):
        """Tanh: nén giá trị về (-1, 1). Dùng trong RNN."""
        return np.tanh(x)

    @staticmethod
    def tanh_derivative(x):
        """Đạo hàm tanh."""
        return 1 - np.tanh(x) ** 2


# =============================================================================
# LOSS SERVICE - Các hàm mất mát
# =============================================================================

class LossService:
    """
    Tập hợp các hàm mất mát (loss functions).

    Loss function đo "mô hình sai bao nhiêu".
    Mục tiêu: giảm loss xuống thấp nhất.
    """

    @staticmethod
    def binary_cross_entropy(y_true, y_pred):
        """
        Binary Cross-Entropy: dùng cho phân loại 2 lớp.
        Phạt nặng khi mô hình "tự tin sai" (dự đoán 0.99 nhưng thực tế là 0).
        """
        y_pred = np.clip(y_pred, 1e-8, 1 - 1e-8)
        return -np.mean(y_true * np.log(y_pred) + (1 - y_true) * np.log(1 - y_pred))

    @staticmethod
    def categorical_cross_entropy(y_true, y_pred):
        """
        Categorical Cross-Entropy: dùng cho phân loại nhiều lớp.
        y_true: one-hot encoded (vd: [0, 1, 0] = lớp thứ 2).
        """
        y_pred = np.clip(y_pred, 1e-8, 1 - 1e-8)
        return -np.mean(np.sum(y_true * np.log(y_pred), axis=1))

    @staticmethod
    def mse(y_true, y_pred):
        """
        Mean Squared Error: dùng cho bài toán regression (dự đoán giá trị liên tục).
        """
        return np.mean((y_true - y_pred) ** 2)


# =============================================================================
# DATA SERVICE - Xử lý dữ liệu
# =============================================================================

class DataService:
    """
    Tập hợp các hàm xử lý dữ liệu phổ biến.
    """

    @staticmethod
    def normalize(X):
        """
        Chuẩn hóa dữ liệu về mean=0, std=1 (Z-score normalization).

        Rất quan trọng cho Neural Network vì:
          - Giúp gradient descent hội tụ nhanh hơn
          - Tránh features có giá trị lớn "lấn át" features nhỏ

        Returns:
            X_normalized, mean, std
        """
        mean = np.mean(X, axis=0)
        std = np.std(X, axis=0)
        std[std == 0] = 1  # Tránh chia cho 0
        return (X - mean) / std, mean, std

    @staticmethod
    def train_test_split(X, y, test_ratio=0.2, seed=42):
        """
        Chia dữ liệu thành tập train và test.

        Args:
            X: dữ liệu đầu vào
            y: nhãn
            test_ratio: tỷ lệ dữ liệu test (mặc định 20%)
            seed: random seed để kết quả tái lập được

        Returns:
            X_train, X_test, y_train, y_test
        """
        np.random.seed(seed)
        n = len(X)
        indices = np.random.permutation(n)
        test_size = int(n * test_ratio)
        test_idx = indices[:test_size]
        train_idx = indices[test_size:]
        return X[train_idx], X[test_idx], y[train_idx], y[test_idx]

    @staticmethod
    def train_val_test_split(X, y, val_ratio=0.15, test_ratio=0.15, seed=42):
        """
        Chia dữ liệu thành 3 tập: train / validation / test.

        Validation dùng để:
          - Chọn hyperparameters (learning rate, số tầng...)
          - Early stopping (dừng train khi val loss không giảm)

        Returns:
            X_train, y_train, X_val, y_val, X_test, y_test
        """
        np.random.seed(seed)
        n = len(X)
        idx = np.random.permutation(n)

        n_test = int(n * test_ratio)
        n_val = int(n * val_ratio)

        test_idx = idx[:n_test]
        val_idx = idx[n_test:n_test + n_val]
        train_idx = idx[n_test + n_val:]

        return (X[train_idx], y[train_idx],
                X[val_idx], y[val_idx],
                X[test_idx], y[test_idx])

    @staticmethod
    def one_hot_encode(y, n_classes):
        """
        One-hot encoding: chuyển nhãn số thành vector nhị phân.

        Ví dụ: nhãn 2 với 3 lớp → [0, 0, 1]
        Neural network cần dạng này để tính loss với softmax.
        """
        encoded = np.zeros((len(y), n_classes))
        for i, label in enumerate(y):
            encoded[i, int(label)] = 1
        return encoded
