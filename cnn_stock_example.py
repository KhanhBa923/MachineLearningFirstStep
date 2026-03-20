"""
Convolutional Neural Network (CNN) - Mạng Nơ-ron Tích chập

CNN ban đầu được thiết kế cho xử lý ảnh, nhưng cũng rất mạnh với dữ liệu
chuỗi thời gian (time series) như giá chứng khoán.

Ý tưởng cốt lõi:
  - Convolution (tích chập): dùng "bộ lọc" (filter/kernel) trượt qua dữ liệu
    để phát hiện CÁC MẪU CỤC BỘ (local patterns)
  - Trong chứng khoán: filter có thể phát hiện các mẫu giá như
    "đầu vai" (head & shoulders), "cốc tay cầm" (cup & handle),
    xu hướng tăng/giảm ngắn hạn, v.v.

Tại sao CNN phù hợp với chứng khoán:
  1. Phát hiện mẫu giá (price patterns) tự động
  2. Không cần feature engineering thủ công
  3. 1D CNN xử lý chuỗi thời gian hiệu quả
  4. Có thể kết hợp nhiều chỉ báo kỹ thuật cùng lúc

Ví dụ trong file này:
  1. Dự đoán xu hướng giá cổ phiếu (Tăng/Giảm) bằng 1D CNN
  2. Phân loại mẫu nến (Candlestick Pattern Recognition)
  3. Dự đoán biến động (Volatility) từ dữ liệu lịch sử
"""

import numpy as np
from services import ActivationService, LossService, DataService


# =============================================================================
# PHẦN 1: CÁC THÀNH PHẦN CỦA CNN
# =============================================================================

class Conv1D:
    """
    Lớp Convolution 1D - thành phần cốt lõi của CNN.

    Cách hoạt động (ví dụ với giá cổ phiếu):
      - Input: chuỗi giá 30 ngày [p1, p2, ..., p30]
      - Filter (kernel_size=5): cửa sổ trượt 5 ngày
      - Filter trượt qua chuỗi: [p1..p5], [p2..p6], ..., [p26..p30]
      - Tại mỗi vị trí: nhân element-wise với weights rồi cộng lại
      - Kết quả: mỗi vị trí cho 1 giá trị = "mức độ khớp với mẫu"

    Ví dụ trực quan:
      Nếu filter học được weights = [-1, -0.5, 0, 0.5, 1]
      → Filter này phát hiện XU HƯỚNG TĂNG (giá tăng dần qua 5 ngày)
      Nếu filter học được weights = [1, 0.5, 0, -0.5, -1]
      → Filter này phát hiện XU HƯỚNG GIẢM
    """

    def __init__(self, n_filters, kernel_size, n_channels=1):
        """
        Args:
            n_filters: số bộ lọc (mỗi filter phát hiện 1 loại mẫu khác nhau)
            kernel_size: kích thước cửa sổ trượt (bao nhiêu ngày)
            n_channels: số kênh đầu vào (1 = chỉ giá, nhiều hơn = giá + volume + indicators)
        """
        self.n_filters = n_filters
        self.kernel_size = kernel_size
        self.n_channels = n_channels

        # Khởi tạo trọng số (He initialization)
        scale = np.sqrt(2.0 / (kernel_size * n_channels))
        self.filters = np.random.randn(n_filters, kernel_size, n_channels) * scale
        self.biases = np.zeros(n_filters)

    def forward(self, X):
        """
        Forward pass: trượt filter qua dữ liệu.

        Args:
            X: shape (batch_size, sequence_length, n_channels)
               Ví dụ: (32 mẫu, 30 ngày, 4 kênh [open, high, low, close])

        Returns:
            output: shape (batch_size, output_length, n_filters)
        """
        self.input = X
        batch_size, seq_len, _ = X.shape
        output_len = seq_len - self.kernel_size + 1  # Chiều dài sau convolution

        output = np.zeros((batch_size, output_len, self.n_filters))

        # Trượt filter qua từng vị trí
        for i in range(output_len):
            # Cắt cửa sổ tại vị trí i
            window = X[:, i:i + self.kernel_size, :]  # (batch, kernel_size, channels)

            for f in range(self.n_filters):
                # Nhân element-wise rồi cộng tất cả → 1 giá trị
                # Giống như "đo độ tương đồng" giữa cửa sổ dữ liệu và filter
                output[:, i, f] = np.sum(window * self.filters[f], axis=(1, 2)) + self.biases[f]

        self.output = output
        return output

    def backward(self, d_output, learning_rate):
        """Backward pass: tính gradient và cập nhật filter weights."""
        batch_size, seq_len, _ = self.input.shape
        output_len = seq_len - self.kernel_size + 1

        d_filters = np.zeros_like(self.filters)
        d_biases = np.zeros_like(self.biases)
        d_input = np.zeros_like(self.input)

        for i in range(output_len):
            window = self.input[:, i:i + self.kernel_size, :]

            for f in range(self.n_filters):
                # Gradient cho filter
                d_out_f = d_output[:, i, f]  # (batch,)
                d_filters[f] += np.sum(
                    window * d_out_f[:, np.newaxis, np.newaxis],
                    axis=0
                )
                d_biases[f] += np.sum(d_out_f)

                # Gradient cho input
                d_input[:, i:i + self.kernel_size, :] += (
                    self.filters[f] * d_out_f[:, np.newaxis, np.newaxis]
                )

        # Cập nhật trọng số
        self.filters -= learning_rate * d_filters / batch_size
        self.biases -= learning_rate * d_biases / batch_size

        return d_input


class MaxPool1D:
    """
    Max Pooling 1D - giảm kích thước dữ liệu.

    Cách hoạt động:
      - Chia output thành các nhóm (pool_size phần tử)
      - Lấy giá trị LỚN NHẤT trong mỗi nhóm

    Trong chứng khoán:
      - Giữ lại tín hiệu MẠNH NHẤT (đỉnh giá, đáy giá)
      - Bỏ qua nhiễu nhỏ, giữ lại xu hướng chính
      - Giảm số lượng tham số → tránh overfitting
    """

    def __init__(self, pool_size=2):
        self.pool_size = pool_size

    def forward(self, X):
        """
        Args:
            X: shape (batch_size, seq_len, n_filters)
        Returns:
            output: shape (batch_size, seq_len // pool_size, n_filters)
        """
        self.input = X
        batch_size, seq_len, n_filters = X.shape
        output_len = seq_len // self.pool_size

        output = np.zeros((batch_size, output_len, n_filters))
        self.max_indices = np.zeros((batch_size, output_len, n_filters), dtype=int)

        for i in range(output_len):
            start = i * self.pool_size
            end = start + self.pool_size
            pool_region = X[:, start:end, :]

            # Lấy giá trị lớn nhất và lưu vị trí (để backward)
            output[:, i, :] = np.max(pool_region, axis=1)
            self.max_indices[:, i, :] = np.argmax(pool_region, axis=1) + start

        self.output = output
        return output

    def backward(self, d_output, learning_rate=None):
        """Backward: gradient chỉ truyền qua vị trí max."""
        d_input = np.zeros_like(self.input)
        batch_size, output_len, n_filters = d_output.shape

        for i in range(output_len):
            for f in range(n_filters):
                for b in range(batch_size):
                    idx = self.max_indices[b, i, f]
                    d_input[b, idx, f] += d_output[b, i, f]

        return d_input


class Flatten:
    """
    Flatten: chuyển tensor đa chiều thành vector 1 chiều.

    Sau Conv + Pool, dữ liệu có shape (batch, seq_len, filters).
    Flatten biến thành (batch, seq_len * filters) để đưa vào Dense layer.
    """

    def forward(self, X):
        self.input_shape = X.shape
        return X.reshape(X.shape[0], -1)

    def backward(self, d_output, learning_rate=None):
        return d_output.reshape(self.input_shape)


class Dense:
    """
    Dense (Fully Connected) Layer - tầng kết nối đầy đủ.

    Mỗi nơ-ron kết nối với TẤT CẢ nơ-ron tầng trước.
    Thường đặt sau Conv+Pool để tổng hợp features và đưa ra quyết định.
    """

    def __init__(self, n_input, n_output):
        scale = np.sqrt(2.0 / n_input)
        self.weights = np.random.randn(n_input, n_output) * scale
        self.biases = np.zeros((1, n_output))

    def forward(self, X):
        self.input = X
        self.output = X @ self.weights + self.biases
        return self.output

    def backward(self, d_output, learning_rate):
        d_input = d_output @ self.weights.T
        d_weights = self.input.T @ d_output / self.input.shape[0]
        d_biases = np.mean(d_output, axis=0, keepdims=True)

        self.weights -= learning_rate * d_weights
        self.biases -= learning_rate * d_biases

        return d_input


class ReLU:
    """ReLU activation: giữ giá trị dương, đặt âm = 0. (Dùng ActivationService)"""

    def forward(self, X):
        self.input = X
        return ActivationService.relu(X)

    def backward(self, d_output, learning_rate=None):
        return d_output * ActivationService.relu_derivative(self.input)


class Sigmoid:
    """Sigmoid activation: nén về (0, 1) cho phân loại nhị phân. (Dùng ActivationService)"""

    def forward(self, X):
        self.output = ActivationService.sigmoid(X)
        return self.output

    def backward(self, d_output, learning_rate=None):
        return d_output * self.output * (1 - self.output)


class Softmax:
    """Softmax activation: chuyển thành phân phối xác suất cho nhiều lớp. (Dùng ActivationService)"""

    def forward(self, X):
        exp_x = np.exp(X - np.max(X, axis=1, keepdims=True))
        self.output = exp_x / np.sum(exp_x, axis=1, keepdims=True)
        return self.output

    def backward(self, d_output, learning_rate=None):
        return d_output


# =============================================================================
# PHẦN 2: LỚP CNN HOÀN CHỈNH
# =============================================================================

class StockCNN:
    """
    CNN cho dữ liệu chứng khoán.

    Kiến trúc:
        Input (chuỗi giá) → Conv1D → ReLU → MaxPool → Flatten → Dense → Output

    Ví dụ:
        30 ngày giá × 4 kênh (OHLC) → 8 filters × kernel 5 → Pool 2
        → Flatten → Dense 64 → Dense 1 (sigmoid) → Tăng/Giảm
    """

    def __init__(self, seq_length, n_channels, n_filters, kernel_size,
                 pool_size, dense_size, n_output, task='binary',
                 learning_rate=0.001):
        """
        Args:
            seq_length: chiều dài chuỗi (số ngày)
            n_channels: số kênh (1=close, 4=OHLC, nhiều hơn nếu thêm indicators)
            n_filters: số filter trong Conv layer
            kernel_size: kích thước filter
            pool_size: kích thước max pooling
            dense_size: số nơ-ron tầng Dense
            n_output: số output (1 cho binary, nhiều hơn cho multi-class)
            task: 'binary' hoặc 'multiclass'
            learning_rate: tốc độ học
        """
        self.learning_rate = learning_rate
        self.task = task

        # Tính kích thước sau mỗi layer
        conv_output_len = seq_length - kernel_size + 1
        pool_output_len = conv_output_len // pool_size
        flatten_size = pool_output_len * n_filters

        # Xây dựng các tầng
        self.layers = [
            Conv1D(n_filters, kernel_size, n_channels),  # Phát hiện mẫu giá
            ReLU(),                                       # Kích hoạt phi tuyến
            MaxPool1D(pool_size),                         # Giữ tín hiệu mạnh
            Flatten(),                                    # Duỗi thẳng
            Dense(flatten_size, dense_size),              # Tổng hợp features
            ReLU(),                                       # Kích hoạt
            Dense(dense_size, n_output),                  # Quyết định cuối
        ]

        # Tầng output tùy theo bài toán
        if task == 'binary':
            self.layers.append(Sigmoid())
        else:
            self.layers.append(Softmax())

        self.loss_history = []

    def forward(self, X):
        """Đưa dữ liệu qua toàn bộ mạng."""
        current = X
        for layer in self.layers:
            current = layer.forward(current)
        return current

    def backward(self, y_true, y_pred):
        """Truyền gradient ngược qua mạng."""
        # Gradient ban đầu: y_pred - y_true (với cross-entropy)
        d = y_pred - y_true

        # Truyền ngược qua từng layer
        for layer in reversed(self.layers):
            d = layer.backward(d, self.learning_rate)

    def train(self, X, y, epochs=100, batch_size=32, verbose=True):
        """Huấn luyện CNN."""
        n_samples = X.shape[0]

        for epoch in range(epochs):
            indices = np.random.permutation(n_samples)
            X_shuffled = X[indices]
            y_shuffled = y[indices]

            epoch_loss = 0

            for start in range(0, n_samples, batch_size):
                end = min(start + batch_size, n_samples)
                X_batch = X_shuffled[start:end]
                y_batch = y_shuffled[start:end]

                # Forward
                y_pred = self.forward(X_batch)

                # Loss
                y_pred_clip = np.clip(y_pred, 1e-8, 1 - 1e-8)
                if self.task == 'binary':
                    loss = LossService.binary_cross_entropy(y_batch, y_pred_clip)
                else:
                    loss = LossService.categorical_cross_entropy(y_batch, y_pred_clip)

                epoch_loss += loss * (end - start)

                # Backward
                self.backward(y_batch, y_pred)

            epoch_loss /= n_samples
            self.loss_history.append(epoch_loss)

            if verbose and (epoch + 1) % max(1, epochs // 5) == 0:
                acc = self.accuracy(X, y)
                print(f"  Epoch {epoch + 1:4d}/{epochs} | Loss: {epoch_loss:.4f} | Accuracy: {acc:.1f}%")

    def predict(self, X):
        """Dự đoán."""
        probs = self.forward(X)
        if self.task == 'binary':
            return (probs >= 0.5).astype(int).flatten()
        else:
            return np.argmax(probs, axis=1)

    def predict_proba(self, X):
        """Trả về xác suất dự đoán."""
        return self.forward(X)

    def accuracy(self, X, y):
        """Tính độ chính xác."""
        preds = self.predict(X)
        if y.ndim > 1:
            y_labels = np.argmax(y, axis=1)
        else:
            y_labels = y.flatten()
        return np.mean(preds == y_labels) * 100


# =============================================================================
# HÀM TẠO DỮ LIỆU CHỨNG KHOÁN MÔ PHỎNG
# =============================================================================

def tao_du_lieu_co_phieu(n_stocks=5, n_days=500, seed=42):
    """
    Tạo dữ liệu giá cổ phiếu mô phỏng với OHLC (Open, High, Low, Close).

    Mô phỏng 3 loại xu hướng:
      - Uptrend: giá tăng dần (mô phỏng cổ phiếu tăng trưởng)
      - Downtrend: giá giảm dần (mô phỏng cổ phiếu suy thoái)
      - Sideways: giá dao động quanh 1 mức (mô phỏng cổ phiếu đi ngang)
    """
    np.random.seed(seed)

    all_data = []

    for _ in range(n_stocks):
        # Giá khởi điểm ngẫu nhiên
        base_price = np.random.uniform(20, 200)

        # Chọn ngẫu nhiên drift (xu hướng) cho cổ phiếu
        # drift > 0: tăng, drift < 0: giảm, drift ≈ 0: đi ngang
        drift = np.random.choice([-0.001, 0, 0.001])

        prices = [base_price]
        for day in range(1, n_days):
            # Mô hình Random Walk với drift (Geometric Brownian Motion đơn giản)
            daily_return = drift + np.random.normal(0, 0.02)
            new_price = prices[-1] * (1 + daily_return)
            prices.append(max(new_price, 1))  # Giá không âm

        prices = np.array(prices)

        # Tạo OHLC từ close price
        ohlc = np.zeros((n_days, 4))
        for day in range(n_days):
            close = prices[day]
            daily_range = close * np.random.uniform(0.01, 0.04)
            ohlc[day, 0] = close + np.random.uniform(-daily_range, daily_range)  # Open
            ohlc[day, 1] = max(close, ohlc[day, 0]) + np.random.uniform(0, daily_range)  # High
            ohlc[day, 2] = min(close, ohlc[day, 0]) - np.random.uniform(0, daily_range)  # Low
            ohlc[day, 3] = close  # Close

        all_data.append(ohlc)

    return all_data


def tao_dataset_xu_huong(window_size=30, predict_days=5, n_samples=800):
    """
    Tạo dataset cho bài toán dự đoán xu hướng.

    Cách tạo:
      - Lấy cửa sổ window_size ngày làm input
      - Nhãn: giá sau predict_days ngày TĂNG hay GIẢM so với ngày cuối cửa sổ

    Args:
        window_size: số ngày dùng để dự đoán
        predict_days: dự đoán xu hướng sau bao nhiêu ngày
        n_samples: số mẫu cần tạo
    """
    np.random.seed(42)

    X_list = []
    y_list = []

    for _ in range(n_samples):
        # Tạo chuỗi giá với xu hướng rõ ràng
        trend = np.random.choice([-1, 1])  # -1: giảm, 1: tăng
        total_len = window_size + predict_days

        base = np.random.uniform(50, 200)
        noise = np.random.normal(0, 0.015, total_len)
        trend_strength = np.random.uniform(0.001, 0.005) * trend

        prices = [base]
        for t in range(1, total_len):
            ret = trend_strength + noise[t]
            prices.append(prices[-1] * (1 + ret))
        prices = np.array(prices)

        # OHLC cho cửa sổ input
        ohlc = np.zeros((window_size, 4))
        for d in range(window_size):
            c = prices[d]
            r = c * np.random.uniform(0.005, 0.02)
            ohlc[d] = [
                c + np.random.uniform(-r, r),      # Open
                c + np.random.uniform(0, r * 1.5),  # High
                c - np.random.uniform(0, r * 1.5),  # Low
                c                                    # Close
            ]

        # Chuẩn hóa theo giá đầu tiên (biến thành % thay đổi)
        # Rất quan trọng: giúp mô hình học MẪU HÌNH thay vì giá tuyệt đối
        first_close = ohlc[0, 3]
        ohlc_normalized = (ohlc - first_close) / first_close

        X_list.append(ohlc_normalized)

        # Nhãn: giá cuối kỳ so với giá cuối cửa sổ
        future_price = prices[window_size + predict_days - 1]
        current_price = prices[window_size - 1]
        label = 1 if future_price > current_price else 0
        y_list.append(label)

    return np.array(X_list), np.array(y_list).reshape(-1, 1).astype(float)


def tao_dataset_mau_nen(n_samples=600):
    """
    Tạo dataset phân loại mẫu nến (Candlestick Patterns).

    3 loại mẫu phổ biến:
      - Hammer (Búa): thân nhỏ phía trên, bóng dưới dài → tín hiệu đảo chiều tăng
      - Doji (Do dự): thân rất nhỏ, bóng trên/dưới dài → thị trường do dự
      - Engulfing (Nhấn chìm): nến sau "nuốt" nến trước → tín hiệu mạnh
    """
    np.random.seed(42)

    X_list = []
    y_list = []
    window = 10  # Nhìn 10 nến để nhận dạng

    for _ in range(n_samples // 3):
        base = np.random.uniform(50, 200)

        # --- MẪU 0: HAMMER (Búa) ---
        # Đặc điểm: giá giảm rồi xuất hiện nến búa → đảo chiều tăng
        candles = np.zeros((window, 4))
        for d in range(window):
            c = base * (1 - 0.01 * d + np.random.normal(0, 0.005))
            r = c * 0.02
            if d == window - 1:
                # Nến cuối là hammer: thân nhỏ phía trên, bóng dưới dài
                o = c * (1 + 0.003)
                h = max(o, c) * (1 + 0.005)
                l = min(o, c) * (1 - 0.03)   # Bóng dưới rất dài
                candles[d] = [o, h, l, c]
            else:
                candles[d] = [c + r * 0.5, c + r, c - r * 0.5, c]

        candles_norm = (candles - candles[0, 3]) / candles[0, 3]
        X_list.append(candles_norm)
        y_list.append(0)

        # --- MẪU 1: DOJI (Do dự) ---
        # Đặc điểm: open ≈ close, bóng trên/dưới dài
        candles = np.zeros((window, 4))
        for d in range(window):
            c = base * (1 + np.random.normal(0, 0.008))
            r = c * 0.02
            if d == window - 1:
                # Nến cuối là doji
                o = c * (1 + 0.001)  # Open ≈ Close
                h = c * (1 + 0.025)  # Bóng trên dài
                l = c * (1 - 0.025)  # Bóng dưới dài
                candles[d] = [o, h, l, c]
            else:
                candles[d] = [c - r * 0.3, c + r, c - r, c]

        candles_norm = (candles - candles[0, 3]) / candles[0, 3]
        X_list.append(candles_norm)
        y_list.append(1)

        # --- MẪU 2: BULLISH ENGULFING (Nhấn chìm tăng) ---
        # Đặc điểm: nến đỏ nhỏ, sau đó nến xanh lớn nuốt trọn
        candles = np.zeros((window, 4))
        for d in range(window):
            c = base * (1 - 0.005 * d + np.random.normal(0, 0.005))
            r = c * 0.015
            if d == window - 2:
                # Nến áp chót: nến đỏ nhỏ (giảm)
                o = c * (1 + 0.008)
                candles[d] = [o, o * 1.005, c * 0.995, c]
            elif d == window - 1:
                # Nến cuối: nến xanh lớn nuốt trọn nến trước
                prev_open = candles[d - 1, 0]
                prev_close = candles[d - 1, 3]
                o = prev_close * 0.998
                c_new = prev_open * 1.01
                candles[d] = [o, c_new * 1.005, o * 0.995, c_new]
            else:
                candles[d] = [c + r, c + r * 1.5, c - r, c]

        candles_norm = (candles - candles[0, 3]) / candles[0, 3]
        X_list.append(candles_norm)
        y_list.append(2)

    return np.array(X_list), np.array(y_list)


train_test_split = DataService.train_test_split
one_hot = DataService.one_hot_encode


# =============================================================================
# VÍ DỤ 1: DỰ ĐOÁN XU HƯỚNG GIÁ CỔ PHIẾU
# =============================================================================
# Bối cảnh: Cho dữ liệu OHLC (Open, High, Low, Close) 30 ngày gần nhất,
# dự đoán giá sẽ TĂNG hay GIẢM trong 5 ngày tiếp theo.

def vi_du_du_doan_xu_huong():
    print("=" * 60)
    print("VI DU 1: DU DOAN XU HUONG GIA CO PHIEU")
    print("=" * 60)
    print("Input: 30 ngay OHLC (Open, High, Low, Close)")
    print("Output: Tang (1) hay Giam (0) sau 5 ngay\n")

    # Tạo dữ liệu
    X, y = tao_dataset_xu_huong(window_size=30, predict_days=5, n_samples=800)
    print(f"Dataset: {X.shape[0]} mau, moi mau {X.shape[1]} ngay x {X.shape[2]} kenh (OHLC)")

    # Chia train/test
    X_train, X_test, y_train, y_test = train_test_split(X, y)

    # Tạo CNN
    # Kiến trúc: Conv(8 filters, kernel 5) → ReLU → MaxPool(2) → Dense(16) → Dense(1)
    # - 8 filters: mỗi filter phát hiện 1 mẫu giá khác nhau
    # - kernel_size=5: nhìn 5 ngày liên tiếp để tìm mẫu
    # - pool_size=2: giảm chiều, giữ tín hiệu mạnh
    np.random.seed(42)
    model = StockCNN(
        seq_length=30,
        n_channels=4,        # 4 kênh: Open, High, Low, Close
        n_filters=8,         # 8 bộ lọc phát hiện 8 loại mẫu
        kernel_size=5,       # Cửa sổ 5 ngày
        pool_size=2,         # Max pooling kích thước 2
        dense_size=16,       # 16 nơ-ron tầng Dense
        n_output=1,          # 1 output: xác suất tăng
        task='binary',
        learning_rate=0.005
    )

    print("\nKien truc: Input(30x4) → Conv1D(8, k=5) → ReLU → MaxPool(2) → Dense(16) → Dense(1)")
    print("Huan luyen...\n")

    model.train(X_train, y_train, epochs=50, batch_size=32)

    # Đánh giá
    train_acc = model.accuracy(X_train, y_train)
    test_acc = model.accuracy(X_test, y_test)
    print(f"\nKet qua: Train = {train_acc:.1f}% | Test = {test_acc:.1f}%")

    # Mô phỏng giao dịch đơn giản
    print("\n--- Mo phong giao dich ---")
    preds = model.predict(X_test)
    y_test_flat = y_test.flatten().astype(int)

    n_trade = 0
    n_win = 0
    for i in range(len(preds)):
        if preds[i] == 1:  # Chỉ mua khi dự đoán tăng
            n_trade += 1
            if y_test_flat[i] == 1:
                n_win += 1

    if n_trade > 0:
        win_rate = n_win / n_trade * 100
        print(f"  So lenh mua: {n_trade}/{len(preds)} (chi mua khi du doan TANG)")
        print(f"  Thang: {n_win}/{n_trade} ({win_rate:.1f}%)")
        print(f"  Thua:  {n_trade - n_win}/{n_trade} ({100 - win_rate:.1f}%)")
    else:
        print("  Khong co lenh mua nao.")

    # Hiển thị xác suất dự đoán cho vài mẫu
    print("\n--- Du doan chi tiet (5 mau dau) ---")
    probs = model.predict_proba(X_test[:5])
    for i in range(5):
        prob_tang = probs[i, 0]
        actual = "TANG" if y_test_flat[i] == 1 else "GIAM"
        signal = "MUA" if prob_tang >= 0.5 else "KHONG MUA"
        dung_sai = "V" if preds[i] == y_test_flat[i] else "X"
        print(f"  Mau {i + 1}: P(tang)={prob_tang:.1%} → {signal:10s} | Thuc te: {actual} [{dung_sai}]")


# =============================================================================
# VÍ DỤ 2: PHÂN LOẠI MẪU NẾN (CANDLESTICK PATTERNS)
# =============================================================================
# Bối cảnh: Nhận dạng 3 mẫu nến phổ biến trong phân tích kỹ thuật.

def vi_du_mau_nen():
    print("\n" + "=" * 60)
    print("VI DU 2: PHAN LOAI MAU NEN (CANDLESTICK PATTERNS)")
    print("=" * 60)
    print("3 mau: Hammer (Bua), Doji (Do du), Bullish Engulfing (Nhan chim)\n")

    # Tạo dữ liệu
    X, y = tao_dataset_mau_nen(n_samples=600)
    y_oh = one_hot(y, 3)

    print(f"Dataset: {X.shape[0]} mau, {X.shape[1]} nen x {X.shape[2]} kenh")
    print(f"Phan bo: Hammer={np.sum(y == 0)}, Doji={np.sum(y == 1)}, Engulfing={np.sum(y == 2)}")

    # Chia train/test
    X_train, X_test, y_train, y_test = train_test_split(X, y_oh)

    # Tạo CNN cho multi-class classification
    np.random.seed(42)
    model = StockCNN(
        seq_length=10,
        n_channels=4,
        n_filters=12,       # Nhiều filter hơn để phát hiện các mẫu nến đa dạng
        kernel_size=3,       # Cửa sổ nhỏ hơn (3 nến) - mẫu nến thường 1-3 nến
        pool_size=2,
        dense_size=16,
        n_output=3,          # 3 loại mẫu nến
        task='multiclass',
        learning_rate=0.005
    )

    print("\nKien truc: Input(10x4) → Conv1D(12, k=3) → ReLU → MaxPool(2) → Dense(16) → Softmax(3)")
    print("Huan luyen...\n")

    model.train(X_train, y_train, epochs=80, batch_size=32)

    # Đánh giá
    train_acc = model.accuracy(X_train, y_train)
    test_acc = model.accuracy(X_test, y_test)
    print(f"\nKet qua: Train = {train_acc:.1f}% | Test = {test_acc:.1f}%")

    # Hiển thị kết quả chi tiết
    ten_mau = ["Hammer", "Doji", "Engulfing"]
    preds = model.predict(X_test)
    y_test_labels = np.argmax(y_test, axis=1)

    print("\n--- Ma tran nham lan (Confusion Matrix) ---")
    print(f"{'':15s} {'Pred Hammer':>12s} {'Pred Doji':>12s} {'Pred Engulf':>12s}")
    for i, name in enumerate(ten_mau):
        row = []
        for j in range(3):
            count = np.sum((y_test_labels == i) & (preds == j))
            row.append(count)
        print(f"  That {name:10s} {row[0]:>10d} {row[1]:>10d} {row[2]:>10d}")

    # Mô tả ý nghĩa giao dịch
    print("\n--- Y nghia giao dich ---")
    probs = model.predict_proba(X_test[:5])
    for i in range(5):
        pred_class = preds[i]
        actual_class = y_test_labels[i]
        confidence = probs[i, pred_class]

        signals = {
            0: "Hammer → Co the DAO CHIEU TANG, can nhac MUA",
            1: "Doji → Thi truong DO DU, CHO THEM tin hieu",
            2: "Engulfing → Tin hieu TANG MANH, co the MUA"
        }

        dung = "V" if pred_class == actual_class else "X"
        print(f"  Mau {i + 1}: {ten_mau[pred_class]:10s} (conf={confidence:.1%}) | "
              f"That: {ten_mau[actual_class]:10s} [{dung}]")
        print(f"         → {signals[pred_class]}")


# =============================================================================
# VÍ DỤ 3: DỰ ĐOÁN MỨC ĐỘ BIẾN ĐỘNG (VOLATILITY)
# =============================================================================
# Bối cảnh: Dự đoán ngày mai sẽ biến động MẠNH, TRUNG BÌNH, hay NHẸ.
# Quan trọng cho quản lý rủi ro và định giá quyền chọn (options).

def vi_du_bien_dong():
    print("\n" + "=" * 60)
    print("VI DU 3: DU DOAN BIEN DONG (VOLATILITY PREDICTION)")
    print("=" * 60)
    print("Input: 20 ngay OHLCV (Open, High, Low, Close, Volume)")
    print("Output: Bien dong Nhe / Trung binh / Manh\n")

    np.random.seed(42)
    n_samples = 600
    window = 20

    X_list = []
    y_list = []

    for _ in range(n_samples):
        base = np.random.uniform(30, 300)

        # Chọn mức biến động
        vol_class = np.random.choice([0, 1, 2])
        vol_levels = [0.005, 0.015, 0.035]  # Nhe, Trung binh, Manh
        volatility = vol_levels[vol_class]

        # Tạo chuỗi giá với mức biến động tương ứng
        prices = [base]
        volumes = []

        for d in range(window):
            ret = np.random.normal(0, volatility)
            prices.append(prices[-1] * (1 + ret))

            # Volume thường tăng khi biến động mạnh
            base_vol = 1000000
            vol_multiplier = 1 + abs(ret) * 20 + np.random.uniform(0, 0.5)
            volumes.append(base_vol * vol_multiplier)

        prices = prices[1:]
        volumes = np.array(volumes)

        # Tạo OHLCV (5 kênh)
        ohlcv = np.zeros((window, 5))
        for d in range(window):
            c = prices[d]
            r = c * volatility * np.random.uniform(0.5, 1.5)
            ohlcv[d] = [
                c + np.random.uniform(-r, r),
                c + abs(np.random.normal(0, r)),
                c - abs(np.random.normal(0, r)),
                c,
                volumes[d]
            ]

        # Chuẩn hóa: giá theo % thay đổi, volume theo z-score
        price_base = ohlcv[0, 3]
        ohlcv[:, :4] = (ohlcv[:, :4] - price_base) / price_base
        vol_mean = np.mean(ohlcv[:, 4])
        vol_std = np.std(ohlcv[:, 4])
        if vol_std > 0:
            ohlcv[:, 4] = (ohlcv[:, 4] - vol_mean) / vol_std

        X_list.append(ohlcv)
        y_list.append(vol_class)

    X = np.array(X_list)
    y = np.array(y_list)
    y_oh = one_hot(y, 3)

    print(f"Dataset: {X.shape[0]} mau, {X.shape[1]} ngay x {X.shape[2]} kenh")
    print(f"Phan bo: Nhe={np.sum(y == 0)}, TB={np.sum(y == 1)}, Manh={np.sum(y == 2)}")

    # Chia train/test
    X_train, X_test, y_train, y_test = train_test_split(X, y_oh)

    # Tạo CNN
    np.random.seed(42)
    model = StockCNN(
        seq_length=20,
        n_channels=5,        # 5 kênh: OHLCV
        n_filters=10,
        kernel_size=5,
        pool_size=2,
        dense_size=16,
        n_output=3,
        task='multiclass',
        learning_rate=0.005
    )

    print("\nKien truc: Input(20x5) → Conv1D(10, k=5) → ReLU → MaxPool(2) → Dense(16) → Softmax(3)")
    print("Huan luyen...\n")

    model.train(X_train, y_train, epochs=80, batch_size=32)

    # Đánh giá
    train_acc = model.accuracy(X_train, y_train)
    test_acc = model.accuracy(X_test, y_test)
    print(f"\nKet qua: Train = {train_acc:.1f}% | Test = {test_acc:.1f}%")

    # Ứng dụng quản lý rủi ro
    ten_vol = ["Nhe", "Trung binh", "Manh"]
    preds = model.predict(X_test[:8])
    y_test_labels = np.argmax(y_test[:8], axis=1)

    print("\n--- Ung dung quan ly rui ro ---")
    for i in range(8):
        pred = ten_vol[preds[i]]
        actual = ten_vol[y_test_labels[i]]
        dung = "V" if preds[i] == y_test_labels[i] else "X"

        # Khuyến nghị dựa trên mức biến động
        if preds[i] == 0:
            advice = "Giu vi the, khong can dieu chinh"
        elif preds[i] == 1:
            advice = "Can nhac dat stop-loss chat hon"
        else:
            advice = "CANH BAO! Giam vi the, tang hedge"

        print(f"  Ngay {i + 1}: Du doan={pred:12s} | Thuc te={actual:12s} [{dung}]")
        print(f"         → {advice}")


# =============================================================================
# VÍ DỤ 4: TRỰC QUAN HÓA FILTER ĐÃ HỌC
# =============================================================================

def vi_du_truc_quan_filter():
    print("\n" + "=" * 60)
    print("VI DU 4: TRUC QUAN HOA FILTER DA HOC")
    print("=" * 60)
    print("Xem CNN da hoc duoc nhung MAU GIA nao\n")

    # Tạo CNN đã train từ ví dụ 1
    X, y = tao_dataset_xu_huong(window_size=30, predict_days=5, n_samples=400)
    np.random.seed(42)
    model = StockCNN(
        seq_length=30, n_channels=4, n_filters=6, kernel_size=5,
        pool_size=2, dense_size=16, n_output=1, task='binary',
        learning_rate=0.005
    )
    model.train(X, y, epochs=30, batch_size=32, verbose=False)

    # Lấy filter weights từ Conv layer
    conv_layer = model.layers[0]  # Conv1D
    filters = conv_layer.filters   # Shape: (n_filters, kernel_size, n_channels)

    print(f"So filter: {filters.shape[0]}, Kich thuoc: {filters.shape[1]} ngay, Kenh: {filters.shape[2]}")
    print("\nMoi filter la 1 'mat kinh' nhin du lieu theo cach khac nhau:")
    print("(Hien thi weights cua kenh Close - kenh thu 4)\n")

    for f_idx in range(filters.shape[0]):
        # Lấy weights cho kênh Close (index 3)
        w = filters[f_idx, :, 3]

        # Vẽ filter dạng text
        w_norm = w / (np.max(np.abs(w)) + 1e-8)  # Chuẩn hóa về [-1, 1]

        # Xác định loại mẫu mà filter phát hiện
        if np.corrcoef(w, np.arange(len(w)))[0, 1] > 0.5:
            pattern = "XU HUONG TANG (gia tang dan)"
        elif np.corrcoef(w, np.arange(len(w)))[0, 1] < -0.5:
            pattern = "XU HUONG GIAM (gia giam dan)"
        elif w[0] > 0 and w[-1] > 0 and w[len(w) // 2] < 0:
            pattern = "HINH CHU V (giam roi tang)"
        elif w[0] < 0 and w[-1] < 0 and w[len(w) // 2] > 0:
            pattern = "HINH NUI (tang roi giam)"
        else:
            pattern = "MAU PHUC TAP"

        # Vẽ bar chart đơn giản
        bar = ""
        for val in w_norm:
            n_chars = int(abs(val) * 10)
            if val > 0:
                bar += " +" + "|" * n_chars
            else:
                bar += " -" + "|" * n_chars

        print(f"  Filter {f_idx + 1}: {pattern}")
        print(f"    Weights: [{', '.join(f'{v:+.3f}' for v in w)}]")
        print(f"    Visual: {bar}")
        print()


# =============================================================================
# CHẠY TẤT CẢ VÍ DỤ
# =============================================================================

if __name__ == "__main__":
    vi_du_du_doan_xu_huong()
    vi_du_mau_nen()
    vi_du_bien_dong()
    vi_du_truc_quan_filter()

    print("\n" + "=" * 60)
    print("TOM TAT CNN TRONG CHUNG KHOAN:")
    print("=" * 60)
    print("""
    1. TAI SAO CNN PHU HOP VOI CHUNG KHOAN:
       - Filter phat hien mau gia cuc bo (5-10 ngay)
       - Tu dong hoc features, khong can thiet ke thu cong
       - 1D CNN xu ly chuoi thoi gian hieu qua
       - Max Pooling giu lai tin hieu manh, loai nhieu

    2. KIEN TRUC THUONG DUNG:
       Input (OHLCV) → Conv1D → ReLU → MaxPool → Dense → Output
       - Nhieu Conv layers: phat hien mau phuc tap hon
       - Nhieu filters: nhin du lieu tu nhieu goc do
       - Kernel size nho (3-5): mau ngan han
       - Kernel size lon (10-20): mau dai han

    3. LUU Y THUC TE:
       - Luon chuan hoa du lieu (% thay doi, khong dung gia tuyet doi)
       - Them nhieu kenh: OHLCV + chi bao ky thuat (RSI, MACD, BB)
       - Kiem soat overfitting: dropout, regularization
       - Backtest ky truoc khi giao dich that
       - Thi truong thay doi → can retrain mo hinh dinh ky

    4. HAN CHE:
       - Du lieu qua khu khong dam bao tuong lai
       - Thieu tin hieu co ban (tin tuc, bao cao tai chinh)
       - Can ket hop voi quan ly rui ro nghiem ngat
    """)
