"""
Recurrent Neural Network (RNN) - Mạng Nơ-ron Hồi quy

RNN được thiết kế để xử lý DỮ LIỆU TUẦN TỰ (sequential data):
  - Chuỗi thời gian (nhiệt độ, giá cổ phiếu)
  - Văn bản (từng từ trong câu)
  - Âm thanh (từng frame âm thanh)

Điểm khác biệt so với Neural Network thường:
  - NN thường: mỗi input ĐỘC LẬP, không nhớ gì từ input trước
  - RNN: có BỘ NHỚ (hidden state), truyền thông tin từ bước này sang bước sau

Minh họa:
  NN thường:   Input → [Layer] → Output    (mỗi input riêng lẻ)

  RNN:         x1 → [Cell] → h1
                       ↓
               x2 → [Cell] → h2     (h1 được truyền sang bước 2)
                       ↓
               x3 → [Cell] → h3 → Output   (h2 truyền sang bước 3)

  h (hidden state) = "bộ nhớ" lưu thông tin từ các bước trước

Các biến thể:
  - Vanilla RNN: đơn giản nhất, dễ quên thông tin xa (vanishing gradient)
  - LSTM: có 3 cổng (forget, input, output) → nhớ được thông tin DÀI HẠN
  - GRU: phiên bản đơn giản hơn LSTM, 2 cổng (reset, update)

Ví dụ trong file này:
  1. Dự đoán nhiệt độ ngày tiếp theo (time series)
  2. Sinh tên người Việt Nam (character-level language model)
  3. Phân loại cảm xúc câu review (sequence classification)
  4. So sánh Vanilla RNN vs LSTM vs GRU
"""

import numpy as np


# =============================================================================
# PHẦN 1: VANILLA RNN CELL
# =============================================================================

class RNNCell:
    """
    Vanilla RNN Cell - ô RNN cơ bản nhất.

    Công thức:
        h_t = tanh(W_xh @ x_t + W_hh @ h_{t-1} + b_h)

    Trong đó:
        x_t: input tại bước t
        h_{t-1}: hidden state từ bước trước (bộ nhớ)
        W_xh: trọng số input → hidden
        W_hh: trọng số hidden → hidden (kết nối hồi quy)
        tanh: hàm kích hoạt, nén giá trị về (-1, 1)

    Nhược điểm: khi chuỗi dài, gradient bị TRIỆT TIÊU (vanishing gradient)
    → RNN "quên" thông tin ở đầu chuỗi
    """

    def __init__(self, input_size, hidden_size):
        self.hidden_size = hidden_size
        scale_xh = np.sqrt(2.0 / input_size)
        scale_hh = np.sqrt(2.0 / hidden_size)

        self.W_xh = np.random.randn(input_size, hidden_size) * scale_xh   # Input → Hidden
        self.W_hh = np.random.randn(hidden_size, hidden_size) * scale_hh  # Hidden → Hidden
        self.b_h = np.zeros((1, hidden_size))

    def forward(self, x_t, h_prev):
        """
        Một bước forward của RNN.

        Args:
            x_t: input tại thời điểm t, shape (batch, input_size)
            h_prev: hidden state từ bước trước, shape (batch, hidden_size)

        Returns:
            h_next: hidden state mới, shape (batch, hidden_size)
        """
        # Tổ hợp tuyến tính: input mới + bộ nhớ cũ
        self.raw = x_t @ self.W_xh + h_prev @ self.W_hh + self.b_h

        # tanh nén về (-1, 1): giá trị dương = "kích hoạt", âm = "ức chế"
        h_next = np.tanh(self.raw)
        return h_next


# =============================================================================
# PHẦN 2: LSTM CELL (Long Short-Term Memory)
# =============================================================================

class LSTMCell:
    """
    LSTM Cell - giải quyết vấn đề vanishing gradient của Vanilla RNN.

    LSTM có thêm CELL STATE (c_t) - "băng chuyền thông tin" chạy song song
    với hidden state, cho phép thông tin đi XA mà không bị mất.

    3 cổng (gates) kiểm soát luồng thông tin:

    1. FORGET GATE (cổng quên):
       f_t = sigmoid(W_f @ [h_{t-1}, x_t] + b_f)
       → Quyết định BỎ thông tin nào từ cell state cũ
       → f_t ≈ 0: quên hết, f_t ≈ 1: nhớ hết
       Ví dụ: khi đọc "Trời hôm nay nắng", quên thông tin "hôm qua mưa"

    2. INPUT GATE (cổng nhập):
       i_t = sigmoid(W_i @ [h_{t-1}, x_t] + b_i)
       c_candidate = tanh(W_c @ [h_{t-1}, x_t] + b_c)
       → Quyết định THÊM thông tin mới nào vào cell state
       → i_t kiểm soát "bao nhiêu", c_candidate là "thông tin gì"
       Ví dụ: thêm thông tin "hôm nay nắng" vào bộ nhớ

    3. OUTPUT GATE (cổng xuất):
       o_t = sigmoid(W_o @ [h_{t-1}, x_t] + b_o)
       h_t = o_t * tanh(c_t)
       → Quyết định XUẤT thông tin nào từ cell state ra hidden state
       Ví dụ: khi cần dự đoán thời tiết, xuất thông tin thời tiết
    """

    def __init__(self, input_size, hidden_size):
        self.hidden_size = hidden_size
        combined = input_size + hidden_size
        scale = np.sqrt(2.0 / combined)

        # 4 bộ trọng số: forget, input, candidate, output
        # Gộp chung thành 1 ma trận lớn cho hiệu quả tính toán
        self.W = np.random.randn(combined, 4 * hidden_size) * scale
        self.b = np.zeros((1, 4 * hidden_size))

        # Khởi tạo bias forget gate = 1 (mặc định NHỚ, không quên)
        # Trick quan trọng: nếu forget gate = 0 từ đầu → gradient vanish ngay
        self.b[0, :hidden_size] = 1.0

    def forward(self, x_t, h_prev, c_prev):
        """
        Một bước forward của LSTM.

        Args:
            x_t: input tại thời điểm t
            h_prev: hidden state trước
            c_prev: cell state trước (bộ nhớ dài hạn)

        Returns:
            h_next: hidden state mới
            c_next: cell state mới
        """
        # Gộp input và hidden state trước
        combined = np.concatenate([x_t, h_prev], axis=1)

        # Tính tất cả 4 gates cùng lúc (hiệu quả hơn tính riêng)
        gates = combined @ self.W + self.b
        H = self.hidden_size

        # Tách ra 4 phần
        f_gate = self._sigmoid(gates[:, 0:H])        # Forget gate
        i_gate = self._sigmoid(gates[:, H:2*H])      # Input gate
        c_candidate = np.tanh(gates[:, 2*H:3*H])     # Candidate
        o_gate = self._sigmoid(gates[:, 3*H:4*H])    # Output gate

        # Cập nhật cell state:
        # c_new = (quên bớt cái cũ) + (thêm cái mới)
        c_next = f_gate * c_prev + i_gate * c_candidate

        # Tính hidden state từ cell state (lọc qua output gate)
        h_next = o_gate * np.tanh(c_next)

        # Lưu lại để backward
        self.cache = (combined, f_gate, i_gate, c_candidate, o_gate, c_prev, c_next)

        return h_next, c_next

    @staticmethod
    def _sigmoid(x):
        x = np.clip(x, -500, 500)
        return 1 / (1 + np.exp(-x))


# =============================================================================
# PHẦN 3: GRU CELL (Gated Recurrent Unit)
# =============================================================================

class GRUCell:
    """
    GRU Cell - phiên bản đơn giản hóa của LSTM.

    GRU gộp forget gate và input gate thành UPDATE GATE (z_t):
      - z_t ≈ 1: giữ hidden state cũ (nhớ)
      - z_t ≈ 0: dùng hidden state mới (cập nhật)

    RESET GATE (r_t) quyết định bao nhiêu hidden state cũ được dùng
    để tính hidden state mới.

    Công thức:
      r_t = sigmoid(W_r @ [h_{t-1}, x_t])           # Reset gate
      z_t = sigmoid(W_z @ [h_{t-1}, x_t])           # Update gate
      h_candidate = tanh(W_h @ [r_t * h_{t-1}, x_t]) # Candidate
      h_t = (1 - z_t) * h_{t-1} + z_t * h_candidate # Trộn cũ + mới

    So với LSTM:
      - Ít tham số hơn (2 gates vs 3 gates, không có cell state riêng)
      - Nhanh hơn, ít bộ nhớ hơn
      - Hiệu quả tương đương LSTM trên nhiều bài toán
    """

    def __init__(self, input_size, hidden_size):
        self.hidden_size = hidden_size
        combined = input_size + hidden_size
        scale = np.sqrt(2.0 / combined)

        # Reset gate weights
        self.W_r = np.random.randn(combined, hidden_size) * scale
        self.b_r = np.zeros((1, hidden_size))

        # Update gate weights
        self.W_z = np.random.randn(combined, hidden_size) * scale
        self.b_z = np.zeros((1, hidden_size))

        # Candidate hidden state weights
        self.W_h = np.random.randn(combined, hidden_size) * scale
        self.b_h = np.zeros((1, hidden_size))

    def forward(self, x_t, h_prev):
        """
        Một bước forward của GRU.

        Args:
            x_t: input tại thời điểm t
            h_prev: hidden state trước

        Returns:
            h_next: hidden state mới
        """
        combined = np.concatenate([x_t, h_prev], axis=1)

        # Reset gate: "quên bao nhiêu hidden state cũ khi tính candidate"
        r = self._sigmoid(combined @ self.W_r + self.b_r)

        # Update gate: "trộn bao nhiêu cũ vs mới"
        z = self._sigmoid(combined @ self.W_z + self.b_z)

        # Candidate: tính hidden state mới (có thể reset bớt h cũ)
        combined_reset = np.concatenate([x_t, r * h_prev], axis=1)
        h_candidate = np.tanh(combined_reset @ self.W_h + self.b_h)

        # Trộn: giữ (1-z) phần cũ + z phần mới
        h_next = (1 - z) * h_prev + z * h_candidate

        return h_next

    @staticmethod
    def _sigmoid(x):
        x = np.clip(x, -500, 500)
        return 1 / (1 + np.exp(-x))


# =============================================================================
# PHẦN 4: MÔ HÌNH RNN HOÀN CHỈNH
# =============================================================================

class RNNModel:
    """
    Mô hình RNN hoàn chỉnh: RNN/LSTM/GRU + Dense output layer.

    Hỗ trợ 3 loại cell và 2 loại bài toán:
      - sequence: dự đoán giá trị tiếp theo trong chuỗi (dùng hidden state cuối)
      - classification: phân loại toàn bộ chuỗi (dùng hidden state cuối)
    """

    def __init__(self, input_size, hidden_size, output_size,
                 cell_type='lstm', learning_rate=0.01):
        self.hidden_size = hidden_size
        self.lr = learning_rate
        self.cell_type = cell_type

        # Tạo RNN cell
        if cell_type == 'rnn':
            self.cell = RNNCell(input_size, hidden_size)
        elif cell_type == 'lstm':
            self.cell = LSTMCell(input_size, hidden_size)
        else:
            self.cell = GRUCell(input_size, hidden_size)

        # Dense layer: hidden state cuối → output
        scale = np.sqrt(2.0 / hidden_size)
        self.W_out = np.random.randn(hidden_size, output_size) * scale
        self.b_out = np.zeros((1, output_size))

        self.loss_history = []

    def forward(self, X):
        """
        Forward pass qua toàn bộ chuỗi.

        Args:
            X: shape (batch_size, seq_length, input_size)

        Returns:
            output: shape (batch_size, output_size)
        """
        batch_size, seq_len, _ = X.shape

        # Khởi tạo hidden state = 0
        h = np.zeros((batch_size, self.hidden_size))
        if self.cell_type == 'lstm':
            c = np.zeros((batch_size, self.hidden_size))

        # Lưu tất cả hidden states (để backward)
        self.h_states = [h.copy()]
        self.inputs = []

        # Duyệt qua từng bước thời gian
        for t in range(seq_len):
            x_t = X[:, t, :]  # Input tại bước t
            self.inputs.append(x_t)

            if self.cell_type == 'lstm':
                h, c = self.cell.forward(x_t, h, c)
            elif self.cell_type == 'gru':
                h = self.cell.forward(x_t, h)
            else:
                h = self.cell.forward(x_t, h)

            self.h_states.append(h.copy())

        # Dùng hidden state CUỐI CÙNG để dự đoán
        # (chứa thông tin tổng hợp từ toàn bộ chuỗi)
        self.final_h = h
        output = h @ self.W_out + self.b_out

        return output

    def train(self, X, y, epochs=100, batch_size=32, task='regression', verbose=True):
        """
        Huấn luyện RNN.

        Args:
            task: 'regression' (dự đoán số) hoặc 'classification' (phân loại)
        """
        n_samples = X.shape[0]

        for epoch in range(epochs):
            idx = np.random.permutation(n_samples)
            X_shuffled = X[idx]
            y_shuffled = y[idx]

            epoch_loss = 0

            for start in range(0, n_samples, batch_size):
                end = min(start + batch_size, n_samples)
                X_batch = X_shuffled[start:end]
                y_batch = y_shuffled[start:end]

                # Forward
                output = self.forward(X_batch)

                if task == 'classification':
                    # Softmax + Cross-entropy
                    exp_out = np.exp(output - np.max(output, axis=1, keepdims=True))
                    probs = exp_out / np.sum(exp_out, axis=1, keepdims=True)
                    probs_clip = np.clip(probs, 1e-8, 1 - 1e-8)
                    loss = -np.mean(np.sum(y_batch * np.log(probs_clip), axis=1))
                    d_output = probs - y_batch
                else:
                    # MSE Loss
                    loss = np.mean((output - y_batch) ** 2)
                    d_output = 2 * (output - y_batch) / y_batch.shape[0]

                epoch_loss += loss * (end - start)

                # Backward (simplified: chỉ cập nhật W_out và bias)
                # Full BPTT (Backpropagation Through Time) phức tạp hơn nhiều
                m = X_batch.shape[0]
                dW_out = self.final_h.T @ d_output / m
                db_out = np.mean(d_output, axis=0, keepdims=True)

                # Gradient cho hidden → truyền ngược qua cell
                d_h = d_output @ self.W_out.T

                # Cập nhật output layer
                self.W_out -= self.lr * dW_out
                self.b_out -= self.lr * db_out

                # Cập nhật cell weights (simplified gradient)
                self._update_cell_weights(d_h, m)

            epoch_loss /= n_samples
            self.loss_history.append(epoch_loss)

            if verbose and (epoch + 1) % max(1, epochs // 5) == 0:
                print(f"  Epoch {epoch + 1:4d}/{epochs} | Loss: {epoch_loss:.4f}")

    def _update_cell_weights(self, d_h, m):
        """Cập nhật trọng số cell (simplified BPTT cho bước cuối)."""
        if self.cell_type == 'rnn':
            # d_raw = d_h * (1 - tanh^2)
            d_raw = d_h * (1 - self.final_h ** 2)
            last_input = self.inputs[-1]
            prev_h = self.h_states[-2]

            dW_xh = last_input.T @ d_raw / m
            dW_hh = prev_h.T @ d_raw / m
            db_h = np.mean(d_raw, axis=0, keepdims=True)

            self.cell.W_xh -= self.lr * dW_xh
            self.cell.W_hh -= self.lr * dW_hh
            self.cell.b_h -= self.lr * db_h

        elif self.cell_type == 'lstm':
            combined, f, i, c_cand, o, c_prev, c_next = self.cell.cache

            d_o = d_h * np.tanh(c_next)
            d_c = d_h * o * (1 - np.tanh(c_next) ** 2)
            d_f = d_c * c_prev
            d_i = d_c * c_cand
            d_c_cand = d_c * i

            H = self.hidden_size
            d_gates = np.zeros_like(self.cell.b)
            d_gates = np.tile(d_gates, (combined.shape[0], 1))
            d_gates[:, 0:H] = d_f * f * (1 - f)
            d_gates[:, H:2*H] = d_i * i * (1 - i)
            d_gates[:, 2*H:3*H] = d_c_cand * (1 - c_cand ** 2)
            d_gates[:, 3*H:4*H] = d_o * o * (1 - o)

            dW = combined.T @ d_gates / m
            db = np.mean(d_gates, axis=0, keepdims=True)

            self.cell.W -= self.lr * dW
            self.cell.b -= self.lr * db

        else:  # GRU
            # Simplified: chỉ cập nhật W_h
            last_input = self.inputs[-1]
            prev_h = self.h_states[-2]
            combined = np.concatenate([last_input, prev_h], axis=1)
            d_raw = d_h * (1 - self.final_h ** 2)

            dW_h = combined.T @ d_raw / m
            db_h = np.mean(d_raw, axis=0, keepdims=True)

            self.cell.W_h -= self.lr * dW_h
            self.cell.b_h -= self.lr * db_h

    def predict(self, X):
        return self.forward(X)

    def predict_class(self, X):
        output = self.forward(X)
        return np.argmax(output, axis=1)


# =============================================================================
# VÍ DỤ 1: DỰ ĐOÁN NHIỆT ĐỘ
# =============================================================================
# Bối cảnh: Cho nhiệt độ 7 ngày liên tiếp, dự đoán ngày thứ 8.
# RNN phù hợp vì nhiệt độ có TÍNH TUẦN TỰ (hôm nay ảnh hưởng ngày mai).

def vi_du_nhiet_do():
    print("=" * 65)
    print("VI DU 1: DU DOAN NHIET DO (Time Series)")
    print("=" * 65)
    print("Input: nhiet do 7 ngay lien tiep")
    print("Output: nhiet do ngay thu 8\n")

    np.random.seed(42)

    # Tạo dữ liệu nhiệt độ mô phỏng 365 ngày
    # Nhiệt độ theo mùa: sin wave + nhiễu
    n_days = 365
    days = np.arange(n_days)

    # Nhiệt độ cơ bản: dao động theo mùa (sin), trung bình 25°C
    temp_base = 25 + 8 * np.sin(2 * np.pi * days / 365)
    # Thêm nhiễu ngẫu nhiên (thời tiết không hoàn toàn theo mùa)
    noise = np.random.normal(0, 2, n_days)
    temperatures = temp_base + noise

    print(f"  Du lieu: {n_days} ngay nhiet do")
    print(f"  Min: {temperatures.min():.1f} C | Max: {temperatures.max():.1f} C | TB: {temperatures.mean():.1f} C")

    # Tạo dataset: cửa sổ 7 ngày → dự đoán ngày 8
    window = 7
    X_list = []
    y_list = []

    for i in range(len(temperatures) - window):
        X_list.append(temperatures[i:i + window])
        y_list.append(temperatures[i + window])

    X = np.array(X_list).reshape(-1, window, 1)  # (samples, seq_len, features=1)
    y = np.array(y_list).reshape(-1, 1)

    # Chuẩn hóa
    temp_mean = np.mean(temperatures)
    temp_std = np.std(temperatures)
    X_norm = (X - temp_mean) / temp_std
    y_norm = (y - temp_mean) / temp_std

    # Chia train/test (80/20 theo thứ tự thời gian, KHÔNG shuffle)
    # Quan trọng: với time series, KHÔNG được shuffle vì dữ liệu có thứ tự
    split = int(len(X_norm) * 0.8)
    X_train, X_test = X_norm[:split], X_norm[split:]
    y_train, y_test = y_norm[:split], y_norm[split:]

    print(f"  Train: {len(X_train)} mau | Test: {len(X_test)} mau")

    # Train LSTM
    print("\n--- Train LSTM ---")
    np.random.seed(42)
    model = RNNModel(
        input_size=1,
        hidden_size=16,
        output_size=1,
        cell_type='lstm',
        learning_rate=0.005
    )

    model.train(X_train, y_train, epochs=100, batch_size=32, task='regression')

    # Dự đoán và đánh giá
    y_pred_norm = model.predict(X_test)

    # Chuyển về nhiệt độ thật
    y_pred = y_pred_norm * temp_std + temp_mean
    y_actual = y_test * temp_std + temp_mean

    # MAE (Mean Absolute Error)
    mae = np.mean(np.abs(y_pred - y_actual))
    print(f"\n  MAE (sai so trung binh): {mae:.2f} C")

    # Hiển thị kết quả dự đoán
    print("\n--- Du doan 10 ngay cuoi ---")
    print(f"  {'Ngay':<6s} {'Thuc te':>10s} {'Du doan':>10s} {'Sai so':>10s}")
    print("  " + "-" * 40)
    for i in range(-10, 0):
        actual = y_actual[i, 0]
        pred = y_pred[i, 0]
        err = abs(actual - pred)
        print(f"  {len(y_actual) + i + 1:<6d} {actual:>9.1f}C {pred:>9.1f}C {err:>9.1f}C")

    # Vẽ biểu đồ text
    print("\n--- Bieu do nhiet do (20 ngay cuoi) ---")
    print(f"  {'':6s} 15C {'':5s} 20C {'':5s} 25C {'':5s} 30C {'':5s} 35C")
    for i in range(-20, 0):
        actual = y_actual[i, 0]
        pred = y_pred[i, 0]

        # Vẽ bar chart
        a_pos = int((actual - 15) * 2)
        p_pos = int((pred - 15) * 2)
        line = [' '] * 45
        if 0 <= a_pos < 45:
            line[a_pos] = '*'
        if 0 <= p_pos < 45:
            line[p_pos] = 'o' if line[p_pos] == ' ' else '@'
        print(f"  {len(y_actual) + i + 1:>4d}: |{''.join(line)}|")
    print(f"  (* = thuc te, o = du doan, @ = trung nhau)")


# =============================================================================
# VÍ DỤ 2: SINH TÊN NGƯỜI VIỆT NAM
# =============================================================================
# Bối cảnh: Cho RNN học từ danh sách tên → tự sinh ra tên mới.
# Character-level language model: dự đoán ký tự tiếp theo.

def vi_du_sinh_ten():
    print("\n" + "=" * 65)
    print("VI DU 2: SINH TEN NGUOI VIET NAM (Character-Level RNN)")
    print("=" * 65)
    print("RNN hoc quy luat ten Viet → tu sinh ten moi\n")

    # Danh sách tên phổ biến
    ten_list = [
        "an", "anh", "binh", "chi", "cuong", "dung", "dat", "duc",
        "giang", "ha", "hai", "hang", "hanh", "hieu", "hoa", "hoang",
        "hong", "hue", "hung", "huong", "khanh", "lam", "lan", "linh",
        "long", "mai", "minh", "my", "nam", "nga", "ngan", "ngoc",
        "nhi", "nhu", "phat", "phong", "phu", "phuong", "quang", "quy",
        "son", "tam", "than", "thanh", "thao", "thu", "thuy", "tien",
        "trang", "trung", "tu", "tuan", "van", "vy", "xuan", "yen",
    ]

    # Xây dựng bộ ký tự
    all_chars = sorted(set(''.join(ten_list)))
    all_chars = ['<PAD>', '<START>', '<END>'] + all_chars
    char_to_idx = {c: i for i, c in enumerate(all_chars)}
    idx_to_char = {i: c for c, i in char_to_idx.items()}
    vocab_size = len(all_chars)

    print(f"  So ten mau: {len(ten_list)}")
    print(f"  Bo ky tu ({vocab_size}): {all_chars}")

    # Chuẩn bị dữ liệu: mỗi tên → chuỗi one-hot characters
    # Input: <START> + tên, Target: tên + <END>
    max_len = max(len(t) for t in ten_list) + 2  # +2 cho START và END

    X_list = []
    y_list = []

    for ten in ten_list:
        # Input sequence: <START> + ký tự tên
        input_seq = [char_to_idx['<START>']] + [char_to_idx[c] for c in ten]
        # Target sequence: ký tự tên + <END>
        target_seq = [char_to_idx[c] for c in ten] + [char_to_idx['<END>']]

        # Padding
        while len(input_seq) < max_len:
            input_seq.append(char_to_idx['<PAD>'])
            target_seq.append(char_to_idx['<PAD>'])

        X_list.append(input_seq)
        y_list.append(target_seq)

    # One-hot encode input
    X = np.zeros((len(X_list), max_len, vocab_size))
    for i, seq in enumerate(X_list):
        for t, idx in enumerate(seq):
            X[i, t, idx] = 1

    # Train RNN để dự đoán ký tự tiếp theo
    # Dùng hidden state cuối để predict (simplified)
    np.random.seed(42)
    hidden_size = 32

    # Tạo LSTM cell và output layer
    cell = LSTMCell(vocab_size, hidden_size)
    W_out = np.random.randn(hidden_size, vocab_size) * np.sqrt(2.0 / hidden_size)
    b_out = np.zeros((1, vocab_size))

    lr = 0.01

    print("\n--- Huan luyen ---")
    for epoch in range(150):
        total_loss = 0

        for i in range(len(X)):
            h = np.zeros((1, hidden_size))
            c = np.zeros((1, hidden_size))

            seq_loss = 0
            n_valid = 0

            for t in range(max_len - 1):
                x_t = X[i:i+1, t, :]
                target_idx = y_list[i][t]

                if target_idx == char_to_idx['<PAD>']:
                    continue

                h, c = cell.forward(x_t, h, c)

                # Predict next char
                logits = h @ W_out + b_out
                exp_logits = np.exp(logits - np.max(logits))
                probs = exp_logits / np.sum(exp_logits)

                # Cross-entropy loss
                seq_loss += -np.log(np.clip(probs[0, target_idx], 1e-8, 1))
                n_valid += 1

                # Backward (simplified: chỉ update output layer)
                d_probs = probs.copy()
                d_probs[0, target_idx] -= 1

                dW_out = h.T @ d_probs
                db_out_step = d_probs

                W_out -= lr * dW_out
                b_out -= lr * db_out_step

            if n_valid > 0:
                total_loss += seq_loss / n_valid

        avg_loss = total_loss / len(X)
        if (epoch + 1) % 30 == 0:
            print(f"  Epoch {epoch + 1:4d}/150 | Loss: {avg_loss:.4f}")

    # Sinh tên mới
    print("\n--- Sinh ten moi ---")

    def sinh_ten(temperature=0.8):
        """
        Sinh tên bằng cách dự đoán ký tự tiếp theo liên tục.

        temperature: kiểm soát "sáng tạo"
          - temperature thấp (0.3): chọn ký tự phổ biến nhất → tên an toàn
          - temperature cao (1.5): ngẫu nhiên hơn → tên sáng tạo (có thể lạ)
        """
        h = np.zeros((1, hidden_size))
        c = np.zeros((1, hidden_size))

        # Bắt đầu bằng <START>
        x_t = np.zeros((1, vocab_size))
        x_t[0, char_to_idx['<START>']] = 1

        ten = ""
        for _ in range(max_len):
            h, c = cell.forward(x_t, h, c)
            logits = h @ W_out + b_out

            # Áp dụng temperature
            logits = logits / temperature
            exp_logits = np.exp(logits - np.max(logits))
            probs = exp_logits / np.sum(exp_logits)

            # Chọn ký tự tiếp theo (sampling)
            next_idx = np.random.choice(vocab_size, p=probs.flatten())
            next_char = idx_to_char[next_idx]

            if next_char == '<END>' or next_char == '<PAD>':
                break

            ten += next_char

            # Chuẩn bị input cho bước tiếp theo
            x_t = np.zeros((1, vocab_size))
            x_t[0, next_idx] = 1

        return ten

    # Sinh với các mức temperature khác nhau
    for temp in [0.5, 0.8, 1.2]:
        print(f"\n  Temperature = {temp} ({'an toan' if temp < 0.7 else 'sang tao' if temp > 1 else 'can bang'}):")
        for _ in range(5):
            ten = sinh_ten(temperature=temp)
            co_trong_list = " (co san)" if ten in ten_list else " (MOI!)"
            print(f"    → {ten}{co_trong_list}")


# =============================================================================
# VÍ DỤ 3: PHÂN LOẠI CẢM XÚC REVIEW (Sequence Classification)
# =============================================================================
# Bối cảnh: Phân loại review sản phẩm thành Tích cực / Tiêu cực.
# RNN đọc từng từ một, tổng hợp ý nghĩa → phân loại.

def vi_du_phan_loai_cam_xuc():
    print("\n" + "=" * 65)
    print("VI DU 3: PHAN LOAI CAM XUC REVIEW")
    print("=" * 65)
    print("RNN doc tung tu → tong hop → phan loai Tich cuc / Tieu cuc\n")

    # Dataset reviews đơn giản
    reviews = [
        # Tích cực (1)
        ("tot dep nhanh", 1), ("rat tot hay lam", 1), ("hang dep chat luong", 1),
        ("giao nhanh hang tot", 1), ("rat hai long", 1), ("tuyet voi xin", 1),
        ("dep lam thich qua", 1), ("chat luong tot gia re", 1),
        ("shop nhiet tinh hang dep", 1), ("san pham tot recommend", 1),
        ("hang chinh hang tot lam", 1), ("mua lan hai van tot", 1),
        ("rat ung hang dep nhanh", 1), ("tot lam se mua lai", 1),
        ("hang xin gia hop ly", 1), ("dep tot nhanh chong", 1),

        # Tiêu cực (0)
        ("hang loi te lam", 0), ("chat luong kem", 0), ("giao cham hang hu", 0),
        ("that vong san pham te", 0), ("hang xau khong dung mo ta", 0),
        ("loi te that vong", 0), ("hang gia fake xau", 0),
        ("khong tot giao cham", 0), ("san pham kem chat luong", 0),
        ("te lam khong mua nua", 0), ("hang hu loi nhieu", 0),
        ("giao sai hang te qua", 0), ("chat luong te gia dat", 0),
        ("khong hai long that vong", 0), ("hang cu ban xau", 0),
        ("san pham te khong nen mua", 0),
    ]

    # Xây từ vựng
    all_words = set()
    for text, _ in reviews:
        all_words.update(text.split())
    all_words = ['<PAD>'] + sorted(all_words)
    word_to_idx = {w: i for i, w in enumerate(all_words)}
    vocab_size = len(all_words)

    print(f"  So review: {len(reviews)}")
    print(f"  Tu vung: {vocab_size} tu")

    # Chuyển review thành chuỗi one-hot
    max_len = max(len(text.split()) for text, _ in reviews)

    X_list = []
    y_list = []

    for text, label in reviews:
        words = text.split()
        # One-hot encode từng từ
        seq = np.zeros((max_len, vocab_size))
        for t, word in enumerate(words):
            seq[t, word_to_idx[word]] = 1
        # Pad phần còn lại
        for t in range(len(words), max_len):
            seq[t, word_to_idx['<PAD>']] = 1

        X_list.append(seq)
        y_list.append(label)

    X = np.array(X_list)
    y_labels = np.array(y_list)

    # One-hot encode labels
    y_onehot = np.zeros((len(y_labels), 2))
    for i, l in enumerate(y_labels):
        y_onehot[i, l] = 1

    # Chia train/test
    np.random.seed(42)
    idx = np.random.permutation(len(X))
    split = int(len(X) * 0.75)
    X_train, X_test = X[idx[:split]], X[idx[split:]]
    y_train, y_test = y_onehot[idx[:split]], y_onehot[idx[split:]]
    y_test_labels = y_labels[idx[split:]]
    test_texts = [reviews[i][0] for i in idx[split:]]

    print(f"  Train: {len(X_train)} | Test: {len(X_test)}")

    # Train LSTM cho classification
    print("\n--- Train LSTM ---")
    np.random.seed(42)
    model = RNNModel(
        input_size=vocab_size,
        hidden_size=16,
        output_size=2,
        cell_type='lstm',
        learning_rate=0.01
    )

    model.train(X_train, y_train, epochs=100, batch_size=8, task='classification')

    # Đánh giá
    preds = model.predict_class(X_test)
    acc = np.mean(preds == y_test_labels) * 100
    print(f"\n  Test Accuracy: {acc:.1f}%")

    # Hiển thị kết quả
    label_names = ["TIEU CUC", "TICH CUC"]
    print(f"\n--- Ket qua du doan ---")
    for i in range(len(test_texts)):
        actual = label_names[y_test_labels[i]]
        predicted = label_names[preds[i]]
        mark = "V" if preds[i] == y_test_labels[i] else "X"
        print(f"  [{mark}] \"{test_texts[i]:<30s}\" → {predicted:<10s} (that: {actual})")


# =============================================================================
# VÍ DỤ 4: SO SÁNH RNN vs LSTM vs GRU
# =============================================================================

def vi_du_so_sanh():
    print("\n" + "=" * 65)
    print("VI DU 4: SO SANH VANILLA RNN vs LSTM vs GRU")
    print("=" * 65)
    print("Bai toan: du doan gia tri tiep theo cua chuoi sin co nhieu\n")

    np.random.seed(42)

    # Tạo chuỗi sin + nhiễu (khó hơn ví dụ 1)
    n = 500
    t = np.linspace(0, 20 * np.pi, n)
    signal = np.sin(t) + 0.3 * np.sin(3 * t) + np.random.normal(0, 0.1, n)

    # Chuẩn hóa
    sig_mean = np.mean(signal)
    sig_std = np.std(signal)
    signal_norm = (signal - sig_mean) / sig_std

    # Tạo dataset
    window = 10
    X_list = []
    y_list = []
    for i in range(len(signal_norm) - window):
        X_list.append(signal_norm[i:i + window])
        y_list.append(signal_norm[i + window])

    X = np.array(X_list).reshape(-1, window, 1)
    y = np.array(y_list).reshape(-1, 1)

    split = int(len(X) * 0.8)
    X_train, X_test = X[:split], X[split:]
    y_train, y_test = y[:split], y[split:]

    print(f"  Du lieu: {n} diem, window = {window}")
    print(f"  Train: {len(X_train)} | Test: {len(X_test)}")

    # Train 3 loại cell
    results = {}

    for cell_type in ['rnn', 'lstm', 'gru']:
        print(f"\n--- {cell_type.upper()} ---")
        np.random.seed(42)
        model = RNNModel(
            input_size=1,
            hidden_size=16,
            output_size=1,
            cell_type=cell_type,
            learning_rate=0.005
        )

        model.train(X_train, y_train, epochs=100, batch_size=32, task='regression', verbose=False)

        y_pred = model.predict(X_test)
        mae = np.mean(np.abs(y_pred - y_test))
        mse = np.mean((y_pred - y_test) ** 2)
        final_loss = model.loss_history[-1]

        results[cell_type] = {
            'mae': mae, 'mse': mse, 'loss': final_loss,
            'n_params': _count_params(model, cell_type),
        }

        print(f"  MAE: {mae:.4f} | MSE: {mse:.4f} | Final Loss: {final_loss:.4f}")

    # Bảng so sánh
    print("\n" + "=" * 65)
    print("BANG SO SANH:")
    print("=" * 65)
    print(f"  {'Model':<8s} {'MAE':>8s} {'MSE':>8s} {'Params':>8s}  {'Ghi chu':<30s}")
    print("  " + "-" * 65)

    notes = {
        'rnn': 'Don gian, de vanishing gradient',
        'lstm': '3 gates, nho dai han tot',
        'gru': '2 gates, nhanh hon LSTM',
    }

    for cell_type in ['rnn', 'lstm', 'gru']:
        r = results[cell_type]
        print(f"  {cell_type.upper():<8s} {r['mae']:>8.4f} {r['mse']:>8.4f} {r['n_params']:>8d}  {notes[cell_type]}")

    print(f"""
  Nhan xet:
    - Vanilla RNN: it tham so nhat, nhung de mat thong tin xa
    - LSTM: nhieu tham so nhat (4x weights), nhung nho DAI HAN tot
    - GRU: trung gian, it tham so hon LSTM, hieu qua tuong duong

  Khi nao dung gi:
    - Chuoi ngan (< 20 buoc): Vanilla RNN du tot
    - Chuoi dai, can nho lau: LSTM
    - Can nhanh + hieu qua: GRU (thuong la lua chon mac dinh tot)
    """)


def _count_params(model, cell_type):
    """Đếm số tham số của model."""
    total = model.W_out.size + model.b_out.size

    if cell_type == 'rnn':
        total += model.cell.W_xh.size + model.cell.W_hh.size + model.cell.b_h.size
    elif cell_type == 'lstm':
        total += model.cell.W.size + model.cell.b.size
    else:
        total += (model.cell.W_r.size + model.cell.b_r.size +
                  model.cell.W_z.size + model.cell.b_z.size +
                  model.cell.W_h.size + model.cell.b_h.size)

    return total


# =============================================================================
# CHẠY TẤT CẢ
# =============================================================================

if __name__ == "__main__":
    vi_du_nhiet_do()
    vi_du_sinh_ten()
    vi_du_phan_loai_cam_xuc()
    vi_du_so_sanh()

    print("\n" + "=" * 65)
    print("TOM TAT RNN:")
    print("=" * 65)
    print("""
    1. RNN LA GI:
       - Mang no-ron co BO NHO (hidden state)
       - Xu ly du lieu TUAN TU: moi buoc nhan input moi + nho buoc truoc
       - Phu hop: chuoi thoi gian, van ban, am thanh

    2. 3 BIEN THE CHINH:
       - Vanilla RNN: don gian, de vanishing gradient
       - LSTM: 3 cong (forget/input/output) + cell state → nho dai han
       - GRU: 2 cong (reset/update), nhanh hon LSTM, hieu qua tuong duong

    3. BAI TOAN PHO BIEN:
       - Many-to-One: chuoi → 1 gia tri (phan loai cam xuc, du doan)
       - Many-to-Many: chuoi → chuoi (dich may, sinh van ban)
       - One-to-Many: 1 gia tri → chuoi (sinh nhac, mo ta anh)

    4. LUU Y THUC TE:
       - Voi chuoi dai (> 100 buoc): dung LSTM hoac GRU
       - Gradient clipping: cat gradient qua lon de tranh exploding
       - Bidirectional RNN: doc ca 2 chieu (trai→phai + phai→trai)
       - Attention mechanism: cho model TAP TRUNG vao phan quan trong
       - Transformer (BERT, GPT): thay the RNN trong nhieu bai toan NLP
    """)
