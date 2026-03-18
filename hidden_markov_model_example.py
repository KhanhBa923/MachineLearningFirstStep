"""
Hidden Markov Model (HMM) - Mô hình Markov ẩn

HMM là mô hình xác suất dùng để mô tả hệ thống có:
  - Trạng thái ẩn (hidden states): những gì ta KHÔNG quan sát trực tiếp được
  - Quan sát (observations): những gì ta THẤY được từ bên ngoài

Ý tưởng cốt lõi:
  Ta không biết trạng thái thật sự của hệ thống, nhưng dựa vào
  những gì quan sát được, ta có thể SUY LUẬN trạng thái ẩn.

Ví dụ thực tế trong file này:
  1. Dự đoán thời tiết dựa trên hoạt động của người (Viterbi Algorithm)
  2. Gán nhãn từ loại (POS Tagging) cho câu tiếng Việt
  3. Phát hiện trạng thái sức khỏe dựa trên triệu chứng

Các thành phần của HMM:
  - States (S): tập trạng thái ẩn
  - Observations (O): tập quan sát
  - Transition Probability (A): xác suất chuyển từ trạng thái này sang trạng thái khác
  - Emission Probability (B): xác suất phát ra quan sát từ mỗi trạng thái
  - Initial Probability (π): xác suất bắt đầu ở mỗi trạng thái
"""

import numpy as np


# =============================================================================
# PHẦN 1: XÂY DỰNG LỚP HMM TỪ ĐẦU
# =============================================================================

class HiddenMarkovModel:
    """
    Lớp Hidden Markov Model cài đặt từ đầu.

    Tham số:
        states: danh sách trạng thái ẩn
        observations: danh sách các quan sát có thể
        transition_prob: ma trận chuyển trạng thái A[i][j] = P(state_j | state_i)
        emission_prob: ma trận phát xạ B[i][j] = P(obs_j | state_i)
        initial_prob: xác suất ban đầu π[i] = P(state_i lúc t=0)
    """

    def __init__(self, states, observations, transition_prob, emission_prob, initial_prob):
        self.states = states
        self.observations = observations
        self.n_states = len(states)
        self.n_obs = len(observations)

        # Tạo mapping tên -> index để tra cứu nhanh
        self.state_idx = {s: i for i, s in enumerate(states)}
        self.obs_idx = {o: i for i, o in enumerate(observations)}

        # Ma trận xác suất (dạng numpy array)
        self.A = np.array(transition_prob)   # Ma trận chuyển trạng thái
        self.B = np.array(emission_prob)     # Ma trận phát xạ
        self.pi = np.array(initial_prob)     # Xác suất ban đầu

    def viterbi(self, obs_sequence):
        """
        Thuật toán Viterbi: tìm chuỗi trạng thái ẩn CÓ KHẢ NĂNG NHẤT
        cho một chuỗi quan sát đã biết.

        Ý tưởng: Quy hoạch động (Dynamic Programming)
          - Tại mỗi bước, tính xác suất cao nhất để đến mỗi trạng thái
          - Lưu lại đường đi tốt nhất (backtrack) để truy vết

        Args:
            obs_sequence: chuỗi quan sát (ví dụ: ["đi_dạo", "mua_sắm", "dọn_nhà"])

        Returns:
            best_path: chuỗi trạng thái ẩn tốt nhất
            best_prob: xác suất của chuỗi đó
        """
        T = len(obs_sequence)  # Chiều dài chuỗi quan sát

        # Chuyển chuỗi quan sát thành index
        obs_idx_seq = [self.obs_idx[o] for o in obs_sequence]

        # Bảng lưu xác suất cao nhất tại mỗi (thời điểm, trạng thái)
        # dp[t][s] = xác suất cao nhất để ở trạng thái s tại thời điểm t
        dp = np.zeros((T, self.n_states))

        # Bảng lưu trạng thái trước đó (để truy vết đường đi)
        backtrack = np.zeros((T, self.n_states), dtype=int)

        # --- Bước khởi tạo (t=0) ---
        # Xác suất = xác suất ban đầu × xác suất phát ra quan sát đầu tiên
        dp[0] = self.pi * self.B[:, obs_idx_seq[0]]

        # --- Bước đệ quy (t=1 đến T-1) ---
        for t in range(1, T):
            for s in range(self.n_states):
                # Với mỗi trạng thái s tại thời điểm t:
                # Thử tất cả trạng thái trước đó, chọn cái cho xác suất cao nhất
                # P = dp[t-1][prev] × A[prev→s] × B[s→obs_t]
                probs = dp[t - 1] * self.A[:, s] * self.B[s, obs_idx_seq[t]]

                # Lưu xác suất cao nhất và trạng thái trước đó tương ứng
                dp[t][s] = np.max(probs)
                backtrack[t][s] = np.argmax(probs)

        # --- Bước truy vết (backtracking) ---
        # Bắt đầu từ trạng thái có xác suất cao nhất ở bước cuối
        best_path_idx = np.zeros(T, dtype=int)
        best_path_idx[T - 1] = np.argmax(dp[T - 1])
        best_prob = np.max(dp[T - 1])

        # Đi ngược lại để tìm toàn bộ đường đi
        for t in range(T - 2, -1, -1):
            best_path_idx[t] = backtrack[t + 1][best_path_idx[t + 1]]

        # Chuyển index thành tên trạng thái
        best_path = [self.states[i] for i in best_path_idx]

        return best_path, best_prob

    def forward(self, obs_sequence):
        """
        Thuật toán Forward: tính xác suất của một chuỗi quan sát.

        Trả lời câu hỏi: "Chuỗi quan sát này có khả năng xảy ra bao nhiêu?"

        Ý tưởng:
          - Tại mỗi bước t, tính tổng xác suất đến được mỗi trạng thái
          - Khác Viterbi: Forward tính TỔNG tất cả đường đi, Viterbi chọn đường đi TỐT NHẤT
        """
        T = len(obs_sequence)
        obs_idx_seq = [self.obs_idx[o] for o in obs_sequence]

        # alpha[t][s] = tổng xác suất của tất cả đường đi đến trạng thái s tại thời điểm t
        alpha = np.zeros((T, self.n_states))

        # Bước khởi tạo
        alpha[0] = self.pi * self.B[:, obs_idx_seq[0]]

        # Bước đệ quy
        for t in range(1, T):
            for s in range(self.n_states):
                # Tổng xác suất từ TẤT CẢ trạng thái trước đó (khác Viterbi lấy max)
                alpha[t][s] = np.sum(alpha[t - 1] * self.A[:, s]) * self.B[s, obs_idx_seq[t]]

        # Tổng xác suất = tổng alpha ở bước cuối
        return np.sum(alpha[T - 1])

    def generate_sequence(self, length):
        """
        Sinh chuỗi ngẫu nhiên từ mô hình HMM.

        Mô phỏng quá trình:
          1. Chọn trạng thái ban đầu theo π
          2. Phát ra quan sát theo B
          3. Chuyển sang trạng thái mới theo A
          4. Lặp lại bước 2-3
        """
        states_seq = []
        obs_seq = []

        # Chọn trạng thái ban đầu
        current_state = np.random.choice(self.n_states, p=self.pi)

        for _ in range(length):
            states_seq.append(self.states[current_state])

            # Phát ra quan sát từ trạng thái hiện tại
            obs = np.random.choice(self.n_obs, p=self.B[current_state])
            obs_seq.append(self.observations[obs])

            # Chuyển sang trạng thái tiếp theo
            current_state = np.random.choice(self.n_states, p=self.A[current_state])

        return states_seq, obs_seq


# =============================================================================
# VÍ DỤ 1: DỰ ĐOÁN THỜI TIẾT TỪ HOẠT ĐỘNG CỦA NGƯỜI
# =============================================================================
# Bối cảnh: Bạn ở trong phòng không có cửa sổ, không biết thời tiết bên ngoài.
# Nhưng bạn quan sát được đồng nghiệp làm gì: đi dạo, mua sắm, hay dọn nhà.
# Từ đó suy luận thời tiết bên ngoài là nắng hay mưa.

def vi_du_thoi_tiet():
    print("=" * 60)
    print("VÍ DỤ 1: DỰ ĐOÁN THỜI TIẾT TỪ HOẠT ĐỘNG")
    print("=" * 60)

    # Trạng thái ẩn: thời tiết (ta không quan sát trực tiếp)
    states = ["Nắng", "Mưa"]

    # Quan sát: hoạt động của đồng nghiệp (ta thấy được)
    observations = ["đi_dạo", "mua_sắm", "dọn_nhà"]

    # Ma trận chuyển trạng thái: P(thời tiết ngày mai | thời tiết hôm nay)
    # Nếu hôm nay Nắng → ngày mai: 70% Nắng, 30% Mưa
    # Nếu hôm nay Mưa  → ngày mai: 40% Nắng, 60% Mưa
    transition = [
        [0.7, 0.3],  # Từ Nắng → [Nắng, Mưa]
        [0.4, 0.6],  # Từ Mưa  → [Nắng, Mưa]
    ]

    # Ma trận phát xạ: P(hoạt động | thời tiết)
    # Nếu Nắng → 60% đi dạo, 30% mua sắm, 10% dọn nhà
    # Nếu Mưa  → 10% đi dạo, 40% mua sắm, 50% dọn nhà
    emission = [
        [0.6, 0.3, 0.1],  # Nắng → [đi_dạo, mua_sắm, dọn_nhà]
        [0.1, 0.4, 0.5],  # Mưa  → [đi_dạo, mua_sắm, dọn_nhà]
    ]

    # Xác suất ban đầu: 60% bắt đầu bằng Nắng, 40% bắt đầu bằng Mưa
    initial = [0.6, 0.4]

    # Tạo mô hình HMM
    model = HiddenMarkovModel(states, observations, transition, emission, initial)

    # --- Bài toán: Quan sát được chuỗi hoạt động, suy ra thời tiết ---
    chuoi_quan_sat = ["đi_dạo", "mua_sắm", "dọn_nhà", "dọn_nhà", "đi_dạo"]

    print(f"\nQuan sát hoạt động: {chuoi_quan_sat}")

    # Dùng Viterbi để tìm chuỗi thời tiết có khả năng nhất
    thoi_tiet, xac_suat = model.viterbi(chuoi_quan_sat)

    print(f"Thời tiết dự đoán:  {thoi_tiet}")
    print(f"Xác suất:           {xac_suat:.6f}")

    # Tính xác suất quan sát bằng Forward
    prob = model.forward(chuoi_quan_sat)
    print(f"P(chuỗi quan sát):  {prob:.6f}")

    # Giải thích kết quả
    print("\nGiải thích:")
    for i, (obs, state) in enumerate(zip(chuoi_quan_sat, thoi_tiet)):
        print(f"  Ngày {i + 1}: Thấy '{obs}' → Suy ra '{state}'")


# =============================================================================
# VÍ DỤ 2: GÁN NHÃN TỪ LOẠI (POS TAGGING) ĐƠN GIẢN
# =============================================================================
# Bối cảnh: Cho một câu, xác định mỗi từ là Danh từ, Động từ, hay Tính từ.
# Trạng thái ẩn = từ loại, Quan sát = từ trong câu.

def vi_du_pos_tagging():
    print("\n" + "=" * 60)
    print("VÍ DỤ 2: GÁN NHÃN TỪ LOẠI (POS TAGGING)")
    print("=" * 60)

    # Trạng thái ẩn: từ loại
    states = ["Danh_từ", "Động_từ", "Tính_từ"]

    # Từ vựng đơn giản
    observations = ["mèo", "chó", "ăn", "ngủ", "chạy", "đẹp", "nhanh", "lớn"]

    # Ma trận chuyển: P(từ loại tiếp theo | từ loại hiện tại)
    # Sau Danh từ → thường là Động từ (Mèo ĂN, Chó CHẠY)
    # Sau Động từ → có thể là Tính từ hoặc Danh từ
    # Sau Tính từ → thường là Danh từ
    transition = [
        [0.1, 0.7, 0.2],  # Danh_từ → [Danh_từ, Động_từ, Tính_từ]
        [0.4, 0.1, 0.5],  # Động_từ → [Danh_từ, Động_từ, Tính_từ]
        [0.6, 0.3, 0.1],  # Tính_từ → [Danh_từ, Động_từ, Tính_từ]
    ]

    # Ma trận phát xạ: P(từ | từ loại)
    # Danh từ phát ra "mèo", "chó" nhiều hơn
    # Động từ phát ra "ăn", "ngủ", "chạy" nhiều hơn
    # Tính từ phát ra "đẹp", "nhanh", "lớn" nhiều hơn
    emission = [
        [0.35, 0.35, 0.02, 0.02, 0.02, 0.08, 0.08, 0.08],  # Danh_từ
        [0.02, 0.02, 0.30, 0.30, 0.30, 0.02, 0.02, 0.02],  # Động_từ
        [0.02, 0.02, 0.02, 0.02, 0.02, 0.30, 0.30, 0.30],  # Tính_từ
    ]

    # Xác suất ban đầu: câu thường bắt đầu bằng Danh từ hoặc Tính từ
    initial = [0.5, 0.2, 0.3]

    model = HiddenMarkovModel(states, observations, transition, emission, initial)

    # --- Thử gán nhãn cho các câu ---
    cau_1 = ["mèo", "ăn", "nhanh"]           # Mèo ăn nhanh
    cau_2 = ["chó", "chạy", "nhanh"]          # Chó chạy nhanh
    cau_3 = ["đẹp", "mèo", "ngủ"]            # Đẹp mèo ngủ (câu lạ)

    for cau in [cau_1, cau_2, cau_3]:
        nhan, prob = model.viterbi(cau)
        print(f"\nCâu: {' '.join(cau)}")
        for tu, loai in zip(cau, nhan):
            print(f"  '{tu}' → {loai}")
        print(f"  Xác suất: {prob:.6f}")


# =============================================================================
# VÍ DỤ 3: PHÁT HIỆN TRẠNG THÁI SỨC KHỎE TỪ TRIỆU CHỨNG
# =============================================================================
# Bối cảnh: Bệnh nhân có trạng thái sức khỏe ẩn (Khỏe, Cảm cúm, Sốt nặng).
# Bác sĩ quan sát triệu chứng hàng ngày để suy luận trạng thái.

def vi_du_suc_khoe():
    print("\n" + "=" * 60)
    print("VÍ DỤ 3: THEO DÕI SỨC KHỎE TỪ TRIỆU CHỨNG")
    print("=" * 60)

    # Trạng thái ẩn: tình trạng sức khỏe thực sự
    states = ["Khỏe", "Cảm_cúm", "Sốt_nặng"]

    # Quan sát: triệu chứng bệnh nhân báo cáo
    observations = ["bình_thường", "mệt_mỏi", "chóng_mặt", "ho", "đau_đầu"]

    # Ma trận chuyển trạng thái sức khỏe
    # Khỏe → phần lớn vẫn khỏe, có thể chuyển sang cảm cúm
    # Cảm cúm → có thể hồi phục hoặc nặng hơn
    # Sốt nặng → khó tự hồi phục ngay, cần thời gian
    transition = [
        [0.7, 0.25, 0.05],  # Khỏe     → [Khỏe, Cảm_cúm, Sốt_nặng]
        [0.3, 0.4, 0.3],    # Cảm_cúm  → [Khỏe, Cảm_cúm, Sốt_nặng]
        [0.1, 0.3, 0.6],    # Sốt_nặng → [Khỏe, Cảm_cúm, Sốt_nặng]
    ]

    # Ma trận phát xạ: P(triệu chứng | trạng thái sức khỏe)
    emission = [
        [0.6, 0.2, 0.05, 0.1, 0.05],   # Khỏe     → chủ yếu bình thường
        [0.1, 0.3, 0.15, 0.25, 0.2],    # Cảm_cúm  → hay mệt mỏi, ho
        [0.02, 0.18, 0.3, 0.2, 0.3],    # Sốt_nặng → chóng mặt, đau đầu nhiều
    ]

    # Xác suất ban đầu: đa số bắt đầu khỏe mạnh
    initial = [0.7, 0.2, 0.1]

    model = HiddenMarkovModel(states, observations, transition, emission, initial)

    # --- Theo dõi triệu chứng trong 7 ngày ---
    trieu_chung_7_ngay = [
        "bình_thường", "mệt_mỏi", "ho", "chóng_mặt",
        "đau_đầu", "mệt_mỏi", "bình_thường"
    ]

    print(f"\nTriệu chứng 7 ngày: {trieu_chung_7_ngay}")

    trang_thai, prob = model.viterbi(trieu_chung_7_ngay)

    print(f"\nKết quả chẩn đoán:")
    for ngay, (tc, tt) in enumerate(zip(trieu_chung_7_ngay, trang_thai), 1):
        # Đánh dấu cảnh báo nếu trạng thái nguy hiểm
        canh_bao = " ⚠️" if tt == "Sốt_nặng" else ""
        print(f"  Ngày {ngay}: triệu chứng '{tc}' → trạng thái '{tt}'{canh_bao}")

    print(f"\nXác suất chuỗi trạng thái: {prob:.8f}")

    # --- Sinh chuỗi mô phỏng bệnh nhân ---
    print("\n--- Mô phỏng bệnh nhân (sinh ngẫu nhiên) ---")
    np.random.seed(42)
    states_gen, obs_gen = model.generate_sequence(10)
    print(f"Trạng thái thật:  {states_gen}")
    print(f"Triệu chứng:      {obs_gen}")

    # Thử suy luận ngược từ triệu chứng
    inferred, _ = model.viterbi(obs_gen)
    print(f"Suy luận từ HMM:  {inferred}")

    # So sánh độ chính xác
    correct = sum(1 for a, b in zip(states_gen, inferred) if a == b)
    print(f"Độ chính xác:     {correct}/{len(states_gen)} ({correct / len(states_gen) * 100:.0f}%)")


# =============================================================================
# VÍ DỤ 4: SO SÁNH CÁC MÔ HÌNH HMM (MODEL SELECTION)
# =============================================================================
# Bối cảnh: Cho một chuỗi quan sát, mô hình nào giải thích tốt hơn?
# Ứng dụng: nhận dạng giọng nói (mỗi từ 1 HMM, chọn HMM cho xác suất cao nhất)

def vi_du_so_sanh_mo_hinh():
    print("\n" + "=" * 60)
    print("VÍ DỤ 4: SO SÁNH MÔ HÌNH - AI LÀ THỦ PHẠM?")
    print("=" * 60)
    print("Bối cảnh: Camera ghi lại hành vi trong cửa hàng.")
    print("Hai mô hình: 'Khách bình thường' vs 'Kẻ trộm'")

    observations = ["nhìn_quanh", "cầm_đồ", "bỏ_lại", "đi_ra", "giấu_đồ"]

    # Mô hình 1: Khách hàng bình thường
    # Trạng thái ẩn: Xem hàng, Cân nhắc
    model_khach = HiddenMarkovModel(
        states=["Xem_hàng", "Cân_nhắc"],
        observations=observations,
        transition_prob=[
            [0.6, 0.4],  # Xem_hàng → [Xem_hàng, Cân_nhắc]
            [0.5, 0.5],  # Cân_nhắc → [Xem_hàng, Cân_nhắc]
        ],
        emission_prob=[
            [0.3, 0.4, 0.1, 0.15, 0.05],  # Xem_hàng: hay cầm đồ, nhìn quanh
            [0.2, 0.2, 0.3, 0.25, 0.05],  # Cân_nhắc: hay bỏ lại, đi ra
        ],
        initial_prob=[0.7, 0.3],
    )

    # Mô hình 2: Kẻ trộm
    # Trạng thái ẩn: Thăm dò, Hành động
    model_trom = HiddenMarkovModel(
        states=["Thăm_dò", "Hành_động"],
        observations=observations,
        transition_prob=[
            [0.5, 0.5],  # Thăm_dò  → [Thăm_dò, Hành_động]
            [0.3, 0.7],  # Hành_động → [Thăm_dò, Hành_động]
        ],
        emission_prob=[
            [0.5, 0.2, 0.1, 0.15, 0.05],  # Thăm_dò: nhìn quanh nhiều
            [0.1, 0.2, 0.05, 0.15, 0.5],   # Hành_động: giấu đồ nhiều
        ],
        initial_prob=[0.8, 0.2],
    )

    # --- Thử các chuỗi hành vi khác nhau ---
    hanh_vi_1 = ["nhìn_quanh", "cầm_đồ", "bỏ_lại", "đi_ra"]       # Bình thường
    hanh_vi_2 = ["nhìn_quanh", "nhìn_quanh", "cầm_đồ", "giấu_đồ"]  # Đáng ngờ

    for hanh_vi in [hanh_vi_1, hanh_vi_2]:
        p_khach = model_khach.forward(hanh_vi)
        p_trom = model_trom.forward(hanh_vi)

        print(f"\nHành vi: {hanh_vi}")
        print(f"  P(khách bình thường) = {p_khach:.6f}")
        print(f"  P(kẻ trộm)          = {p_trom:.6f}")

        if p_khach > p_trom:
            print(f"  → Kết luận: Khách bình thường (tỷ lệ {p_khach / p_trom:.1f}x)")
        else:
            print(f"  → Kết luận: ĐÁNG NGỜ! (tỷ lệ {p_trom / p_khach:.1f}x)")


# =============================================================================
# CHẠY TẤT CẢ VÍ DỤ
# =============================================================================

if __name__ == "__main__":
    vi_du_thoi_tiet()
    vi_du_pos_tagging()
    vi_du_suc_khoe()
    vi_du_so_sanh_mo_hinh()

    print("\n" + "=" * 60)
    print("TÓM TẮT CÁC BÀI TOÁN CHÍNH CỦA HMM:")
    print("=" * 60)
    print("""
    1. Decoding (Viterbi): Cho chuỗi quan sát → tìm chuỗi trạng thái ẩn tốt nhất
       → Ứng dụng: nhận dạng giọng nói, POS tagging, chẩn đoán bệnh

    2. Evaluation (Forward): Cho chuỗi quan sát → tính xác suất xảy ra
       → Ứng dụng: so sánh mô hình, phân loại, phát hiện bất thường

    3. Learning (Baum-Welch): Cho dữ liệu → học tham số mô hình
       → Ứng dụng: huấn luyện HMM từ dữ liệu thực tế
       (Chưa cài đặt trong file này - cần thuật toán EM)
    """)
