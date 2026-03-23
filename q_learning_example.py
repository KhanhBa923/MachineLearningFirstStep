"""
Q-Learning - Học tăng cường (Reinforcement Learning)

REINFORCEMENT LEARNING LÀ GÌ?
  Agent (tác nhân) tương tác với Environment (môi trường):
    - Agent thực hiện Action (hành động)
    - Environment trả về State mới + Reward (phần thưởng)
    - Mục tiêu: học Policy (chiến lược) để tối đa hóa tổng reward

  Khác biệt với Supervised Learning:
    - Không có nhãn đúng/sai cho sẵn
    - Agent tự khám phá bằng thử-sai (trial and error)
    - Reward có thể bị trễ (delayed reward)

Q-LEARNING:
  - Thuật toán RL kinh điển, off-policy, model-free
  - Học bảng Q(state, action) = giá trị kỳ vọng của reward tương lai
  - Công thức cập nhật:
    Q(s, a) ← Q(s, a) + α * [r + γ * max Q(s', a') - Q(s, a)]

    Trong đó:
      α (alpha): learning rate - tốc độ học
      γ (gamma): discount factor - mức quan trọng của reward tương lai
      r: reward nhận được
      s': state mới sau khi thực hiện action a

Ví dụ trong file này:
  1. Mê cung (Grid World) - tìm đường đi tối ưu
  2. Taxi đón khách - bài toán phức tạp hơn với pickup/dropoff
  3. Frozen Lake - đi trên mặt hồ đóng băng, tránh hố
  4. So sánh các chiến lược Exploration vs Exploitation
"""

import numpy as np


# =============================================================================
# PHẦN 1: Q-LEARNING AGENT
# =============================================================================

class QLearningAgent:
    """
    Agent học bằng Q-Learning.

    Q-Table: bảng tra cứu giá trị Q(state, action)
      - Mỗi ô = "hành động này ở trạng thái này tốt cỡ nào?"
      - Ban đầu = 0 (chưa biết gì)
      - Dần dần cập nhật qua trải nghiệm

    Epsilon-Greedy: chiến lược cân bằng khám phá vs khai thác
      - Xác suất ε: chọn hành động ngẫu nhiên (exploration)
      - Xác suất 1-ε: chọn hành động tốt nhất theo Q-table (exploitation)
    """

    def __init__(self, n_states, n_actions, learning_rate=0.1,
                 discount_factor=0.95, epsilon=1.0, epsilon_decay=0.995,
                 epsilon_min=0.01):
        self.n_states = n_states
        self.n_actions = n_actions
        self.lr = learning_rate
        self.gamma = discount_factor
        self.epsilon = epsilon
        self.epsilon_decay = epsilon_decay
        self.epsilon_min = epsilon_min

        # Khởi tạo Q-table = 0
        self.q_table = np.zeros((n_states, n_actions))

    def choose_action(self, state):
        """
        Epsilon-Greedy: chọn hành động.
          - Random < epsilon → khám phá (chọn ngẫu nhiên)
          - Ngược lại → khai thác (chọn action có Q cao nhất)
        """
        if np.random.random() < self.epsilon:
            return np.random.randint(self.n_actions)
        return int(np.argmax(self.q_table[state]))

    def update(self, state, action, reward, next_state, done):
        """
        Cập nhật Q-value theo công thức Bellman:
        Q(s,a) ← Q(s,a) + α * [r + γ * max Q(s',a') - Q(s,a)]

        - done=True: không có state tiếp theo → target = reward
        - done=False: target = reward + giá trị tương lai tốt nhất
        """
        if done:
            target = reward
        else:
            target = reward + self.gamma * np.max(self.q_table[next_state])

        # TD error: sai lệch giữa dự đoán và thực tế
        td_error = target - self.q_table[state, action]
        self.q_table[state, action] += self.lr * td_error

    def decay_epsilon(self):
        """Giảm epsilon theo thời gian: càng về sau càng ít khám phá."""
        self.epsilon = max(self.epsilon_min, self.epsilon * self.epsilon_decay)


# =============================================================================
# PHẦN 2: MÔI TRƯỜNG MÊ CUNG (GRID WORLD)
# =============================================================================

class GridWorld:
    """
    Mê cung dạng lưới đơn giản.

    Bản đồ (5x5):
      S . . # .       S = Start (xuất phát)
      . # . . .       G = Goal (đích)
      . . . # .       # = Wall (tường, không đi được)
      . # . . .       . = Đường đi được
      . . # . G

    Agent có 4 hành động: lên(0), xuống(1), trái(2), phải(3)

    Reward:
      - Đến đích: +100
      - Mỗi bước đi: -1 (khuyến khích đi ngắn nhất)
      - Đâm vào tường: -5
    """

    def __init__(self):
        self.rows = 5
        self.cols = 5
        self.n_states = self.rows * self.cols  # 25 states
        self.n_actions = 4  # lên, xuống, trái, phải

        self.start = (0, 0)
        self.goal = (4, 4)

        # Vị trí tường
        self.walls = {(0, 3), (1, 1), (2, 3), (3, 1), (4, 2)}

        self.state = self.start

    def reset(self):
        """Đặt lại về vị trí xuất phát."""
        self.state = self.start
        return self._pos_to_state(self.state)

    def step(self, action):
        """
        Thực hiện hành động, trả về (next_state, reward, done).

        Actions: 0=lên, 1=xuống, 2=trái, 3=phải
        """
        row, col = self.state

        # Tính vị trí mới theo hành động
        moves = {0: (-1, 0), 1: (1, 0), 2: (0, -1), 3: (0, 1)}
        dr, dc = moves[action]
        new_row, new_col = row + dr, col + dc

        # Kiểm tra hợp lệ
        if (0 <= new_row < self.rows and 0 <= new_col < self.cols
                and (new_row, new_col) not in self.walls):
            self.state = (new_row, new_col)
            reward = -1  # Phạt nhẹ mỗi bước để khuyến khích đi ngắn
        else:
            reward = -5  # Phạt nặng hơn khi đâm tường

        # Kiểm tra đến đích
        done = self.state == self.goal
        if done:
            reward = 100

        return self._pos_to_state(self.state), reward, done

    def _pos_to_state(self, pos):
        """Chuyển tọa độ (row, col) thành state index."""
        return pos[0] * self.cols + pos[1]

    def _state_to_pos(self, state):
        """Chuyển state index thành tọa độ (row, col)."""
        return state // self.cols, state % self.cols

    def render(self, agent_pos=None):
        """Vẽ mê cung ra console."""
        action_symbols = {0: "↑", 1: "↓", 2: "←", 3: "→"}
        grid = []
        for r in range(self.rows):
            row_str = []
            for c in range(self.cols):
                if agent_pos and (r, c) == agent_pos:
                    row_str.append("A")
                elif (r, c) == self.goal:
                    row_str.append("G")
                elif (r, c) == self.start:
                    row_str.append("S")
                elif (r, c) in self.walls:
                    row_str.append("#")
                else:
                    row_str.append(".")
            grid.append(" ".join(row_str))
        return "\n".join(grid)

    def render_policy(self, q_table):
        """Vẽ policy (chiến lược tối ưu) từ Q-table."""
        action_symbols = {0: "↑", 1: "↓", 2: "←", 3: "→"}
        grid = []
        for r in range(self.rows):
            row_str = []
            for c in range(self.cols):
                if (r, c) == self.goal:
                    row_str.append("G")
                elif (r, c) in self.walls:
                    row_str.append("#")
                else:
                    state = self._pos_to_state((r, c))
                    best_action = int(np.argmax(q_table[state]))
                    row_str.append(action_symbols[best_action])
            grid.append(" ".join(row_str))
        return "\n".join(grid)


# =============================================================================
# PHẦN 3: MÔI TRƯỜNG TAXI
# =============================================================================

class TaxiEnv:
    """
    Bài toán Taxi đơn giản hóa.

    Lưới 5x5 với 4 địa điểm đặc biệt:
      R(0,0)  G(0,4)
      Y(4,0)  B(4,3)

    Agent (taxi) cần:
      1. Di chuyển đến vị trí khách
      2. Đón khách (pickup)
      3. Di chuyển đến đích
      4. Trả khách (dropoff)

    6 hành động: lên(0), xuống(1), trái(2), phải(3), đón(4), trả(5)

    State = (taxi_row, taxi_col, passenger_loc, destination)
      - passenger_loc: 0-3 = ở địa điểm R/G/Y/B, 4 = trên xe
      - destination: 0-3 = đích là R/G/Y/B

    Reward:
      - Mỗi bước: -1
      - Đón/trả sai: -10
      - Trả đúng đích: +20
    """

    LOCATIONS = [(0, 0), (0, 4), (4, 0), (4, 3)]  # R, G, Y, B
    LOCATION_NAMES = ["R", "G", "Y", "B"]

    def __init__(self):
        self.rows = 5
        self.cols = 5
        # State: (taxi_row, taxi_col, passenger_loc, destination)
        # passenger_loc: 0-3 = ở location, 4 = trên xe
        # destination: 0-3
        self.n_states = self.rows * self.cols * 5 * 4  # 500 states
        self.n_actions = 6

        self.state = None

    def reset(self, seed=None):
        """Random vị trí taxi, khách, đích."""
        if seed is not None:
            np.random.seed(seed)

        self.taxi_row = np.random.randint(self.rows)
        self.taxi_col = np.random.randint(self.cols)
        self.pass_loc = np.random.randint(4)  # Khách ở 1 trong 4 điểm

        # Đích khác vị trí khách
        self.dest = np.random.randint(4)
        while self.dest == self.pass_loc:
            self.dest = np.random.randint(4)

        return self._encode_state()

    def _encode_state(self):
        """Mã hóa state thành 1 số nguyên."""
        return (self.taxi_row * self.cols * 5 * 4
                + self.taxi_col * 5 * 4
                + self.pass_loc * 4
                + self.dest)

    def step(self, action):
        """Thực hiện hành động."""
        reward = -1  # Phạt mỗi bước
        done = False

        if action < 4:
            # Di chuyển
            moves = {0: (-1, 0), 1: (1, 0), 2: (0, -1), 3: (0, 1)}
            dr, dc = moves[action]
            new_r = max(0, min(self.rows - 1, self.taxi_row + dr))
            new_c = max(0, min(self.cols - 1, self.taxi_col + dc))
            self.taxi_row, self.taxi_col = new_r, new_c

        elif action == 4:
            # Đón khách
            if (self.pass_loc < 4 and
                    (self.taxi_row, self.taxi_col) == self.LOCATIONS[self.pass_loc]):
                self.pass_loc = 4  # Khách lên xe
            else:
                reward = -10  # Đón sai chỗ

        elif action == 5:
            # Trả khách
            if (self.pass_loc == 4 and
                    (self.taxi_row, self.taxi_col) == self.LOCATIONS[self.dest]):
                done = True
                reward = 20  # Trả đúng đích!
            else:
                reward = -10  # Trả sai chỗ

        return self._encode_state(), reward, done

    def describe_state(self):
        """Mô tả state hiện tại."""
        pass_status = (f"tai {self.LOCATION_NAMES[self.pass_loc]}"
                       if self.pass_loc < 4 else "tren xe")
        return (f"Taxi({self.taxi_row},{self.taxi_col}) | "
                f"Khach: {pass_status} | "
                f"Dich: {self.LOCATION_NAMES[self.dest]}")


# =============================================================================
# PHẦN 4: MÔI TRƯỜNG FROZEN LAKE
# =============================================================================

class FrozenLake:
    """
    Frozen Lake - Đi trên hồ đóng băng.

    Bản đồ 4x4:
      S F F F       S = Start
      F H F H       F = Frozen (an toàn)
      F F F H       H = Hole (hố, rơi = thua)
      H F F G       G = Goal (đích)

    Đặc biệt: mặt băng TRƠN (slippery)!
      - Xác suất đi đúng hướng: 70%
      - Xác suất trượt sang 2 hướng vuông góc: 15% mỗi hướng
      → Agent phải học cách đi an toàn dù bị trượt

    Reward:
      - Đến đích: +1
      - Rơi hố: -1
      - Mỗi bước: -0.01 (khuyến khích đi nhanh)
    """

    def __init__(self, slippery=True):
        self.rows = 4
        self.cols = 4
        self.n_states = 16
        self.n_actions = 4  # lên, xuống, trái, phải
        self.slippery = slippery

        # Bản đồ: S=start, F=frozen, H=hole, G=goal
        self.grid = [
            ['S', 'F', 'F', 'F'],
            ['F', 'H', 'F', 'H'],
            ['F', 'F', 'F', 'H'],
            ['H', 'F', 'F', 'G'],
        ]

        self.holes = set()
        for r in range(4):
            for c in range(4):
                if self.grid[r][c] == 'H':
                    self.holes.add((r, c))

        self.start = (0, 0)
        self.goal = (3, 3)
        self.state = self.start

    def reset(self):
        self.state = self.start
        return self._pos_to_state(self.state)

    def step(self, action):
        """
        Thực hiện hành động. Nếu slippery=True, có thể bị trượt.

        Trượt: intended action 70%, 2 hướng vuông góc mỗi hướng 15%
        """
        if self.slippery:
            # Có thể bị trượt
            rand = np.random.random()
            if rand < 0.7:
                actual_action = action  # Đi đúng hướng
            elif rand < 0.85:
                actual_action = (action + 1) % 4  # Trượt hướng 1
            else:
                actual_action = (action - 1) % 4  # Trượt hướng 2
        else:
            actual_action = action

        row, col = self.state
        moves = {0: (-1, 0), 1: (1, 0), 2: (0, -1), 3: (0, 1)}
        dr, dc = moves[actual_action]
        new_row = max(0, min(self.rows - 1, row + dr))
        new_col = max(0, min(self.cols - 1, col + dc))

        self.state = (new_row, new_col)

        # Kiểm tra kết quả
        if self.state in self.holes:
            return self._pos_to_state(self.state), -1.0, True
        elif self.state == self.goal:
            return self._pos_to_state(self.state), 1.0, True
        else:
            return self._pos_to_state(self.state), -0.01, False

    def _pos_to_state(self, pos):
        return pos[0] * self.cols + pos[1]

    def _state_to_pos(self, state):
        return state // self.cols, state % self.cols

    def render_policy(self, q_table):
        """Vẽ policy tối ưu."""
        action_symbols = {0: "↑", 1: "↓", 2: "←", 3: "→"}
        grid = []
        for r in range(self.rows):
            row_str = []
            for c in range(self.cols):
                if (r, c) == self.goal:
                    row_str.append("G")
                elif (r, c) in self.holes:
                    row_str.append("H")
                else:
                    state = self._pos_to_state((r, c))
                    best = int(np.argmax(q_table[state]))
                    row_str.append(action_symbols[best])
            grid.append(" ".join(row_str))
        return "\n".join(grid)


# =============================================================================
# PHẦN 5: HÀM TRAIN VÀ ĐÁNH GIÁ
# =============================================================================

def train_agent(env, agent, n_episodes=1000, max_steps=200, verbose=True):
    """
    Train agent trên môi trường.

    Mỗi episode:
      1. Reset môi trường
      2. Agent chọn action → nhận reward → cập nhật Q
      3. Lặp lại đến khi done hoặc hết bước
      4. Giảm epsilon (ít khám phá hơn theo thời gian)

    Returns:
        rewards_history: tổng reward mỗi episode
    """
    rewards_history = []

    for episode in range(n_episodes):
        state = env.reset()
        total_reward = 0

        for step in range(max_steps):
            action = agent.choose_action(state)
            next_state, reward, done = env.step(action)

            agent.update(state, action, reward, next_state, done)

            state = next_state
            total_reward += reward

            if done:
                break

        agent.decay_epsilon()
        rewards_history.append(total_reward)

        if verbose and (episode + 1) % (n_episodes // 5) == 0:
            avg_reward = np.mean(rewards_history[-100:])
            print(f"  Episode {episode + 1:5d}/{n_episodes} | "
                  f"Avg Reward (100 ep): {avg_reward:7.2f} | "
                  f"Epsilon: {agent.epsilon:.4f}")

    return rewards_history


def evaluate_agent(env, agent, n_episodes=100, max_steps=200):
    """
    Đánh giá agent (không exploration, không cập nhật Q).

    Returns:
        success_rate, avg_reward, avg_steps
    """
    successes = 0
    total_reward = 0
    total_steps = 0

    old_epsilon = agent.epsilon
    agent.epsilon = 0  # Tắt exploration khi đánh giá

    for _ in range(n_episodes):
        state = env.reset()
        ep_reward = 0

        for step in range(max_steps):
            action = agent.choose_action(state)
            next_state, reward, done = env.step(action)

            state = next_state
            ep_reward += reward

            if done:
                if reward > 0:
                    successes += 1
                total_steps += step + 1
                break
        else:
            total_steps += max_steps

        total_reward += ep_reward

    agent.epsilon = old_epsilon

    success_rate = successes / n_episodes * 100
    avg_reward = total_reward / n_episodes
    avg_steps = total_steps / n_episodes

    return success_rate, avg_reward, avg_steps


# =============================================================================
# VÍ DỤ 1: MÊ CUNG - TÌM ĐƯỜNG ĐI TỐI ƯU
# =============================================================================

def vi_du_me_cung():
    print("=" * 60)
    print("VI DU 1: ME CUNG (GRID WORLD)")
    print("=" * 60)

    env = GridWorld()
    print("\nBan do me cung:")
    print(env.render())

    print("\nAgent hoc Q-table = bang gia tri (state, action)...")
    print("Bat dau: khong biet gi → thu-sai → hoc dan\n")

    # Train agent
    agent = QLearningAgent(
        n_states=env.n_states,
        n_actions=env.n_actions,
        learning_rate=0.1,
        discount_factor=0.95,
        epsilon=1.0,
        epsilon_decay=0.995,
        epsilon_min=0.01,
    )

    rewards = train_agent(env, agent, n_episodes=2000, max_steps=100)

    # Đánh giá
    success_rate, avg_reward, avg_steps = evaluate_agent(env, agent, n_episodes=200)
    print(f"\nKet qua danh gia (200 episodes):")
    print(f"  Ty le thanh cong: {success_rate:.1f}%")
    print(f"  Reward trung binh: {avg_reward:.2f}")
    print(f"  So buoc trung binh: {avg_steps:.1f}")

    # Hiển thị policy
    print(f"\nPolicy toi uu (huong di tai moi o):")
    print(env.render_policy(agent.q_table))

    # Demo 1 episode
    print(f"\nDemo 1 lan chay:")
    state = env.reset()
    agent.epsilon = 0
    path = [env.start]
    for _ in range(50):
        action = agent.choose_action(state)
        state, reward, done = env.step(action)
        path.append(env.state)
        if done:
            break

    action_names = {0: "len", 1: "xuong", 2: "trai", 3: "phai"}
    print(f"  Duong di: {' → '.join([f'({r},{c})' for r, c in path])}")
    print(f"  So buoc: {len(path) - 1}")


# =============================================================================
# VÍ DỤ 2: TAXI ĐÓN KHÁCH
# =============================================================================

def vi_du_taxi():
    print("\n" + "=" * 60)
    print("VI DU 2: TAXI DON KHACH")
    print("=" * 60)

    env = TaxiEnv()

    print("\nLuoi 5x5 voi 4 dia diem: R(0,0) G(0,4) Y(4,0) B(4,3)")
    print("Taxi can: di den khach → don → di den dich → tra")
    print("6 hanh dong: len, xuong, trai, phai, don, tra\n")

    agent = QLearningAgent(
        n_states=env.n_states,
        n_actions=env.n_actions,
        learning_rate=0.1,
        discount_factor=0.99,
        epsilon=1.0,
        epsilon_decay=0.9995,
        epsilon_min=0.01,
    )

    rewards = train_agent(env, agent, n_episodes=5000, max_steps=200)

    # Đánh giá
    success_rate, avg_reward, avg_steps = evaluate_agent(
        env, agent, n_episodes=200, max_steps=200
    )
    print(f"\nKet qua danh gia (200 episodes):")
    print(f"  Ty le thanh cong: {success_rate:.1f}%")
    print(f"  Reward trung binh: {avg_reward:.2f}")
    print(f"  So buoc trung binh: {avg_steps:.1f}")

    # Demo
    print(f"\nDemo 1 chuyen xe:")
    state = env.reset(seed=7)
    agent.epsilon = 0
    print(f"  Ban dau: {env.describe_state()}")

    action_names = {0: "len", 1: "xuong", 2: "trai", 3: "phai", 4: "don", 5: "tra"}
    for step in range(50):
        action = agent.choose_action(state)
        state, reward, done = env.step(action)
        if action >= 4 or done:
            print(f"  Buoc {step + 1}: {action_names[action]:5s} | "
                  f"reward={reward:+.0f} | {env.describe_state()}")
        if done:
            print(f"  → Hoan thanh trong {step + 1} buoc!")
            break


# =============================================================================
# VÍ DỤ 3: FROZEN LAKE
# =============================================================================

def vi_du_frozen_lake():
    print("\n" + "=" * 60)
    print("VI DU 3: FROZEN LAKE (MAT HO DONG BANG)")
    print("=" * 60)

    print("\nBan do:")
    print("  S F F F    S=Start, F=Frozen (an toan)")
    print("  F H F H    H=Hole (ho, roi = thua)")
    print("  F F F H    G=Goal (dich)")
    print("  H F F G")
    print("\nMat bang TRON: 70% di dung huong, 30% bi truot!")

    # So sánh: có trượt vs không trượt
    for slippery, label in [(False, "KHONG TRUOT"), (True, "CO TRUOT (thuc te)")]:
        print(f"\n--- {label} ---")

        env = FrozenLake(slippery=slippery)
        agent = QLearningAgent(
            n_states=env.n_states,
            n_actions=env.n_actions,
            learning_rate=0.1,
            discount_factor=0.95,
            epsilon=1.0,
            epsilon_decay=0.999,
            epsilon_min=0.01,
        )

        n_ep = 3000 if slippery else 2000
        rewards = train_agent(env, agent, n_episodes=n_ep, max_steps=100,
                              verbose=False)

        success_rate, avg_reward, avg_steps = evaluate_agent(
            env, agent, n_episodes=500, max_steps=100
        )
        print(f"  Ty le den dich: {success_rate:.1f}%")
        print(f"  Reward trung binh: {avg_reward:.3f}")

        print(f"  Policy:")
        for line in env.render_policy(agent.q_table).split("\n"):
            print(f"    {line}")


# =============================================================================
# VÍ DỤ 4: SO SÁNH EXPLORATION VS EXPLOITATION
# =============================================================================

def vi_du_exploration():
    print("\n" + "=" * 60)
    print("VI DU 4: SO SANH EXPLORATION VS EXPLOITATION")
    print("=" * 60)

    print("""
    Exploration (kham pha): thu hanh dong ngau nhien de tim phuong an moi
    Exploitation (khai thac): dung hanh dong tot nhat da biet

    Can doi: qua nhieu exploration → cham hoi tu
             qua it exploration → mac ket o phuong an tam
    """)

    env = GridWorld()

    configs = [
        {"name": "Epsilon cao (0.5 → 0.01)", "epsilon": 0.5,
         "epsilon_decay": 0.995, "epsilon_min": 0.01},
        {"name": "Epsilon vua (1.0 → 0.01)", "epsilon": 1.0,
         "epsilon_decay": 0.995, "epsilon_min": 0.01},
        {"name": "Epsilon thap (0.1 → 0.01)", "epsilon": 0.1,
         "epsilon_decay": 0.999, "epsilon_min": 0.01},
        {"name": "Greedy (khong kham pha)", "epsilon": 0.0,
         "epsilon_decay": 1.0, "epsilon_min": 0.0},
    ]

    results = []
    for cfg in configs:
        agent = QLearningAgent(
            n_states=env.n_states,
            n_actions=env.n_actions,
            learning_rate=0.1,
            discount_factor=0.95,
            epsilon=cfg["epsilon"],
            epsilon_decay=cfg["epsilon_decay"],
            epsilon_min=cfg["epsilon_min"],
        )

        rewards = train_agent(env, agent, n_episodes=2000, max_steps=100,
                              verbose=False)

        success_rate, avg_reward, avg_steps = evaluate_agent(
            env, agent, n_episodes=200
        )

        results.append({
            "name": cfg["name"],
            "success_rate": success_rate,
            "avg_reward": avg_reward,
            "avg_steps": avg_steps,
        })

    # In bảng so sánh
    print(f"{'Chien luoc':<35} {'Thanh cong':>10} {'Avg Reward':>12} {'Avg Steps':>10}")
    print("-" * 70)
    for r in results:
        print(f"{r['name']:<35} {r['success_rate']:>9.1f}% "
              f"{r['avg_reward']:>11.2f} {r['avg_steps']:>10.1f}")

    # So sánh learning rate
    print(f"\n--- Anh huong cua Learning Rate ---")
    lr_configs = [0.01, 0.1, 0.5, 0.9]
    lr_results = []

    for lr in lr_configs:
        agent = QLearningAgent(
            n_states=env.n_states,
            n_actions=env.n_actions,
            learning_rate=lr,
            discount_factor=0.95,
            epsilon=1.0,
            epsilon_decay=0.995,
            epsilon_min=0.01,
        )

        rewards = train_agent(env, agent, n_episodes=2000, max_steps=100,
                              verbose=False)

        success_rate, avg_reward, _ = evaluate_agent(env, agent, n_episodes=200)
        lr_results.append((lr, success_rate, avg_reward))

    print(f"{'Learning Rate':>15} {'Thanh cong':>12} {'Avg Reward':>12}")
    print("-" * 42)
    for lr, sr, ar in lr_results:
        print(f"{lr:>15.2f} {sr:>11.1f}% {ar:>11.2f}")

    print("""
    Nhan xet:
      - Epsilon qua thap/greedy: de bi mac ket, khong kham pha du
      - Epsilon qua cao: mat nhieu thoi gian kham pha vo ich
      - Learning rate qua nho: hoc cham
      - Learning rate qua lon: Q-value dao dong, khong on dinh
      → Can doi la chìa khoa!
    """)


# =============================================================================
# MAIN
# =============================================================================

if __name__ == "__main__":
    vi_du_me_cung()
    vi_du_taxi()
    vi_du_frozen_lake()
    vi_du_exploration()

    print("\n" + "=" * 60)
    print("TOM TAT Q-LEARNING:")
    print("=" * 60)
    print("""
    1. Q-LEARNING LA GI:
       - Thuat toan Reinforcement Learning co ban nhat
       - Hoc bang Q(state, action) = gia tri ky vong cua reward tuong lai
       - Model-free: khong can biet cach moi truong hoat dong
       - Off-policy: hoc tu hanh dong toi uu, khong phai hanh dong da chon

    2. CONG THUC CAP NHAT:
       Q(s,a) ← Q(s,a) + α * [r + γ * max Q(s',a') - Q(s,a)]
       - α (learning rate): toc do hoc (0.1 la pho bien)
       - γ (discount factor): quan trong cua reward tuong lai (0.9-0.99)
       - TD error = [r + γ * max Q(s',a')] - Q(s,a)

    3. EXPLORATION VS EXPLOITATION:
       - Epsilon-Greedy: xac suat ε chon ngau nhien, 1-ε chon tot nhat
       - Epsilon decay: giam ε theo thoi gian (bat dau kham pha nhieu,
         sau do tap trung khai thac)
       - Boltzmann exploration: chon theo phan phoi softmax cua Q-values

    4. HAN CHE VA MO RONG:
       - Q-table chi dung cho state/action roi rac, so luong nho
       - State lien tuc hoac qua lon → dung Deep Q-Network (DQN)
       - DQN: thay Q-table bang Neural Network
       - Double DQN, Dueling DQN, Rainbow: cai tien cua DQN
       - Policy Gradient (REINFORCE, PPO, A3C): hoc truc tiep policy

    5. UNG DUNG THUC TE:
       - Game AI (AlphaGo, Atari, StarCraft)
       - Robot tu hanh, dieu khien xe tu dong
       - Quan ly tai nguyen, toi uu lich trinh
       - He thong goi y (recommendation system)
       - Giao dich tu dong (algorithmic trading)
    """)
