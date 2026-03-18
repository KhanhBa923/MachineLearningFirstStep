"""
Sentiment Analysis (Phân tích cảm xúc) - Xây dựng từ đầu

Sentiment Analysis là bài toán xác định CẢM XÚC trong văn bản:
  - Tích cực (Positive): "Sản phẩm rất tốt, giao hàng nhanh"
  - Tiêu cực (Negative): "Hàng lỗi, dịch vụ tệ"
  - Trung tính (Neutral): "Tôi đã mua sản phẩm này hôm qua"

Ứng dụng thực tế:
  - Phân tích đánh giá sản phẩm trên Shopee, Lazada, Tiki
  - Theo dõi dư luận mạng xã hội về thương hiệu
  - Phân loại email khiếu nại / khen ngợi
  - Phân tích phản hồi khách hàng tự động

Các phương pháp trong file này:
  1. Rule-Based: dùng từ điển cảm xúc (đơn giản, nhanh)
  2. Bag of Words + Naive Bayes (ML cổ điển)
  3. TF-IDF + Neural Network (nâng cao hơn)
  4. So sánh tất cả phương pháp
"""

import numpy as np
import re
from collections import Counter


# =============================================================================
# PHẦN 1: TIỀN XỬ LÝ VĂN BẢN (TEXT PREPROCESSING)
# =============================================================================

class TextPreprocessor:
    """
    Tiền xử lý văn bản trước khi phân tích.

    Tại sao cần tiền xử lý:
      - Máy tính không hiểu chữ, chỉ hiểu số
      - Cần loại bỏ nhiễu (dấu câu, từ vô nghĩa)
      - Chuẩn hóa văn bản (viết thường, bỏ ký tự đặc biệt)
    """

    # Stopwords tiếng Việt: những từ xuất hiện nhiều nhưng không mang ý nghĩa cảm xúc
    STOPWORDS = {
        "và", "của", "là", "có", "được", "cho", "với", "này", "đã",
        "các", "trong", "một", "những", "để", "tôi", "mình", "bạn",
        "thì", "mà", "khi", "nếu", "từ", "hay", "hoặc", "vì",
        "ở", "đến", "về", "ra", "lên", "xuống", "vào", "theo",
        "cũng", "đều", "rồi", "nó", "họ", "ta", "em", "anh",
    }

    @staticmethod
    def clean_text(text):
        """
        Làm sạch văn bản:
          1. Chuyển thành chữ thường
          2. Giữ lại chữ cái và khoảng trắng
          3. Bỏ khoảng trắng thừa
        """
        text = text.lower()
        text = re.sub(r'[^\w\sáàảãạăắằẳẵặâấầẩẫậéèẻẽẹêếềểễệíìỉĩịóòỏõọôốồổỗộơớờởỡợúùủũụưứừửữựýỳỷỹỵđ]', ' ', text)
        text = re.sub(r'\s+', ' ', text).strip()
        return text

    @staticmethod
    def tokenize(text):
        """
        Tách từ (tokenization).
        Đơn giản: tách theo khoảng trắng.
        Thực tế tiếng Việt cần tool chuyên dụng (VnCoreNLP, underthesea)
        vì tiếng Việt có từ ghép: "hài lòng", "chất lượng"...
        """
        return text.split()

    @classmethod
    def preprocess(cls, text, remove_stopwords=True):
        """Pipeline đầy đủ: clean → tokenize → bỏ stopwords."""
        text = cls.clean_text(text)
        tokens = cls.tokenize(text)
        if remove_stopwords:
            tokens = [t for t in tokens if t not in cls.STOPWORDS]
        return tokens


# =============================================================================
# PHẦN 2: PHƯƠNG PHÁP 1 - RULE-BASED (Dùng từ điển)
# =============================================================================

class RuleBasedSentiment:
    """
    Phân tích cảm xúc dựa trên TỪ ĐIỂN CẢM XÚC.

    Cách hoạt động:
      1. Đếm số từ tích cực và tiêu cực trong câu
      2. Tính điểm: score = tích_cực - tiêu_cực
      3. Xét thêm từ phủ định ("không", "chẳng") đảo ngược cảm xúc
      4. Xét thêm từ tăng cường ("rất", "cực kỳ") nhân đôi điểm

    Ưu điểm: đơn giản, nhanh, không cần dữ liệu train
    Nhược điểm: phụ thuộc từ điển, không hiểu ngữ cảnh
    """

    # Từ điển cảm xúc tiếng Việt
    POSITIVE_WORDS = {
        # Tính từ tích cực
        "tốt", "hay", "đẹp", "nhanh", "rẻ", "ngon", "tuyệt", "xuất_sắc",
        "ổn", "ok", "nice", "good", "great", "perfect",
        "thích", "yêu", "hài_lòng", "ưng", "phê",
        "chất_lượng", "bền", "đáng_tiền", "xịn", "pro",
        # Động từ / trạng thái tích cực
        "recommend", "giới_thiệu", "quay_lại", "ủng_hộ", "khen",
        "hợp", "vừa", "tiện", "dễ", "nhanh_chóng",
        "chuyên_nghiệp", "nhiệt_tình", "chu_đáo", "thân_thiện",
        "hấp_dẫn", "ấn_tượng", "thoải_mái", "sạch_sẽ",
    }

    NEGATIVE_WORDS = {
        # Tính từ tiêu cực
        "tệ", "xấu", "chậm", "đắt", "dở", "tồi", "kém", "lỗi",
        "hỏng", "rách", "bẩn", "ồn", "nóng", "bad", "fail",
        "thất_vọng", "chán", "ghét", "tức", "bực",
        # Vấn đề
        "lừa_đảo", "fake", "giả", "nhái", "gian_lận",
        "chờ", "trễ", "muộn", "delay", "hủy",
        "vỡ", "gãy", "méo", "trầy", "móp",
        "thiếu", "sai", "nhầm", "lộn", "hết_hạn",
        "khó_chịu", "phiền", "rắc_rối", "phức_tạp",
    }

    # Từ phủ định: đảo ngược cảm xúc của từ đứng sau
    NEGATION_WORDS = {"không", "chẳng", "chả", "đừng", "chưa", "hết", "mất", "thiếu"}

    # Từ tăng cường: nhân đôi điểm cảm xúc
    INTENSIFIERS = {"rất", "cực", "cực_kỳ", "quá", "siêu", "vô_cùng", "hết_sức", "khá"}

    def analyze(self, text):
        """
        Phân tích cảm xúc 1 câu.

        Returns:
            label: "positive", "negative", "neutral"
            score: điểm cảm xúc (> 0: tích cực, < 0: tiêu cực)
            details: chi tiết từ nào đóng góp
        """
        tokens = TextPreprocessor.preprocess(text, remove_stopwords=False)

        score = 0
        details = []

        i = 0
        while i < len(tokens):
            word = tokens[i]
            multiplier = 1

            # Kiểm tra từ tăng cường đứng trước
            if i > 0 and tokens[i - 1] in self.INTENSIFIERS:
                multiplier = 2

            # Kiểm tra phủ định đứng trước (trong 2 từ)
            negated = False
            for j in range(max(0, i - 2), i):
                if tokens[j] in self.NEGATION_WORDS:
                    negated = True
                    break

            if word in self.POSITIVE_WORDS:
                if negated:
                    score -= 1 * multiplier
                    details.append(f"'{word}' (bi phu dinh → -{multiplier})")
                else:
                    score += 1 * multiplier
                    details.append(f"'{word}' (+{multiplier})")
            elif word in self.NEGATIVE_WORDS:
                if negated:
                    score += 0.5 * multiplier
                    details.append(f"'{word}' (bi phu dinh → +{0.5 * multiplier})")
                else:
                    score -= 1 * multiplier
                    details.append(f"'{word}' (-{multiplier})")

            i += 1

        # Phân loại dựa trên tổng điểm
        if score > 0:
            label = "positive"
        elif score < 0:
            label = "negative"
        else:
            label = "neutral"

        return label, score, details


# =============================================================================
# PHẦN 3: PHƯƠNG PHÁP 2 - BAG OF WORDS + NAIVE BAYES
# =============================================================================

class BagOfWords:
    """
    Bag of Words (BoW): chuyển văn bản thành vector số.

    Cách hoạt động:
      1. Xây dựng từ vựng (vocabulary) từ tất cả văn bản
      2. Mỗi văn bản → vector đếm số lần xuất hiện mỗi từ

    Ví dụ:
      Vocabulary: ["tốt", "xấu", "rất", "hàng"]
      "hàng rất tốt"  → [1, 0, 1, 1]  (tốt:1, xấu:0, rất:1, hàng:1)
      "hàng xấu"      → [0, 1, 0, 1]  (tốt:0, xấu:1, rất:0, hàng:1)

    Nhược điểm: mất thứ tự từ ("tôi thích bạn" = "bạn thích tôi")
    """

    def __init__(self, max_features=500):
        self.max_features = max_features
        self.vocabulary = {}

    def fit(self, texts):
        """Xây dựng từ vựng từ danh sách văn bản."""
        word_counts = Counter()
        for text in texts:
            tokens = TextPreprocessor.preprocess(text)
            word_counts.update(tokens)

        # Lấy max_features từ phổ biến nhất
        most_common = word_counts.most_common(self.max_features)
        self.vocabulary = {word: idx for idx, (word, _) in enumerate(most_common)}

    def transform(self, texts):
        """Chuyển danh sách văn bản thành ma trận BoW."""
        matrix = np.zeros((len(texts), len(self.vocabulary)))

        for i, text in enumerate(texts):
            tokens = TextPreprocessor.preprocess(text)
            for token in tokens:
                if token in self.vocabulary:
                    matrix[i, self.vocabulary[token]] += 1

        return matrix


class NaiveBayes:
    """
    Naive Bayes Classifier cho phân loại văn bản.

    Dựa trên Định lý Bayes:
      P(class | document) ∝ P(document | class) × P(class)

    "Naive" vì giả sử các từ ĐỘC LẬP với nhau:
      P(doc | class) = P(word1 | class) × P(word2 | class) × ...

    Giả sử này SAI trong thực tế (từ có liên quan nhau),
    nhưng Naive Bayes vẫn hoạt động tốt đáng ngạc nhiên!

    Ưu điểm: nhanh, ít dữ liệu vẫn chạy tốt, dễ hiểu
    Nhược điểm: giả sử độc lập, không hiểu ngữ cảnh
    """

    def __init__(self, alpha=1.0):
        """
        Args:
            alpha: Laplace smoothing - tránh xác suất = 0 khi gặp từ mới
                   Nếu 1 từ chưa xuất hiện trong class nào → P = 0 → nhân cả chuỗi = 0
                   Thêm alpha (thường = 1) vào mọi đếm để tránh điều này
        """
        self.alpha = alpha

    def fit(self, X, y):
        """
        Huấn luyện: tính xác suất P(word | class) cho mỗi từ và mỗi lớp.
        """
        self.classes = np.unique(y)
        n_features = X.shape[1]

        # P(class): xác suất tiên nghiệm (prior)
        # Ví dụ: 60% review tích cực, 40% tiêu cực
        self.class_prior = {}

        # P(word | class): xác suất từ xuất hiện trong mỗi class
        self.word_prob = {}

        for c in self.classes:
            # Lấy tất cả documents thuộc class c
            X_c = X[y == c]

            # P(class) = số docs class c / tổng số docs
            self.class_prior[c] = len(X_c) / len(y)

            # Đếm tổng số lần xuất hiện mỗi từ trong class c
            word_counts = np.sum(X_c, axis=0) + self.alpha  # Laplace smoothing
            total_words = np.sum(word_counts)

            # P(word | class) = (đếm từ + alpha) / (tổng từ + alpha × vocab_size)
            self.word_prob[c] = word_counts / total_words

    def predict(self, X):
        """
        Dự đoán: tính P(class | doc) cho mỗi class, chọn class có P cao nhất.

        Dùng log để tránh underflow (nhân nhiều xác suất nhỏ → gần 0):
          log P(class | doc) = log P(class) + Σ log P(word_i | class)
        """
        predictions = []

        for x in X:
            best_class = None
            best_log_prob = float("-inf")

            for c in self.classes:
                # log P(class)
                log_prob = np.log(self.class_prior[c])

                # + Σ count(word) × log P(word | class)
                # Chỉ tính cho các từ xuất hiện (count > 0)
                log_prob += np.sum(x * np.log(self.word_prob[c]))

                if log_prob > best_log_prob:
                    best_log_prob = log_prob
                    best_class = c

            predictions.append(best_class)

        return np.array(predictions)

    def accuracy(self, X, y):
        preds = self.predict(X)
        return np.mean(preds == y) * 100


# =============================================================================
# PHẦN 4: PHƯƠNG PHÁP 3 - TF-IDF + NEURAL NETWORK
# =============================================================================

class TFIDF:
    """
    TF-IDF: Term Frequency - Inverse Document Frequency.

    Cải tiến Bag of Words bằng cách:
      - TF (Term Frequency): từ xuất hiện nhiều trong 1 doc → quan trọng
      - IDF (Inverse Document Frequency): từ xuất hiện ở NHIỀU docs → ít quan trọng

    Ví dụ:
      Từ "sản phẩm" xuất hiện ở hầu hết reviews → IDF thấp → ít quan trọng
      Từ "tuyệt vời" chỉ xuất hiện ở review tốt → IDF cao → RẤT quan trọng

    Công thức:
      TF(t, d) = số lần từ t xuất hiện trong doc d / tổng số từ trong doc d
      IDF(t) = log(tổng số docs / số docs chứa từ t)
      TF-IDF(t, d) = TF(t, d) × IDF(t)
    """

    def __init__(self, max_features=500):
        self.max_features = max_features
        self.vocabulary = {}
        self.idf = None

    def fit(self, texts):
        """Xây dựng vocabulary và tính IDF."""
        # Xây vocabulary giống BoW
        word_counts = Counter()
        doc_counts = Counter()  # Đếm số docs chứa mỗi từ

        tokenized = []
        for text in texts:
            tokens = TextPreprocessor.preprocess(text)
            tokenized.append(tokens)
            word_counts.update(tokens)
            doc_counts.update(set(tokens))  # set() để mỗi từ chỉ đếm 1 lần/doc

        most_common = word_counts.most_common(self.max_features)
        self.vocabulary = {word: idx for idx, (word, _) in enumerate(most_common)}

        # Tính IDF cho mỗi từ trong vocabulary
        n_docs = len(texts)
        self.idf = np.zeros(len(self.vocabulary))
        for word, idx in self.vocabulary.items():
            # +1 để tránh chia cho 0
            self.idf[idx] = np.log((n_docs + 1) / (doc_counts.get(word, 0) + 1)) + 1

    def transform(self, texts):
        """Chuyển texts thành ma trận TF-IDF."""
        matrix = np.zeros((len(texts), len(self.vocabulary)))

        for i, text in enumerate(texts):
            tokens = TextPreprocessor.preprocess(text)
            n_tokens = len(tokens) if tokens else 1

            # Đếm TF
            token_counts = Counter(tokens)
            for token, count in token_counts.items():
                if token in self.vocabulary:
                    idx = self.vocabulary[token]
                    tf = count / n_tokens
                    matrix[i, idx] = tf * self.idf[idx]

        # Chuẩn hóa L2 (mỗi vector có độ dài = 1)
        norms = np.sqrt(np.sum(matrix ** 2, axis=1, keepdims=True))
        norms[norms == 0] = 1
        matrix = matrix / norms

        return matrix


class SentimentNN:
    """
    Neural Network đơn giản cho Sentiment Analysis.
    Input: vector TF-IDF → Hidden → Output (3 classes: neg, neu, pos)
    """

    def __init__(self, n_input, n_hidden, n_output, learning_rate=0.01):
        self.lr = learning_rate

        self.W1 = np.random.randn(n_input, n_hidden) * np.sqrt(2.0 / n_input)
        self.b1 = np.zeros((1, n_hidden))
        self.W2 = np.random.randn(n_hidden, n_output) * np.sqrt(2.0 / n_hidden)
        self.b2 = np.zeros((1, n_output))

    def softmax(self, x):
        exp_x = np.exp(x - np.max(x, axis=1, keepdims=True))
        return exp_x / np.sum(exp_x, axis=1, keepdims=True)

    def forward(self, X):
        self.z1 = X @ self.W1 + self.b1
        self.a1 = np.maximum(0, self.z1)  # ReLU
        self.z2 = self.a1 @ self.W2 + self.b2
        self.a2 = self.softmax(self.z2)
        return self.a2

    def train(self, X, y_onehot, epochs=100, verbose=True):
        for epoch in range(epochs):
            # Forward
            output = self.forward(X)

            # Loss
            output_clip = np.clip(output, 1e-8, 1 - 1e-8)
            loss = -np.mean(np.sum(y_onehot * np.log(output_clip), axis=1))

            # Backward
            m = X.shape[0]
            d2 = output - y_onehot
            dW2 = self.a1.T @ d2 / m
            db2 = np.mean(d2, axis=0, keepdims=True)

            d1 = (d2 @ self.W2.T) * (self.z1 > 0).astype(float)
            dW1 = X.T @ d1 / m
            db1 = np.mean(d1, axis=0, keepdims=True)

            self.W2 -= self.lr * dW2
            self.b2 -= self.lr * db2
            self.W1 -= self.lr * dW1
            self.b1 -= self.lr * db1

            if verbose and (epoch + 1) % (epochs // 5) == 0:
                acc = self.accuracy(X, np.argmax(y_onehot, axis=1))
                print(f"  Epoch {epoch + 1:4d}/{epochs} | Loss: {loss:.4f} | Acc: {acc:.1f}%")

    def predict(self, X):
        probs = self.forward(X)
        return np.argmax(probs, axis=1)

    def predict_proba(self, X):
        return self.forward(X)

    def accuracy(self, X, y):
        return np.mean(self.predict(X) == y) * 100


# =============================================================================
# DỮ LIỆU MẪU: ĐÁNH GIÁ SẢN PHẨM TRÊN SÀN TMĐT
# =============================================================================

def tao_du_lieu_review():
    """
    Tạo dataset đánh giá sản phẩm (mô phỏng Shopee/Lazada/Tiki).

    3 loại:
      0 = negative (tiêu cực)
      1 = neutral (trung tính)
      2 = positive (tích cực)
    """
    reviews = [
        # === POSITIVE (2) ===
        ("Sản phẩm rất tốt, đóng gói cẩn thận, giao hàng nhanh", 2),
        ("Chất lượng tuyệt vời, đúng như mô tả", 2),
        ("Mình rất hài lòng, sẽ quay lại mua tiếp", 2),
        ("Giá rẻ mà chất lượng tốt, đáng tiền", 2),
        ("Shop giao hàng nhanh, đóng gói đẹp, hàng chất lượng", 2),
        ("Sản phẩm đẹp, dùng rất thích, recommend cho mọi người", 2),
        ("Tuyệt vời, đúng hàng, giao nhanh, sẽ ủng hộ shop", 2),
        ("Hàng xịn, giá hợp lý, shop nhiệt tình", 2),
        ("Rất đẹp, chất liệu tốt, may đẹp, vừa vặn", 2),
        ("Sản phẩm ok, giao hàng siêu nhanh, shop thân thiện", 2),
        ("Mua lần 2 rồi, vẫn rất ưng, chất lượng ổn định", 2),
        ("Đẹp lắm, mọi người nên mua, shop chuyên nghiệp", 2),
        ("Hàng chính hãng, chất lượng cao, rất hài lòng", 2),
        ("Sản phẩm bền đẹp, dùng thoải mái, giá phải chăng", 2),
        ("Shop chu đáo, tư vấn nhiệt tình, hàng đẹp", 2),
        ("Rất tốt luôn, vượt mong đợi, sẽ giới thiệu bạn bè", 2),
        ("Chất lượng xứng đáng với giá tiền, hàng đẹp", 2),
        ("Giao hàng cực nhanh, sản phẩm tốt, shop uy tín", 2),
        ("Dùng rất ổn, không có gì phàn nàn, tốt lắm", 2),
        ("Hàng đẹp, đúng mô tả, đóng gói kỹ càng", 2),
        ("Mình thích sản phẩm này, chất lượng rất ok", 2),
        ("Sản phẩm dùng rất tiện, thiết kế đẹp mắt", 2),
        ("Shop tuyệt vời, hàng xịn, giá tốt, giao nhanh", 2),
        ("Hàng đẹp xuất sắc, đóng gói cẩn thận, rất ưng", 2),
        ("Lần đầu mua mà ấn tượng, sẽ quay lại ủng hộ", 2),

        # === NEGATIVE (0) ===
        ("Hàng lỗi, giao chậm, shop không hỗ trợ đổi trả", 0),
        ("Sản phẩm tệ, không đúng mô tả, thất vọng", 0),
        ("Chất lượng kém, dùng được 2 ngày đã hỏng", 0),
        ("Giao hàng chậm, đóng gói sơ sài, hàng bị méo", 0),
        ("Shop lừa đảo, hàng fake, không giống hình", 0),
        ("Sản phẩm xấu, chất liệu rẻ tiền, không đáng mua", 0),
        ("Đặt size M gửi size S, gọi shop không nghe máy", 0),
        ("Hàng bị vỡ khi nhận, đóng gói quá tệ", 0),
        ("Chờ 2 tuần mới nhận hàng, sản phẩm lại lỗi", 0),
        ("Không recommend, hàng kém chất lượng, phí tiền", 0),
        ("Mua về không dùng được, hàng lỗi hoàn toàn", 0),
        ("Shop thái độ tệ, hàng xấu, không bao giờ mua lại", 0),
        ("Sản phẩm bị trầy xước, đóng gói cẩu thả", 0),
        ("Hàng nhái rẻ tiền, chất lượng tồi, thất vọng", 0),
        ("Giao sai hàng, shop không chịu đổi, rất bực", 0),
        ("Dùng một tuần đã hỏng, chất lượng quá kém", 0),
        ("Hàng không đúng màu, chất liệu mỏng, xấu", 0),
        ("Shop giao hàng trễ, hàng đến bị rách bao bì", 0),
        ("Sản phẩm rất tệ, không đáng đồng nào", 0),
        ("Mình rất thất vọng, hàng khác xa hình ảnh", 0),
        ("Chất lượng dở, hàng bị lỗi ngay khi mở hộp", 0),
        ("Đóng gói sơ sài, hàng bị móp, shop vô trách nhiệm", 0),
        ("Hàng giả, fake rõ ràng, shop lừa khách", 0),
        ("Giao chậm, hàng cũ, bẩn, không thể chấp nhận", 0),
        ("Không bao giờ mua lại, quá tệ, phí tiền", 0),

        # === NEUTRAL (1) ===
        ("Sản phẩm bình thường, không tốt không xấu", 1),
        ("Giao hàng bình thường, đóng gói bình thường", 1),
        ("Mình đã nhận hàng, chưa dùng nên chưa đánh giá", 1),
        ("Hàng đúng mô tả, giao đúng hẹn, bình thường", 1),
        ("Tạm được, không có gì đặc biệt", 1),
        ("Sản phẩm tầm trung, giá hợp lý", 1),
        ("Đã nhận hàng, đang dùng thử", 1),
        ("Shop giao đúng hẹn, hàng đủ số lượng", 1),
        ("Chưa dùng nhiều nên chưa biết tốt hay không", 1),
        ("Sản phẩm giống hình, chất lượng tầm giá", 1),
        ("Giao hàng nhanh, nhưng sản phẩm thì bình thường", 1),
        ("Được cái giao nhanh, hàng thì tạm ổn", 1),
        ("Lần đầu mua ở shop này, hàng tạm được", 1),
        ("Nhận hàng rồi, hàng đúng như mô tả", 1),
        ("Sản phẩm đúng giá tiền, không hơn không kém", 1),
        ("Mới dùng được vài ngày, tạm thời chưa có vấn đề", 1),
        ("Hàng OK, đóng gói bình thường, giao hơi lâu", 1),
        ("Sản phẩm tầm này thì vậy thôi, không kỳ vọng nhiều", 1),
        ("Đã mua, đang chờ dùng lâu hơn mới đánh giá", 1),
        ("Hàng bình thường, không nổi bật cũng không tệ", 1),
    ]

    texts = [r[0] for r in reviews]
    labels = np.array([r[1] for r in reviews])

    return texts, labels


# =============================================================================
# VÍ DỤ 1: RULE-BASED SENTIMENT
# =============================================================================

def vi_du_rule_based():
    print("=" * 65)
    print("VI DU 1: RULE-BASED SENTIMENT (Tu dien cam xuc)")
    print("=" * 65)
    print("Phuong phap: dem tu tich cuc/tieu cuc, xet phu dinh/tang cuong\n")

    analyzer = RuleBasedSentiment()

    # Test với nhiều câu review
    test_reviews = [
        "Sản phẩm rất tốt, giao hàng nhanh",
        "Hàng lỗi, chất lượng tệ, thất vọng",
        "Bình thường, không có gì đặc biệt",
        "Không tốt, giao chậm, đóng gói sơ sài",        # Phủ định
        "Rất đẹp, cực kỳ hài lòng, tuyệt vời",          # Tăng cường
        "Hàng không xấu, shop nhiệt tình",                # Phủ định + tích cực
        "Giá hơi đắt nhưng chất lượng tốt",              # Hỗn hợp
    ]

    for review in test_reviews:
        label, score, details = analyzer.analyze(review)

        # Hiển thị kết quả
        emoji = {"positive": "[+]", "negative": "[-]", "neutral": "[=]"}
        print(f"  {emoji[label]} \"{review}\"")
        print(f"      Ket qua: {label:8s} | Score: {score:+.1f}")
        if details:
            print(f"      Chi tiet: {', '.join(details)}")
        print()

    # Đánh giá trên toàn bộ dataset
    print("--- Danh gia tren toan bo dataset ---")
    texts, labels = tao_du_lieu_review()
    label_map = {"negative": 0, "neutral": 1, "positive": 2}

    correct = 0
    for text, true_label in zip(texts, labels):
        pred_label, _, _ = analyzer.analyze(text)
        if label_map[pred_label] == true_label:
            correct += 1

    print(f"  Accuracy: {correct}/{len(texts)} ({correct / len(texts) * 100:.1f}%)")
    print(f"  (Rule-based khong can train, nhung phu thuoc tu dien)")


# =============================================================================
# VÍ DỤ 2: BAG OF WORDS + NAIVE BAYES
# =============================================================================

def vi_du_naive_bayes():
    print("\n" + "=" * 65)
    print("VI DU 2: BAG OF WORDS + NAIVE BAYES")
    print("=" * 65)
    print("Phuong phap: dem tu (BoW) + xac suat Bayes\n")

    texts, labels = tao_du_lieu_review()

    # Chia train/test
    np.random.seed(42)
    n = len(texts)
    idx = np.random.permutation(n)
    split = int(n * 0.8)

    train_texts = [texts[i] for i in idx[:split]]
    test_texts = [texts[i] for i in idx[split:]]
    y_train = labels[idx[:split]]
    y_test = labels[idx[split:]]

    print(f"  Train: {len(train_texts)} reviews | Test: {len(test_texts)} reviews")

    # Tạo Bag of Words
    bow = BagOfWords(max_features=200)
    bow.fit(train_texts)
    X_train = bow.transform(train_texts)
    X_test = bow.transform(test_texts)

    print(f"  Vocabulary size: {len(bow.vocabulary)} tu")
    print(f"  Vector size: {X_train.shape[1]}\n")

    # Train Naive Bayes
    print("--- Train Naive Bayes ---")
    nb = NaiveBayes(alpha=1.0)
    nb.fit(X_train, y_train)

    train_acc = nb.accuracy(X_train, y_train)
    test_acc = nb.accuracy(X_test, y_test)
    print(f"  Train accuracy: {train_acc:.1f}%")
    print(f"  Test accuracy:  {test_acc:.1f}%")

    # Dự đoán reviews mới
    print("\n--- Du doan reviews moi ---")
    label_names = ["NEGATIVE", "NEUTRAL", "POSITIVE"]
    new_reviews = [
        "Hàng đẹp, shop nhiệt tình, sẽ mua lại",
        "Giao hàng chậm, hàng bị lỗi",
        "Tạm được, bình thường",
        "Rất thất vọng, hàng fake",
        "Chất lượng tuyệt vời, đáng tiền",
    ]

    X_new = bow.transform(new_reviews)
    preds = nb.predict(X_new)

    for review, pred in zip(new_reviews, preds):
        print(f"  [{label_names[pred]:8s}] \"{review}\"")


# =============================================================================
# VÍ DỤ 3: TF-IDF + NEURAL NETWORK
# =============================================================================

def vi_du_tfidf_nn():
    print("\n" + "=" * 65)
    print("VI DU 3: TF-IDF + NEURAL NETWORK")
    print("=" * 65)
    print("Phuong phap: TF-IDF (trong so tu thong minh hon) + NN\n")

    texts, labels = tao_du_lieu_review()

    # Chia train/test
    np.random.seed(42)
    n = len(texts)
    idx = np.random.permutation(n)
    split = int(n * 0.8)

    train_texts = [texts[i] for i in idx[:split]]
    test_texts = [texts[i] for i in idx[split:]]
    y_train = labels[idx[:split]]
    y_test = labels[idx[split:]]

    # Tạo TF-IDF features
    tfidf = TFIDF(max_features=200)
    tfidf.fit(train_texts)
    X_train = tfidf.transform(train_texts)
    X_test = tfidf.transform(test_texts)

    print(f"  Train: {len(train_texts)} | Test: {len(test_texts)}")
    print(f"  TF-IDF vector size: {X_train.shape[1]}\n")

    # Hiển thị top TF-IDF words cho 1 review
    print("--- Vi du TF-IDF cho 1 review ---")
    sample = "Sản phẩm rất tốt, chất lượng tuyệt vời"
    sample_vec = tfidf.transform([sample])[0]
    word_scores = []
    for word, idx_w in tfidf.vocabulary.items():
        if sample_vec[idx_w] > 0:
            word_scores.append((word, sample_vec[idx_w]))
    word_scores.sort(key=lambda x: x[1], reverse=True)
    print(f"  Review: \"{sample}\"")
    print(f"  Top tu quan trong:")
    for word, score in word_scores[:5]:
        bar = "|" * int(score * 30)
        print(f"    {word:<15s} {score:.4f} {bar}")

    # One-hot encode labels
    y_train_oh = np.zeros((len(y_train), 3))
    for i, l in enumerate(y_train):
        y_train_oh[i, l] = 1

    # Train Neural Network
    print("\n--- Train Neural Network ---")
    np.random.seed(42)
    nn = SentimentNN(
        n_input=X_train.shape[1],
        n_hidden=32,
        n_output=3,
        learning_rate=0.1
    )

    nn.train(X_train, y_train_oh, epochs=200)

    train_acc = nn.accuracy(X_train, y_train)
    test_acc = nn.accuracy(X_test, y_test)
    print(f"\n  Train accuracy: {train_acc:.1f}%")
    print(f"  Test accuracy:  {test_acc:.1f}%")

    # Dự đoán với xác suất chi tiết
    print("\n--- Du doan voi xac suat chi tiet ---")
    label_names = ["NEGATIVE", "NEUTRAL", "POSITIVE"]
    new_reviews = [
        "Sản phẩm tốt, rất hài lòng",
        "Hàng tệ, không bao giờ mua lại",
        "Bình thường, tạm được",
        "Shop nhiệt tình, hàng xịn, giá tốt",
        "Giao chậm, hàng lỗi, thất vọng",
    ]

    X_new = tfidf.transform(new_reviews)
    probs = nn.predict_proba(X_new)
    preds = nn.predict(X_new)

    for i, review in enumerate(new_reviews):
        print(f"\n  \"{review}\"")
        print(f"    → {label_names[preds[i]]}")
        print(f"    P(neg)={probs[i, 0]:.1%}  P(neu)={probs[i, 1]:.1%}  P(pos)={probs[i, 2]:.1%}")

    return tfidf, nn


# =============================================================================
# VÍ DỤ 4: SO SÁNH TẤT CẢ PHƯƠNG PHÁP
# =============================================================================

def vi_du_so_sanh():
    print("\n" + "=" * 65)
    print("VI DU 4: SO SANH TAT CA PHUONG PHAP")
    print("=" * 65)

    texts, labels = tao_du_lieu_review()

    np.random.seed(42)
    n = len(texts)
    idx = np.random.permutation(n)
    split = int(n * 0.8)

    train_texts = [texts[i] for i in idx[:split]]
    test_texts = [texts[i] for i in idx[split:]]
    y_train = labels[idx[:split]]
    y_test = labels[idx[split:]]

    label_map = {"negative": 0, "neutral": 1, "positive": 2}
    label_names = ["NEGATIVE", "NEUTRAL", "POSITIVE"]

    # --- 1. Rule-Based ---
    rule_analyzer = RuleBasedSentiment()
    rule_preds = []
    for text in test_texts:
        pred_label, _, _ = rule_analyzer.analyze(text)
        rule_preds.append(label_map[pred_label])
    rule_preds = np.array(rule_preds)
    rule_acc = np.mean(rule_preds == y_test) * 100

    # --- 2. BoW + Naive Bayes ---
    bow = BagOfWords(max_features=200)
    bow.fit(train_texts)
    X_train_bow = bow.transform(train_texts)
    X_test_bow = bow.transform(test_texts)

    nb = NaiveBayes(alpha=1.0)
    nb.fit(X_train_bow, y_train)
    nb_acc = nb.accuracy(X_test_bow, y_test)
    nb_preds = nb.predict(X_test_bow)

    # --- 3. TF-IDF + NN ---
    tfidf = TFIDF(max_features=200)
    tfidf.fit(train_texts)
    X_train_tfidf = tfidf.transform(train_texts)
    X_test_tfidf = tfidf.transform(test_texts)

    y_train_oh = np.zeros((len(y_train), 3))
    for i, l in enumerate(y_train):
        y_train_oh[i, l] = 1

    np.random.seed(42)
    nn = SentimentNN(X_train_tfidf.shape[1], 32, 3, learning_rate=0.1)
    nn.train(X_train_tfidf, y_train_oh, epochs=200, verbose=False)
    nn_acc = nn.accuracy(X_test_tfidf, y_test)
    nn_preds = nn.predict(X_test_tfidf)

    # --- Bảng so sánh ---
    print(f"\n  {'Phuong phap':<25s} {'Test Acc':>10s}  {'Can train?':>10s}  {'Uu diem':>20s}")
    print("  " + "-" * 70)
    print(f"  {'Rule-Based (Tu dien)':<25s} {rule_acc:>9.1f}%  {'Khong':>10s}  {'Nhanh, de hieu':>20s}")
    print(f"  {'BoW + Naive Bayes':<25s} {nb_acc:>9.1f}%  {'Co':>10s}  {'Don gian, it data':>20s}")
    print(f"  {'TF-IDF + Neural Net':<25s} {nn_acc:>9.1f}%  {'Co':>10s}  {'Hoc duoc pattern':>20s}")

    # Chi tiết từng mẫu test
    print(f"\n--- Chi tiet du doan tung review ---")
    print(f"  {'Review':<45s} {'That':>8s} {'Rule':>8s} {'NB':>8s} {'NN':>8s}")
    print("  " + "-" * 80)

    for i in range(min(len(test_texts), 14)):
        review_short = test_texts[i][:42] + "..." if len(test_texts[i]) > 42 else test_texts[i]
        actual = label_names[y_test[i]][:3]
        rule = label_names[rule_preds[i]][:3]
        nb_p = label_names[nb_preds[i]][:3]
        nn_p = label_names[nn_preds[i]][:3]

        # Đánh dấu đúng/sai
        r_mark = "V" if rule_preds[i] == y_test[i] else "X"
        nb_mark = "V" if nb_preds[i] == y_test[i] else "X"
        nn_mark = "V" if nn_preds[i] == y_test[i] else "X"

        print(f"  {review_short:<45s} {actual:>5s}  {rule:>4s}({r_mark}) {nb_p:>4s}({nb_mark}) {nn_p:>4s}({nn_mark})")


# =============================================================================
# VÍ DỤ 5: ỨNG DỤNG THỰC TẾ - PHÂN TÍCH ĐÁNH GIÁ SẢN PHẨM
# =============================================================================

def vi_du_ung_dung():
    print("\n" + "=" * 65)
    print("VI DU 5: UNG DUNG - PHAN TICH DANH GIA SAN PHAM")
    print("=" * 65)
    print("Kich ban: Shop Shopee muon biet khach hang nghi gi\n")

    # Giả lập đánh giá từ nhiều sản phẩm
    reviews_by_product = {
        "Tai nghe Bluetooth X1": [
            "Âm thanh hay, pin trâu, đeo thoải mái",
            "Kết nối bluetooth ổn định, dùng rất thích",
            "Âm bass tốt, thiết kế đẹp",
            "Dùng 2 tuần pin vẫn tốt, âm thanh rõ",
            "Hơi ồn khi gọi điện thoại, tạm được",
            "Đeo lâu hơi đau tai, chất lượng bình thường",
        ],
        "Áo thun nam basic": [
            "Chất vải mỏng, nhanh rách",
            "Màu phai sau 2 lần giặt, thất vọng",
            "Size không đúng, đặt L gửi M",
            "Vải mỏng, mặc nóng, không thoáng",
            "Tạm được, giá rẻ thì vậy thôi",
            "Vải mềm, mặc thoải mái, giá ok",
        ],
        "Sạc dự phòng 10000mAh": [
            "Sạc nhanh, dung lượng đúng như quảng cáo",
            "Pin trâu, sạc được 3 lần iPhone",
            "Thiết kế nhỏ gọn, tiện mang theo",
            "Dùng tốt, giá hợp lý, giao nhanh",
            "Bình thường, sạc hơi nóng",
            "Sạc chậm, không nhanh như quảng cáo",
        ],
    }

    # Dùng Rule-Based để phân tích nhanh (không cần train)
    analyzer = RuleBasedSentiment()

    for product, reviews in reviews_by_product.items():
        print(f"\n  --- {product} ---")

        pos_count = 0
        neg_count = 0
        neu_count = 0
        total_score = 0

        for review in reviews:
            label, score, _ = analyzer.analyze(review)
            total_score += score
            if label == "positive":
                pos_count += 1
            elif label == "negative":
                neg_count += 1
            else:
                neu_count += 1

        n = len(reviews)
        avg_score = total_score / n

        # Tổng kết sản phẩm
        print(f"    Tong reviews: {n}")
        print(f"    Tich cuc: {pos_count} ({pos_count / n * 100:.0f}%) | "
              f"Trung tinh: {neu_count} ({neu_count / n * 100:.0f}%) | "
              f"Tieu cuc: {neg_count} ({neg_count / n * 100:.0f}%)")
        print(f"    Diem cam xuc TB: {avg_score:+.1f}")

        # Đánh giá tổng thể
        if avg_score > 0.5:
            verdict = "SAN PHAM TOT - Khach hang hai long"
        elif avg_score < -0.5:
            verdict = "CAN CAI THIEN - Nhieu phan hoi tieu cuc"
        else:
            verdict = "TRUNG BINH - Can theo doi them"

        # Thanh biểu đồ
        bar_pos = "|" * (pos_count * 4)
        bar_neu = ":" * (neu_count * 4)
        bar_neg = "x" * (neg_count * 4)
        print(f"    [{bar_pos}{bar_neu}{bar_neg}]")
        print(f"    → {verdict}")


# =============================================================================
# CHẠY TẤT CẢ
# =============================================================================

if __name__ == "__main__":
    vi_du_rule_based()
    vi_du_naive_bayes()
    vi_du_tfidf_nn()
    vi_du_so_sanh()
    vi_du_ung_dung()

    print("\n" + "=" * 65)
    print("TOM TAT SENTIMENT ANALYSIS:")
    print("=" * 65)
    print("""
    1. CAC PHUONG PHAP:
       - Rule-Based: nhanh, khong can data, nhung cung nhac
       - Naive Bayes: ML co dien, it data van chay tot
       - TF-IDF + NN: hoc duoc pattern phuc tap hon
       - (Thuc te: dung BERT/PhoBERT cho tieng Viet → chinh xac nhat)

    2. PIPELINE XU LY VAN BAN:
       Raw text → Clean → Tokenize → Remove stopwords
       → Vectorize (BoW/TF-IDF) → Model → Predict

    3. LUU Y TIENG VIET:
       - Can tach tu (word segmentation): "hài lòng" la 1 tu, khong phai 2
       - Dung tool: underthesea, VnCoreNLP, pyvi
       - Xu ly tu long/viet tat: "sp" = "san pham", "gh" = "giao hang"
       - Emoji cung mang cam xuc: 😍 = tich cuc, 😡 = tieu cuc

    4. UNG DUNG THUC TE:
       - Phan tich review san pham (Shopee, Lazada, Tiki)
       - Giam sat thuong hieu tren mang xa hoi
       - Phan loai email khieu nai tu dong
       - Chatbot hieu cam xuc khach hang
    """)
