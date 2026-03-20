# Python for AI

Tập hợp các ví dụ minh họa về Machine Learning và AI, xây dựng từ đầu bằng Python + NumPy.

## Danh sách ví dụ

| File | Chủ đề | Nội dung |
|------|--------|----------|
| `dataset_example.py` | Dataset | Tạo và xử lý dữ liệu cho ML |
| `classification_example.py` | Classification | Phân loại dữ liệu (KNN, Decision Tree...) |
| `clustering_example.py` | Clustering | Phân cụm dữ liệu (K-Means, DBSCAN...) |
| `neural_network_example.py` | Neural Network | Mạng nơ-ron từ đầu (phân loại hoa, bệnh tiểu đường, chữ số viết tay) |
| `cnn_stock_example.py` | CNN | Mạng tích chập 1D cho chứng khoán (xu hướng giá, mẫu nến, biến động) |
| `hidden_markov_model_example.py` | HMM | Mô hình Markov ẩn (thời tiết, POS tagging, sức khỏe) |
| `rnn_example.py` | RNN | Mạng hồi quy: Vanilla RNN, LSTM, GRU (nhiệt độ, sinh tên, cảm xúc) |
| `sentiment_analysis_example.py` | Sentiment Analysis | Phân tích cảm xúc review (Rule-Based, Naive Bayes, TF-IDF + NN) |
| `model_save_load_improve.py` | Model Save/Load | Lưu, tải và cải tiến model (Dropout, L2, Early Stopping) |
| `tensor.py` | Tensor | Các phép toán tensor cơ bản |
| `services.py` | Services | Các class tiện ích dùng chung (Activation, Loss, Data) |

## Yêu cầu

- Python 3.8+
- NumPy
- TensorFlow (cho `dataset_example.py`, `classification_example.py`, `clustering_example.py`, `tensor.py`)

## Cài đặt

```bash
pip install -r requirements.txt
```

## Chạy ví dụ

```bash
python dataset_example.py
python classification_example.py
python clustering_example.py
python neural_network_example.py
python cnn_stock_example.py
python hidden_markov_model_example.py
python rnn_example.py
python sentiment_analysis_example.py
python model_save_load_improve.py
```
