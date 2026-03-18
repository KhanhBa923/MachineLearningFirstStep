import tensorflow as tf


def tensor_basics():
	# Tạo 2 tensor hằng số (2x2)
	a = tf.constant([[1.0, 2.0], [3.0, 4.0]])
	b = tf.constant([[5.0, 6.0], [7.0, 8.0]])
	t = tf.reshape(a, [2,1,1, 1, 1, 2])  # Thay đổi hình dạng tensor a thành vector 1 chiều có 4 phần tử

	# In tensor gốc
	print("Tensor a:\n", a)
	print("Tensor b:\n", b)
	print("Reshaped tensor a (t):\n", t)

	# Cộng 2 tensor theo từng phần tử
	print("a + b:\n", a + b)

	# Nhân từng phần tử (element-wise multiply)
	print("a * b (element-wise):\n", a * b)

	# Nhân ma trận (matrix multiplication)
	print("a @ b (matrix multiply):\n", tf.matmul(a, b))


def simple_linear_regression():
	# Dữ liệu mẫu: y = 2x + 1
	x = tf.constant([1.0, 2.0, 3.0, 4.0])
	y_true = tf.constant([3.0, 5.0, 7.0, 9.0])

	# 2 tham số cần học của mô hình y = w*x + b
	w = tf.Variable(0.0)
	b = tf.Variable(0.0)

	# Tốc độ học (mỗi bước cập nhật tham số bao nhiêu)
	learning_rate = 0.1

	for step in range(1, 101):
		# Ghi lại phép tính để TensorFlow tự tính gradient
		with tf.GradientTape() as tape:
			# Dự đoán từ mô hình hiện tại
			y_pred = w * x + b

			# Hàm mất mát MSE = trung bình bình phương sai số
			loss = tf.reduce_mean(tf.square(y_pred - y_true))

		# Tính đạo hàm của loss theo w và b
		dw, db = tape.gradient(loss, [w, b])

		# Cập nhật tham số theo gradient descent
		w.assign_sub(learning_rate * dw)
		b.assign_sub(learning_rate * db)

		# In log mỗi 20 bước để theo dõi quá trình học
		if step % 20 == 0:
			print(f"Step {step:3d} | loss={loss.numpy():.4f} | w={w.numpy():.4f} | b={b.numpy():.4f}")

	# Kết quả cuối cùng: w gần 2, b gần 1
	print("\nFinal model: y = w*x + b")
	print(f"w = {w.numpy():.4f}, b = {b.numpy():.4f}")


if __name__ == "__main__":
	print("=== TensorFlow Tensor Basics ===")
	tensor_basics()

	print("\n=== Simple Linear Regression ===")
	simple_linear_regression()
