import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
import matplotlib.pyplot as plt

# 生成 300 個範圍在 [0, 1000] 的隨機整數
np.random.seed(0)
X = np.random.randint(0, 1001, 300)

# 根據範圍 [500, 800] 為數據分配二分類標籤
Y = np.where((X >= 500) & (X <= 800), 1, 0)

# 將數據集分為訓練集和測試集
X_train, X_test, Y_train, Y_test = train_test_split(X.reshape(-1, 1), Y, test_size=0.2, random_state=42)

# 訓練 Logistic Regression 模型
logreg = LogisticRegression()
logreg.fit(X_train, Y_train)
y1 = logreg.predict(X_test)

# 訓練 SVM 模型
svm = SVC()
svm.fit(X_train, Y_train)
y2 = svm.predict(X_test)

# 按 X 值升序排列測試集數據
sorted_indices = np.argsort(X_test.flatten())
X_test_sorted = X_test[sorted_indices]
Y_test_sorted = Y_test[sorted_indices]
y1_sorted = y1[sorted_indices]
y2_sorted = y2[sorted_indices]

# 繪製圖形
plt.figure(figsize=(12, 6))

# Logistic Regression 預測結果
plt.subplot(1, 2, 1)
plt.scatter(X_test_sorted, Y_test_sorted, color='blue', marker='o', label='True')
plt.scatter(X_test_sorted, y1_sorted, color='green', marker='x', label='Logistic Regression Prediction')
plt.plot(X_test_sorted, y1_sorted, color='green', linestyle='--')
plt.xlabel('X')
plt.ylabel('y')
plt.title('Logistic Regression Prediction')
plt.legend()

# SVM 預測結果
plt.subplot(1, 2, 2)
plt.scatter(X_test_sorted, Y_test_sorted, color='blue', marker='o', label='True')
plt.scatter(X_test_sorted, y2_sorted, color='red', marker='s', label='SVM Prediction')
plt.plot(X_test_sorted, y2_sorted, color='red', linestyle='--')
plt.xlabel('X')
plt.ylabel('y')
plt.title('SVM Prediction')
plt.legend()

plt.tight_layout()
plt.show()
