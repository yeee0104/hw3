import streamlit as st
import numpy as np
import plotly.graph_objects as go
from sklearn.svm import LinearSVC

st.title("3D Scatter Plot with Separating Hyperplane")

# 生成資料點的數量和距離閾值滑桿
num_points = 600
distance_threshold = st.slider("Distance Threshold for Classification", 0.1, 10.0, 4.0, step=0.1)

# 生成高斯分布的隨機數據
np.random.seed(42)
mean = 0
variance = 10
c1_x = np.random.normal(mean, np.sqrt(variance), num_points)
c1_y = np.random.normal(mean, np.sqrt(variance), num_points)
distances = np.sqrt(c1_x**2 + c1_y**2)
Y = np.where(distances < distance_threshold, 0, 1)

# 定義高斯函數作為第三維度
def gaussian_function(x1, x2):
    return np.exp(-0.1 * (x1**2 + x2**2))

x3 = gaussian_function(c1_x, c1_y)

# 特徵矩陣和標籤
X = np.column_stack((c1_x, c1_y, x3))

# 使用線性 SVM 訓練模型
svc = LinearSVC()
svc.fit(X, Y)

# 獲取 SVM 的係數和截距
coef = svc.coef_[0]
intercept = svc.intercept_[0]

# 創建 3D 散點圖和分隔超平面
fig = go.Figure()

# 類別 0 的點
fig.add_trace(go.Scatter3d(
    x=X[Y == 0, 0], y=X[Y == 0, 1], z=X[Y == 0, 2],
    mode='markers', marker=dict(size=5, color='blue'),
    name='Y=0'
))

# 類別 1 的點
fig.add_trace(go.Scatter3d(
    x=X[Y == 1, 0], y=X[Y == 1, 1], z=X[Y == 1, 2],
    mode='markers', marker=dict(size=5, color='red'),
    name='Y=1'
))

# 分隔超平面
xx, yy = np.meshgrid(np.linspace(-10, 10, 10), np.linspace(-10, 10, 10))
zz = (-coef[0] * xx - coef[1] * yy - intercept) / coef[2]
fig.add_trace(go.Surface(x=xx, y=yy, z=zz, showscale=False, opacity=0.5, colorscale='gray'))

# 更新圖表配置
fig.update_layout(scene=dict(
    xaxis_title="X1",
    yaxis_title="X2",
    zaxis_title="X3"
), title="3D Scatter Plot with Y Color and Separating Hyperplane")

# 顯示 3D 圖表
st.plotly_chart(fig)