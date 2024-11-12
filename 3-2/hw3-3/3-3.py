import streamlit as st
import numpy as np
import plotly.graph_objects as go
from sklearn.svm import LinearSVC

st.title("3D Scatter Plot with Rectangular Distribution and SVM Hyperplane")

# 定義矩形的寬度和高度
width = st.slider("Rectangle Width", 0.1, 10.0, 5.0, step=0.1)  # 矩形的寬度
height = st.slider("Rectangle Height", 0.1, 10.0, 3.0, step=0.1)  # 矩形的高度

# 設定距離閾值的範圍，使其在合理範圍內
min_distance_threshold = 0.1
max_distance_threshold = min(width, height)
distance_threshold = st.slider("Distance Threshold for Classification", 
                               min_distance_threshold, max_distance_threshold, 
                               (min_distance_threshold + max_distance_threshold) / 2, 
                               step=0.1)

# 生成矩形範圍內的隨機點
np.random.seed(42)
num_points = 600
c1_x = np.random.uniform(-width / 2, width / 2, num_points)  # x 範圍
c1_y = np.random.uniform(-height / 2, height / 2, num_points)  # y 範圍

# 計算每個點到原點的距離，並進行分類
distances = np.sqrt(c1_x**2 + c1_y**2)
Y = np.where(distances < distance_threshold, 0, 1)

# 定義高斯函數來生成第三維度
def gaussian_function(x1, x2):
    return np.exp(-0.1 * (x1**2 + x2**2))

x3 = gaussian_function(c1_x, c1_y)

# 特徵矩陣和標籤
X = np.column_stack((c1_x, c1_y, x3))

# 使用線性 SVM 訓練模型
svc = LinearSVC()
svc.fit(X, Y)

# 獲取 SVM 模型的係數和截距
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
xx, yy = np.meshgrid(np.linspace(-width / 2, width / 2, 10), 
                     np.linspace(-height / 2, height / 2, 10))
zz = (-coef[0] * xx - coef[1] * yy - intercept) / coef[2]
fig.add_trace(go.Surface(x=xx, y=yy, z=zz, showscale=False, opacity=0.5, colorscale='gray'))

# 更新圖表配置
fig.update_layout(scene=dict(
    xaxis_title="X1",
    yaxis_title="X2",
    zaxis_title="X3"
), title="3D Scatter Plot with Rectangular Distribution and SVM Hyperplane")

# 顯示 3D 圖表
st.plotly_chart(fig)
