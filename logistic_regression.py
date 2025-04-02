import matplotlib as mpl
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
from sklearn.preprocessing import StandardScaler, PolynomialFeatures
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline
from mpl_toolkits.mplot3d import Axes3D
from matplotlib.colors import ListedColormap
mpl.use('TkAgg')

path = 'data/iris.data'
data = pd.read_csv(path)

x = data.values[:, :-1]
y = data.values[:, -1]

print("x's shape", x.shape)
print("y's shape", y.shape)

y = pd.Categorical(y).codes
# Categorical()计算列表型数据中的类别；codes 将类别信息转化成数值信息

# 特征选择
x = x[:, :3]

lr = Pipeline([('sc', StandardScaler()), # 数据标准化
               ('poly', PolynomialFeatures(degree=3)), # 添加多项式特征
               ('clf', LogisticRegression(penalty='l1', solver='liblinear', C=0.1))]) # L1正则化
# StandardScaler()标准化处理；PolynomialFeatures 映射到 n 维多项式特征集, degree参数用来控制多项式的度；LogisticRegression()逻辑回归模型

model = lr.fit(x, y)

y_pred = model.predict(x)

np.set_printoptions(suppress=True)
# 设置输出的精度, suppress 是否压缩由科学计数法表示的浮点数

accuracy = 100 * np.mean(y_pred == y)
print("准确度：%.2f%%" % accuracy)

# 绘制决策边界和散点图
fig = plt.figure(facecolor='w')
ax = fig.add_subplot(111, projection='3d')

# 生成网格点坐标
N, M = 100, 100
x1_min, x1_max = x[:, 0].min(), x[:, 0].max()
x2_min, x2_max = x[:, 1].min(), x[:, 1].max()
x3_min, x3_max = x[:, 2].min(), x[:, 2].max()
t1 = np.linspace(x1_min, x1_max, N)
t2 = np.linspace(x2_min, x2_max, M)
t1, t2 = np.meshgrid(t1, t2)
t3 = np.zeros_like(t1)
x_test = np.stack((t1.flat, t2.flat, t3.flat), axis=1)
y_hat = model.predict(x_test)
y_hat = y_hat.reshape(t1.shape)
# 绘制决策边界
ax.plot_surface(t1, t2, t3, rstride=1, cstride=1, cmap=plt.cm.binary, alpha=0.5)
# 设置不同类别的颜色
colors = ['#77E0A0', '#FF8080', '#A0A0FF']
cmap = ListedColormap(colors[:len(np.unique(y))])

# 绘制散点图
for i, color in zip(np.unique(y), colors):
    ax.scatter(x[y == i, 0], x[y == i, 1], x[y == i, 2], c=color, edgecolors='k', s=50)

# 调整坐标范围
ax.set_xlim(x1_min, x1_max)
ax.set_ylim(x2_min, x2_max)
ax.set_zlim(x3_min, x3_max)
ax.set_xlabel(u'calyx_length', fontsize=14)
ax.set_ylabel(u'calyx_width', fontsize=14)
ax.set_zlabel(u'petal_length', fontsize=14)
plt.title(u'Iris Logistic graph - optimized', fontsize=17)
plt.show()

# N, M = 500, 500 # 横纵各采样 500 个值
# x1_min, x1_max = x[:, 0].min(), x[:, 0].max() # 第 0 列的范围
# x2_min, x2_max = x[:, 1].min(), x[:, 1].max() # 第 1 列的范围
#
# t1 = np.linspace(x1_min, x1_max, N)
# # linspace()生成等间隔数列；x1_min 作为数列的开头；x1_max 作为结尾，N 作为数列元素的个数
# t2 = np.linspace(x2_min, x2_max, M)
# x1, x2 = np.meshgrid(t1, t2) # 生成网格坐标点
# x_test = np.stack((x1.flat, x2.flat), axis=1)
# # 测试点；flat 方法，返回数组的 flatiter 迭代器；
# cm_light = mpl.colors.ListedColormap(['#77E0A0', '#FF8080', '#A0A0FF'])
# cm_dark = mpl.colors.ListedColormap(['g', 'r', 'b'])
# y_hat = model.predict(x_test) # 预测值
# y_hat = y_hat.reshape(x1.shape) # 使之与输入的形状相同
# plt.figure(facecolor='w') # 可设置控制 dpi、边界颜色、图形大小、和子区( subplot)
# plt.pcolormesh(x1, x2, y_hat, cmap=cm_light) # 预测值的分类显示
# plt.scatter(x[:, 0], x[:, 1], c=y, edgecolors='k', s=50, cmap=cm_dark)
#
# # 样本的显示
# plt.xlabel(u'calyx_length', fontsize=14)
# plt.ylabel(u'calyx_width', fontsize=14)
#
# # 来调整 x,y 坐标范围
# plt.xlim(x1_min, x1_max)
# plt.ylim(x2_min, x2_max)
# plt.grid()
#
# patchs = [mpatches.Patch(color='#77E0A0', label='Iris-setosa'),
#           mpatches.Patch(color='#FF8080', label='Iris-versicolor'),
#           mpatches.Patch(color='#A0A0FF', label='Iris-virginica'),]
#
# plt.legend(handles=patchs, fancybox=True, framealpha=0.8)
# plt.title(u'Iris Logistic graph - standard', fontsize=17)
# plt.show()