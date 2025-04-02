import numpy as np
import pandas as pd
from sklearn import svm
import matplotlib.colors
import matplotlib.pyplot as plt
from PIL import Image
from sklearn.metrics import accuracy_score
import os
from sklearn.model_selection import train_test_split
from sklearn.model_selection import GridSearchCV
from time import time
matplotlib.use('TkAgg')

# 使用 pandas 加载数据
train_data = pd.read_csv('data/optdigits.tra', header=None)
test_data = pd.read_csv('data/optdigits.tes', header=None)

# 将数据分割为特征和标签
x = train_data.iloc[:, :-1].values
y = train_data.iloc[:, -1].values
x_test = test_data.iloc[:, :-1].values
y_test = test_data.iloc[:, -1].values

# 将特征重塑为图像
images = x.reshape(-1, 8, 8)
images_test = x_test.reshape(-1, 8, 8)

# 将数据分割为训练集、验证集和测试集
x_train, x_val, y_train, y_val = train_test_split(x, y, test_size=0.4, random_state=1)

# 定义 GridSearchCV 的参数网格
params = {'C': [0.1, 1, 10, 100, 1000], 'gamma': [1e-5, 1e-4, 1e-3, 1e-2, 1e-1, 1]}

# 创建并训练模型（使用 GridSearchCV）
model = GridSearchCV(svm.SVC(kernel='rbf'), param_grid=params, cv=3)
print('开始学习...')
t0 = time()
model.fit(x_train, y_train)
t1 = time()
t = t1 - t0
print('训练+cv耗时：%d 分钟 %.3f 秒' % (int(t / 60), t - 60 * int(t / 60)))

# 打印最佳参数
print('最佳参数：\t', model.best_params_)

# 计算并打印准确率
print('学习完成...')
train_accuracy = accuracy_score(y_train, model.predict(x_train))
val_accuracy = accuracy_score(y_val, model.predict(x_val))
test_accuracy = accuracy_score(y_test, model.predict(x_test))
print('训练集准确率：', train_accuracy)
print('验证集准确率：', val_accuracy)
print('测试集准确率：', test_accuracy)

# 显示图像样例
fig, axes = plt.subplots(nrows=4, ncols=8, figsize=(12, 6))
for ax, image, label in zip(axes.flat, images[:16], y[:16]):
    ax.imshow(image, cmap='gray')
    ax.set_title('train image: %i' % label)
    ax.axis('off')

for ax, image, label in zip(axes.flat[16:], images_test[:16], y_test[:16]):
    ax.imshow(image, cmap='gray')
    ax.set_title('test image: %i' % label)
    ax.axis('off')

plt.tight_layout()
plt.show()

# data = np.loadtxt('data/optdigits.tra', dtype=float, delimiter=',')
# # print(data.shape)
#
# x, y = np.split(data, (-1, ), axis=1)
# # print(x.shape, y.shape)
#
# y = y.ravel().astype(int)
#
# # print(np.unique(y))
#
# images = x.reshape(-1, 8, 8)
# # print(images.shape)
#
# print('Loading...')
# data = np.loadtxt('data/optdigits.tes', dtype=float, delimiter=',')
# x_test, y_test = np.split(data, (-1, ), axis=1)
# print(y_test.shape)
# images_test = x_test.reshape(-1, 8, 8)
# y_test = y_test.ravel().astype(int)
# print('finish')
#
# x, x_test, y, y_test = train_test_split(x, y, test_size=0.4, random_state=1)
#
# for index, image in enumerate(images[:16]):
#     plt.subplot(4,8,index+1)
#     plt.imshow(image, cmap=plt.cm.gray_r,interpolation='nearest')
#     plt.title(u'train_images: %i' %y[index])
# for index, image in enumerate(images_test[:16]):
#     plt.subplot(4,8,index+17)
#     plt.imshow(image,cmap=plt.cm.gray_r,interpolation='nearest')
#     plt.title(u'test_images: %i' % y_test[index])
#     plt.tight_layout()
# plt.show()
#
# params = {'C': np.logspace(0, 3, 7), 'gamma': np.logspace(-5, 0, 11)}
#
# model = GridSearchCV(svm.SVC(kernel='rbf'), param_grid=params, cv=3)
#
# print('Start Learning...')
# t0 = time()
# model.fit(x, y)
#
# t1 = time()
# t = t1 - t0
# print('训练+CV耗时：%d分钟%.3f秒' % (int(t/60), t - 60*int(t/60)))
#
#
# print('最优参数：\t', model.best_params_)
#
# print('Learning is OK...')
# print('训练集准确率：', accuracy_score(y, model.predict(x)))
# y_hat = model.predict(x_test)
# print('测试集准确率：', accuracy_score(y_test, model.predict(x_test)))
# print(y_hat)
# print(y_test)
