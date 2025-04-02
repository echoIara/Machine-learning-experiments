import matplotlib
import pandas as pd
import numpy as np
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn import metrics
from sklearn.model_selection import train_test_split
from sklearn.model_selection import GridSearchCV
from sklearn.preprocessing import label_binarize  # 将label二值化
import matplotlib as mpl
import matplotlib.pyplot as plt
matplotlib.use('TkAgg')

data = pd.read_csv('data/car.data', header=None)	 # header=None没有标题

print(data.head(2))

print(data.shape)

n_columns = len(data.columns)  # 获得数据集的列的个数
columns = ['buy', 'maintain', 'doors', 'persons', 'boot', 'safety', 'accept']  # 自定义列名
new_columns = dict(zip(np.arange(n_columns), columns))  # 将列名与整数映射
data.rename(columns=new_columns, inplace=True)  # 替换数据集中的列名为columns中的值
for col in columns:
	data[col] = pd.Categorical(data[col]).codes   # Categorical方法，获取list的类别；codes方法赋给每个类别对应的类别编码值

x = data.loc[:, columns[:-1]]  # 得到样本特征
y = data['accept']		# 取到标签值
print(x.shape)  # 结果是(1728, 6)
print(y.shape)  # 结果是(1728,)

x, x_test, y, y_test = train_test_split(x, y, test_size=0.3)  # 将x,y都切分成训练集数据、测试集数据，其中训练集占%30

# clf = DecisionTreeClassifier(criterion='gini', max_depth=12, min_samples_split=5, max_features=5)
#
# clf.fit(x, y)
#
# y_hat = clf.predict(x)  # 在训练集上进行预测
# print('训练集精确度：', metrics.accuracy_score(y, y_hat))  # 评估成绩
# y_test_hat = clf.predict(x_test)  # 测试集预测
# print('测试集精确度：', metrics.accuracy_score(y_test, y_test_hat))  # 评估
#
# n_class = len(data['accept'].unique())
# y_test_one_hot = label_binarize(y_test, classes=np.arange(n_class))	 # 将标签值映射成one-hot编码
# print(y_test_one_hot.shape)  # 结果是(519, 4)
#
# y_test_one_hot_hat = clf.predict_proba(x_test)  # 测试集预测分类的概率
# print(y_test_one_hot_hat)
#
# fpr, tpr, _ = metrics.roc_curve(y_test_one_hot.ravel(), y_test_one_hot_hat.ravel())
#
# print('Micro AUC:\t', metrics.auc(fpr, tpr)) # AUC ROC意思是ROC曲线下方的面积(Area under the Curve of ROC)
# print('Micro AUC(System):\t', metrics.roc_auc_score(y_test_one_hot, y_test_one_hot_hat, average='micro'))
# auc = metrics.roc_auc_score(y_test_one_hot, y_test_one_hot_hat, average='macro')
# print('Macro AUC:\t', auc)

# plt.figure(figsize=(8, 7), dpi=80, facecolor='w')  # dpi:每英寸长度的像素点数；facecolor 背景颜色
# plt.xlim((-0.01, 1.02))  # x,y 轴刻度的范围
# plt.ylim((-0.01, 1.02))
# plt.xticks(np.arange(0, 1.1, 0.1))  # 绘制刻度
# plt.yticks(np.arange(0, 1.1, 0.1))
#
# plt.plot(fpr, tpr, 'r-', lw=2, label='AUC=%.4f' % auc)  # 绘制AUC 曲线
# plt.legend(loc='lower right')    # 设置显示标签的位置
#
# plt.xlabel('False Positive Rate', fontsize=14)   # 绘制x,y 坐标轴对应的标签
# plt.ylabel('True Positive Rate', fontsize=14)
#
# plt.grid(visible=True, ls=':')  # 绘制网格作为底板; visible是否显示网格线；ls表示line style
#
# plt.title(u'DecisionTree ROC curve And  AUC', fontsize=18)  # 打印标题
# plt.show()

# n_estimators:设置树的个数为100，criterion:使用gini系数作为信息熵，max_depth:树的最大深度为12，min_samples_split:少于5个样本不分裂，max_features:最多使用5个特征
clf = RandomForestClassifier(n_estimators=100, criterion='gini', max_depth=12, min_samples_split=5, max_features=5)

clf.fit(x, y)

y_hat = clf.predict(x)  # 在训练集上进行预测
print('训练集精确度：', metrics.accuracy_score(y, y_hat))  # 评估成绩
y_test_hat = clf.predict(x_test)  # 测试集预测
print('测试集精确度：', metrics.accuracy_score(y_test, y_test_hat))  # 评估
n_class = len(data['accept'].unique())     # 将标签值映射成one-hot编码
y_test_one_hot = label_binarize(y_test, classes=np.arange(n_class))

y_test_one_hot_hat = clf.predict_proba(x_test)  # 测试集预测分类的概率

fpr, tpr, _ = metrics.roc_curve(y_test_one_hot.ravel(), y_test_one_hot_hat.ravel())  # 获取auc， roc 曲线需要的值

print('Micro AUC:\t', metrics.auc(fpr, tpr))
print('Micro AUC(System):\t', metrics.roc_auc_score(y_test_one_hot, y_test_one_hot_hat, average='micro'))  # 计算roc曲线下的面积
auc = metrics.roc_auc_score(y_test_one_hot, y_test_one_hot_hat, average='macro')
print('Macro AUC:\t', auc)

plt.figure(figsize=(8, 7), dpi=80, facecolor='w')  # dpi:每英寸长度的像素点数；facecolor 背景颜色
plt.xlim((-0.01, 1.02))  # x,y 轴刻度的范围
plt.ylim((-0.01, 1.02))
plt.xticks(np.arange(0, 1.1, 0.1))  # 绘制刻度
plt.yticks(np.arange(0, 1.1, 0.1))

plt.plot(fpr, tpr, 'r-', lw=2, label='AUC=%.4f' % auc)  # 绘制AUC 曲线
plt.legend(loc='lower right')    # 设置显示标签的位置

plt.xlabel('False Positive Rate', fontsize=14)   # 绘制x,y 坐标轴对应的标签
plt.ylabel('True Positive Rate', fontsize=14)

plt.grid(visible=True, ls=':')  # 绘制网格作为底板; visible是否显示网格线；ls表示line style

plt.title(u'RandomForest ROC curve And  AUC', fontsize=18)  # 打印标题
plt.show()
