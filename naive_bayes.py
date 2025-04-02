import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import GaussianNB

path = 'data/iris2.data'
iris = pd.read_csv(path)  # 读取csv文件

# print(iris.head())
#
# print(iris.shape)
#
# print(iris.columns)

X_iris = iris.drop(['species'], axis=1)
# print(X_iris.head())

y_iris = np.ravel(iris[['species']])
# print(y_iris)

# print(y_iris.shape)

X_train, X_test, y_train, y_test = train_test_split(X_iris, y_iris, random_state=1)  # 将数据分为训练集，测试集
# print(X_train.head())    # 获取数据前5行
#
# print(X_train.shape)
# print(X_test.shape)
# print(y_train.shape)
# print(y_test.shape)
#
# print(y_train)



gb = GaussianNB()
model_GaussinaNB = gb.fit(X_train, y_train)
# predict(X)：直接输出测试集预测的类标记,X_test为测试集
y_predict_GaussianNB = model_GaussinaNB.predict(X_test)
# print("y_predict_GaussianNB", y_predict_GaussianNB)

z_data = {'sepal_length': ['5'],'sepal_width': ['3'],'petal_length': ['3'],'petal_width': ['1.8']}
Z_data = pd.DataFrame(z_data, columns=['sepal_length', 'sepal_width', 'petal_length', 'petal_width'])
# print(Z_data)

Z_model_predict = model_GaussinaNB.predict(Z_data)
# print('Z_model_predict', Z_model_predict)

# print(y_predict_GaussianNB == y_test)

y_test_mean = np.mean(y_predict_GaussianNB == y_test)
print('y_test_GaussianNB_mean', y_test_mean)

print(gb.score(X_train, y_train))


