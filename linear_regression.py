import matplotlib
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
import numpy as np
import sys
matplotlib.use('TkAgg')

path = 'data/Advertising.csv'
data = pd.read_csv(path)

# print(data.head(10))

# print(data.shape)

# x = data[['TV', 'Radio', 'Newspaper']]
y = data['Sales']

# plt.figure(figsize=(9, 12))
# plt.subplot(311)
# plt.plot(data['TV'], y, 'ro')
# plt.title('TV')
# plt.grid()
# plt.subplot(312)
# plt.plot(data['Radio'], y, 'b*')
# plt.title('Radio')
# plt.grid()
# plt.subplot(313)
# plt.plot(data['Newspaper'], y, 'g^')
# plt.title('Newspaper')
# plt.grid()
# plt.show()

x = data[['TV', 'Radio']]

x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.3, random_state=23)

lr = LinearRegression(n_jobs=-1)
model = lr.fit(x_train, y_train)

# print(lr.intercept_)
# print(lr.coef_)

y_pred = model.predict(x_test)

# mse = mean_squared_error(y_test, y_pred)
# print("MSE  : ",mse)
# print("RMSE :" ,np.sqrt(mse))

# 使用mse、mae、rmse、R-squared评估模型
# mse = mean_squared_error(y_test, y_pred)
# mae = mean_absolute_error(y_test, y_pred)
# rmse = np.sqrt(mse)
# r2 = r2_score(y_test, y_pred)
#
# print("MSE: {:.16f}".format(mse))
# print("MAE: {:.16f}".format(mae))
# print("RMSE: {:.16f}".format(rmse))
# print("R-squared: {:.16f}".format(r2))

plt.figure()
plt.plot(range(len(y_pred)), y_pred, 'b', label='predict')
plt.plot(range(len(y_test)), y_test, 'r', label='test')
plt.legend(loc='upper right')
plt.xlabel("the num of sales")
plt.ylabel("value of sales")
plt.title("sales real with pred")
plt.show()

