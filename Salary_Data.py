import numpy as np
import pandas as pd


df = pd.read_csv("D:\\DS\\books\\ASSIGNMENTS\\Simple Linear Regression\\Salary_Data.csv")
df
df.head()
df.shape
df.isnull().sum()
df.describe()

# Step - 2 Split the variables in X and Y
Y = df[["Salary"]]
X = df[["YearsExperience"]]

#EDA
 # Scatter Plot
import matplotlib.pyplot as plt
plt.scatter (X.iloc[:,0],Y,color = 'red')
plt.xlabel("YearsExperience")
plt.ylabel("Salary")
plt.show()

# Boxplot
plt.boxplot(df["YearsExperience"])
plt.boxplot(df["Salary"])

# Histogram
plt.hist(df["YearsExperience"],bins=5)
plt.hist(df["Salary"],bins=5)

df.corr()
# Model Fitting
from sklearn.linear_model import LinearRegression
LR = LinearRegression()
LR.fit(X,Y)
LR.intercept_ #Bo
LR.coef_ #B1

# Predict the value
Y_pred = LR.predict(X)
Y_pred

# Scatter Plot with Plot
plt.scatter (X.iloc[:,0],Y,color = 'red')
plt.plot (X.iloc[:,0],Y_pred,color = 'blue')
plt.xlabel("YearsExperience")
plt.ylabel("Salary")
plt.show()

# Then Finding Error
from sklearn.metrics import mean_squared_error,r2_score
mse = mean_squared_error(Y, Y_pred)
RMSE = np.sqrt(mse)
print("Root mean square error: ", RMSE.round(3))
print("Rsquare: ", r2_score(Y, Y_pred).round(3)*100)

"""
There is a Difference of RMSE is 5592.04 and the R2 is 95.7
"""

# Transformation
# Model 2
from sklearn.linear_model import LinearRegression
LR=LinearRegression()
LR.fit(np.log(X),Y)
y1 = LR.predict(np.log(X))

mse= mean_squared_error(Y, y1)
RMSE=np.sqrt(mse).round(3)
print("Root mean square error: ", RMSE)
print("Rsquare: ", r2_score(Y, y1).round(3)*100)

plt.scatter (X.iloc[:,0],Y,color = 'red')
plt.plot (X.iloc[:,0],y1,color = 'blue')
plt.xlabel("YearsExperience")
plt.ylabel("Salary")
plt.show()

"""
There is a Difference of RMSE is 10302.894 and the R2 is 85.399
"""

# Model 3
from sklearn.linear_model import LinearRegression
LR=LinearRegression()
LR.fit(np.sqrt(X),Y)
y1 = LR.predict(np.sqrt(X))

mse= mean_squared_error(Y, y1)
RMSE=np.sqrt(mse).round(3)
print("Root mean square error: ", RMSE)
print("Rsquare: ", r2_score(Y, y1).round(3)*100)

plt.scatter (X.iloc[:,0],Y,color = 'red')
plt.plot (X.iloc[:,0],y1,color = 'blue')
plt.xlabel("YearsExperience")
plt.ylabel("Salary")
plt.show()

"""
There is a Difference of RMSE is 7080.096 and the R2 is 93.10
"""

# Model 4
from sklearn.linear_model import LinearRegression
LR=LinearRegression()
LR.fit(X**2,Y)
y1 = LR.predict(X**2)

mse= mean_squared_error(Y, y1)
RMSE=np.sqrt(mse).round(3)
print("Root mean square error: ", RMSE)
print("Rsquare: ", r2_score(Y, y1).round(3)*100)

plt.scatter (X.iloc[:,0],Y,color = 'red')
plt.plot (X.iloc[:,0],y1,color = 'blue')
plt.xlabel("YearsExperience")
plt.ylabel("Salary")
plt.show()

"""
There is a Difference of RMSE is 7843.471 and the R2 is 91.5
"""

# Model 5

from sklearn.linear_model import LinearRegression
LR=LinearRegression()
LR.fit(X**3,Y)
y1 = LR.predict(X**3)

mse= mean_squared_error(Y, y1)
RMSE=np.sqrt(mse).round(3)
print("Root mean square error: ", RMSE)
print("Rsquare: ", r2_score(Y, y1).round(3)*100)

plt.scatter (X.iloc[:,0],Y,color = 'red')
plt.plot (X.iloc[:,0],Y_pred,color = 'blue')
plt.xlabel("YearsExperience")
plt.ylabel("Salary")
plt.show()

"""
There is a Difference of RMSE is 10973.906 and the R2 is 83.399
"""

"""
Inference :  A prediction model is built and the best model selected is model 1 since its r2score is 95%
"""










