import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
df=pd.read_csv('fuelcons.csv')
df.columns

#x=df.ENGINESIZE
#y=df.CO2EMISSIONS
#plt.scatter(x,y)
#plt.show()

#histograms
df.hist()
plt.show()

#generate train and test data

msk=np.random.rand(len(df)) <0.8
train=df[msk]
test=df[~msk]
train_x=np.asanyarray(train[['ENGINESIZE','CYLINDERS','FUELCONSUMPTION_COMB']])
test_x=np.asanyarray(test[['ENGINESIZE','CYLINDERS','FUELCONSUMPTION_COMB']])
train_y=np.asanyarray(train[['CO2EMISSIONS']])
test_y=np.asanyarray(test[['CO2EMISSIONS']])

#model train data
from sklearn import linear_model
regr = linear_model.LinearRegression()
regr.fit(train_x,train_y)
print('coef:',regr.coef_)
print('intercept:',regr.intercept_)

plt.scatter(train[['ENGINESIZE']],train_y,color='red',label='Engine size')
plt.scatter(train[['CYLINDERS']],train_y,color='blue',label='Cylinders')
plt.scatter(train[['FUELCONSUMPTION_COMB']],train_y,color='green',label='Fuel consumption')
plt.ylabel('CO2 Emission')
plt.legend()
#plt.plot(train[['FUELCONSUMPTION_COMB']],(train[['ENGINESIZE']]*10.85421)+(train[['CYLINDERS']]*7.6)+(train[['FUELCONSUMPTION_COMB']]*9.7)+63.6)
plt.savefig('training-data.png', dpi=300)
plt.show()

#evaluation

#mean square error

np.mean((regr.predict(test_x)-test_y)**2)

from sklearn.metrics import r2_score
r2_score(test_y,regr.predict(test_x))

