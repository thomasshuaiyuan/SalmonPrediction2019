import numpy as np
from pandas import read_csv
from datetime import datetime



# load data
#def parse(x):
	#return datetime.strptime(x, '%Y %m %d %H')
dataset = read_csv('SalmonPredictionData3.csv' , index_col=1 )
dataset.drop('ID', axis=1, inplace=True)
dataset.drop('Place', axis=1, inplace=True)

# manually specify column names
dataset.columns = ['Estimation', 'Temperature', 'Water Disharge' , 'Precipitation', 'WaterTemperature']
dataset.index.name = 'date'

# mark all NA values with 0
dataset['Estimation'].fillna(0, inplace=True)

# drop the first 24 hours

#dataset = dataset[24:]

# summarize first 5 rows
print(dataset.head(5))

# save to file
dataset.to_csv('Salmon.csv')

from math import sqrt
from numpy import concatenate
from matplotlib import pyplot
from pandas import read_csv
from pandas import DataFrame
from pandas import concat
from sklearn.preprocessing import MinMaxScaler
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import mean_squared_error
from sklearn.metrics import mean_absolute_error
from keras.models import Sequential
from keras.layers import Dense
from keras.layers import LSTM
 
# convert series to supervised learning
def series_to_supervised(data, n_in=1, n_out=1, dropnan=True):
	n_vars = 1 if type(data) is list else data.shape[1]
	df = DataFrame(data)
	cols, names = list(), list()
	# input sequence (t-n, ... t-1)d
	for i in range(n_in, 0, -1):
		cols.append(df.shift(i))
		names += [('var%d(t-%d)' % (j+1, i)) for j in range(n_vars)]
	# forecast sequence (t, t+1, ... t+n)
	for i in range(0, n_out):
		cols.append(df.shift(-i))
		if i == 0:
			names += [('var%d(t)' % (j+1)) for j in range(n_vars)]
		else:
			names += [('var%d(t+%d)' % (j+1, i)) for j in range(n_vars)]
	# put it all together
	agg = concat(cols, axis=1)
	agg.columns = names
	# drop rows with NaN values
	if dropnan:
		agg.dropna(inplace=True)
	return agg
 
# load dataset
dataset = read_csv('Salmon.csv', header=0, index_col=0)
values = dataset.values

# integer encode direction
#encoder = LabelEncoder()
#values[:,4] = encoder.fit_transform(values[:,4])
# ensure all data is float
values = values.astype('float32')

# normalize features
scaler = MinMaxScaler(feature_range=(0, 1))

#scale from (any number) to between 0 and 1
scaled = scaler.fit_transform(values)

# frame as supervised learning
reframed = series_to_supervised(scaled, 1, 1)
print(reframed)

# drop columns we don't want to predict
reframed.drop(reframed.columns[[6,7,8,9]], axis=1, inplace=True)
print(reframed.head())
 
# split into train and test sets
values = reframed.values
n_train_hours = 44
train = values[:n_train_hours, :]
test = values[n_train_hours:, :]

# split into input and outputs
train_X, train_y = train[:, :-1], train[:, -1]
test_X, test_y = test[:, :-1], test[:, -1]

# reshape input to be 3D [samples, timesteps, features]
train_X = train_X.reshape((train_X.shape[0], 1, train_X.shape[1]))
test_X = test_X.reshape((test_X.shape[0], 1, test_X.shape[1]))
print(train_X.shape, train_y.shape, test_X.shape, test_y.shape)
 
# design network
model = Sequential()
model.add(LSTM(200, input_shape=(train_X.shape[1], train_X.shape[2])))
model.add(Dense(1))
model.compile(loss='mae', optimizer='adam')

# fit network

history = model.fit(train_X, train_y, epochs=200, validation_data=(test_X, test_y), verbose=2, shuffle=False)
# plot history
pyplot.plot(history.history['loss'], label='train')

pyplot.plot(history.history['val_loss'], label='test')
pyplot.ylabel('mean percent error')
pyplot.xlabel('number of epochs')
pyplot.legend()
pyplot.show()

# make a prediction
yhat = model.predict(test_X)
test_X = test_X.reshape((test_X.shape[0], test_X.shape[2]))

# invert scaling for forecast
inv_yhat = concatenate((yhat, test_X[:, 1:]), axis=1)
inv_yhat = scaler.inverse_transform(inv_yhat)
inv_yhat = inv_yhat[:,0]

# invert scaling for actual
test_y = test_y.reshape((len(test_y), 1))
inv_y = concatenate((test_y, test_X[:, 1:]), axis=1)
inv_y = scaler.inverse_transform(inv_y)
inv_y = inv_y[:,0]

# calculate RMSE
#rmse = sqrt(mean_squared_error(inv_y, inv_yhat))
mape = np.mean(np.abs((inv_y - inv_yhat)/inv_y))*100
print('Test MAPE: %.3f' % mape)
print(inv_yhat)

pyplot.subplot(223)

pyplot.plot(inv_yhat, label="Actual")
pyplot.legend(bbox_to_anchor=(1.05, 1), loc=2, borderaxespad=0.)

pyplot.show()

fig = pyplot.figure()
pyplot.plot(x, np.log(x), '.')

fig2 = pyplot.figure()
pyplot.plot(a, np.log(a), '.');

# Analyzing the results of the prediction

count = 0 
for i in range(0, len(inv_yhat)):
    if int(inv_yhat[i]) > 2:
        print("The Salmon population is exceeding the normal average return")
        print(inv_yhat[i])
        count = count + 1
print("There are" + " " + str(count) + " " + "years where the Salmon population was predicted to be normal")


# Root Mean Squared Error(RMSE) Look this up











































































































































































































































































































