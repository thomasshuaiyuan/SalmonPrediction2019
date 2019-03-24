#Artificial Neural  Network

#Installing Theano
#Pip install --upgrade--no deps git+git://github.com/Theano/Theano.git

#Installing TensorFlow 

#Install Tensorflow from the website:https://www.tensorflow.org/versions/roll/get-started

#Installing Keras

#pip install

import numpy as np
import matplotlib.pyplot as plt
import pandas as pd


# Importing the dataset
dataset = pd.read_csv('TestCase1.csv')
X = dataset.iloc[:, 0:3].values
y = dataset.iloc[:, 3].values


# Encoding categorical data
#from sklearn.preprocessing import LabelEncoder, OneHotEncoder
#labelencoder_X_1 = LabelEncoder()
#X[:, 1] = labelencoder_X_1.fit_transform(X[:, 1])
#labelencoder_X_2 = LabelEncoder()
#X[:, 2] = labelencoder_X_2.fit_transform(X[:, 2])
#onehotencoder = OneHotEncoder(categorical_features = [1])
#X = onehotencoder.fit_transform(X).toarray()
#X = X[:, 1:]


# Splitting the dataset into the Training set and Test set
from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.2 , random_state = 0)

print(X_test)
# Feature Scaling takes maximum value and minimal value and fit it into a range of 0 and 1

from sklearn.preprocessing import StandardScaler
sc = StandardScaler()
X_train = sc.fit_transform(X_train)
X_test = sc.transform(X_test)
print(X_test)
print(y_test)
X_test = np.array([(1, 1, 1),(0,0,0)])
print(X_test)
y_test = np.array([1,0])
# Part 2 - Now let's make the ANN!

# Importing the Keras libraries and packages
import keras
from keras.models import Sequential
from keras.layers import Dense

from keras.utils import to_categorical
#y_binary = to_categorical(y)



# Initialising the ANN
classifier = Sequential()

# Adding the input layer and the first hidden layer
classifier.add(Dense(output_dim = 6, init = 'uniform', input_dim = 3))

# Adding the second hidden layer
#classifier.add(Dense(output_dim = 6, init = 'uniform') )

# Adding the output layer
classifier.add(Dense(output_dim = 1, init = 'uniform', activation = 'sigmoid'))

# Compiling the ANN
#classifier.compile(optimizer = 'adam', loss = 'mean_squared_error', metrics = ['mse'])

classifier.compile(optimizer = 'adam', loss = 'binary_crossentropy', metrics = ['accuracy'])


# Fitting the ANN to the Training set
classifier.fit(X_train, y_train, batch_size = 10, nb_epoch = 100)


# Part 3 - Making the predictions and evaluating the model



# Predicting the Test set results
y_pred = classifier.predict(X_test)
y_pred = (y_pred > 0.5)



# Making the Confusion Matrix
#from sklearn.metrics import confusion_matrix
#cm = confusion_matrix(y_test, y_pred)
#print(cm)



#Testing the accuracy
from sklearn.metrics import accuracy_score
results = accuracy_score(y_test, y_pred)
print(results)


from matplotlib import animation

def barlist(n): 
    return [1/float(n*k) for k in range(1,6)]

fig=plt.figure()

n=100 #Number of frames
#x=range(1,6)

barcollection = plt.bar(inv_yhat, barlist(1))

def animate(i):
    y=barlist(i+1)
    for i, b in enumerate(barcollection):
        b.set_height(y[i])

anim=animation.FuncAnimation(fig,animate,repeat=False,blit=False,frames=n,
                             interval=100)

anim.save('mymovie.mp4',writer=animation.FFMpegWriter(fps=10))
plt.show()


