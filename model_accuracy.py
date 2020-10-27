import numpy as np
import os
from tensorflow.keras.models import load_model
from sklearn.preprocessing import StandardScaler
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Activation


# Define the model architecture 
model = Sequential()
model.add(Dense(90, input_shape=(11,)))
model.add(Activation('tanh'))

for laye in range(1,6):
    model.add(Dense(90))
    model.add(Activation('tanh'))

model.add(Dense(1))
model.add(Activation('linear'))

# Load weights from already trained model
model.load_weights("model.h5")
print("Loaded model from disk")

# Print model summary
model.summary()

# Load training and testing data
X_train 	= 	np.load('data_set' + os.sep + 'X_train.npy')
X_test 		= 	np.load('data_set' + os.sep + 'X_test.npy')
y_train 	= 	np.load('data_set' + os.sep + 'y_train.npy')
y_test 		= 	np.load('data_set' + os.sep + 'y_test.npy')

# Standard scaling before providing it as input to the model
scaler 		= 	StandardScaler()
X_train 	= 	scaler.fit_transform(X_train)
X_test 		= 	scaler.transform(X_test)

# Calculate training and testing error
train_error = 	100*abs(model.predict(X_train) - y_train)/y_train
test_error 	= 	100*abs(model.predict(X_test) - y_test)/y_test

acc = np.sum(train_error<10)/len(X_train)

print('Training Accuracy: {0:0.2f}'.format(acc*100) + '% to predict withiin 10% of actual value')

acc = np.sum(test_error<10)/len(X_test)

print('Testing Accuracy: {0:0.2f}'.format(acc*100) + '% to predict withiin 10% of actual value')
