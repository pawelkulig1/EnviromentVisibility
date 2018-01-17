from keras.models import Sequential
from keras.layers import Dense

import pickle
import numpy as np

#get pickled data
outputData = pickle.load(open("data.pickle", "rb")) #average outData

#data ready we have 100x50 px image now and split in frames
mariackiFrames = pickle.load(open("mariackiFrames.pickle", "rb"))
wojciechFrames = pickle.load(open("wojciechFrames.pickle", "rb"))

# print(outputData[0])

#data stored in mariackiFrames[frameNumber][rowNumber][pixelNumber]

dataForNN = []

print("concatenating...")
for i in range(len(wojciechFrames)):
	dataForNN.append(np.concatenate((mariackiFrames[i], wojciechFrames[i]), axis=0))

		# dataForNN.append(mariackiFrames[i][j] + wojciechFrames[i][j])


print(len(dataForNN), len(dataForNN[0]))

y = []

print("calculating means...")
def calculateMean(arr, count, poz):
    s = 0
    if len(arr) < count:
        count = len(arr)-1

    for i in range(count):
        s += arr[poz-i]

    return s/count

average = 50

for i in range(len(outputData[0])):
	calculation = (calculateMean(dataForNN[0], average, i) + calculateMean(dataForNN[1], average, i))/2
	if calculation >0.15:
		y.append([0,0,1])
	elif calculation > 0.8:
		y.append([0,1,0])
	else:
		y.append([1,0,0])

# for i in range(len(outputData[0])):
	
# 	if (outputData[0][i]+outputData[1][i])/2 <0.05:
# 		y.append([1,0,0])
# 	elif (outputData[0][i]+outputData[1][i])/2 <0.1:
# 		y.append([0,1,0])
# 	else:
# 		y.append([0,0,1])

y_test = np.array(y[:-100], "float32")
y = np.array(y[:-100], "float32")

test_data = np.array(dataForNN[:-100], "float32")
dataForNN = np.array(dataForNN[:-100], "float32")

print("building model...")

#NN MODEL NOW
model = Sequential()
model.add(Dense(3000, input_dim=5000, activation = "sigmoid"))
model.add(Dense(2000, input_dim=3000, activation= "sigmoid"))
model.add(Dense(3, input_dim=2000, activation="sigmoid"))

model.compile(loss="mean_squared_error", optimizer="adam", metrics=['binary_accuracy'])

model.fit(dataForNN, y, epochs = 50, verbose=1)
print(model.predict(dataForNN).round())




# model = Sequential()
# model.add(Dense())


# model = Sequential()
# model.add(Convolution2D(32, 3, 3, activation='relu', input_shape=(1, 100, 50)))
# model.add(MaxPooling2D(pool_size=(2,2)))
# model.add(Dropout(0.25))


# model.add(Flatten())
# model.add(Dense(128, activation='relu'))
# model.add(Dropout(0.5))
# model.add(Dense(2, activation='softmax'))


# model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])
# model.fit(dataForNN, y, batch_size=32, nb_epoch=10, verbose = 1)


# model.evaluate(test_data, y_test, verbose=0)





