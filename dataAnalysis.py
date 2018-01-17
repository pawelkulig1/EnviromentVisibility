import matplotlib.pyplot as plt
import pickle
import time as tm
import datetime

data = pickle.load(open("data.pickle", "rb"))

dt = round(3600*24/len(data[0]))
#print(dt)
print(data)


#x = [str((datetime.datetime.min + datetime.timedelta(0,120*i*10,0,0)).time()) for i in range(round(len(data[0])/10))]

#plt.plot(data[0], 'g-', data[1], 'r-', data[2], 'b-')
plt.plot(data[1])
plt.show()

### average of 3 inputs
outData = []
outData.append([])
for i in range(len(data[0])):
    outData[0].append((data[0][i] + data[1][i] + data[2][i])/3)

def calculateMean(arr, count, poz):
    s = 0
    if len(arr) < count:
        count = len(arr)-1

    for i in range(count):
        s += arr[poz-i]

    return s/count

### average of ten pasts
outData.append([])
for i in range(len(data[0])):
    outData[1].append((calculateMean(data[0], 10, i) + calculateMean(data[1], 10, i) +calculateMean(data[2], 10, i))/3)

outData.append([])
for i in range(len(data[0])):
    outData[2].append((calculateMean(data[0], 30, i) + calculateMean(data[1], 30, i) +calculateMean(data[2], 30, i))/3)

average = 300
outData.append([])
for i in range(len(data[0])):
    outData[3].append((calculateMean(data[0], average, i) + calculateMean(data[1], average, i) +calculateMean(data[2], average, i))/3)


plt.plot(outData[3])
plt.show()

#save outData[3] as pickle for NN

pickle.dump(outData[3], open("NNdata.pickle", "wb"))


'''plt.plot(outData[0])
plt.show()

plt.plot(outData[1])
plt.show()

plt.plot(outData[2])
plt.show()'''
