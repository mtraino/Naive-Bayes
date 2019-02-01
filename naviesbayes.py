import numpy as np
import ipdb
from sklearn.metrics import confusion_matrix

#loading the files needed
traininglabels = np.loadtxt("traininglabels.txt", dtype=int)
trainingdata = np.loadtxt("trainingdata.txt", dtype=int)
testinglabels = np.loadtxt("testinglabels.txt", dtype=int)
testingdata = np.loadtxt("testingdata.txt",dtype=int)


maxword = int(testingdata.max())
maxnum = int(traininglabels.max())
Pclass = np.empty(maxnum)
Pword = np.ndarray(shape=(maxword, maxnum))

for i in range(len(traininglabels)):
    Pclass[traininglabels[i]-1] += 1
Pclass = Pclass/len(traininglabels)

for i in range(len(trainingdata)):
    count = trainingdata[i, 2]
    word = trainingdata[i, 1]-1
    pos = traininglabels[trainingdata[i, 0]-1]-1
    Pword[word, pos] += count
Pword += .06 #Alpha value being added to the whole matrix

#sums up the colum
for i in range(traininglabels.max()):
    summ = sum(Pword[:, i])
    Pword[:, i] = Pword[:, i]/(summ + (.06 * maxword))

Pword = np.log(Pword)
Pclass = np.log(Pclass)

#Product of P(Xj | Yi) get the P from training data and raises that P to the count in testing data
PwordTesting = np.zeros((np.amax(testingdata[:, 0]), maxnum))
for i in range(len(testingdata)):
    print(i)
    word = testingdata[i, 1]
    for x in range(traininglabels.max()):
        pvalue = Pword[(word-1), x]
        pvalue *= testingdata[i, 2]
        PwordTesting[testingdata[i, 0]-1, x] += pvalue
PwordTesting += Pclass

a = np.argmax(PwordTesting, axis=1) + 1
c = (testinglabels == a).mean()*100
print("Accuracy: " + str(c))
matrix = confusion_matrix(testinglabels, a)
print(matrix)
