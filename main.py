from sklearn import metrics
from sklearn.model_selection import train_test_split, KFold
from function.generateImage import *
from function.toCategorical import *

import os
import sys
script_dir = os.path.dirname(__file__)
mymodule_dir = os.path.join(script_dir, 'class')
sys.path.append(mymodule_dir)

from CNN import CNN
from layer.ConvolutionalLayer import ConvolutionalLayer
from layer.FlattenLayer import FlattenLayer
from layer.DenseLayer import DenseLayer

if __name__ == "__main__":
  objectLabelDictionary = {
    0: 'bear',
    1: 'panda'
  }
  dataInput, dataInputLabel = generateImage()
  dataInputLabel = toCategorical(dataInputLabel, 1)

  # Melakukan pembelajaran dengan skema split train 90% dan test 10%, dan menampilkan kinerja serta confusion matrixnya
  
  print("\n2. Implement training using 10-fold cross validation, and show its performance.\n")
  kf = KFold(n_splits=10,shuffle=True)
  best_accuracy = 0
  best_model = None
  i = 1
  for train_index, test_index in kf.split(dataInput):
    print("SPLIT - ", i)
    X_train, X_test = dataInput[train_index], dataInput[test_index]
    dataInputLabel = np.array(dataInputLabel)
    y_train, y_test = dataInputLabel[train_index], dataInputLabel[test_index]

    cnnKfold = CNN()
    print("Reading (load) model from external file (model.json)")
    cnnKfold.loadModel('model')

    output = np.array([])
    for data in X_test:
        forward_cnn = cnnKfold.forward(data)
        output = np.append(output, np.rint(forward_cnn))
    
    cnnKfold.predict(features = X_train, target = y_train, batchSize = 5, epoch = 10, learningRate = 0.5)
    accuracy = metrics.accuracy_score(y_test, output)
    print("\nAccuracy:", accuracy)
    print("Confusion matrix:\n", metrics.confusion_matrix(y_test, output), "\n")
    if accuracy > best_accuracy:
        best_accuracy = accuracy
    i = i + 1
        
  print("\nBest Accuracy:", best_accuracy)