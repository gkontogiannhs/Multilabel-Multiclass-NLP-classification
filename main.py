import re
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.preprocessing import MinMaxScaler, StandardScaler
# from sklearn.model_selection import train_test_split
from sklearn.model_selection import KFold
import numpy as np
import keras
import matplotlib.pyplot  as plt
from sklearn.metrics import accuracy_score
from numpy import mean
from numpy import std


def read_file(filename):
    arr = []

    try:
        with open(filename, 'r') as file:
            lines = file.readlines()
            for line in lines:
                arr.append(line.split(',')[0])
    except FileNotFoundError:
        raise "Could not read file:" + filename
        
    return arr

def read_data(fn1 , fn2, docs=None):

    # labels = read_file('Data/labels.txt')
    
    try:
        with open(fn1) as fd, open(fn2) as fl:
            X = []
            y = []
            words = []
            cnt = 0

            # safe mode
            if docs is None:
                docs = 10

            while True:

                # get each line from the files
                data_line = fd.readline().strip('\n')
                label_line = fl.readline().strip('\n') 
                
                # check if EOF
                if len(data_line) == 0:
                    return X, y

                # labels
                y.append(label_line.split(' '))

                # convert to list and throw 2 first elem
                temp = data_line.split(' ')

                # get every word to create sentece
                # The total of sentences create a document
                for char in temp:
                    if not re.findall('<(.*?)>', char):
                        words.append(char)

                # append document
                X.append(' '.join(words))

                words.clear()

                # cnt += 1
                # if cnt >= docs:
                #    return X, y
                    
    except FileNotFoundError:
        raise "Count not read file"


def main():
   
    X, y = read_data('Data/train-data.dat', 'Data/train-label.dat', 5000)

    voc = [str(i) for i in range(8520)]

    # create vectorizer and pass existed corpus
    vectorizer = CountVectorizer(vocabulary=voc)

    # tranform X, y to numpy arrays
    X = vectorizer.transform(X).toarray()
    y = np.asarray(y, dtype=int)

    ################ NORMALIZATION ########################################
    
    # X = MinMaxScaler().fit_transform(X)

    ################ STANDARDIZATION ########################################
    # X = StandardScaler().fit_transform(X)

     # neural network's dims
    input_dims, out_dims = X.shape[1], y.shape[1]
    hidden_dims = 20
    results = []

    # Split the data to training and testing data 5-Fold
    kfold = KFold(n_splits=2, shuffle=True)
    for i, (train, test) in enumerate(kfold.split(X)):
        
        # create model
        model = keras.Sequential([
            keras.layers.Dense(hidden_dims, input_shape=(input_dims,), activation='relu'),
            keras.layers.Dense(out_dims, activation='sigmoid')
        ])

        keras.optimizers.SGD(learning_rate=0.001)
        model.compile(loss='binary_crossentropy', optimizer='sgd', metrics=['mse', 'accuracy'])

        # Fit model
        model.fit(X[train], y[train], epochs=30, batch_size=10)

        #evalute model
        print(i, model.evaluate(X[test], y[test]))

    
main()