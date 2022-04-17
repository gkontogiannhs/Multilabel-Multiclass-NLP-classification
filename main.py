import re
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.preprocessing import MinMaxScaler, StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.model_selection import KFold
import numpy as np
from tensorflow import keras
from keras import backend as K
import matplotlib.pyplot  as plt
from sklearn.metrics import accuracy_score

# plot loss during training
def plot(h, label):
    plt.plot(h.history['binary_accuracy'])
    plt.plot(h.history['val_binary_accuracy'])
    plt.title(label + ' - Accuracy')
    plt.ylabel('accuracy')
    plt.xlabel('epoch')
    plt.legend(['train', 'test'], loc='upper left')
    plt.show()

    # summarize history for loss
    plt.plot(h.history['loss'])
    plt.plot(h.history['val_loss'])
    plt.title(label + ' - Loss')
    plt.ylabel('loss')
    plt.xlabel('epoch')
    plt.legend(['train', 'test'], loc='upper left')
    plt.show()

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

                cnt += 1
                if cnt >= docs:
                    return X, y
                    
    except FileNotFoundError:
        raise "Count not read file"

# get the model
def get_model(n_inputs, n_outputs, loss_f, n_hidden1, n_hidden2=None):
    model = keras.models.Sequential()
    model.add(keras.Input(shape=(n_inputs,)))
    model.add(keras.layers.Dense(n_hidden1, activation='relu'))
    if n_hidden2:
        model.add(keras.layers.Dense(n_hidden2, activation='relu'))
    model.add(keras.layers.Dense(n_outputs, activation='sigmoid'))
    opt = keras.optimizers.SGD(learning_rate=0.01, momentum=0.6)
    model.compile(optimizer=opt, loss=loss_f, metrics=[keras.metrics.BinaryAccuracy(threshold=0.5)])
    return model

def evaluate_model(X, y):
    history = []
    es = keras.callbacks.EarlyStopping(monitor='val_loss', mode='min', verbose=1)
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    # Split the data to training and testing data 5-Fold
    kfold = KFold(n_splits=5, shuffle=True)
    for i, (train, test) in enumerate(kfold.split(X_train)):
        # create model
        model = get_model(X.shape[1], y.shape[1], 'mean_squared_error', 20)
   
        # Fit model
        h = model.fit(X_train[train], y_train[train], validation_data=(X_train[test], y_train[test]), epochs=150, callbacks=[], verbose=1)
        history.append(h.history)

        # evaluate model
        model.evaluate(X_train[test], y_train[test])

        plot(h, 'MSE')
        
        # make predict to unseen data
        yhat = model.predict(X_test)    
        yhat = yhat.round()
       
		# calculate accuracy
        acc = accuracy_score(y_test, yhat)

		# store result
        print('>%.3f' % acc)

def main():
    # load data
    X, y = read_data('Data/train-data.dat', 'Data/train-label.dat', 8250)

    # create corpus
    voc = [str(i) for i in range(8520)]

    # tranform X, y to numpy arrays
    X = CountVectorizer(vocabulary=voc).transform(X).toarray()
    y = np.asarray(y, dtype=int)

    ################ CENTERING  ##########################################
    # row_means = np.mean(X, axis=1)
    # X = np.subtract(X, row_means.reshape((row_means.shape[0], 1)))
    
    ################ NORMALIZATION ########################################
    X = MinMaxScaler().fit_transform(X)

    ################ STANDARDIZATION ########################################
    # X = StandardScaler().fit_transform(X)

    evaluate_model(X, y)


main()