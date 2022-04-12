import re
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.preprocessing import MinMaxScaler, StandardScaler
# from sklearn.model_selection import train_test_split
from sklearn.model_selection import KFold
import numpy as np
import keras
from keras.models import Sequential
from keras.layers import Flatten, Dense, Embedding
import matplotlib.pyplot  as plt

# plot loss during training
def plot(h):
    plt.title('Loss / Mean Squared Error')
    plt.plot(h.history['loss'], label='train')
    plt.plot(h.history['val_loss'], label='test')
    plt.legend()
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

    # plot loss during training
        def plot():
            plt.title('Loss / Mean Squared Error')
            plt.plot(h.history['loss'], label='train')
            plt.plot(h.history['val_loss'], label='test')
            plt.legend()
            plt.show()

    # plot loss during training
        def plot():
            plt.title('Loss / Mean Squared Error')
            plt.plot(h.history['loss'], label='train')
            plt.plot(h.history['val_loss'], label='test')
            plt.legend()
            plt.show()


def main():
    
    # load data
    X, y = read_data('Data/train-data.dat', 'Data/train-label.dat', 8000)

    # create corpus
    voc = [str(i) for i in range(8520)]

    # tranform X, y to numpy arrays
    X = CountVectorizer(vocabulary=voc).transform(X).toarray()
    y = np.asarray(y, dtype=int)
    
    ################ NORMALIZATION ########################################
    # X = MinMaxScaler().fit_transform(X)

    ################ STANDARDIZATION ########################################
    # X = StandardScaler().fit_transform(X)
   
    # model's dims
    input_dims, out_dims = X.shape[1], y.shape[1]
    hidden_dims = 8540
    results = []

    # Split the data to training and testing data 5-Fold
    kfold = KFold(n_splits=5, shuffle=True)
    for i, (train, test) in enumerate(kfold.split(X)):
       
        # create model
        model = Sequential([
            Dense(hidden_dims, input_dim=input_dims, activation='relu'),
            Dense(out_dims, activation='sigmoid')
        ])

        loss_fn = keras.losses.CategoricalCrossentropy()
        opt = keras.optimizers.SGD(learning_rate=1e-3)
        model.compile(loss=loss_fn, optimizer=opt, metrics=['accuracy'])

        # Fit model
        h = model.fit(X[train], y[train], epochs=50, batch_size=64, verbose=1)

        # evaluate the model
        test_mse = model.evaluate(X[test], y[test], verbose=0)
        print('Test: %.3f' % test_mse[0])

        results.append(test_mse[0])

        print("Fold :", i, " RMSE:", test_mse[0])
    
main()