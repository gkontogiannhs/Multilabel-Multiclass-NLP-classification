import re
from sklearn.feature_extraction.text import CountVectorizer
# from sklearn.preprocessing import MinMaxScaler, StandardScaler
# from sklearn.model_selection import train_test_split
from sklearn.model_selection import KFold
import numpy as np
import keras
from keras.models import Sequential
from keras.layers import Dense
from keras import backend as K

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
                y.append(list(label_line.split(' ')))

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


def get_model(n_inputs, n_outputs, loss_f):
	model = Sequential()
	model.add(Dense(20, input_dim=n_inputs, kernel_initializer='he_uniform', activation='relu'))
	model.add(Dense(n_outputs, activation='sigmoid'))
	model.compile(loss=loss_f, optimizer='sgd')
	return model

def main():
   
    X, y = read_data('Data/test-data.dat', 'Data/test-label.dat', 10)
  
    voc = [str(i) for i in range(8520)]

    # create vectorizer and pass existed corpus
    vectorizer = CountVectorizer(vocabulary=voc)

    # tranform X, y to numpy arrays
    X_bow = vectorizer.transform(X).toarray()
    y = np.asarray(y, dtype=int)
    
    ################ NORMALIZATION ########################################
    
    # normalizer = MinMaxScaler()
    # print(normalizer.fit_transform(df_bow))

    ################ STANDARDIZATION ########################################

    # standardizer = StandardScaler()
    #print(standardizer.fit_transform(df_bow))

    #################################### CROSS VALIDATION ################################
    # X_train, X_test, y_train, y_test = train_test_split(df_bow.iloc[:, :-1], df_bow['y'], test_size=0.4, random_state=0)
    # Split the data to training and testing data 5-Fold


    # Split the data to training and testing data 5-Fold
    kfold = KFold(n_splits=5, shuffle=True)
    rmseList = []
    n_inputs = X_bow.shape
    n_outputs = y.shape

    for i, (train, test) in enumerate(kfold.split(X_bow)):
        model = Sequential()
        model.add(Dense(20, input_dim=n_inputs, kernel_initializer='he_uniform', activation='relu'))
        model.add(Dense(n_outputs, activation='sigmoid'))
        model.compile(loss=loss_f, optimizer='sgd')

        keras.optimizers.SGD(lr=1e-3, momentum=0.0, decay=0.0, nesterov=False)
        model.compile(loss='mean_squared_error', optimizer='sgd', metrics=[rmse])

        # Fit model
        model.fit(X[train], Y[train], epochs=500, batch_size=500, verbose=0)

        # Evaluate model
        scores = model.evaluate(X[test], Y[test], verbose=0)
        rmseList.append(scores[0])
        print("Fold :", i, " RMSE:", scores[0])

    print("RMSE: ", np.mean(rmseList))
    
main()