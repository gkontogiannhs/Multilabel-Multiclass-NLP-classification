import re
from sklearn.feature_extraction.text import CountVectorizer
# from sklearn.preprocessing import MinMaxScaler, StandardScaler
# from sklearn.model_selection import train_test_split
# from sklearn.model_selection import KFold
from tensorflow.keras.layers import Dense
from tensorflow.keras.models import Sequential

import pandas as pd

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

    labels = read_file('Data/labels.txt')
    
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

                # convert to list and throw 2 first elem
                temp = data_line.split(' ')

                # convert to list and search for '1s' indexes
                ones = label_line.split(' ')
                indices = [i for i in range(len(ones)) if ones[i] == '1']
               
                # get labels for every document
                y.append([labels[index] for index in indices])

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


def main():
   
    X, y = read_data('Data/test-data.dat', 'Data/test-label.dat', 10)
    voc = [str(i) for i in range(8520)]
    # create vectorizer and pass existed corpus
    vectorizer = CountVectorizer(vocabulary=voc)
    # nparray
    X_bow = vectorizer.transform(X)
    df_bow = pd.DataFrame(X_bow.toarray(), columns=voc)
    df_bow['y'] = y
    
    ################ NORMALIZATION ########################################
    
    # normalizer = MinMaxScaler()
    # print(normalizer.fit_transform(df_bow))

    ################ STANDARDIZATION ########################################

    # standardizer = StandardScaler()
    #print(standardizer.fit_transform(df_bow))
    

    #################################### CROSS VALIDATION ################################
    # X_train, X_test, y_train, y_test = train_test_split(df_bow.iloc[:, :-1], df_bow['y'], test_size=0.4, random_state=0)
    # Split the data to training and testing data 5-Fold

    kfold = KFold(n_splits=5, shuffle=True)
    rmseList = []
    rrseList = []

    for i, (train, test) in enumerate(kfold.split(X)):
        # Create model
        model = Sequential()

        model.add(Dense(10, activation="relu", input_dim=8))
        model.add(Dense(1, activation="linear", input_dim=10))

        # Compile model
        def rmse(y_true, y_pred):
            return K.sqrt(K.mean(K.square(y_pred - y_true)))

        keras.optimizers.SGD(lr=0.08, momentum=0.2, decay=0.0, nesterov=False)
        model.compile(loss='mean_squared_error', optimizer='sgd', metrics=[rmse])

        # Fit model
        model.fit(X[train], Y[train], epochs=500, batch_size=500, verbose=0)

        # Evaluate model
        scores = model.evaluate(X[test], Y[test], verbose=0)
        rmseList.append(scores[0])
        print("Fold :", i, " RMSE:", scores[0])

        # print("RMSE: ", np.mean(rmseList))
    
main()