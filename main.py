# Convert encoded data to text
# make Bag of Words (BoW)
import re
from sklearn.feature_extraction.text import CountVectorizer
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
    X_bow = vectorizer.transform(X)
    df_bow = pd.DataFrame(X_bow.toarray(), columns=voc)
    df_bow['y'] = y
    print(df_bow)

main()