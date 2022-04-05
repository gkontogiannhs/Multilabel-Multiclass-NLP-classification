# Convert encoded data to text
# make Bag of Words (BoW)
import re

"""
train_x = ['Hello there', 'How are you today?']

vectorizer = CountVectorizer()
x = vectorizer.fit(train_x)
x = vectorizer.transform(train_x)
print(vectorizer.get_feature_names_out())
print(x.toarray())

"""

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

def read_data(fn1 , fn2, samples):

    vocabulary = read_file('Data/vocabs.txt')
    labels = read_file('Data/labels.txt')
    
    try:
        with open(fn1) as fd, open(fn2) as fl:
            X = []
            y = []
            words = []
            doc = []
            cnt = 0

            while True:
                # get each line from the files
                data_line = fd.readline().strip('\n')
                label_line = fl.readline().strip('\n') 

                # check if EOF
                if len(data_line) == 0:
                    return X, y

                # convert to list and throw 2 first elem
                temp = data_line.split(' ')[2:]

                # convert to list and search for '1s' indexes
                ones = label_line.split(' ')
                indices = [i for i in range(len(ones)) if ones[i] == '1']
               
                # get labels for every document
                y.append([labels[index] for index in indices])

                # get every word to create sentece
                # The total of sentences create a document
                for char in temp:
                    m = re.findall('<(.*?)>', char)
                    if not m:
                        words.append(vocabulary[int(char)])
                    else:
                        doc.append(' '.join(words))
                        words.clear()

                # append document
                X.append(doc)
                doc = []
               
                cnt += 1
                if cnt >= samples:
                    return X, y
                    
    except FileNotFoundError:
        raise "Count not read file"


def main():
    X, y = read_data('Data/test-data.dat', 'Data/test-label.dat', 10)
    print(X, y)
    print(len(X), len(y))

main()