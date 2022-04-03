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

def read_data(fname):

    row = []
    column = []
    data = []
    bags = dict()
    fp = open(fname)
    d = 0
    while True:
        ln = fp.readline()
        if len(ln)==0:
            break
        ln_split = ln.split(',')
        wrds = re.findall(r',([0-9\.]*):[0-9\.]*', ln)
        cnts = re.findall(r',[0-9\.]*:([0-9\.]*)', ln)

        for i,wrd in enumerate(wrds):
            row.append(d)
            column.append(int(wrd)-1)
            data.append(float(cnts[i]))

        doc = int(ln_split[1].lstrip('d'))
        try:
            bags[doc].append(d)
        except KeyError:
            bags.update({doc:[d]})

        d += 1
        if d >= 20:
            break

    fp.close()

    # X = csr_matrix((np.array(data), (np.array(row), np.array(column))), shape=(len(set(row)),N))

    return bags
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

def read_data(fn1 , fn2):

    vocabulary = read_file('Data/vocabs.txt')
    # labels = read_file('Data/labels.txt')

    try:
        with open(fn1) as fd, open(fn2) as fl:
            line_d = fd.readline()
            # line_l = fl.readline()
            text = line_d.split(' ')
            print(text)
            lengths = re.findall('<(.*?)>', line_d)
            print(lengths)
            k = 2
            index = 1
            arr = []
            sents = int(lengths[0])
            sentence = ''
            for i in range(sents-1):
                l = int(lengths[index]) + 1
                for j in range(k, l):
                    sentence += vocabulary[int(text[j])]
                index += 1
                k = l + 2
                arr.append(sentence)
            return arr
    except FileNotFoundError:
        raise "Count not read file"


def main():

    v = read_data('Data/test-data.dat', 'Data/test-label.dat')
    print(v)

main()