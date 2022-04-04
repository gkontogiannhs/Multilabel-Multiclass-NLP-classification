# Convert encoded data to text
# make Bag of Words (BoW)
import re
from xml.dom.minidom import Document

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
    arr = []

    try:
        with open(fn1) as fd, open(fn2) as fl:
            arr = []
            words = []
            doc = []
            line_d = fd.readline().strip('\n')
            # line_l = fl.readline()      
            temp = line_d.split(' ')
            temp.pop(0)
    
            for char in temp:
                m = re.findall('<(.*?)>', char)
                if not m:
                    words.append(vocabulary[int(char)])
                else:
                    doc.append(' '.join(words))
                    words.clear()
            return arr.append(doc)
                    
    except FileNotFoundError:
        raise "Count not read file"


def main():

    v = read_data('Data/test-data.dat', 'Data/test-label.dat')
    print(v)

main()