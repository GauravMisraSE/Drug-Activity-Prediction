import numpy as np
from sklearn.decomposition import PCA
from sklearn.svm import SVC

fh1 = open('train.dat', 'r')
fh2 = open('test.dat', 'r')
trainData = fh1.readlines()
testData = fh2.readlines()
classes = [] # stores the test classes
fin = np.zeros((9, 350)) # stores the output matrix generated

# create a sparse matrix out of train data
def get_train(data):
    sparse_arr = np.zeros((800, 100001))
    counter = 0

    for i in data:
        splitted = i.split(' ')
        splitted = splitted[:-1]  # removing the newline character
        zero_element = splitted[0].split('\t')
        splitted[0] = zero_element[1]
        classes.append(int(zero_element[0]))  # storing classes

        for y in splitted:
            y = int(y)
            sparse_arr[counter, y] = 1

        counter = counter + 1

    return sparse_arr

# create a sparse matrix out of test data
def get_test(data):
    sparse_arr = np.zeros((350, 100001))
    counter = 0

    for i in data:
        splitted = i.split(' ')
        splitted = splitted[:-1]

        for y in splitted:
            y = int(y)
            sparse_arr[counter, y] = 1

        counter = counter + 1

    return sparse_arr


train_sparse = get_train(trainData)
test_sparse = get_test(testData)

train_pruned = np.zeros((156, 100001)) # contains the data used while classification
train_zeros = np.zeros((722, 100001)) # contains records with 0 class
train_unity = np.zeros((78, 100001)) # contains records with 1 class
zero_holder = np.zeros((78, 100001)) # holds the 0 record buckets of size 78

counter = 0
class_pruned = [0]*156 # contains classes for train_pruned
for x in range(78):
    class_pruned[x] = 0
    class_pruned[x + 78] = 1

for x in range(800): # get the 0 class records
    if classes[x] == 0:
        train_zeros[counter] = train_sparse[x]
        counter = counter + 1

counter = 0
for x in range(800): # get the 1 class records
    if classes[x] == 1:
        train_unity[counter] = train_sparse[x]
        counter = counter + 1

# using PCA for dimensionality reduction with size 300(350 also gave the same result)
pca = PCA(n_components=300)
# used SVC with linear kernel for classification
clf = SVC(kernel='linear', random_state=25)
one_pruned = train_unity[:78, :]

for x in range(78):
    train_pruned[x + 78] = one_pruned[x] # last 78 records are of class 1, first 78 will be of class 0

inc = 78
for x in range(9): # creating 9 buckets of class 0 data(each contains 78 records), and using each bucket one by one for classification
    zero_holder = train_zeros[inc - 78:inc, :]
    inc = inc + 78
    for y in range(78):
        train_pruned[y] = zero_holder[y] # inserting 0 records in train_pruned

    pca.fit_transform(train_pruned)
    clf.fit(train_pruned, class_pruned)

    ans = clf.predict(test_sparse)
    #     print ans
    fin[x, :] = ans

sum_ans = fin.sum(axis = 0)
print sum_ans
f = open('format.dat', 'w')
for x in sum_ans: # if a record has been classified as 1 in any iteration, classify it as 1 in the final classification
    if x > 0.0:
        f.write(str(1))
        f.write('\n')
    else:
        f.write(str(0))
        f.write('\n')

f.close()
