import os,pickle
import numpy as np
from scipy.misc import imread,imresize,imshow
inputs=[]
targets=[]

# 13CS027
# animesh1234*
X = 32
Y = 32
char = 'E'
datapath = str(X)+'x'+str(Y)+char


for file_name in os.listdir(datapath+"/"):

    # # read only jpg files
    # print file_name
    number = int(file_name.split('_')[1])
    targ = number
    # if(extension == ".jpg"):
    img = imread(datapath+"/"+file_name)             
#     # resize to 16x8x3
#     img = imresize(img,(16,8,))
    linear_img = []
    linear_img = np.reshape(img, (X*Y*1, 1))
#     # normalize 
    linear_img =np.true_divide(linear_img, 255.0)
#     # add entry to inputs and target
    inputs.append(linear_img)
    targets.append(targ)


from random import shuffle
# Given list1 and list2
list1_shuf = []
list2_shuf = []
index_shuf = range(len(inputs))
shuffle(index_shuf)
for i in index_shuf:
    list1_shuf.append(inputs[i])
    list2_shuf.append(targets[i])

inputs = list1_shuf
targets = list2_shuf

pickle.dump(inputs ,open('inputs.pkl','wb'))
pickle.dump(targets, open('targets.pkl','wb'))

