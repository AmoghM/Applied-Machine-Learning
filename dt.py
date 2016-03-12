import csv
import operator
import nltk
from nltk.corpus import stopwords
from collections import Counter
import random
import math


fileobj =open('out1.csv', "rt")  #reading csv file
reader = csv.reader(fileobj)
dataset=[]
for row in reader:
    dataset.append(row)
random.shuffle(dataset)

#create three list for each class containing the tweets
stop = stopwords.words('english')

class1=[]
class2=[]
class3=[]
for item in dataset:
    if(item[1]=="1"):
        class1.append(item[0])
    elif(item[1]=="2"):
        class2.append(item[0])
    else:
        class3.append(item[0])
class1_words=[]
class2_words=[]
class3_words=[]

class1_wordlist=[]
class2_wordlist=[]
class3_wordlist=[]
finalwords1=[]
finalwords2=[]
finalwords3=[]

for i in class1:
    class1_words.append(i.split(" "))
for j in class1_words:
    for k in j:
        class1_wordlist.append(k)
class1_wordlist1=list(set(class1_wordlist))

for i in class2:
    class2_words.append(i.split(" "))
for j in class2_words:
    for k in j:
        class2_wordlist.append(k)
class2_wordlist1=list(set(class2_wordlist))

for i in class3:
    class3_words.append(i.split(" "))
for j in class3_words:
    for k in j:
        class3_wordlist.append(k)
class3_wordlist1=list(set(class3_wordlist))



for words in class1_wordlist1[1:]:
    if words.lower() not in stop:
        finalwords1.append(words)  #finalwords1 keeps all the words from category 1 after removing stop words.


for words in class2_wordlist1[1:]:
    if words.lower() not in stop:
        finalwords2.append(words)

for words in class3_wordlist1[1:]:
    if words.lower() not in stop:
        finalwords3.append(words) 
allwords=[] #allwords is a list of list that contains all the words from all three categories in different list.
allwords.append(finalwords1)
allwords.append(finalwords2)
allwords.append(finalwords3)

feature_vector=[] #it is a list which contains all the words from all the categories. (count is 3142)
for no in allwords:
    for ele in no:
        feature_vector.append(ele)
feature_vector=list(set(feature_vector))


for i in feature_vector:
    if(len(i)<2):
        feature_vector.remove(i)



#now we are trying to express each tweet in terms of feature vector as a dictionary
alldata=[]
for items in dataset:
    tweet=items[0]
    dic={}
    for words in feature_vector:
        if words in tweet:
            dic[words]=1
        else:
            dic[words]=0
    dic["target_attr"]=items[1]
    alldata.append(dic)

#now splitting training and testing data.
train_data=alldata[:1200]
test_data=alldata[1201:]
best_attr_list=[]
best_attr=0


def entropy(data, target_attr):
    """
    Calculates the entropy of the given data set for the target attribute.
    """
    val_freq     = {}
    data_entropy = 0.0

    # Calculate the frequency of each of the values in the target attr
    for record in data:
        if (record[target_attr] in val_freq):
            val_freq[record[target_attr]] += 1.0
        else:
            val_freq[record[target_attr]]  = 1.0

    # Calculate the entropy of the data for the target attribute
    for freq in val_freq.values():
        data_entropy += (-freq/len(data)) * math.log(freq/len(data), 2) 
        
    return data_entropy

def gain(data, attr, target_attr):
    """
    Calculates the information gain (reduction in entropy) that would
    result by splitting the data on the chosen attribute (attr).
    """
    val_freq       = {}
    subset_entropy = 0.0

    # Calculate the frequency of each of the values in the target attribute
    for record in data:
        if (record[attr] in val_freq):
            val_freq[record[attr]] += 1.0
        else:
            val_freq[record[attr]]  = 1.0

    # Calculate the sum of the entropy for each subset of records weighted
    # by their probability of occuring in the training set.
    for val in val_freq.keys():
        val_prob        = val_freq[val] / sum(val_freq.values())
        data_subset     = [record for record in data if record[attr] == val]
        subset_entropy += val_prob * entropy(data_subset, target_attr)

    # Subtract the entropy of the chosen attribute from the entropy of the
    # whole data set with respect to the target attribute (and return it)
    return (entropy(data, target_attr) - subset_entropy)

def majority_value(data, target_attr):
    count_1=0
    count_2=0
    count_3=0

    for i in data:
        for k,v in i.items():
            if k=="target_attr":
                if int(v)==1:
                    count_1+=1
                if int(v)==2:
                    count_2+=1
                if int(v)==3:
                    count_3+=1
    maximum = max(count_1, count_2, count_3)
    return maximum		

	
def choose_attribute(data, attributes, target_attr, fitness_func):
        best_attr="-1"
        best_gain = -1
        for i in attributes:
                this_gain=fitness_func(data, i, target_attr)
                if this_gain > best_gain:
                        best_gain=this_gain
                        best_attr=i
        return best_attr

def get_values(data,best):
    l=[]
    for i in data:
        l.append(i[best])
    return list(set(l))

def get_examples(data,best,val):
    examples = []
    for i in data:
        if i[best]==val:
            examples.append(i)
    return examples					 

def create_decision_tree(data, attributes, target_attr, fitness_func):
    """
    Returns a new decision tree based on the examples given.
    """
    data    = data[:]
    vals    = [record[target_attr] for record in data]
    default = majority_value(data, target_attr)
    global best_attr
    best_attr=0
    # If the dataset is empty or the attributes list is empty, return the
    # default value. When checking the attributes list for emptiness, we
    # need to subtract 1 to account for the target attribute.
    if not data or (len(attributes) - 1) <= 0:
        return default
    # If all the records in the dataset have the same classification,
    # return that classification.
    elif vals.count(vals[0]) == len(vals):    
        return vals[0]
    else:
	#entropy = Entropy(data, target_attr)
        # Choose the next best attribute to best classify our data
        best = choose_attribute(data, attributes, target_attr, fitness_func)
        best_attr=best
        best_attr_list.append(best_attr)

        # Create a new decision tree/node with the best attribute and an empty
        # dictionary object--we'll fill that up next.
        tree = {best:{}}
        

        # Create a new decision tree/sub-node for each of the values in the
        # best attribute field
        for val in get_values(data, best):
            # Create a subtree for the current value under the "best" field
            subtree = create_decision_tree(
                get_examples(data, best, val),
                [attr for attr in attributes if attr != best],
                target_attr,
                fitness_func)

            # Add the new subtree to the empty dictionary object in our new
            # tree/node we just created.
            tree[best][val] = subtree

    return tree

def get_label(dtree, i):
        node=next(iter(tree.keys()))
        val=dtree
        finished=0
        
        while finished==0:
                val1=val[node][i[node]]
                
                if val1=="1" or val1=="2" or val1=="3":
                        finished=1
                else:
                        node=next(iter(val.keys()))
        return val

def accuracy_check(dtree, test):
        correct=0
        for i in test:
                prediction=get_label(dtree, i)
                if prediction==i["target_attr"]:
                        correct+=1
        accuracy=float(correct)/len(test)*100
        print("accuracy for decision tree classifier:", "%.2f"%accuracy,"%")
              

tree= create_decision_tree(train_data, feature_vector, "target_attr", gain)
accuracy_check(tree, test_data)

print("Attribute with most discriminatory power:",best_attr_list[0])

