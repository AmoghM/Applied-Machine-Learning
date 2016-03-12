import csv
import nltk
import re

count=0
l1=[]
words=[]   #bag of words
posterior_positive={}
posterior_negative={}
stopwords=[]
positive=[]
negative=[]
pos_words=[]
neg_words=[]
l2=[] # Testing data
 
f=open("stopwords.txt","r")  #stopwords file
patt="\r\n"
s=""
for line in f:
	s=(''.join(re.sub(patt,"",line)))
	stopwords.append(s)

def preprocess(row): #Only for training purpose, Can't be used for preprocessing of testing data
	l=[]
	row[0]=row[0].lower()
	x = (' '.join(re.sub("(@[A-Za-z0-9]+)|([^0-9A-Za-z \t])|(\w+:/\/\/S+)"," ",row[0]).split()))
	x=x.lower()
	y = x.split(" ")

	for i in y:
		if i in stopwords:
			while i in y:
				y.remove(i)
	l.append(y)
	for j in y:
		words.append(j)
	l.append(row[1])
	
	return l

stopwords.append("rt")
with open('dataset_v1.csv','r') as csvfile: # Training of classifier
	tweet=csv.reader(csvfile)
	for row in tweet:
		count+=1
		ans=preprocess(row)
		if count<1201:
			l1.append(ans)
		else:
			l2.append(ans)
words=set(words)
words=list(words)	#bag of words completed

for i in l1:
	if i[1]=='2':
		positive.append(i[0]) # making a list of all positive tweets
		for j in i[0]:
			pos_words.append(j) # making a list of all words of positive tweets
	else:
		for j in i[0]: 
			neg_words.append(j) # making a list of all words of positive tweets
		negative.append(i) # making a list of all negative tweets

prior_positive = float(len(positive))/1200 # positive prior probability
prior_negative = 1.0-prior_positive # negative prior probability

for i in words:						#Calculating positive posterior probability for all words 
		c1=pos_words.count(i)
		total=pos_words.count(i)+neg_words.count(i)
		if total==0:
			total=len(pos_words)
		if c1!=0:
			p1=float(c1)/total
			posterior_positive[i]=p1
		else:
			p1=1.0/total
			posterior_positive[i]=p1
for i in words:						#Calculating negative posterior probability for all words
		c2=neg_words.count(i)
		total1=pos_words.count(i)+neg_words.count(i)
		if total1==0:
			total1=len(neg_words)
		if c2!=0:
			p2=float(c2)/total1
			posterior_negative[i]=p2
		else:
			p2=1.0/total1
			posterior_negative[i]=p2

def classify(in_tweet):				#Classification function
	pos_prob=1
	neg_prob=1
	in_tweet=in_tweet.lower()
	wire = (' '.join(re.sub("(@[A-Za-z0-9]+)|([^0-9A-Za-z \t])|(\w+:/\/\/S+)"," ",in_tweet).split()))
	wire=(''.join(re.sub("[0-9]","",wire)).split())
	for i in wire:
		if i=="rt":
			wire.remove(i)	
	for i in wire:
		if i in stopwords:
			while i in wire:
				wire.remove(i)	
	for i in wire:
		if i not in words:
			continue
		pos_prob=pos_prob*posterior_positive[i] # Calculating probability of each words occuring in test tweet being positive
		neg_prob=neg_prob*posterior_negative[i] # Calculating probability of each words occuring in test tweet being negative

	pos_prob=pos_prob*prior_positive # Final positive probability
	neg_prob*=prior_negative # FInal negative probality
	if pos_prob>neg_prob:
		return 2
	else:
		return 0
	
cp=0
actual_pos=0

for i in l2:					#Testing
	s = i[0][0]
	for j in range(1,len(i[0])):
		s = s+" "+i[0][j]
	var = classify(s)
	actual = i[1]
	actual=int(actual)
	if actual == var:
		cp+=1
	if var==0:
		if actual==1 or actual ==3:
			cp+=1

acc=float(cp)/300
acc*=100
acc="{0:.2f}".format(acc)

print "Accuracy is",acc,"%"

