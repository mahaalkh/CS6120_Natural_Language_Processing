from collections import Counter, OrderedDict
# import operator
# import random
import math
import itertools

## cluster -> [word, ...]
cluster2wordDict = OrderedDict()

### cluster -> [cluster1, cluster2]
mergedClusters = OrderedDict()

clusteringDomain = set()

###  the initial clustering
## Word -> cluster 
word2clusterDict = OrderedDict()


nextAdd = None

## reduced amount of the sentences  

### the sentences 

sentences = []

### the ranked unigramWords with the count  (the word unigram counts)
## word -> count
unigramWords = OrderedDict()

### the word bigram counts 
## (w1, w2) -> count
bigramWords = OrderedDict()

### the cluster unigram counts 
## cluster -> count 
unigramClusters = OrderedDict()

### the cluster bigram counts
## '(c1, c2)' -> count
bigramClusters = OrderedDict() 


################################# get from the files ########################

def getVocab(filename = 'corpraVocab.txt'): 
	"""
	gets the unigramWords from the ranked list in the file 
	"""
	# print "getting ranked vocab from {}".format(filename)
	global unigramWords
	fileObject = open(filename, "r")
	for line in fileObject: 
		l = list(line.split())
		if len(l) == 2: 
			unigramWords[l[0]] = int(l[1]) 
	print "done getting ranked vocab"


def removeTag(word): 
	"""
	removes the tag
	the word is in the format "abcd/tag"
	"""
	wl = word.split("/")
	return wl[0]


def readFromFile(filename):
	"""
	read from the file 
	"""
	filename = "brown/{}".format(filename)
	fileObject = open(filename, "r")
	low = []
	for line in fileObject: 
		low.append(line)
	return low

def readNamesFromFile(filename):
	"""
	"""
	fileObject = open(filename, "r")
	lol = []
	for line in fileObject: 
		l = line.split()
		lol.extend(l)
	return lol


def changeText(text):
	"""
	changes the text to get the sentences based on the /. tag  and gets the sentences 
	"""
	# print "changing the text"
	textNew = text.replace('./.', './. !END').replace('?/.', '?/. !END').replace('!/.', '!/. !END').replace(':/.', ':/. !END').replace(';/.', ';/. !END')
	s = textNew.split('!END')
	newSent = map(lambda x: map(lambda y: removeTag(y).lower(), x.split()), s)
	return newSent

def getSentences(): 
	"""
	gets the sentences from the corpra
	"""
	global sentences
	# print "getting the sentences"
	lof = readNamesFromFile("fileNames.txt")
	words = []
	for fn in lof: 
		low = readFromFile(fn)
		words.extend(low)
	text = ' '.join(words)
	sentences = changeText(text)
	print "got the sentences"


### need to call this 
def getTheSentenceAndVocab():
	getSentences()
	getVocab()

##############################################

### Helpers 


##########
def addToDict(c_from, c_to):
	# print "ADDING TO DICT"
	global word2clusterDict, cluster2wordDict
	# assert  isinstance(cluster2wordDict[c_from], list) and len(cluster2wordDict[c_from]) == 1 
	cluster2wordDict[c_to].extend(cluster2wordDict[c_from])
	for w in cluster2wordDict[c_from]:
		word2clusterDict[w] = c_to

	cluster2wordDict[c_from] = []

# def reverseAddToDict(c_from, c_to): 
# 	global word2clusterDict, cluster2wordDict

# 	assert  isinstance(cluster2wordDict[c_from], str) 
# 	assert cluster2wordDict[c_from] == 'DONE'
# 	assert isinstance(cluster2wordDict[c_to], list)

# 	lastWord = cluster2wordDict[c_to][-1:][0]
# 	cluster2wordDict[c_to].remove(lastWord)
# 	cluster2wordDict[c_from] = [lastWord]
# 	word2clusterDict[lastWord] = c_from
################

def insertByCountToDict(d, ky): 
	"""
	"""
	if d.has_key(ky):
		d[ky] += 1.0
	else: 
		d[ky] = 1.0

def getCluster(w): 
	"""
	"""
	return word2clusterDict[w]

def getWord(w):
	"""
	"""
	if unigramWords.has_key(w): 
		word = w
	else: 
		word = 'UNK'
	return word

############################

# def singletonClusters(): 
	# """

	# """ 
	# global cluster2wordDict
	# print "making the singleton clusters of all words"
	# n = len(unigramWords.keys())
	# i = 1
	# for w, _ in unigramWords.iteritems(): 
	# 	cluster2wordDict[i] = [w]
	# 	mergedClusters[i] = [i, None]
	# 	i += 1
	# print "done making the singleton clusters of all words"

def initializeClusterDomain(k = 200):
	global clusteringDomain 
	for i in range(1, k+1): 
		clusteringDomain.add(i)


def makeInitialClusters():
	"""
	makes the initial random clusters 
	""" 
	global word2clusterDict, cluster2wordDict, unigramWords
	# print "making the initial  clusters "
	n = len(unigramWords.keys())
	i = 1
	for w, c in unigramWords.iteritems(): 
		word2clusterDict[w] = i 
		cluster2wordDict[i] = [w]
		mergedClusters[i] = [i, None]
		i += 1
	# print "done making the initial clusters "

def getClusterGram(): 
	"""
	getting the unigram of the clusters 
	"""
	global unigramWords, cluster2wordDict, unigramClusters

	# print "getting the cluster unigrams"
	for c, low in cluster2wordDict.iteritems(): 
		unigramClusters[c] = sum(map(lambda x: unigramWords[x], low))
	# print "done getting the cluster unigrams"


def getBigrams(): 
	"""
	"""
	# print "getting n grams"
	global word2clusterDict, sentences, bigramClusters, bigramWords
	# print "getting word and cluster bigrams"

	### maybe fix the counts? 

	# print bigramClusters

	for sent in sentences:
		# for w in sent:  
		# 	insertByCountToDict(unigramClusters, getCluster(getWord(w)))
		for i in range(len(sent) - 1): 
			w_i_1 = getWord(sent[i])
			w_i  = getWord(sent[i + 1])
			word_tuple = (w_i, w_i_1)
			cluster_tuple = (getCluster(w_i), getCluster(w_i_1))
			insertByCountToDict(bigramClusters, cluster_tuple)
			insertByCountToDict(bigramWords, word_tuple)  

def initialize(k = 200): 
	global nextAdd
	getTheSentenceAndVocab()
	makeInitialClusters()
	getClusterGram()
	getBigrams()
	initializeClusterDomain(k = k)
	nextAdd = k+1

initialize()



########################### done initializing ####################
 
############################ FOR THE MERGING ####################


def getTuples(clusterTuples, p, q): 
	"""
	gets the tuples that the counts of get affected when merging p and q 
	p : to
	q : from
	"""

	P_K_Tuples = [bg for bg in clusterTuples if (bg[0] == p) and (bg[1] not in [p, q])] 
	K_P_Tuples = [bg for bg in clusterTuples if (bg[1] == p) and (bg[0] not in [p, q])]
	
	Q_K_Tuples = [bg for bg in clusterTuples if (bg[0] == q) and (bg[1] not in [p, q])] 
	K_Q_Tuples = [bg for bg in clusterTuples if (bg[1] == q) and (bg[0] not in [p, q])]
	
	P_Q_Tuples = [bg for bg in clusterTuples if (bg[0] == p) and (bg[1] == q)]

	Q_P_Tuples = [bg for bg in clusterTuples if (bg[0] == q) and (bg[1] == p)]


	return P_K_Tuples, K_P_Tuples, Q_K_Tuples, K_Q_Tuples, P_Q_Tuples, Q_P_Tuples


def getUpdatedBigram(c_from, c_to): 
	"""
	p : to
	q : from
	"""
	newBigram = OrderedDict()
	clusterTuples = bigramClusters.keys()
	ts = getTuples(clusterTuples, c_to, c_from)
	P_K_Tuples, K_P_Tuples, Q_K_Tuples, K_Q_Tuples, P_Q_Tuples, Q_P_Tuples  = ts
	assert len(P_Q_Tuples) in [1, 0] 
	assert len(Q_P_Tuples) in [1, 0]

	for p, k in P_K_Tuples: 
		count = 0
		if (c_from, k) in Q_K_Tuples:
			count = bigramClusters[(c_from, k)] 
		newBigram[(p, k)] = bigramClusters[(p, k)] + count

	for k, p in K_P_Tuples: 
		count = 0
		if (k, c_from) in K_Q_Tuples:
			count = bigramClusters[(k, c_from)]
		newBigram[(k, p)] = bigramClusters[(k, p)] + count

	for q, k in Q_K_Tuples: 
		newBigram[(q, k)] = 0 

	for k, q in K_Q_Tuples: 
		newBigram[(k, q)] = 0

	newBigram[(c_from, c_to)] = 0
	newBigram[(c_to, c_from)] = 0

	return newBigram

def getUpdatedUnigram(c_from, c_to): 
	"""
	"""
	newUnigram = OrderedDict()
	newUnigram[c_from] = 0 
	newUnigram[c_to] = unigramClusters[c_to] + unigramClusters[c_from]

	return newUnigram

def getCount(p_q, newGram, originalGram):
	"""
	"""
	if newGram.has_key(p_q): 
		return newGram[p_q]
	else:
		return originalGram[p_q]

def getProbLog(c1, c2, N, newBigram, newUnigram):
	"""
	"""
	prob_c1_c2 = float(float(getCount((c1, c2), newBigram, bigramClusters)) / N) 

	prob_c1 = float(float(getCount(c1, newUnigram, unigramClusters))/ N)
	prob_c2 = float(float(getCount(c2, newUnigram, unigramClusters))/ N)

	if 0 in [prob_c1, prob_c2, prob_c1_c2]: 
		return 0

	# print (prob_c1_c2 / (prob_c1 * prob_c2))
	lg = math.log((prob_c1_c2 / (prob_c1 * prob_c2)))

	return prob_c1_c2 * lg

def calculateI_C(N,  newBigram, newUnigram):
	"""
	I(C) = sum_c1,c2(p(c1,c2) * log((p(c1,c2))/(p(c1) * p(c2))))
	""" 
	# print "calculating I_C"
	I_C = 0 
	for c1, c2 in bigramClusters.keys():
		cluster_log_likelihood = getProbLog(c1, c2, N, newBigram, newUnigram)
		I_C += cluster_log_likelihood
	return I_C

def getProbLogW(w, N, vocab): 
	prob_w = float(float(vocab[w]) / N)
	lg = math.log(prob_w)
	return prob_w * lg

def calculate_H_W(N, vocab): 
	# print "calculating W_C"
	H_W = 0
	for w in vocab.keys():
		word_logProb = getProbLogW(w, N, vocab)
		H_W += word_logProb
	return H_W

def updateEnropy(N, c_from, c_to):
	newBigram = getUpdatedBigram(c_from, c_to)
	newUnigram = getUpdatedUnigram(c_from, c_to)
	I_C = calculateI_C(N, newBigram, newUnigram)
	H_C = calculate_H_W(N, unigramWords)
	return I_C, H_C, newBigram, newUnigram

def updateUnigram(newUnigram, merged, p, q):
	""" 
	""" 
	global unigramClusters
	for c, count in newUnigram.iteritems():
		if c == p: 
			unigramClusters[merged] = count
			unigramClusters[c] = 0
		if c == q: 
			unigramClusters[c] = 0



def updateBigram(newBigram, merged, p, q):
	global bigramClusters
	for c, count in newBigram.iteritems():
		if c[0] == p:
			bigramClusters[c] = 0
			bigramClusters[(merged, c[1])] = count
		if c[1] == p:
			bigramClusters[(c[0], merged)] = count
			bigramClusters[c] = 0
		if c[0] == q:
			bigramClusters[c] = 0
		if c[1] == q:
			bigramClusters[c] = 0



def assignClusters(N): 
	"""
	p : to
	q : from
	"""
	global clusteringDomain
	crossProducts = list(itertools.product(clusteringDomain, repeat = 2))
	crossProductsNonDup = [bi for bi in crossProducts if bi[0] != bi[1]]

	print "assigning clusters"
	## need to update bigram count 

	max_pair = (None, None)
	max_I_C = float('-inf')
	maxNewBigram = None
	maxNewUnigram = None 

	## note contains (c1, c1), ...
	
	for p, q in crossProductsNonDup: 
		I_C, H_C, newBigram, newUnigram = updateEnropy(N, q, p) 
		if I_C > max_I_C: 
			print "greater + ", p, q
			max_pair = (q, p)
			max_I_C = I_C
			maxNewBigram = newBigram
			maxNewUnigram = newUnigram 

	# print newBigram
	# print newUnigram
	# updateUnigram(maxNewUnigram)
	# updateBigram(maxNewBigram)
	return max_pair, maxNewBigram, maxNewUnigram

def mergeClusters(p, q): 
	"""
	p: to 
	q: from 
	"""

	global mergedClusters, unigramClusters, bigramClusters

	clusterName  = max(mergedClusters.keys())
	
	merged = clusterName + 1

	mergedClusters[merged] = [p, q]

	return merged


def updateClusterDomain(p, q, mergedCluster): 
	"""
	p: to 
	q: from
	should be originaly [1 - 201]
	"""

	global clusteringDomain

	clusteringDomain = clusteringDomain - set([p, q])
	clusteringDomain.add(mergedCluster)

def mergeAndUpdate(): 
	"""
	""" 
	global clusteringDomain, nextAdd, unigramWords
	N = sum(unigramWords.values())
	clusteringDomain.add(nextAdd)
	### should be using the clustering domain
	max_pair, maxNewBigram, maxNewUnigram = assignClusters(N)
	
	p, q = max_pair

	mergedCluster = mergeClusters(p, q)

	print "p: ", p, "q: ", q, "merged: ", mergedCluster

	updateUnigram(maxNewUnigram, mergedCluster, q, p)
	updateBigram(maxNewBigram, mergedCluster, q, p)
	updateClusterDomain(p, q, mergedCluster)
	nextAdd += 1

def mainLoop():
	print "in main"
	totalNumberOfWords = len(unigramWords.keys()) 
	while nextAdd <= totalNumberOfWords:
		mergeAndUpdate()
		print nextAdd 

mainLoop()




################################ the end 

def printBinaryString(st, cluster): 
	"""
	the one that works
	""" 
	if cluster is None: 
		return 

	left, right = mergedClusters[cluster]
	if right is None: 
		print  cluster2wordDict[left][0], "\t",  st
		return

	printBinaryString(st + '0', left)
	printBinaryString(st + '1', right)


def getAllBinary():
	global clusteringDomain
	# print "getting all the binary strings"

	for cluster in clusteringDomain: 
		print " ---------------------- preceded by common string for cluster", cluster
		printBinaryString('', cluster)

getAllBinary()
	
#################### testing ##############

# def checkBinary(): 
# 	global cluster2wordDict, mergedClusters, clusteringDomain, finalBinary
# 	words = ['w1', 'w2', 'w3', 'w4', 'w5', 'w6', 'w7', 'w8', 'w9', 'w10']
# 	cluster2wordDict = {1: ['w1'], 2:['w2'], 3: ['w3'], 4: ['w4'], 5: ['w5'], 6: ['w6'], 7: ['w7'], 8: ['w8'], 9: ['w9'], 10: ['w10']}
# 	mergedClusters = {1: [1, None], 2:[2, None], 3: [3, None], 4: [4, None],\
# 					 5: [5, None], 6: [6, None], 7: [7, None], 8: [8, None], 9: [9, None], 10: [10, None], 
# 					 11: [5, 3], 12: [6, 1], 13: [12, 2], 14: [8, 7], 15: [9, 14], 16: [15, 13]} 
# 	clusteringDomain = [16, 11, 4, 10]

# 	getAllBinary()


# checkBinary()
 
# print finalBinary








