"""
getting the vocabulary from the corpra ordered based on frequency 

"""

from collections import Counter, OrderedDict
import operator

sentences = []

def removeTag(word): 
	"""
	removes the tag
	the word is in the format "abcd/tag"
	"""
	wl = word.split("/")
	return wl[0]

def readWordsFromFile(filename):
	"""
	reads the words from the file  (filename)
	filename is the file in the corpra
	"""
	filename = "brown/{}".format(filename)
	fileObject = open(filename, "r")
	low = []
	for line in fileObject: 
		l = line.split()
		l = map(lambda x: removeTag(x).lower(), l)
		# l = map(lambda x: x.lower(), l)
		low.extend(l)
	return low


def readFromFile(filename):
	filename = "brown/{}".format(filename)
	fileObject = open(filename, "r")
	low = []
	for line in fileObject: 
		low.append(line)
	return low

def readNamesFromFile(filename):
	fileObject = open(filename, "r")
	lol = []
	for line in fileObject: 
		l = line.split()
		lol.extend(l)
	return lol

def getWords(): 
	"""
	gets the word from the corpra 
	"""
	print "getting the words"
	lofn = readNamesFromFile("fileNames.txt")
	words = []
	for fn in lofn: 
		low = readWordsFromFile(fn)
		words.extend(low)
	return words 

def changeText(text):
	"""
	changes the text to get the sentences based on the /. tag 
	"""
	print "changing the text"
	# textN = re.sub(r'/[a-zA-Z$*+\x2d]+', '', text).lower()
	textNew = text.replace('./.', './. !END').replace('?/.', '?/. !END').replace('!/.', '!/. !END').replace(':/.', ':/. !END').replace(';/.', ';/. !END')
	s = textNew.split('!END')
	newSent = map(lambda x: map(lambda y: removeTag(y).lower(), x.split()), s)
	return newSent

def getSentences(): 
	"""
	gets the sentences from the corpra
	"""
	global sentences
	print "getting the sentences"
	lof = readNamesFromFile("fileNames.txt")
	words = []
	for fn in lof: 
		low = readFromFile(fn)
		words.extend(low)
	text = ' '.join(words)
	sentences = changeText(text)
	return sentences
	print "got the sentences"

def countFrequencyWords(words): 
	"""
	gets the frequency of the words in the corpra
	"""
	print "getting the frequency of the words"
	counted = Counter(words)
	return counted


def changeInfreq(counted, cutoff = 10):
	"""
	makes the infrequent words 
	"""
	print "changing infrequent"
	modified_counted = dict()
	for w, c in counted.iteritems(): 
		if c <= cutoff:
			w = "UNK"
		if modified_counted.has_key(w):
			modified_counted[w] += c
		else: 
			modified_counted[w] = c
	return modified_counted


def sortedDict(dic): 
	"""
	sorts the dictionary based on frequency and alphabetically 
	"""
	print "sorting dict"
	sorted_c = sorted(dic.items(), key = operator.itemgetter(0), reverse = False)
	sorted_cc = sorted(sorted_c, key = operator.itemgetter(1), reverse = True)
	return sorted_cc

def writeVocabRankedToFile(filename, dic):
	"""
	writes the ranked vocabulary to the file based on the filename 
	"""
	print "writing {} to the file {}".format('ModifiedWordWithCount', filename)
	d = sortedDict(dic)
	fo = open(filename, "w")
	for w, c in d:
		fo.write(str(w) + '\t')
		fo.write(str(c) + '\n')
	fo.close() 


def writeSentencesToFile(filename, lst): 
	"""
	writes the corpra sentences to one file with new lines 
	"""
	print "writing the sentences to {}".format(filename)
	fo = open(filename, "w")
	for sentence in lst:
		fo.write(' '.join(sentence) + '\n\n')
	fo.close()


def main():
	"""
	runs all the methods in the correct order 
	"""
	global sentences
	words = getWords()
	sentences = getSentences()
	print "len of sentences = ", len(sentences)
	counted = countFrequencyWords(words)
	modified_counted = changeInfreq(counted, cutoff = 10)
	writeVocabRankedToFile("corpraVocab.txt", modified_counted)
	writeSentencesToFile("corpraSentences.txt", sentences)

main()




