import random
import argparse

################# Extract arguments ###########
parser = argparse.ArgumentParser()
parser.add_argument(dest = "fileName", default = "grammar1", type = str)
parser.add_argument(dest = "numSentences", default = 5, type = int)
args = parser.parse_args()
args = vars(args)
fileName = "{}.txt".format(args['fileName'])
numSentences = args['numSentences']

################################################

## this code uses weights and grammars to generate English sentences 

vocabulary = dict()

grammerRulesS1 = dict()

grammerRulesS2 = dict()

## {symbol: prob, ...}
startStates = dict()

## symbols that indicate a comment line (can be ignored)
comment = ['#']

## grammer being used 
usedGrammer = dict()

def openFile(fileName):
	"""
	""" 
	f = open(fileName)
	lines = []
	for line in f: 
		l = line.split()
		## when the line is just white space 
		if not l: 
			pass 
		## the line starts with the 
		elif l[0] in comment: 
			pass
		## sort the lines correctly
		else: 
			lines.append(l)

	return lines

def getStartLines(lines, comment, s0):
	start_lines = []
	for l in lines: 
		if (s0 in l) and (comment in l): 
			break
		start_lines.append(l)
	return start_lines


def getCorrectLines(lines, comment, s0, s1):
	ap = False
	correctLines = []
	for l in lines: 
		if (s0 in l) and (comment in l): 
			ap = True
		elif (s1 in l) and (comment in l):  
			ap = False
		elif ap:
			correctLines.append(l)
	return correctLines


def sortLines(lines):
	assert len(lines) != 0
	start_lines = getStartLines(lines, "##", "Beginning")
	grammar_lines_s1 = getCorrectLines(lines, "##", "Beginning", "End")
	vocabulary_lines = getCorrectLines(lines, "##", "Vocabulary", "S2")
	grammar_lines_s2 = getCorrectLines(lines, "##", "S2", "hi")
	return start_lines, grammar_lines_s1, vocabulary_lines, grammar_lines_s2


def populateDics(lst):
	global vocabulary, startStates, grammerRulesS1, grammerRulesS2
	starting, grammerS1, vocab, grammerS2 = lst
	for l in starting: 
		prob = int(l[0])
		startState = l[2]
		startStates[startState] = prob

	for l in grammerS1:
		prob = int(l[0])
		startPoint = l[1]
		rule = l[2:]
		rule_prob = (rule, prob)
		if grammerRulesS1.has_key(startPoint):
			grammerRulesS1[startPoint].append(rule_prob)
		else: 
			grammerRulesS1[startPoint] = [rule_prob]

	for l in grammerS2: 
		prob = int(l[0])
		startPoint = l[1]
		rule = l[2:]
		rule_prob = (rule, prob)
		if grammerRulesS2.has_key(startPoint): 
			grammerRulesS2[startPoint].append(rule_prob)
		else: 
			grammerRulesS2[startPoint] = [rule_prob]

	for l in vocab: 
		prob = int(l[0])
		tag = l[1]
		word = " ".join(l[2:])
		word_prob = (word, prob)
		if vocabulary.has_key(tag): 
			vocabulary[tag].append(word_prob)
		else: 
			vocabulary[tag] = [word_prob]

#===============================================
def chooseStartingSymbol(): 
	"""
	"""
	# return max(startStates, key=startStates.get)
	lst = makeStartDictList()
	return getThingy(lst)

def makeStartDictList(): 
	"""
	"""
	global startStates
	lst = []
	for symbol, weight in startStates.iteritems():
		s_w = (symbol, weight)
		lst.append(s_w)
	return lst

def getMaxStuff(word_list, mx):
	"""
	"""
	words = []
	for word, weight in word_list: 
		if weight == mx: 
			words.append(word)
	return words

def getRule(r): 
	assert r in usedGrammer
	lst = usedGrammer[r]
	rule = getThingy(lst)
	return rule

def getTerminal(st):
	# print "in getTerminal" s
	assert st in vocabulary
	word_list = vocabulary[st]
	word = getThingy(word_list)
	return word

def getThingy(lst):
	newList = []
	for t, weight in lst: 
		for _ in range(weight): 
			newList.append(t)
	thingy = random.choice(newList)
	return thingy

def getMaximizedRandom(lst): 
	weights = map(lambda x: x[1], lst)
	mx = max(weights)
	mx_lst = getMaxStuff(lst, mx)
	thingy = random.choice(mx_lst) 
	return thingy

def getTerminals(lol): 
	l = []
	# print lol
	for lst in lol: 
		if isinstance(lst, list):
			l.extend(getTerminals(lst))
		elif isinstance(lst, str): 
			if lst in vocabulary: 
				# print "in vocab {}".format(lst)
				st = getTerminal(lst)
				l.append(st)
			else: 
				"print ERROR not in vocab: {}".format(lst)
				l.append(lst)
	return l


#-----------------------------
def applyRule(rule): 
	lst = []
	for r in rule: 
		if isinstance(r, list): 
			appliedRule = applyRule(r)
			lst.append(appliedRule)
		elif isinstance(r, str):
			if r in usedGrammer: 
				rule = getRule(r)
				appliedRule = applyRule(rule)
				lst.append(appliedRule)
			else: 
				lst.append(r)
	return lst

def applyRules(rules): 
	lol = []
	for rule in rules: 
		lst = applyRule(rule)
		lol.append(lst)
	return lol

#######################################################

def applyGrammerRules(statingSymbol): 
	# print "------------using-------------- {}".format(statingSymbol) 
	rule1 = getRule(statingSymbol)
	lstRules = applyRule(rule1)
	lol = applyRules(lstRules)
	terminals = getTerminals(lol)
	sentence = " ".join(terminals)
	return sentence
		
def apply(): 
	global usedGrammer, grammerRulesS1, grammerRulesS2
	statingSymbol = chooseStartingSymbol()
	sentence = "NONE"
	if statingSymbol == 'S1': 
		usedGrammer = grammerRulesS1 
		sentence = applyGrammerRules(statingSymbol)
	elif statingSymbol == 'S2': 
		usedGrammer = grammerRulesS2 
		sentence = applyGrammerRules(statingSymbol)
	else: 
		### should not get here
		raise ValueError()
	print sentence

def run(file, numTimes): 
	lines = openFile(file)
	lst = sortLines(lines) 
	populateDics(lst)
	for _ in range(numTimes): 
		apply()
	
run(fileName, numSentences)