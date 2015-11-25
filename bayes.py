from common import *

class TrainingDataset(Dataset):
	def __init__(self, listArray, stopWords, minimum_word_lengh, minimum_word_count, p1Key, p0Key):
		Dataset.__init__(self, listArray, stopWords, minimum_word_lengh, minimum_word_count)
		self.p1Vect = []; self.p0Vect = []; self.p10 = 0.0
		self.p1Numerator = []; self.p0Numerator = []
		self.p1Keywords = p1Key; self.p0Keywords = p0Key

	def words2Vector(self, lst, mode):
		result = [0]*len(self.wordList)
		for word in lst:
			if word in self.wordList: 
				if mode == 0: 	result[self.wordList.index(word)] += 1
				else: 			result[self.wordList.index(word)] = 1
		return result

	def prepare(self, cate, mode, ratio=100):
		trainKey = []; trainMat = []; trainCat = []
		samplKey=[]; samplMat=[]; samplCat = []

		for key, value in self.dataset.items():
			category = 0
			if key in cate: 
				category = 1
			trainKey.append(key)
			trainMat.append(self.words2Vector(value, mode))
			trainCat.append(category)

		if ratio < 100:
			for i in range(int(len(trainCat)*ratio/100)):
				randIndex = int(random.uniform(0,len(trainCat)))
				samplKey.append(trainKey[randIndex]); samplMat.append(trainMat[randIndex]); samplCat.append(trainCat[randIndex])
				del(trainKey[randIndex]); del(trainMat[randIndex]); del(trainCat[randIndex])
			trainKey = samplKey; trainMat = samplMat; trainCat = samplCat			
		########################################################################################
		# for key, value in self.dataset.items():
		# 	if key in cate: 
		# 		samplKey.append(key)
		# 		samplMat.append(self.words2Vector(value, mode))
		# 		samplCat.append(1)
		# 	else:
		# 		trainKey.append(key)
		# 		trainMat.append(self.words2Vector(value, mode))
		# 		trainCat.append(0)

		# if ratio < 100:
		# 	for i in range(int(len(samplCat)*(100-ratio)/100)):
		# 		randIndex = int(random.uniform(0,len(samplCat)))
		# 		del(samplKey[randIndex]); del(samplMat[randIndex]); del(samplCat[randIndex])

		# for i in range(int(len(samplCat))):
		# 	randIndex = int(random.uniform(0,len(trainCat)))
		# 	samplKey.append(trainKey[randIndex]); samplMat.append(trainMat[randIndex]); samplCat.append(trainCat[randIndex])
		# 	del(trainKey[randIndex]); del(trainMat[randIndex]); del(trainCat[randIndex])
		# trainKey = samplKey; trainMat = samplMat; trainCat = samplCat			
		########################################################################################

		self.p10 = sum(trainCat) / float(len(trainCat))

		vectLen = len(self.wordList)
		p1Num = ones(vectLen); p0Num = ones(vectLen)
		# p1Num = zeros(vectLen); p0Num = zeros(vectLen)

		p1Denom = 0.0; p0Denom = 0.0
		for i in range(len(trainCat)):
			if trainCat[i] == 1: p1Num += trainMat[i]; p1Denom += sum(trainMat[i])
			else:				 p0Num += trainMat[i]; p0Denom += sum(trainMat[i])

		if mode == 0: 	p1Denom += vectLen; 		 p0Denom += vectLen
		elif mode == 1: p1Denom = sum(trainCat); 	 p0Denom = (len(trainCat)-sum(trainCat))
		elif mode == 2: p1Denom += (p1Num>1).sum();	 p0Denom += (p0Num>1).sum()
		
		self.p1Numerator = p1Num; self.p0Numerator = p0Num

		return {"num": p1Num, "denom": p1Denom}, {"num":p0Num, "denom": p0Denom}

	def training(self, p1, p0, min1=0.0, weight1=0.0, min0=0.0, weight0=0.0):
		p1Num = p1['num']; p1Denom = p1['denom']
		p0Num = p0['num']; p0Denom = p0['denom']

		# p1a = (p1Num*(p1Num/p0Num >= min1).astype(float))*weight1
		# p0a = (p0Num*(p0Num/p1Num >= min0).astype(float))*weight0
		# self.p1Vect = log( (p1Num+p1a)/p1Denom ) 		
		# self.p0Vect = log( (p0Num+p0a)/p0Denom )
		for k in self.p1Keywords:	p1Num[self.wordList.index(k)] *= weight1
		for k in self.p0Keywords:	p0Num[self.wordList.index(k)] *= weight1

		p1a = (p1Num/p0Num >= min1).astype(float)*weight1 + (p1Num/p0Num < min1).astype(float)
		p0a = (p0Num/p1Num >= min0).astype(float)*weight0 + (p0Num/p1Num < min0).astype(float)
		self.p1Vect = log( (p1Num*p1a)/p1Denom ) 		
		self.p0Vect = log( (p0Num*p0a)/p0Denom )

	def testing(self, testSet, p1, p0, mode):
		result = {}; appendix = {}
		p1Denom = p1['denom']; p0Denom = p0['denom']
		for key, values in testSet.dataset.items():
			testVect = array(self.words2Vector(values, mode))
			missedCount = len(values)-testVect.sum()
			p1a = 0; p0a = 0;
			if missedCount != 0:
				p1a = log( (missedCount/p1Denom) )
				p0a = log( (missedCount/p0Denom) )
			p1 = sum(testVect*self.p1Vect) + p1a + log(self.p10)
			p0 = sum(testVect*self.p0Vect) + p0a + log(1.0-self.p10)

			v = 0
			if p1 > p0: v = 1
			result[key] = [v, p1, p0, testSet.dataLine.get(key)]
			appendix[key] = testSet.dataset.get(key)
		return result, appendix

	def report(self, result, checkList, debug, appendix):
		softList = []
		for k, v in result.items():
			if v[0] == 1: softList.append(k)
		truePositive = set(softList) & set(checkList)
		falsePositive = set(softList) - set(checkList)
		falseNegative = set(checkList) - set(softList)
		# print("- False Negative[{0}]: {1}".format(len(falseNegative), falseNegative))
		if debug == True:
			print("- True  Positive[{0}]: {1}".format(len(truePositive), truePositive))
			print("- False Positive[{0}]: {1}".format(len(falsePositive), falsePositive))
			print("- False Negative[{0}]: {1}".format(len(falseNegative), falseNegative))
			print("* Recall : {0}".format(len(truePositive)/float(len(checkList))*100))
			print("* Precision : {0}".format(len(truePositive)/float(len(softList))*100))

			for fn in falseNegative:
				wordVect = []
				max_word = str(); max_value = -1
				for word in appendix.get(fn):
					try:
						idx = self.wordList.index(word)
					except ValueError as e:
						diff = 0
					else:
						diff = self.p1Numerator[idx]/self.p0Numerator[idx]
						if max_value < diff:
							max_word = word; max_value = diff;
					if diff > 1 : 
						wordVect.append([word, diff])
				print("{0} : {1:10s}({2:3.4f}) : {3}".format(fn, max_word, max_value, wordVect))

		return len(truePositive), len(falsePositive), len(falseNegative)

	def resultSave(self, result, filename):
		# for k, v in result.items():
		# 	print(v)
		reports = {}
		reports["Report"] = sorted(dict2list(result), key=itemgetter(1), reverse=True)
		reportResult(reports, filename)
