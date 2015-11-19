from common import *

class TrainingDataset(Dataset):
	def __init__(self, listArray, stopWords, minimum_word_lengh, minimum_word_count):
		Dataset.__init__(self, listArray, stopWords, minimum_word_lengh, minimum_word_count)
		self.p1Vect = []; self.p0Vect = []; self.p10 = 0.0

	def words2Vector(self, lst, modelType):
		result = [0]*len(self.wordList)
		for word in lst:
			if word in self.wordList: 
				if modelType == 0: 	result[self.wordList.index(word)] += 1
				else: 				result[self.wordList.index(word)] = 1
		return result

	def prepare(self, cate, modelType, ratio=100):
		trainKey = []; trainMat = []; trainCat = []
		for key, value in self.dataset.items():
			category = 0
			if key in cate: category = 1
			trainKey.append(key)
			trainMat.append(self.words2Vector(value, modelType))
			trainCat.append(category)

		self.trainSize = len(trainCat)
		self.trainPositive = sum(trainCat)
		self.p10 = sum(trainCat) / float(len(trainCat))

		if ratio < 100:
			samplKey=[]; samplMat=[]; samplCat = []
			for i in range(int(len(trainCat)*ratio/100)):
				randIndex = int(random.uniform(0,len(trainCat)))
				samplKey.append(trainKey[randIndex]); samplMat.append(trainMat[randIndex]); samplCat.append(trainCat[randIndex])
				del(trainKey[randIndex]); del(trainMat[randIndex]); del(trainCat[randIndex])
			trainKey = samplKey; trainMat = samplMat; trainCat = samplCat			

		p1Num = ones(len(self.wordList)); p0Num = ones(len(self.wordList))
		p1Denom = 0.0; p0Denom = 0.0
		for i in range(len(trainCat)):
			if trainCat[i] == 1: p1Num += trainMat[i]; p1Denom += sum(trainMat[i])
			else:				 p0Num += trainMat[i]; p0Denom += sum(trainMat[i])

		if modelType == 0: 	 p1Denom += (p1Num>1).sum(); p0Denom += (p0Num>1).sum()
		elif modelType == 1: p1Denom += 2.0; 			 p0Denom += 2.0
		elif modelType == 2: p1Denom = sum(trainCat); 	 p0Denom = (len(trainCat)-sum(trainCat))
		elif modelType == 3: p1Denom = p1Denom;  		 p0Denom = p0Denom

		return {"num": p1Num, "denom": p1Denom}, {"num":p0Num, "denom": p0Denom}

	def training(self, p1, p0, min1=0.0, weight1=0.0, min0=0.0, weight0=0.0):
		p1a = (p1['num']*(p1['num']/p0['num'] >= min1).astype(float))*p1['denom']*weight1/10
		p0a = (p0['num']*(p0['num']/p1['num'] >= min0).astype(float))*p0['denom']*weight0/10

		self.p1Vect = log((p1['num']+p1a)/p1['denom'])		
		self.p0Vect = log((p0['num']+p0a)/p0['denom'])

	def testing(self, testSet, modelType):
		result = {}
		for key, value in testSet.dataset.items():
			testVect = array(self.words2Vector(value, modelType))
			p1 = sum(testVect * self.p1Vect) + log(self.p10)
			p0 = sum(testVect * self.p0Vect) + log(1.0 - self.p10)
			v = 0
			if p1 > p0: v = 1
			result[key] = [v, p1, p0]
			# result[key] = [v, p1, p0, value]
		return result

	def report(self, result, checkList, filename, debug, report_file):
		softList = []
		for k, v in result.items():
			if v[0] == 1: 
				softList.append(k)
		truePositive = set(softList) & set(checkList)
		falsePositive = set(softList) - set(checkList)
		falseNegative = set(checkList) - set(softList)
		# print("- False Negative[{0}]: {1}".format(len(falseNegative), falseNegative))
		if debug == True:
			print("* Sample(Postive)/Total : {0}({1})/{2}".format(self.trainPositive, self.trainSize, len(self.dataset)))
			print("- True  Positive[{0}]: {1}".format(len(truePositive), truePositive))
			print("- False Positive[{0}]: {1}".format(len(falsePositive), falsePositive))
			print("- False Negative[{0}]: {1}".format(len(falseNegative), falseNegative))
			print("* Recall : {0}".format(len(truePositive)/float(len(checkList))*100))
			print("* Precision : {0}".format(len(truePositive)/float(len(softList))*100))
		if report_file == True:
			reports = {}
			reports["Report"] = sorted(dict2list(result), key=itemgetter(1), reverse=True)
			reportResult(reports, filename)
		return len(truePositive), len(falsePositive), len(falseNegative)

def main():
	MINIMUM_WORD_LENGTH = 6
	MINIMUM_WORD_COUNT  = 3
	BAYES_MODEL_TYPE = 1 
				 	# 0 : Multinominal(Bag of Words)
					# 1 : Set of Words
					# 2 : Bernoulli 
					# 3 : User Defined
	SAMPLING_RATE = 100
	DEBUG = False
	REPORT_FILE = False
	ITERATiON = 1

	dic_filter = loadExcel("./Filter.xlsx")
	list_stopword = dic_filter["Stop Word List"]
	list_stopword.extend(dic_filter["Self Defined"])

	list_arr_train = readFile('./FLAT_RCL_Out_14.txt')
	list_train_cate = loadExcel('./2014RecallNo_Soft.xlsx')['Sheet1']
	list_arr_test = readFile('./FLAT_RCL_Out_15.txt')
	list_test_cate = dic_filter["Check List"]

	tp = 0.0; fp = 0.0; fn = 0.0
	max_tp = 0.0; min_fp = 10000; min_fn = 10000
	print("Start...")
	for length in range(0,4):
		for count in range(0,4):
			trs = TrainingDataset(list_arr_train, list_stopword, length, count)
			tes = Dataset(list_arr_test, list_stopword, length, count)
			for modelType in range(0,4): 
				p1, p0 = trs.prepare(list_train_cate, modelType, SAMPLING_RATE)
				max1 = int((p1['num']/p0['num']).max()); max0 = int((p0['num']/p1['num']).max())
				filename = "model-"+ str(modelType) + ".xlsx"
				for min0 in range(0, 1):
				# for min0 in range(max0, 1, int(max0*-0.3)):
					for weight0 in range(0,10,10):
						# for min1 in range(max1, 15, -12):
						for min1 in range(0, 1):
							for weight1 in range(0, 10, 10):
								trs.training(p1, p0, min1, weight1, min0, weight0)
								result = trs.testing(tes, modelType)
								tp, fp, fn = trs.report(result, list_test_cate, filename, DEBUG, REPORT_FILE)
								if tp > max_tp: max_tp = tp
								if fp < min_fp: min_fp = fp
								if fn < min_fn: min_fn = fn
								# if tp > 20 and fn < 20 and fp < 20: 
								print("[L:{0},C:{1},M:{2}] (P0:{3:3d},{4:3d}) (P1:{5:3d},{6:3d}) TP/FP/FN : {7:3d},{8:3d},{9:3d}".format(length, 
													count, modelType, min0, weight0, min1, weight1, tp, fp, fn))
	print("True Positive/False Positive/False Negative : {0}/{1}/{2}".format(max_tp, min_fp, min_fn))
	# nb.analyze()
main()