from common import *
from sklearn.svm import *
from numpy import *
answer =[]
answer = [0,0,0,0,0,0,0,0,0,0,1,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,1,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,1,0,0,1,0,1,0,1,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,1,0,0,1,0,1,0,1,1,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,1,0,0,0,0,0,0,0,0,0,0,1,0,0,0,0,0,0,0,0,0,1,0,0,0,0,0,0,1,0,0,0,0,0,0,1,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,1,0,0,0,0,0,0,0,1,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,1,0,0,0,0,0,0,0,0,0,0,1,0,0,0,1,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,1,0,1,0,0,0,1,0,0,0,0,0,0,0,1,0,0,0,1,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,1,0,0,0,0,0,0,0,0,0,0,0,0,1,0,0,0,0,0,0,0,0,0,0,0,1,1,1,0,0,0,1,1,0,0,0,0,0,1,1,0,1,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0]
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
		self.trainMat = trainMat
		self.trainCat = trainCat

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
		testMat = []
		for key, value in testSet.dataset.items():
			# testVect = array(self.words2Vector(value, modelType))
			testVect = self.words2Vector(value, modelType)
			testMat.append(testVect)
			# p1 = sum(testVect * self.p1Vect) + log(self.p10)
			# p0 = sum(testVect * self.p0Vect) + log(1.0 - self.p10)
			# v = 0
			# if p1 > p0: v = 1
			# result[key] = [v, p1, p0]
			# result[key] = [v, p1, p0, value]
		self.testMat = testMat
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


	length = 3
	count = 2
	modelType = 0
	svm_r={}
	trs = TrainingDataset(list_arr_train, list_stopword, length, count)
	tes = Dataset(list_arr_test, list_stopword, length, count)
	p1, p0 = trs.prepare(list_train_cate, modelType, SAMPLING_RATE)
	result = trs.testing(tes, modelType)
	clf = NuSVC(kernel='linear').fit(trs.trainMat, trs.trainCat)
	predicted = clf.predict(trs.testMat)
	print("---")
	svm_r["match"] = 0
	svm_r["match0"] = 0
	svm_r["match1"] = 0
	svm_r["fp"] = 0
	svm_r["fn"] = 0
	for k in range(349):
		if(answer[k] == predicted[k]):
			svm_r["match"]+=1
			if(answer[k] == 0):
				svm_r["match0"]+=1
			else:
				svm_r["match1"]+=1
		if(answer[k] > predicted[k]): svm_r["fn"]+=1  #Predicted as Not Software issue even it's Software issue
		if(answer[k] < predicted[k]): svm_r["fp"]+=1  #Predicted as Software issue even it's not Software issue


	print("True Positive/False Positive/False Negative : {0} / {1} / {2}".format(svm_r["match"], svm_r["fp"], svm_r["fn"]))

	print("Match0 / Match1 / Number of 1 : {0} / {1} / {2}".format(svm_r["match0"], svm_r["match1"], (predicted == 1).sum()))


	# nb.analyze()
main()