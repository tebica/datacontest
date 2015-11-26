from bayes import *
from data import *

#MINIMUM_WORD_LENGTH = 6
#MINIMUM_WORD_COUNT  = 4
DEBUG = True
REPORT_FILE = False
SAMPLING_RATE = 100
ITERATION = 1 
BAYES_MODEL_TYPE = 0
			 	# 0 : Multinominal(Bag of Words)
				# 1 : Bernoulli 
				# 2 : User Defined


def subTest(trs, tes, p1, p0, modelType, length, count):
	tp = 0.0; fp = 0.0; fn = 0.0

	filename="Result.xlsx"
	max1 = int((p1['num']/p0['num']).max())
	max0 = int((p0['num']/p1['num']).max())
	# print("{0}:{1}".format(max1, max0))
	# print("{0}:{1}".format(p1['denom'], p0['denom']))
	p1IncidentRange = [26]
	p0IncidentRange = [126]
	# p1Weight = [int(p1['num'].max()), int(p1['denom']), int(p1['denom']*p0['denom']*10)]
	p1Weight = [int(p1['denom']**4)]
	p0Weight = [int(p0['num'].max())]

	for min0 in p0IncidentRange:
		for weight0 in p0Weight:
			for min1 in p1IncidentRange:
				for weight1 in p1Weight:
					trs.training(p1, p0, min1, weight1, min0, weight0)
					result, appendix = trs.testing(tes, p1, p0, modelType)
					tp, fp, fn = trs.report(result, list_test_cate, DEBUG, appendix)
					if DEBUG == False: print("[L:{0},C:{1},M:{2}] (P0:{3:5d},{4:5d}) (P1:{5:5d},{6:5d}) TP/FP/FN : {7:3d},{8:3d},{9:3d}".format(length,
					 										count, modelType, min0, weight0, min1, weight1, tp, fp, fn))

					if fp <= 9 and fn <= 4 or REPORT_FILE == True:
						filename = "result-"+str(fp)+"-"+str(fn)+".xlsx"
						trs.resultSave(result, filename)

def main():
	for length in [3]:
		for count in [2]:
			trs = TrainingDataset(list_arr_train, list_stopword, length, count, sKeywords, nKeywords)
			tes = Dataset(list_arr_test, list_stopword, length, count)
			for x in range(0, ITERATION):
				for modelType in range(0,1): 
					p1, p0 = trs.prepare(list_train_cate, modelType, SAMPLING_RATE)
					subTest(trs, tes, p1, p0, modelType, length, count)
	# nb.analyze()
main()