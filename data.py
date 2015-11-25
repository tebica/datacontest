from common import *
from stop_words import get_stop_words

dic_filter = loadExcel("./Filter.xlsx")
list_stopword = dic_filter["Stop Word List"]
list_stopword.extend(dic_filter["Self Defined"])
list_stopword.extend(get_stop_words("en"))
list_stopword = set(list_stopword)
keywords = dic_filter["Stress Soft"]

list_arr_train = readFile('./FLAT_RCL_Out_14.txt')
list_train_cate = loadExcel('./2014RecallNo_Soft.xlsx')['Sheet1']
list_arr_test = readFile('./FLAT_RCL_Out_15.txt')
list_test_cate = dic_filter["Check List"]

