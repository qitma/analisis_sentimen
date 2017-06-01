from textanalyze.text_analyze import *
from store import *
from collections import Counter

import copy
from term import Term
import timeit
import json
filecsv = "C:\\Users\\qitma\\Dropbox\\Tugas Akhir\\Dataset\\Data Twitter\\Praproses\\Qitma\\anies_json_7_sd_13_2_2017_tweets.csv"
filecsvanislatih = "C:\\Users\\qitma\\Dropbox\\Tugas Akhir\\Dataset\\Data Twitter\\Praproses\\Teh Wulan\\Data_latih_anies_sandiaga.csv"
filecsvanisuji = "C:\\Users\\qitma\\Dropbox\\Tugas Akhir\\Dataset\\Data Twitter\\Praproses\\Teh Wulan\\Data_uji_anies_sandiaga.csv"
filecsvtokoh = "C:\\Users\\qitma\\Dropbox\\Tugas Akhir\\Dataset\\Data Twitter\\Praproses\\Teh Wulan\\CSV\\Data_uji_ahok_djarot.csv"
filecsv1k = "C:\\Users\\qitma\\Dropbox\\Tugas Akhir\\Dataset\\Data Twitter\\Praproses\\Teh Wulan\\CSV_Data_latih\\Data_ahok_djarot.csv"
filecsvtest = "C:\\Users\\qitma\\Dropbox\\Tugas Akhir\\Dataset\\Data Twitter\\Praproses\\Qitma\\test_uji.csv"
filecsvtrain = "C:\\Users\\qitma\\Dropbox\\Tugas Akhir\\Dataset\\Data Twitter\\Praproses\\Qitma\\test_train.csv"
uji_coba2 = "C:\\Users\\qitma\\Dropbox\\Tugas Akhir\\Dataset\\Data Twitter\\Praproses\\Qitma\\uji_coba2.csv"
fileEmoticon ="C:\\Users\\qitma\\Dropbox\\Tugas Akhir\\Dataset\\Preprocessing\\convert_emoticon.csv"
fileStopWord = "C:\\Users\\qitma\Dropbox\\Tugas Akhir\\Dataset\\Preprocessing\\stop_word.txt"
fileProfilTrait = "C:\\Users\\qitma\\Dropbox\\Tugas Akhir\\Dataset\\Preprocessing\\eksplisit_fitur\\profil_trait.csv"
specialStopWords = {'anies','baswedan','sandiaga','uno'}
specialProfilTR1 = ['negasi_korupsi','korupsi']
negation_word = ['ga','tidak','tdk','jangan','jgn','ngggak','g','tak','gak','bukan']
sw=open(fileStopWord,encoding='utf-8', mode='r');stop=sw.readlines();sw.close()
stop=[kata.strip() for kata in stop];stop=set(stop)
stop=stop.union(specialStopWords)
tic=timeit.default_timer()
textAnalyze = TextAnalyze()
store = Store()
textAnalyze.list_of_train_tweet = textAnalyze.import_file_train_to_object(fileName=filecsv1k)
textAnalyze.preprocessing(fileEmoticon=fileEmoticon,negationWord=negation_word,stopWord=stop,listOfTweet=textAnalyze.list_of_train_tweet)
textAnalyze.initialize_profil_trait(fileName=fileProfilTrait,list_tr1=specialProfilTR1)

#------ test make profil ---------------
# textAnalyze.import_json_to_object('dataset/data_1000_anies_sandiaga.json')
# list_of_train_tokens = textAnalyze.initialize_train_tokens(textAnalyze.list_of_train_tweet)
# textAnalyze.feature_extraction(lot=textAnalyze.list_of_train_tweet,list_of_tokens=list_of_train_tokens)
# textAnalyze.grouping_profil(lot=textAnalyze.list_of_train_tweet)
# profil = textAnalyze.calculate_profil(textAnalyze.list_of_train_tweet)
# store.import_profil_to_excel(header_format=['Profil','Jumlah'],filename="profil_anies_sandi_1000",data=profil,sheet_name="profil")
# print(profil)
#----- end test make profil ===================
# namafile="data_1000_ahok_djarot"
# file = open('dataset/'+namafile+'.json', 'w')
# textAnalyze.toJSON(objFile=file)
# toc=timeit.default_timer()
# print("time for preprocessing:{}".format(toc - tic))
#-------------- test active learning -------------------- #
# textAnalyze.import_json_to_object('dataset/data_1000_anies_sandiaga.json')
#data_pool,data_train = textAnalyze.initial_data_train_and_data_pool(data_pool=textAnalyze.list_of_train_tweet,initial_ratio=10)
#textAnalyze.active_learning(data_pool=data_pool,data_train=data_train,query_count=100,cluster_k=100,is_interactive=False,data_gold=textAnalyze.list_of_train_tweet,max_query=10,K_fold=3)

#-------------- end test AL ------------------------------ #

#--------------- Initial data train dan data pool ----------------
# data_pool,data_train = textAnalyze.initial_data_train_and_data_pool(data_pool=textAnalyze.list_of_train_tweet,initial_ratio=20)
# print("data train ---------------")
# textAnalyze.print_train_tweet(data_train)
# print("data pool ---------------")
# textAnalyze.print_train_tweet(data_pool)
# print("length data pool:{},data train:{}".format(len(data_pool),len(data_train)))

#--------------- End data tarin dan data pool --------------------


#textAnalyze.k_fold_cross_validation(textAnalyze.list_of_train_tweet,K=3,averaging="macro",multi_label=True)
#textAnalyze.k_fold_cross_validation(textAnalyze.list_of_train_tweet,K=3,averaging="macro",multi_label=False)
#textAnalyze.k_fold_cross_validation(textAnalyze.list_of_train_tweet,K=8,averaging="micro",multi_label=True)
# textAnalyze.k_fold_cross_validation(textAnalyze.list_of_train_tweet,K=5,averaging="micro",multi_label=False)

#-----------------------test data---------------------------------
# #textAnalyze.print_train_tweet(textAnalyze.list_of_train_tweet)
# test_result = textAnalyze.test_data(fileTest=filecsvtest,fileEmoticon=fileEmoticon,negationWord=negation_word,stopWord=stop,train_token = list_of_train_tokens,train_tweet=textAnalyze.list_of_train_tweet)
# conf_matrix = textAnalyze.make_confusion_matrix(test_result)
# textAnalyze.print_conf_matrix(conf_matrix)
# precision, recall, f_measure = textAnalyze.calculate_precision_recall_f1_measure(conf_matrix, averaging="macro")
# table = textAnalyze.evaluation_performance(param_eval1=precision, param_eval2=recall, param_eval3=f_measure, K=1,
#                                    multi_label=True, averaging="macro")
# textAnalyze.print_test_tweet(test_result)
# print(table)
#-------------end test data -------------------

#------------- Clustering ---------------------
# tic=timeit.default_timer()
# textAnalyze.import_file_train_to_object(fileName=uji_coba2)
# textAnalyze.preprocessing(fileEmoticon=fileEmoticon,negationWord=negation_word,stopWord=stop,listOfTweet=textAnalyze.list_of_train_tweet)
# textAnalyze.initialize_profil_trait(fileName=fileProfilTrait,list_tr1=specialProfilTR1)
# textAnalyze.group_by_sentiment(textAnalyze.list_of_train_tweet)
# toc=timeit.default_timer()
# print("time for preprocessing:{}".format(toc - tic))
# namafile="ujicoba"
# file = open('dataset/'+namafile+'.json', 'w')
# textAnalyze.toJSON(filename=file)
# print("Data latih sudah dirubah ke json")
# tic2 = timeit.default_timer()
# textAnalyze.import_json_to_object('dataset/ujicoba.json')
# list_of_train_tokens = textAnalyze.initialize_train_tokens(textAnalyze.list_of_train_tweet)
# textAnalyze.feature_extraction(list_of_tokens=list_of_train_tokens,lot=textAnalyze.list_of_train_tweet)
# textAnalyze.grouping_profil(textAnalyze.list_of_train_tweet)
# #textAnalyze.print_train_tweet(textAnalyze.list_of_train_tweet)
# list_of_cluster = textAnalyze.clustering_k_means(textAnalyze.list_of_train_tweet,cluster_K=10)
# trans = textAnalyze.transform_cluster(list_of_cluster)
# #for key,value in trans.items():
#
# count_data_pool = sum(len(val) for key,val in trans.items())
# print(count_data_pool)
# #textAnalyze.print_train_tweet(trans_cluster)
# textAnalyze.print_cluster(loc=list_of_cluster)
# toc2=timeit.default_timer()
# print("time for clustering:{}".format(toc2 - tic2))
# print("time for All proses:{}".format(toc2 - tic))
#------------- end Clustering -----------------
#------------- Test Entopy --------------------
# textAnalyze.feature_extraction(list_of_tokens=list_of_train_tokens,lot=textAnalyze.list_of_train_tweet)
# textAnalyze.grouping_profil(textAnalyze.list_of_train_tweet)
# model = textAnalyze.naive_bayes_make_classification_model(lot=textAnalyze.list_of_train_tweet,train_token=list_of_train_tokens)
# list_of_test_tweet = []
# textAnalyze.import_file_train_to_object(fileName=filecsvtrain,lot=list_of_test_tweet)
# textAnalyze.preprocessing(fileEmoticon=fileEmoticon,negationWord=negation_word,stopWord=stop,listOfTweet=list_of_test_tweet)
# for tweet in list_of_test_tweet:
#     for term in tweet.filter_tweet:
#         term.weight =textAnalyze.calculate_tf(token=term.name, filterTweet=tweet.filter_tweet, isTrain=False)
# textAnalyze.grouping_profil(lot=list_of_test_tweet)
# textAnalyze.print_train_tweet(list_of_test_tweet)
# textAnalyze.print_train_tweet(textAnalyze.list_of_train_tweet)
# entp_tw = textAnalyze.uncertain_entropy(lot=list_of_test_tweet,model_classification=model)
# print("id_train:{},tweet:{}".format(entp_tw.train_id,entp_tw.tweet))
#----------- End Test Entropy -------------------
