from textanalyze.text_analyze import *

filecsv = "C:\\Users\\qitma\\Dropbox\\Tugas Akhir\\Dataset\\Data Twitter\\Praproses\\Qitma\\anies_json_7_sd_13_2_2017_tweets.csv"
filecsvanislatih = "C:\\Users\\qitma\\Dropbox\\Tugas Akhir\\Dataset\\Data Twitter\\Praproses\\Teh Wulan\\Data_latih_anies_sandiaga.csv"
filecsvanisuji = "C:\\Users\\qitma\\Dropbox\\Tugas Akhir\\Dataset\\Data Twitter\\Praproses\\Teh Wulan\\Data_uji_anies_sandiaga.csv"
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
textAnalyze = TextAnalyze()
textAnalyze.import_file_train_to_object(fileName=uji_coba2)
textAnalyze.preprocessing(fileEmoticon=fileEmoticon,negationWord=negation_word,stopWord=stop,listOfTweet=textAnalyze.list_of_train_tweet)
textAnalyze.initialize_profil_trait(fileName=fileProfilTrait,list_tr1=specialProfilTR1)
textAnalyze.group_by_sentiment(textAnalyze.list_of_train_tweet)
list_of_train_tokens = textAnalyze.initialize_train_tokens(textAnalyze.list_of_train_tweet)
textAnalyze.k_fold_cross_validation(textAnalyze.list_of_train_tweet,K=5,averaging="macro",multi_label=True)
#textAnalyze.k_fold_cross_validation(textAnalyze.list_of_train_tweet,K=5,averaging="macro",multi_label=False)
#textAnalyze.k_fold_cross_validation(textAnalyze.list_of_train_tweet,K=8,averaging="micro",multi_label=True)
#textAnalyze.k_fold_cross_validation(textAnalyze.list_of_train_tweet,K=8,averaging="micro",multi_label=False)

#-----------------------test data---------------------------------
# textAnalyze.feature_extraction(list_of_tokens=list_of_train_tokens,lot=textAnalyze.list_of_train_tweet)
# textAnalyze.grouping_profil(textAnalyze.list_of_train_tweet)
# #textAnalyze.print_train_tweet(textAnalyze.list_of_train_tweet)
# test_result = textAnalyze.test_data(fileTest=filecsvtest,fileEmoticon=fileEmoticon,negationWord=negation_word,stopWord=stop,train_token = list_of_train_tokens)
# conf_matrix = textAnalyze.make_confusion_matrix(test_result)
# textAnalyze.print_conf_matrix(conf_matrix)
# precision, recall, f_measure = textAnalyze.calculate_precision_recall_f1_measure(conf_matrix, averaging="macro")
# table = textAnalyze.evaluation_performance(param_eval1=precision, param_eval2=recall, param_eval3=f_measure, K=1,
#                                    multi_label=True, averaging="macro")
# textAnalyze.print_test_tweet(test_result)
# print(table)
#-------------end test data -------------------
#textAnalyze.print_test_tweet(test_result)

#textAnalyze.print_tweet()
#textAnalyze.print_token()
