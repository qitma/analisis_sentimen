from textanalyze.text_analyze import *
from store import *

textAnalyze_latih_al = TextAnalyze()
textAnalyze_uji_al = TextAnalyze()
store = Store()
iter_number = 10
# textAnalyze_latih = TextAnalyze()
# textAnalyze_uji = TextAnalyze()
# store = Store()
# #--------------------- Uji non AL -----------------------------------------------------
# textAnalyze_latih.import_json_to_object('dataset/data_latih_anies_sandiaga.json')
# textAnalyze_uji.import_json_to_object('dataset/data_uji_anies_sandiaga.json')
#
# for tweet in textAnalyze_uji.list_of_test_tweet:
#     for term in tweet.filter_tweet:
#         term.weight = textAnalyze_uji.calculate_tf(token=term.name, filterTweet=tweet.filter_tweet, isTrain=False)
#
# textAnalyze_uji.grouping_profil(lot=textAnalyze_uji.list_of_test_tweet)
# textAnalyze_latih.grouping_profil(lot=textAnalyze_latih.list_of_train_tweet)
# list_of_train_tokens = textAnalyze_latih.initialize_train_tokens(textAnalyze_latih.list_of_train_tweet)
# textAnalyze_latih.feature_extraction(textAnalyze_latih.list_of_train_tweet, list_of_train_tokens)
# textAnalyze_latih.grouping_profil(textAnalyze_latih.list_of_train_tweet)
# model_classification = textAnalyze_latih.naive_bayes_make_classification_model(lot=textAnalyze_latih.list_of_train_tweet, train_token=list_of_train_tokens)
# test_result = textAnalyze_uji.naive_bayes_determine_classification(lot=textAnalyze_uji.list_of_test_tweet, model=model_classification)
# conf_matrix = textAnalyze_uji.make_confusion_matrix(lot=test_result)
# textAnalyze_uji.print_conf_matrix(conf_matrix)
# tp,fp,fn = textAnalyze_uji.calculate_precision_recall_f1_measure(conf_matrix=conf_matrix,averaging="micro")
# #print("tp:{},fp:{},fn:{}".format(tp,fp,fn))
# table1,data1 = textAnalyze_uji.evaluation_performance(param_eval1=tp,param_eval2=fp,param_eval3=fn,averaging="micro",multi_label=True,K=1)
# table2,data2 = textAnalyze_uji.evaluation_performance(param_eval1=tp,param_eval2=fp,param_eval3=fn,averaging="micro",multi_label=False,K=1)
# print(table1)
# print(table2)
#----------------------- UJI Active Learning -------------------------------------------
textAnalyze_latih_al.import_json_to_object('dataset/data_latih_agus_sylvi.json')
textAnalyze_uji_al.import_json_to_object('dataset/data_uji_agus_sylvi.json')
data_pool,data_train = textAnalyze_latih_al.initial_data_train_and_data_pool(data_pool=textAnalyze_latih_al.list_of_train_tweet, initial_data_train=100,cluster_k=100)
for tweet in textAnalyze_uji_al.list_of_test_tweet:
    for term in tweet.filter_tweet:
        term.weight = textAnalyze_uji_al.calculate_tf(token=term.name, filterTweet=tweet.filter_tweet, isTrain=False)
list_whole,list_individu,list_rekap_train = textAnalyze_latih_al.active_learning(data_pool=data_pool, data_train=data_train, data_test=textAnalyze_uji_al.list_of_test_tweet,
                                                                                 query_amount=100, cluster_k=100, is_interactive=False,
                                                                                 data_gold=textAnalyze_latih_al.list_of_train_tweet, max_query=10,
                                                                                 cluster_data_pool=False)
store.import_performa_batch_al(header_format=['', 'Precision', 'Recall', 'F-Measure'], filename="test_whole_al",
                                   batch_data=list_whole, worksheet_name="whole", iteration_number=iter_number)
store.import_performa_batch_al(header_format=['Kelas','Precision','Recall','F-Measure'], filename="test_individu_al",
                                   batch_data=list_individu, worksheet_name="individu", iteration_number=iter_number)
# store.import_performa_batch_al(header_format=['','Data Train','Sisa Data','Sisa Query','data per Query'], filename="test_train_al",
#                                    batch_data=list_rekap_train, worksheet_name="data train", iteration_number=iter_number)
store.import_performa_batch_al(header_format=['','Jumlah','',''], filename="test_train_al",
                                   batch_data=list_rekap_train, worksheet_name="data train", iteration_number=iter_number)

