from textanalyze.text_analyze import *
from store import *
import timeit

textAnalyze_latih_al = TextAnalyze()
textAnalyze_uji_al = TextAnalyze()
store = Store()
iter_number = 10
#----------------------- UJI Active Learning -------------------------------------------
tic=timeit.default_timer()
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
toc=timeit.default_timer()
print("time process :{}".format(toc - tic))
