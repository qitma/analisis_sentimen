from textanalyze.text_analyze import *
from store import *
import timeit
textAnalyze_latih = TextAnalyze()
textAnalyze_uji = TextAnalyze()
store = Store()
#--------------------- Uji non AL -----------------------------------------------------
tic=timeit.default_timer()
textAnalyze_latih.import_json_to_object('dataset/data_latih_agus_sylvi.json')
textAnalyze_uji.import_json_to_object('dataset/data_uji_agus_sylvi.json')

for tweet in textAnalyze_uji.list_of_test_tweet:
    for term in tweet.filter_tweet:
        term.weight = textAnalyze_uji.calculate_tf(token=term.name, filterTweet=tweet.filter_tweet, isTrain=False)

textAnalyze_uji.grouping_profil(lot=textAnalyze_uji.list_of_test_tweet)
textAnalyze_latih.grouping_profil(lot=textAnalyze_latih.list_of_train_tweet)
print("lot:{}".format(len(textAnalyze_latih.list_of_train_tweet)))
print("lotest:{}".format(len(textAnalyze_uji.list_of_test_tweet)))
list_of_train_tokens = textAnalyze_latih.initialize_train_tokens(textAnalyze_latih.list_of_train_tweet)
textAnalyze_latih.feature_extraction(textAnalyze_latih.list_of_train_tweet, list_of_train_tokens)
textAnalyze_latih.grouping_profil(textAnalyze_latih.list_of_train_tweet)
model_classification = textAnalyze_latih.naive_bayes_make_classification_model(lot=textAnalyze_latih.list_of_train_tweet, train_token=list_of_train_tokens)
test_result = textAnalyze_uji.naive_bayes_determine_classification(lot=textAnalyze_uji.list_of_test_tweet, model=model_classification)
conf_matrix = textAnalyze_uji.make_confusion_matrix(lot=test_result)
textAnalyze_uji.print_conf_matrix(conf_matrix)
tp,fp,fn = textAnalyze_uji.calculate_precision_recall_f1_measure(conf_matrix=conf_matrix,averaging="micro")
#print("tp:{},fp:{},fn:{}".format(tp,fp,fn))
table1,data1 = textAnalyze_uji.evaluation_performance(param_eval1=tp,param_eval2=fp,param_eval3=fn,averaging="micro",multi_label=True,K=1)
table2,data2 = textAnalyze_uji.evaluation_performance(param_eval1=tp,param_eval2=fp,param_eval3=fn,averaging="micro",multi_label=False,K=1)
print(table1)
print(table2)
toc=timeit.default_timer()
print("time process :{}".format(toc - tic))