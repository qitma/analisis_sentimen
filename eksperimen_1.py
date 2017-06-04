from textanalyze.text_analyze import *
from store import *

# textAnalyze = TextAnalyze()
# store = Store()
# textAnalyze.import_json_to_object('dataset/data_1000_anies_sandiaga.json')
# individual_data,whole_data,profil = textAnalyze.k_fold_cross_validation(textAnalyze.list_of_train_tweet,K=8,averaging="micro")
# store.import_profil_to_excel(header_format=['Profil','Positif','Negatif','Netral'],filename="profil_anies_sandi_1000",data=profil,sheet_name="profil")
#
# whole = {'performa':[value*100 for value in whole_data]}
# print(whole)
# store.import_profil_to_excel(header_format=['','Precision','Recall','F-Measure'],filename="performa_anies_sandi_1000",data=whole,sheet_name="performance")
# individu = {}
# list_perform_pos = []
# list_perform_neg = []
# list_perform_net = []
# for i in range(3):
#     list_perform_pos.append(individual_data[i][0]*100)
#     list_perform_neg.append(individual_data[i][1]*100)
#     list_perform_net.append(individual_data[i][2]*100)
# individu['positif'] = list_perform_pos
# individu['negatif'] = list_perform_neg
# individu['netral'] = list_perform_net
# print(individual_data)
# print(individual_data[0][0])
# print(individu)
# store.import_profil_to_excel(header_format=['Kelas','Precision','Recall','F-Measure'],filename="performa_anies_sandi2_1000",data=individu,sheet_name="performance")

textAnalyze_latih = TextAnalyze()
textAnalyze_uji = TextAnalyze()
store = Store()
#--------------------- Uji non AL -----------------------------------------------------
textAnalyze_latih.import_json_to_object('dataset/data_latih_anies_sandiaga.json')
textAnalyze_uji.import_json_to_object('dataset/data_uji_anies_sandiaga.json')

for tweet in textAnalyze_uji.list_of_test_tweet:
    for term in tweet.filter_tweet:
        term.weight = textAnalyze_uji.calculate_tf(token=term.name, filterTweet=tweet.filter_tweet, isTrain=False)

textAnalyze_uji.grouping_profil(lot=textAnalyze_uji.list_of_test_tweet)
textAnalyze_latih.grouping_profil(lot=textAnalyze_latih.list_of_train_tweet)
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