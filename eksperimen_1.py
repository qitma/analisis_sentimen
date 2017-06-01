from textanalyze.text_analyze import *
from store import *

textAnalyze = TextAnalyze()
store = Store()
textAnalyze.import_json_to_object('dataset/data_1000_anies_sandiaga.json')
individual_data,whole_data,profil = textAnalyze.k_fold_cross_validation(textAnalyze.list_of_train_tweet,K=5,averaging="micro")
store.import_profil_to_excel(header_format=['Profil','Positif','Negatif','Netral'],filename="profil_anies_sandi_1000",data=profil,sheet_name="profil")

whole = {'performa':[value*100 for value in whole_data]}
print(whole)
store.import_profil_to_excel(header_format=['','Precision','Recall','F-Measure'],filename="performa_anies_sandi_1000",data=whole,sheet_name="performance")
individu = {}
list_perform_pos = []
list_perform_neg = []
list_perform_net = []
for i in range(3):
    list_perform_pos.append(individual_data[i][0]*100)
    list_perform_neg.append(individual_data[i][1]*100)
    list_perform_net.append(individual_data[i][2]*100)
individu['positif'] = list_perform_pos
individu['negatif'] = list_perform_neg
individu['netral'] = list_perform_net
print(individual_data)
print(individual_data[0][0])
print(individu)
store.import_profil_to_excel(header_format=['Kelas','Precision','Recall','F-Measure'],filename="performa_anies_sandi2_1000",data=individu,sheet_name="performance")