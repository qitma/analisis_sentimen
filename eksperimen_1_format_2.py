from textanalyze.text_analyze import *
from store import *

list_whole = []
list_individu = []
list_profil = []
fold_number = 10
for i in range(1,11):
    textAnalyze = TextAnalyze()
    store = Store()
    textAnalyze.import_json_to_object('dataset/data_1000_agus_sylvi.json')
    individual_data,whole_data,profil = textAnalyze.k_fold_cross_validation(textAnalyze.list_of_train_tweet,K=fold_number,averaging="micro")
    whole = {'performa':[value*100 for value in whole_data]}
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

    list_whole.append(whole)
    list_individu.append(individu)
    list_profil.append(profil)

store.import_profil_to_excel_batch(header_format=['Profil', 'Positif', 'Negatif', 'Netral'], filename="test_profil",
                                   batch_data=list_profil, worksheet_name="profil", fold_number=fold_number)
store.import_profil_to_excel_batch(header_format=['', 'Precision', 'Recall', 'F-Measure'], filename="test_whole",
                                   batch_data=list_whole, worksheet_name="performance", fold_number=fold_number)
store.import_profil_to_excel_batch(header_format=['Kelas','Precision','Recall','F-Measure'], filename="test_individu",
                                   batch_data=list_individu, worksheet_name="performance", fold_number=fold_number)