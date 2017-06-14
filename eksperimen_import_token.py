from textanalyze.text_analyze import *
from store import *

textAnalyze = TextAnalyze()
store = Store()
textAnalyze.import_json_to_object('dataset/data_1000_agus_sylvi.json')
token_pos,token_neg,token_net = textAnalyze.get_most_feature_class(textAnalyze.list_of_train_tweet)
store.import_token_to_excel(['ID','NAME','COUNT'],'test_token_anies',token_pos,token_neg,token_net,"tokens")