from textanalyze.text_analyze import *

filecsv = "C:\\Users\\qitma\\Dropbox\\Tugas Akhir\\Dataset\\Data Twitter\\Praproses\\Qitma\\anies_json_7_sd_13_2_2017_tweets.csv"
filecsvujicoba = "C:\\Users\\qitma\\Dropbox\\Tugas Akhir\\Dataset\\Data Twitter\\Praproses\\Qitma\\uji_coba.csv"
fileEmoticon ="C:\\Users\\qitma\\Dropbox\\Tugas Akhir\\Dataset\\Preprocessing\\convert_emoticon.csv"
fileStopWord = "C:\\Users\\qitma\Dropbox\\Tugas Akhir\\Dataset\\Preprocessing\\stop_word.txt"
fileProfilTrait = "C:\\Users\\qitma\\Dropbox\\Tugas Akhir\\Dataset\\Preprocessing\\eksplisit_fitur\\profil_trait.csv"
specialStopWords = {'anies','baswedan','sandiaga','uno'}
specialProfilTR1 = ['negasi_korupsi']
negation_word = ['ga','tidak','tdk','jangan','jgn','ngggak','g','tak','gak','bukan']
sw=open(fileStopWord,encoding='utf-8', mode='r');stop=sw.readlines();sw.close()
stop=[kata.strip() for kata in stop];stop=set(stop)
stop=stop.union(specialStopWords)
textAnalyze = TextAnalyze()
textAnalyze.import_file_train_to_object(fileName=filecsv)
textAnalyze.preprocessing(fileEmoticon=fileEmoticon,negationWord=negation_word,stopWord=stop)
textAnalyze.initialize_train_tokens()
#textAnalyze.print_filter_tweet_empty()
textAnalyze.feature_extraction(normalize_euclidean=False,normalize_frequencies=True,feature_selection = True)
textAnalyze.initialize_profil_trait(fileName=fileProfilTrait,list_tr1=specialProfilTR1)
textAnalyze.grouping_profil()
textAnalyze.print_tweet()
textAnalyze.print_token()
# textAnalyze.importFileTrainToObject(filecsv)
# test2 = textAnalyze.pp_tokenizing("saya pikir anies baswedan merupakan pemimpin hebat")
# test3 = textAnalyze.pp_convert_negation(test2, negation_word)
# test4 = textAnalyze.pp_remove_stopword(test3, stop)
# print(test4)
#textanalyze.preprocessing(fileEmoticon)
#textanalyze.printTweet()
# test2 = textanalyze.pptokenizing("saya tidak yakin dengan kamu")
# factory = StemmerFactory()
# stemmer = factory.create_stemmer()
#
# test3 = [textanalyze.ppstemming(word,stemmer) for word in test2]
# test4 = []
# print(test3)
#
# print(test4)

# test = "Detik Ini Penjelasan Anies Tentang Konsep Rumah Tanpa DP Cagub DKI Anies Baswedan ditanya masalah\xe2\x80\xa6\xe2\x80\xa6"
# test2 = textanalyze.ppconvertemoticon(test,fileEmoticon)
# test3 = test2.
# test4 = re.sub(r'[\\x80-\\xf0]+','', test3)
# print(test3)
# print(test4)


#test_str = "Hits\nJawaban Kocak Anies Baswedan Saat Dibilang Dia Adminnya Lambeturah Karena Nyinyirnya Ngeselin\n \\xe\\x0\\x2"
#str = test_str.rstrip('\n')
#str = re.sub('[\\n]+'," ",test_str)
#convertEmot = test_str.replace("\\xe\\x0\\x2"," senang ")
#print(convertEmot.strip('\\n'))


#print(textanalyze.listOfTrainTweet.__len__())