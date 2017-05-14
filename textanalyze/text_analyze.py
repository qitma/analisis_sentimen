import csv , sys , re , string , enum , math
from nltk import word_tokenize
from itertools import count

from nltk.tokenize import RegexpTokenizer
from Sastrawi.Stemmer.StemmerFactory import StemmerFactory
from collections import Counter

class Term(object):
    def __init__(cls):
        cls.id = None
        cls.name = None
        cls.tfidf = None
        cls.profil = None

class TrainTweet(object):
    def __init__(cls):
        cls.train_id = 0
        cls.id_str = None
        cls.created_date = None
        cls.tweet = None
        cls.filter_tweet = None
        cls.sentiment = None
        cls.profil = None

class TextAnalyze(object):
    def __init__(cls):
        cls.listOfTrainTweet = []
        cls.listOfEmoticon = []
        cls.listOfPositiveWord = []
        cls.listOfNegativeWord = []
        cls.listOfTrainTokens = []
        cls.listOfProfilTrait = []

    def import_file_train_to_object(cls, fileName):
        train_id = 1;
        with open(fileName) as csvfile:
            reader = csv.DictReader(csvfile)
            try:
                for row in reader:
                    objTrainTweet = TrainTweet()
                    objTrainTweet.train_id = train_id
                    objTrainTweet.id_str = row['id_str']
                    objTrainTweet.created_date = row['created_at']
                    objTrainTweet.tweet = row['tweet'].strip('\n')
                    objTrainTweet.filter_tweet=""
                    objTrainTweet.sentiment = row['label']
                    cls.listOfTrainTweet.append(objTrainTweet)
                    train_id+=1
            except csv.Error as e:
                sys.exit('file {}, line {}: {}'.format(fileName, reader.line_num, e))

    def print_token(cls):
        for token in cls.listOfTrainTokens:
            print("id_token:{id} ,nama_token :{name}, count_token:{count}".format(id=token.get('id'),name=token.get('name'),count=token.get('count')))

    def print_tweet(cls):
        objListOfTweet = cls.listOfTrainTweet
        for tw in objListOfTweet:
            print("id tweet: {} , id_train : {}, tweet : {} ,profil :{}" .format(tw.id_str,tw.train_id,tw.tweet,tw.profil))
            for term in tw.filter_tweet:
                print("id : {id} , term : {name} , tfidf :{tfidf}".format(id=term.id,name=term.name,tfidf=term.tfidf))

    def initialize_train_tokens(cls):
        """
        :return: membentuk token/kata yang unik dari daftar term pada data latih 
        """
        tokens = []
        list_of_trains_tokens = []
        for tweet in cls.listOfTrainTweet:
            tokens.extend(tweet.filter_tweet)
        list_of_trains_tokens = list(set(tokens))
        index = 1
        for token in list_of_trains_tokens:
            count_token = 0
            for tweet in cls.listOfTrainTweet:
                count_token = count_token + tweet.filter_tweet.count(token)
            if count_token >= 3:
                cls.listOfTrainTokens.append({"id":index, "name":token, "count":count_token})
                index+=1
        print('initialize tokens done..')

    def initialize_profil_trait(cls, fileName = None, list_tr1=[], list_tr2 = [], list_tr3 = [], list_tr4 = [], list_tr5 = [], list_tr6 = [], list_tr7 = []):
        with open(fileName,'r') as csvfile:
            reader = csv.DictReader(csvfile)
            try:
                for row in reader:
                    if row['TR1'] is not '':
                        list_tr1.append(row['TR1'].lower())
                    if row['TR2'] is not '':
                        list_tr2.append(row['TR2'].lower())
                    if row['TR3'] is not '':
                        list_tr3.append(row['TR3'].lower())
                    if row['TR4'] is not '':
                        list_tr4.append(row['TR4'].lower())
                    if row['TR5'] is not '':
                        list_tr5.append(row['TR5'].lower())
                    if row['TR6'] is not '':
                        list_tr6.append(row['TR6'].lower())
                    if row['TR7'] is not '':
                        list_tr7.append(row['TR7'].lower())
                trait = {}
                trait['TR1'] = list(list_tr1)
                trait['TR2'] = list(list_tr2)
                trait['TR3'] = list(list_tr3)
                trait['TR4'] = list(list_tr4)
                trait['TR5'] = list(list_tr5)
                trait['TR6'] = list(list_tr6)
                trait['TR7'] = list(list_tr7)
                cls.listOfProfilTrait.append(trait)
                print(cls.listOfProfilTrait)
            except csv.Error as e:
                sys.exit('file {}, line {}: {}'.format(fileName, reader.line_num, e))

    def grouping_profil(cls):
        for tweet in cls.listOfTrainTweet:
            if cls.is_filter_tweet_empty(tweet.filter_tweet) == False:
                for term in tweet.filter_tweet:
                    term.profil = cls.check_profil(term.name)
                    #print("term : {} , profil : {} , tfidf : {}".format(term.name,term.profil,term.tfidf))
                temp_term = max(tweet.filter_tweet, key=lambda term:term.tfidf)
                tweet.profil = temp_term.profil
                #print("temp_term : {} , profil : {} , tfidf : {}".format(temp_term.name, temp_term.profil, temp_term.tfidf))

    def check_profil(cls, term):
        for i in range(1,8):
            #print(self.listOfProfilTrait[0].get("TR"+str(i)))
            for trait in cls.listOfProfilTrait[0].get('TR'+str(i)):
                if term == trait:
                   return "TR"+str(i)

    def feature_extraction(cls, normalize_euclidean = False, normalize_frequencies = True, feature_selection = False):
        """
        :return: filter_tweet yang berisi object term, dimana untuk masing-masing
        object term pada sebuah tweet sudah diberi bobot tfidf 
        """
        if feature_selection:
            for tweet in cls.listOfTrainTweet:
                tweet.filter_tweet = cls.feature_selection(tweet.filter_tweet,cls.listOfTrainTokens)
        idf = cls.calculate_idf()
        for tweet in cls.listOfTrainTweet:
            tfidf = []
            temp_tfidf = []
            count = 0
            for token in idf.keys():
                temp_term = Term()
                if len(tweet.filter_tweet) > 0 :
                    tf = cls.calculate_tf(token, tweet.filter_tweet, normalized=normalize_frequencies)
                    #tfidf.append({token : float(tf * idf[token])})
                    temp_term.name = token
                    temp_term.tfidf = float(tf*idf[token])
                    if temp_term.tfidf > 0:
                        count+=1
                        temp_term.id = count
                        tfidf.append(temp_term)
            temp_tfidf.extend(tfidf)
            tweet.filter_tweet = list(temp_tfidf)
            if normalize_euclidean:
                tweet.filter_tweet = cls.calculate_norm_euclidean(tweet.filter_tweet)[:]
        print("feature extraction done...")

    def calculate_idf(cls):
        idf = {}
        for tokens in cls.listOfTrainTokens:
            count_token = 0
            for tweet in cls.listOfTrainTweet:
                if tokens['name'] in tweet.filter_tweet:
                    count_token+=1
            idf[tokens['name']] = float(math.log(len(cls.listOfTrainTweet) / count_token, 10)) + 1 #pake smoothing
        return idf

    def calculate_tf(cls, token, filterTweet, normalized = False):
        if not cls.is_filter_tweet_empty(filterTweet):
            return filterTweet.count(token)/len(filterTweet) if normalized else filterTweet.count(token)
        else:
            return 0
    def calculate_norm_euclidean(cls, listOfTerm):
        """
        :param listOfTerm: List of object term dengan attribute berupa name,tfidf dan profil 
        :return: List of object term yang nilai tfidf nya sudah dinormalisasi dengan euclidean
        """
        listOfL2 = listOfTerm
        denominator_value = 0
        for term_denominator in listOfL2:
            denominator_value = denominator_value + math.sqrt(math.pow(term_denominator.tfidf,2))

        for term in listOfL2:
            term.tfidf = term.tfidf/denominator_value

        return listOfL2

    def feature_selection(cls,filter_tweet,tokens):
        feature = []
        for term in filter_tweet:
            for token in tokens:
                if term == token.get("name"):
                    feature.append(term)
                    break
        return feature

    def is_filter_tweet_empty(cls,filter_tweet):
        if len(filter_tweet) == 0:
            return True
        else:
            #print("Filter Tweet Kosong!!")
            return False

    def print_filter_tweet_empty(cls):
        for tweet in cls.listOfTrainTweet:
            if len(tweet.filter_tweet) == 0:
                print("id :{id} , tweet:{tweet} , filter_tweet:{filter}".format(id=tweet.id_str,tweet=tweet.tweet,filter=tweet.filter_tweet))

    def preprocessing(cls, fileEmoticon, negationWord, stopWord):
        factory = StemmerFactory()
        stemmer = factory.create_stemmer()
        cls.initialize_emoticon(fileName=fileEmoticon)
        for tw in cls.listOfTrainTweet:
            tw.filter_tweet = cls.pp_cleansing(tweet = tw.tweet)
            tw.filter_tweet = cls.pp_case_folding(tweet = tw.filter_tweet)
            tw.filter_tweet = cls.pp_convert_emoticon(tweet = tw.filter_tweet)
            tw.filter_tweet = cls.pp_tokenizing(tweet = tw.filter_tweet)
            tw.filter_tweet = [cls.pp_convert_word(tweet = word) for word in tw.filter_tweet]
            tw.filter_tweet = [cls.pp_stemming(tweet = word, stemmer=stemmer) for word in tw.filter_tweet]
            tw.filter_tweet = cls.pp_convert_negation(tweet = tw.filter_tweet, negationWord=negationWord)
            tw.filter_tweet = cls.pp_remove_stopword(tweet=tw.filter_tweet, stopWord=stopWord)
        print("preprocessing done...")

    def initialize_emoticon(cls, fileName):
        with open(fileName,'r') as csvfile:
            reader = csv.DictReader(csvfile)
            try:
                for row in reader:
                    emoticon = {}
                    emoticon['UTF-8'] = row['UTF-8'].lower()
                    emoticon['Representasi'] = row['Representasi']
                    cls.listOfEmoticon.append(emoticon)
            except csv.Error as e:
                sys.exit('file {}, line {}: {}'.format(fileName, reader.line_num, e))

    def initilize_positif_word_negatif_word(cls, filePositive, fileNegative):
        pos = open(filePositive, encoding='utf-8', mode='r');
        positif = pos.readlines()
        pos.close()
        positif = [kata.strip() for kata in positif];
        cls.listOfPositiveWord = set(positif)
        #negatif word begin
        neg = open(fileNegative, encoding='utf-8', mode='r');
        negatif = neg.readlines()
        neg.close()
        negatif = [kata.strip() for kata in negatif];
        cls.listOfNegativeWord = set(negatif)


    def pp_cleansing(cls, tweet):
        """
        :param tweet:  berisi tweet/kalimat
        :return: tweet/kalimat yang sudah dihilangkan url, tanda binary, RT, mention dan tanda baca
        """
        tweet = tweet.rstrip('\n')
        url = re.sub('http[s]?://(?:[a-zA-Z]|[0-9]|[$-_@.&+]|[!*\(\),]|(?:%[0-9a-fA-F][0-9a-fA-F]))+'," ",tweet)
        #return ' '.join(re.sub("(^b)|(RT)|(@[A-Za-z0-9]+)|([^0-9A-Za-z\\\\ \t\n])", " ", url).split())
        return ' '.join(re.sub("(^b)|(RT)|(@[A-Za-z0-9]+)|([^0-9A-Za-z\\\\ \t\n])", " ", url).split())

    # def pp_convert_emoticon(self, tweet, fileName):
    #     with open(fileName,'r') as csvfile:
    #         convertEmot = tweet.encode('ascii').decode()
    #         reader = csv.DictReader(csvfile)
    #         try:
    #             for row in reader:
    #                 convertEmot = convertEmot.replace(row['UTF-8'].lower()," "+row['Representasi'] +" ")
    #             reConvertEmot = re.sub(r'\\[xX][0-9a-fA-F]+', '', convertEmot).strip('\n')
    #             reConvertEmot = reConvertEmot.replace('\\n'," ").replace('\\t'," ")
    #             return reConvertEmot
    #         except csv.Error as e:
    #             sys.exit('file {}, line {}: {}'.format(fileName, reader.line_num, e))
    def pp_convert_emoticon(cls, tweet):
        """
        :param tweet: berisi tweet/kalimat 
        :return: tweet/kalimat yang sudah dirubah kode emot menjadi kata, serta menghilangkan
        hexadecimal, angka, newline dan tab
        """
        convertEmot = tweet.encode('ascii').decode()
        for row in cls.listOfEmoticon:
            convertEmot = convertEmot.replace(row['UTF-8'].lower()," "+row['Representasi'] +" ")
        reConvertEmot = re.sub(r'\\[xX][0-9a-fA-F]+', '', convertEmot)
        reConvertEmot = re.sub(r'\d+', '', reConvertEmot)
        reConvertEmot = reConvertEmot.replace('\\n'," ").replace('\\t'," ")
        return reConvertEmot

    def pp_case_folding(cls, tweet):
        return tweet.lower()

    def pp_tokenizing(cls, tweet):
        #tokenizer = RegexpTokenizer(r'(\w+)')
        tokens = word_tokenize(text=tweet)
        for token in tokens:
            if len(token) <= 2:
                del token
        return tokens

    def pp_convert_word(cls, tweet):
        return re.sub(r'(.)\1+', r'\1\1', tweet) #(.)\1 merubah semua character yang iikuti dgn karakter yg sama menjadi 1 karakter \1

    def pp_stemming(cls, tweet, stemmer):
        if any(i in '_' for i in tweet) == False:
            return stemmer.stem(tweet)
        else:
            return tweet

    # def pp_convert_negation(self, tweet, negationWord):
    #     negationValue = 0
    #     for idx, word in enumerate(tweet):
    #         if word in negationWord:
    #             if idx < len(tweet) - 1:
    #                 if tweet[idx + 1] in self.listOfPositiveWord:
    #                     negationValue +=1 #jika nilai negation value positif, maka kemungkinan besar sentimen merupakan negatif
    #                 elif tweet[idx + 1] in self.listOfNegativeWord:
    #                     negationValue -=1 #jika nilai negation value negatif, maka kemungkinan besar sentimen merupakan positif
    #     return negationValue

    def pp_convert_negation(cls, tweet, negationWord):
        listOfConvertNegation = []
        for idx, word in enumerate(tweet):
            if word in negationWord:
                if idx < len(tweet) - 1:
                    word = "negasi_"+ tweet[idx+1]
                    del tweet[idx+1]
            listOfConvertNegation.append(word)
        return listOfConvertNegation

    def pp_remove_stopword(cls, tweet, stopWord):
        tweet_stop = [word for word in tweet if word not in stopWord]
        return tweet_stop

