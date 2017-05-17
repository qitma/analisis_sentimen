import csv , sys , re , string , enum , math , copy
from nltk import word_tokenize
from itertools import count

from nltk.tokenize import RegexpTokenizer
from Sastrawi.Stemmer.StemmerFactory import StemmerFactory
from collections import Counter

class Term(object):
    def __init__(cls,name=None,id=None,weight = 0,profil = None):
        cls.id = id
        cls.name = name
        cls.weight = weight
        cls.profil = profil

class Tweet(object):
    def __init__(cls):
        cls.id_str = None
        cls.created_date = None
        cls.tweet = None
        cls.filter_tweet = None
        cls.profil = None
        cls.actual_sentiment = None

class TrainTweet(Tweet):
    def __init__(cls):
        cls.train_id = 0
        super().__init__()
        cls.status_train = None

class TestTweet(Tweet):
    def __init__(cls):
        cls.test_id = 0
        super().__init__()
        cls.predicted_sentiment = None

class Classification(object):
    def __init__(cls):
        cls.term_name = None
        cls.term_id = None
        cls.prob_pos = 0
        cls.prob_neg = 0
        cls.prob_net = 0

class TextAnalyze(object):
    def __init__(cls):
        cls.list_of_train_tweet = []
        cls.list_of_test_tweet = []
        cls.list_of_emoticon = []
        cls.list_of_positive_word = []
        cls.list_of_negative_word = []
        cls.list_of_train_tokens = []
        cls.list_of_profile_trait = []
        cls.list_of_classification = []

    def import_file_train_to_object(cls, fileName):
        train_id = 1
        with open(fileName) as csvfile:
            reader = csv.DictReader(csvfile)
            try:
                for row in reader:
                    obj_train_tweet = TrainTweet()
                    obj_train_tweet.train_id = train_id
                    obj_train_tweet.id_str = row['id_str']
                    obj_train_tweet.created_date = row['created_at']
                    obj_train_tweet.tweet = row['tweet'].strip('\n')
                    obj_train_tweet.filter_tweet=""
                    obj_train_tweet.actual_sentiment = row['label']
                    cls.list_of_train_tweet.append(obj_train_tweet)
                    train_id+=1
            except csv.Error as e:
                sys.exit('file {}, line {}: {}'.format(fileName, reader.line_num, e))

    def import_file_test_to_object(cls, fileName):
        test_id = 1
        with open(fileName) as csvfile:
            reader = csv.DictReader(csvfile)
            try:
                for row in reader:
                    obj_test_tweet = TrainTweet()
                    obj_test_tweet.test_id = test_id
                    obj_test_tweet.id_str = row['id_str']
                    obj_test_tweet.created_date = row['created_at']
                    obj_test_tweet.tweet = row['tweet'].strip('\n')
                    obj_test_tweet.filter_tweet= ""
                    obj_test_tweet.actual_sentiment = row['label']
                    obj_test_tweet.predicted_sentiment = ""
                    cls.list_of_test_tweet.append(obj_test_tweet)
                    test_id+=1
            except csv.Error as e:
                sys.exit('file {}, line {}: {}'.format(fileName, reader.line_num, e))

    def train_data(cls):
        pass

    def test_data(cls,fileTest = None,fileEmoticon = None, negationWord = None, stopWord = None):
        if len(cls.list_of_train_tweet) >  0:
            cls.import_file_test_to_object(fileName=fileTest)
            cls.preprocessing(fileEmoticon=fileEmoticon,negationWord=negationWord,stopWord=stopWord,listOfTweet=cls.list_of_test_tweet)
            for tweet in cls.list_of_test_tweet:
                total_prob = {"positif":0,"negatif":0,"netral":0}
                for term in tweet.filter_tweet:
                    term.weight = float(sum(1 for token in tweet.filter_tweet if token.name == term.name)/len(tweet.filter_tweet))
                    for cmp in cls.list_of_classification:
                        if term.name == cmp.term_name:
                            total_prob['positif'] += (cmp.prob_pos*term.weight)
                            total_prob['negatif'] += (cmp.prob_neg*term.weight)
                            total_prob['netral'] += (cmp.prob_net*term.weight)
                print("prob_pos :{},prob_neg:{},prob_net:{}".format(total_prob['positif'],total_prob['negatif'],total_prob['netral']))
                tweet.predicted_sentiment = min(total_prob, key=lambda key: total_prob[key])
        else:
            print("Please make data train first")


    def print_token(cls):
        for token in cls.list_of_train_tokens:
            print("id_token:{id} ,nama_token :{name}, count_token:{count} , idf:{idf}".format(id=token.get('id'),name=token.get('name'),count=token.get('count'),idf=token.get('idf')))

    def print_train_tweet(cls):
        for tw in cls.list_of_train_tweet:
            print("id tweet: {} , id_train : {}, tweet : {} ,profil :{}" .format(tw.id_str,tw.train_id,tw.tweet,tw.profil))
            for term in tw.filter_tweet:
                print("id : {id} , term : {name} , weight :{tfidf}".format(id=term.id,name=term.name,tfidf=term.weight))

    def print_test_tweet(cls):
        for tw in cls.list_of_test_tweet:
            print("id tweet: {} , id_train : {}, tweet : {} ,profil :{},actual sentiment:{} , predicted :{}" .format(tw.id_str,tw.train_id,tw.tweet,tw.profil,tw.actual_sentiment,tw.predicted_sentiment))
            for term in tw.filter_tweet:
                print("id : {id} , term : {name} , weight :{tfidf}".format(id=term.id,name=term.name,tfidf=term.weight))

    def print_cls(cls,loc=[]):
        for term in loc:
            print("term:{}, prob_pos:{}, prob_neg:{}, prob_net:{}".format(term.term_name,term.prob_pos,term.prob_neg,term.prob_net))

    def naive_bayes_classification(cls):
        cls.initialization_classification_model()
        cls.list_of_classification = copy.deepcopy(cls.naive_bayes_normalization(cls.naive_bayes_weight(cls.naive_bayes_complement())))


    def initialization_classification_model(cls):
        for tweet in cls.list_of_train_tweet:
            sentiment = tweet.actual_sentiment.lower()
            for term in tweet.filter_tweet:
                temp_cls = Classification()
                if cls.is_term_in_classification_empty(term.id):
                    temp_cls.term_name = term.name
                    temp_cls.term_id = term.id
                    cls.add_prob_by_sentiment(sentiment=sentiment,obj_cls=temp_cls, tfidf_value=term.weight)
                    cls.list_of_classification.append(temp_cls)
                else:
                    for term_prob in cls.list_of_classification:
                        if term_prob.term_id == term.id:
                            cls.add_prob_by_sentiment(sentiment=sentiment,obj_cls=term_prob, tfidf_value=term.weight)

    def naive_bayes_complement(cls):
        list_of_complement = []
        laplace_smooting = 1
        vocabulary = len(cls.list_of_train_tokens)
       # print("vocab:{}".format(vocabulary))
        total_complement_pos = 0
        total_complement_neg = 0
        total_complement_net = 0
        for term_cls in cls.list_of_classification:
            total_complement_pos += term_cls.prob_neg + term_cls.prob_net
            total_complement_neg += term_cls.prob_pos + term_cls.prob_net
            total_complement_net += term_cls.prob_pos + term_cls.prob_neg
        #print("total_pos:{},total_neg:{},total_net:{}".format(total_complement_pos,total_complement_neg,total_complement_net))
        for term_cls in cls.list_of_classification:
            complement = Classification()
            complement.term_name = term_cls.term_name
            complement.term_id = term_cls.term_id
            complement.prob_pos = float((term_cls.prob_neg+term_cls.prob_net+laplace_smooting)/(total_complement_pos+vocabulary))
            complement.prob_neg = float((term_cls.prob_pos+term_cls.prob_net+laplace_smooting)/(total_complement_neg+vocabulary))
            complement.prob_net = float((term_cls.prob_pos+term_cls.prob_neg+laplace_smooting)/(total_complement_net+vocabulary))
            list_of_complement.append(complement)
        # print("complement")
        # cls.print_cls(list_of_complement)

        return list_of_complement

    def naive_bayes_weight(cls,loc = []):
        list_of_weight = copy.deepcopy(loc)
        for term_cls in list_of_weight:
            term_cls.prob_pos = float(math.log(term_cls.prob_pos,10))
            term_cls.prob_neg = float(math.log(term_cls.prob_neg,10))
            term_cls.prob_net = float(math.log(term_cls.prob_net,10))

        # print("weight")
        # cls.print_cls(list_of_weight)
        return list_of_weight

    def naive_bayes_normalization(cls,low =[]):
        list_of_weight_normalization = copy.deepcopy(low)
        total_norm_pos = 0
        total_norm_neg = 0
        total_norm_net = 0
        for term_cls in low:
            total_norm_pos += math.fabs(term_cls.prob_pos)
            total_norm_neg += math.fabs(term_cls.prob_neg)
            total_norm_net += math.fabs(term_cls.prob_net)

        for term_cls in list_of_weight_normalization:
            term_cls.prob_pos =  float(term_cls.prob_pos/total_norm_pos)
            term_cls.prob_neg =  float(term_cls.prob_neg/total_norm_neg)
            term_cls.prob_net =  float(term_cls.prob_net/total_norm_net)

        # print("Normalize")
        # cls.print_cls(list_of_weight_normalization)
        return list_of_weight_normalization

    def is_term_in_classification_empty(cls,term_id):
        for term_prob in cls.list_of_classification:
            if term_prob.term_id == term_id:
                return False
            else:
                return True
        return True

    def add_prob_by_sentiment(cls,sentiment = None,obj_cls = Classification(),tfidf_value = None):
        if sentiment.lower() == "positif":
            obj_cls.prob_pos += tfidf_value
        elif sentiment.lower() == "negatif":
            obj_cls.prob_neg += tfidf_value
        else:
            obj_cls.prob_net += tfidf_value

    def group_term_by_sentiment(cls):
        pass

    #--------------------------- ekstraksi fitur ------------------------------------------
    #def initialize_term(cls):

    def initialize_train_tokens(cls,base_selection = 0):
        """
        :return: membentuk token/kata yang unik dari daftar term pada data latih 
        """
        tokens = []
        for tweet in cls.list_of_train_tweet:
            for term in tweet.filter_tweet:
                tokens.append(term.name)
        list_of_tokens = list(set(tokens))
        print(list_of_tokens)
        index = 1
        for token in list_of_tokens:
            count_token = 0
            for tweet in cls.list_of_train_tweet:
                #count = Counter(getattr(term,'name') for term in tweet.filter_tweet)
                #count_token = count_token + tweet.filter_tweet.count(token)
                #count_token += count[token]
                count_token += sum(1 for term in tweet.filter_tweet if term.name == token)
            if count_token >= base_selection:
                cls.list_of_train_tokens.append({"id":index, "name":token, "count":count_token , "idf": ""})
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
                cls.list_of_profile_trait.append(trait)
                print(cls.list_of_profile_trait)
            except csv.Error as e:
                sys.exit('file {}, line {}: {}'.format(fileName, reader.line_num, e))

    def grouping_profil(cls,lot = []):
        for tweet in lot:
            if cls.is_filter_tweet_empty(tweet.filter_tweet) == False:
                for term in tweet.filter_tweet:
                    term.profil = cls.check_profil(term.name)
                    #print("term : {} , profil : {} , tfidf : {}".format(term.name,term.profil,term.tfidf))
                temp_term = max(tweet.filter_tweet, key=lambda term:term.weight)
                tweet.profil = temp_term.profil
                #print("temp_term : {} , profil : {} , tfidf : {}".format(temp_term.name, temp_term.profil, temp_term.tfidf))

    def check_profil(cls, term):
        for i in range(1,8):
            #print(self.listOfProfilTrait[0].get("TR"+str(i)))
            for trait in cls.list_of_profile_trait[0].get('TR'+str(i)):
                if term == trait:
                   return "TR"+str(i)

    def feature_extraction(cls, normalize_euclidean = True, feature_selection = True):
        """
        :return: filter_tweet yang berisi object term, dimana untuk masing-masing
        object term pada sebuah tweet sudah diberi bobot tfidf 
        """
        if feature_selection:
            for tweet in cls.list_of_train_tweet:
                tweet.filter_tweet = cls.feature_selection(tweet.filter_tweet, cls.list_of_train_tokens)
        cls.calculate_idf()
        # for tweet in cls.list_of_train_tweet:
        #     tfidf = []
        #     temp_tfidf = []
        #     for token in cls.list_of_train_tokens:
        #         temp_term = Term()
        #         if len(tweet.filter_tweet) > 0 :
        #             tf = cls.calculate_tf(token['name'], tweet.filter_tweet)
        #             #tfidf.append({token : float(tf * idf[token])})
        #             temp_term.name = token
        #             temp_term.weight = float(tf * token['idf'])
        #             if temp_term.weight > 0:
        #                 temp_term.id = token['id']
        #                 tfidf.append(temp_term)
        #     temp_tfidf.extend(tfidf)
        #     tweet.filter_tweet = list(temp_tfidf)
        for tweet in cls.list_of_train_tweet:
            for term in tweet.filter_tweet:
                tf = cls.calculate_tf(term.name, tweet.filter_tweet)
                idf = 0
                id = 0
                for token in cls.list_of_train_tokens:
                    if token['name'] == term.name:
                        idf = token['idf']
                        id = token['id']
                # tfidf.append({token : float(tf * idf[token])})
                term.weight = float(tf * idf )
                print("term {} f:{} ,idf:{} , tf-idf:{}".format(term.name, tf,idf,term.weight))
                term.id = id
                if term.weight <= 0 :
                    del term
            if normalize_euclidean:
                #tweet.filter_tweet = cls.calculate_norm_euclidean(tweet.filter_tweet)[:]
                cls.calculate_norm_euclidean(tweet.filter_tweet)
        print("feature extraction done...")

    def calculate_idf(cls):
        for tokens in cls.list_of_train_tokens:
            count_token = 0
            for tweet in cls.list_of_train_tweet:
                #count = Counter(getattr(term, 'name') for term in tweet.filter_tweet)
                #if tokens['name'] in tweet.filter_tweet:
                if any(term.name == tokens['name'] for term in tweet.filter_tweet):
                    count_token+=1
            if count_token == 0:
                count_token = 1
            tokens['idf'] = float(math.log(len(cls.list_of_train_tweet) / count_token, 10))

    def calculate_tf(cls, token, filterTweet):
        laplace_smoothing = 1
        if not cls.is_filter_tweet_empty(filterTweet):
            #return float(math.log(filterTweet.count(token)+laplace_smoothing,10))
            return float(math.log(sum(1 for term in filterTweet if term.name == token)+laplace_smoothing,10))
        else:
            return 0

    def calculate_norm_euclidean(cls, listOfTerm):
        """
        :param listOfTerm: List of object term dengan attribute berupa name,weight dan profil 
        :return: List of object term yang nilai weight nya sudah dinormalisasi dengan euclidean
        """
        denominator_value = 0
        for term_denominator in listOfTerm:
            denominator_value = denominator_value + float(math.pow(term_denominator.weight,2))

        for term in listOfTerm:
            term.weight = float(term.weight/math.sqrt(denominator_value))


    def feature_selection(cls,filter_tweet,tokens):
        feature = []
        for term in filter_tweet:
            temp_term = Term(term.name)
            for token in tokens:
                if term.name == token.get("name"):
                    feature.append(temp_term)
                    break
        return feature

    def is_filter_tweet_empty(cls,filter_tweet):
        if len(filter_tweet) == 0:
            return True
        else:
            #print("Filter Tweet Kosong!!")
            return False

    def print_filter_tweet_empty(cls):
        for tweet in cls.list_of_train_tweet:
            if len(tweet.filter_tweet) == 0:
                print("id :{id} , tweet:{tweet} , filter_tweet:{filter}".format(id=tweet.id_str,tweet=tweet.tweet,filter=tweet.filter_tweet))

    #-------------------------- Preprocessing ---------------------------------

    def preprocessing(cls, fileEmoticon, negationWord, stopWord,listOfTweet):
        factory = StemmerFactory()
        stemmer = factory.create_stemmer()
        cls.initialize_emoticon(fileName=fileEmoticon)
        for tw in listOfTweet:
            tw.filter_tweet = cls.pp_cleansing(tweet = tw.tweet)
            tw.filter_tweet = cls.pp_case_folding(tweet = tw.filter_tweet)
            tw.filter_tweet = cls.pp_convert_emoticon(tweet = tw.filter_tweet)
            tw.filter_tweet = cls.pp_tokenizing(tweet = tw.filter_tweet)
            tw.filter_tweet = [cls.pp_convert_word(tweet = word) for word in tw.filter_tweet]
            tw.filter_tweet = cls.pp_remove_stopword(tweet=tw.filter_tweet, stopWord=stopWord,negationWord=negationWord)
            tw.filter_tweet = [cls.pp_stemming(tweet = word, stemmer=stemmer) for word in tw.filter_tweet]
            tw.filter_tweet = cls.pp_convert_negation(tweet=tw.filter_tweet, negationWord=negationWord)
            list_of_term = []
            for fitur in tw.filter_tweet:
                temp_term = Term(name = fitur)
                list_of_term.append(temp_term)
            tw.filter_tweet = copy.deepcopy(list_of_term)
            # for term in tw.filter_tweet:
            #     print("name :{}".format(term.name))

        print("preprocessing done...")

    def initialize_emoticon(cls, fileName):
        with open(fileName,'r') as csvfile:
            reader = csv.DictReader(csvfile)
            try:
                for row in reader:
                    emoticon = {}
                    emoticon['UTF-8'] = row['UTF-8'].lower()
                    emoticon['Representasi'] = row['Representasi']
                    cls.list_of_emoticon.append(emoticon)
            except csv.Error as e:
                sys.exit('file {}, line {}: {}'.format(fileName, reader.line_num, e))

    def initilize_positif_word_negatif_word(cls, filePositive, fileNegative):
        pos = open(filePositive, encoding='utf-8', mode='r');
        positif = pos.readlines()
        pos.close()
        positif = [kata.strip() for kata in positif];
        cls.list_of_positive_word = set(positif)
        #negatif word begin
        neg = open(fileNegative, encoding='utf-8', mode='r');
        negatif = neg.readlines()
        neg.close()
        negatif = [kata.strip() for kata in negatif];
        cls.list_of_negative_word = set(negatif)

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
        for row in cls.list_of_emoticon:
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

    def pp_remove_stopword(cls, tweet, stopWord, negationWord):
        tweet_stop=[]
        for idx,word in enumerate(tweet):
            if idx == 0:
                if word not in stopWord:
                    tweet_stop.append(word)
            else:
                if word in stopWord and tweet[idx-1] in negationWord:
                    tweet_stop.append(word)
                elif word not in stopWord:
                    tweet_stop.append(word)
        #tweet_stop = [word for idx,word in tweet if word not in stopWord and word[idx-1] not in negationWord]
        print(tweet_stop)
        return tweet_stop

