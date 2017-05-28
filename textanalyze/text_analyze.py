import csv , sys , re , string , enum , math , copy
import json
from nltk import word_tokenize
from prettytable import PrettyTable
#from itertools import count
import random
from collections import Counter
#from nltk.tokenize import RegexpTokenizer
from Sastrawi.Stemmer.StemmerFactory import StemmerFactory

from tweet import Tweet
from traintweet import TrainTweet
from testtweet import TestTweet
from term import Term
from classification import Classification
from clustering import  Clustering

class TextAnalyze(object):
    def __init__(cls):
        cls.list_of_train_tweet = []
        cls.list_of_test_tweet = []
        cls.list_of_emoticon = []
        # cls.list_of_positive_word = []
        # cls.list_of_negative_word = []
        cls.list_of_data_pool = []
        cls.list_of_gold_tweet = []
        cls.list_of_train_tokens = []
        cls.list_of_profile_trait = []
        cls.list_of_classification = []
        cls.list_of_train_tweet_pos = []
        cls.list_of_train_tweet_neg = []
        cls.list_of_train_tweet_net = []
        #cls.list_of_cluster_tweet = []

    def toJSON(self,filename):
        return json.dump(self,filename, default=lambda o: o.__dict__,
            sort_keys=True, indent=4)

    def import_json_to_object(cls,file):
        with open(file) as data_file:
            data = json.load(data_file)

        for data_cls in data['list_of_classification']:
            temp_cls = Classification(
                name=data_cls['term_name'],
                id=data_cls['term_id'],
                prob_pos=data_cls['prob_pos'],
                prob_neg=data_cls['prob_neg'],
                prob_net=data_cls['prob_net']
            )
            cls.list_of_classification.append(temp_cls)

        for data_emot in data['list_of_emoticon']:
            cls.list_of_emoticon.append(data_emot)

        for jtweet in data['list_of_test_tweet']:
            list_of_filter = cls.json_to_term_object(jtweet['filter_tweet'])
            temp_test = TestTweet(
                id_str=jtweet['id_str'],
                profil=jtweet['profil'],
                tweet=jtweet['tweet'],
                sentiment=jtweet['actual_sentiment'],
                created_date=jtweet['created_date'],
                filter_tweet=copy.deepcopy(list_of_filter),
                test_id=jtweet['test_id'],
                pred_sentiment=jtweet['predicted_sentiment']
            )
            cls.list_of_test_tweet.append(temp_test)

        for jtweet in data['list_of_train_tweet']:
            list_of_filter = cls.json_to_term_object(jtweet['filter_tweet'])
            temp_train = TrainTweet(
                id_str=jtweet['id_str'],
                profil=jtweet['profil'],
                tweet=jtweet['tweet'],
                sentiment=jtweet['actual_sentiment'],
                created_date=jtweet['created_date'],
                filter_tweet=copy.deepcopy(list_of_filter),
                train_id=jtweet['train_id'],
                status_train=jtweet['status_train']

            )
            cls.list_of_train_tweet.append(temp_train)

        cls.list_of_train_tokens = copy.deepcopy(data['list_of_train_tokens'])
        cls.list_of_profile_trait = copy.deepcopy(data['list_of_profile_trait'])



    def json_to_term_object(cls, jterm):
        list_of_filter = []
        for term in jterm:
            temp_term = Term(name=term['name'],id=term['id'],weight=term['weight'],profil=term['profil'])
            list_of_filter.append(temp_term)

        return list_of_filter


    def import_file_train_to_object(cls, fileName,lot,is_data_pool=False):
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
                    #cls.list_of_data_pool.append(obj_train_tweet)
                    obj_train_tweet.actual_sentiment = row['label']
                    #cls.list_of_train_tweet.append(obj_train_tweet)
                    lot.append(obj_train_tweet)
                    train_id+=1
                return lot
            except csv.Error as e:
                sys.exit('file {}, line {}: {}'.format(fileName, reader.line_num, e))

    def import_file_test_to_object(cls, fileName):
        test_id = 1
        list_of_test_tweet = []
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
                    list_of_test_tweet.append(obj_test_tweet)
                    test_id+=1
            except csv.Error as e:
                sys.exit('file {}, line {}: {}'.format(fileName, reader.line_num, e))

        return list_of_test_tweet
    def print_token(cls):
        for token in cls.list_of_train_tokens:
            print("id_token:{id} ,nama_token :{name}, count_token:{count} , idf:{idf}".format(id=token.get('id'),name=token.get('name'),count=token.get('count'),idf=token.get('idf')))

    def print_train_tweet(cls,lot):
        for tw in lot:
            print("id tweet: {} , id_train : {}, tweet : {} ,profil :{},sentiment:{}" .format(tw.id_str,tw.train_id,tw.tweet,tw.profil,tw.actual_sentiment))
            for term in tw.filter_tweet:
                print("id : {id} , term : {name} , weight :{tfidf}".format(id=term.id,name=term.name,tfidf=term.weight))

    def print_test_tweet(cls,lot):
        for tw in lot:
            print("id tweet: {} , id_test : {}, tweet : {} ,profil :{},actual sentiment:{} , predicted :{}" .format(tw.id_str,tw.test_id,tw.tweet,tw.profil,tw.actual_sentiment,tw.predicted_sentiment))
            # for term in tw.filter_tweet:
            #     print("id : {id} , term : {name} , weight :{tfidf}".format(id=term.id,name=term.name,tfidf=term.weight))

    def print_cls(cls,loc):
        for term in loc:
            print("term:{}, prob_pos:{}, prob_neg:{}, prob_net:{}".format(term.term_name,term.prob_pos,term.prob_neg,term.prob_net))

    def print_cluster(cls,loc):
        t = PrettyTable(['Train ID', 'Group'])
        for tweet in loc:
            t.add_row([tweet.cluster_id,tweet.cluster])
        print(t)

    def train_data(cls):
        pass

    def test_data(cls,train_token,fileTest,fileEmoticon , negationWord, stopWord):
        if len(cls.list_of_train_tweet) >  0:
            list_of_test_tweet = cls.import_file_test_to_object(fileName=fileTest)
            cls.preprocessing(fileEmoticon=fileEmoticon,negationWord=negationWord,stopWord=stopWord,listOfTweet=list_of_test_tweet)
            for tweet in list_of_test_tweet:
                for term in tweet.filter_tweet:
                    term.weight =cls.calculate_tf(token=term.name, filterTweet=tweet.filter_tweet, isTrain=False)
            cls.grouping_profil(lot=list_of_test_tweet)
            model_classification = cls.naive_bayes_make_classification_model(lot=cls.list_of_train_tweet,train_token = train_token)
            test_result = cls.naive_bayes_determine_classification(lot=list_of_test_tweet,loc=model_classification)

            return test_result
        else:
            print("Please make data train first")

        return None
#======================== Active Learning ============================

    def initial_data_train_and_data_pool(cls, data_pool, initial_ratio = 10):
        cls.group_by_sentiment(copy.deepcopy(data_pool))
        lot = []
        for i in range(initial_ratio):
            tweet_pos = [tweet for idx, tweet in enumerate(cls.list_of_train_tweet_pos) if idx % initial_ratio == i]
            tweet_neg = [tweet for idx, tweet in enumerate(cls.list_of_train_tweet_neg) if idx % initial_ratio == i]
            tweet_net = [tweet for idx, tweet in enumerate(cls.list_of_train_tweet_net) if idx % initial_ratio == i]
            lot.extend(tweet_pos)
            lot.extend(tweet_neg)
            lot.extend(tweet_net)
            data_pool = [tweet for idx, tweet in enumerate(data_pool) if tweet not in lot]
        lod = copy.deepcopy(data_pool)
        for tweet in lod:
            tweet.actual_sentiment = ""

        return lod,lot

    def active_learning(cls,data_pool,data_train,max_query = 5,query_method = "entropy"):
        lot = copy.deepcopy(data_train)
        lod = copy.deepcopy(data_pool)
        for i in range(max_query):
            list_of_train_tokens = cls.initialize_train_tokens(lot)
            cls.feature_extraction(lot, list_of_train_tokens)
            cls.grouping_profil(lot)
            model_classification = cls.naive_bayes_make_classification_model(lot=lot,train_token=list_of_train_tokens)


#======================== End Active Learning ========================
#======================== Clustering =================================
    # def set_min_distance_centroid_to_centroid(cls,lis_of_cluster,cluster_K):
    #     loc = copy.deepcopy(lis_of_cluster)
    #     list_tweet_by_cluster = []
    #     for i in range(cluster_K):
    #         for tc in loc:
    #             if tc.cluster == "group"+str(i+1):
    #                 list_tweet_by_cluster.append(tc)
    #


    def clustering_k_means (cls,list_of_tweet,cluster_K = 3, is_random = True):
        list_of_centroid = []
        list_of_clustering = []
        lot = copy.deepcopy(list_of_tweet)
        centroid_candidate = copy.deepcopy(list_of_tweet)
        for i in range(cluster_K):
            idx = random.randint(0,len(centroid_candidate)-1)
            #cluster = {}
            cluster = {'group_name':'group'+str(i+1),'centroid':centroid_candidate[idx]}
            list_of_centroid.append(cluster)
            centroid_candidate.pop(idx)

        for tweet in lot:
            temp_cluster = Clustering(cluster_id=tweet.train_id,tweet=tweet)
            list_of_clustering.append(temp_cluster)

        convergen = False
        max_iteration = 20
        iteration = 0
        while(not convergen):
            count = 0
            #t = PrettyTable(['Train ID',"Distance","Distance to Group",'Group'])
            for tw_clus in list_of_clustering:
                distance = {}
                for centroid in list_of_centroid:
                    distance[centroid['group_name']] = cls.euclidean_distance(tw_clus.tweet,centroid['centroid'])
                tw_clus.distance = distance
                temp_cluster = copy.deepcopy(tw_clus.cluster)
                tw_clus.cluster = min(tw_clus.distance, key=tw_clus.distance.get)
                tw_clus.distance_to_current_cluster = min(tw_clus.distance.values())
                #t.add_row([tw_clus.cluster_id,tw_clus.distance,tw_clus.distance_to_current_cluster,tw_clus.cluster])
                if temp_cluster == tw_clus.cluster:
                    count +=1
            #print(t)
            if count == len(list_of_clustering) or iteration > max_iteration :
                convergen = True
            list_of_centroid = copy.deepcopy(cls.generate_new_centroid(cluster_K=cluster_K,list_of_clustering=list_of_clustering))
            iteration +=1
            #cls.print_cluster(list_of_clustering)
        #print("iterasi:{}".format(str(iteration)))
        return list_of_clustering

    def generate_new_centroid (cls,cluster_K,list_of_clustering):
        loc = copy.deepcopy(list_of_clustering)
        list_of_centroid = []
        for i in range(0,cluster_K):
            temp_dict = Counter({})
            denominator = 0
            for tw_c in loc:
                if tw_c.cluster == 'group'+str(i+1):
                    dict_c = Counter(cls.create_dict_term_by_id(tw_c.tweet))
                    temp_dict = temp_dict + dict_c
                    denominator += 1
            filter_tw = [Term(id=key,weight=float(temp_dict[key]/denominator)) for key in temp_dict]
            cluster = {'group_name': 'group' + str(i + 1), 'centroid': Tweet(filter_tweet=filter_tw)}
            list_of_centroid.append(cluster)

        return list_of_centroid

    def euclidean_distance (cls, tweet, centroid):
        """
        :param tweet: berupa message yang sudah di filter alias list of term dari tweet yang ingin dicari jaraknya 
        :param centroid:  berupa message yang sudah di filter alias list of term dari tweet sebagai centroid
        :return: 
        """
        dict_t = Counter(cls.create_dict_term_by_id(tweet))
        dict_c = Counter(cls.create_dict_term_by_id(centroid))
        dict_t.subtract(dict_c)
        distance = 0
        for val in dict_t.values():
            distance += math.pow(val,2)

        return math.sqrt(distance)
        # temp_distance = 0
        # for term_t in tweet:
        #     check = True
        #     for  term_c in centroid:
        #         if term_t.id == term_c.id:
        #             temp_distance += math.pow((term_t.weight - term_c.weight),2)
        #             check = False
        #     if check:
        #         temp_distance += math.pow(term_t.weight,2)
        #
        # distance = math.sqrt(temp_distance)
        #
        # return distance

    def create_dict_term_by_id (cls, tweet):
        dict_t = {}
        test_dict ={}
        for term in tweet.filter_tweet:
            dict_t[str(term.id)] = term.weight
            dict_t[term.id,"_profil"] = cls.convert_profil_to_int(term.profil)
        dict_t['tweet_profile'] = cls.convert_profil_to_int(tweet.profil)
        return dict_t

    def convert_profil_to_int(cls,profil):
        for i in range(1,8):
            if profil is not None:
                if profil == "TR"+str(i):
                    return i
            else:
                return 0
#======================== END Clustering =============================

#======================== Classificatioon ============================

    def naive_bayes_make_classification_model(cls, lot,train_token):
        loc = cls.initialization_classification_model(lot)
        list_of_classification = copy.deepcopy(cls.naive_bayes_normalization(cls.naive_bayes_weight(cls.naive_bayes_complement(loc=loc,list_of_train_tokens=train_token))))
        return list_of_classification

    def naive_bayes_determine_classification (cls,lot,loc):
        list_of_test_tweet = copy.deepcopy(lot)
        for tweet in list_of_test_tweet:
            total_prob = {"positif": 0, "negatif": 0, "netral": 0}
            for term in tweet.filter_tweet:
                # term.weight = float(
                #     sum(1 for token in tweet.filter_tweet if token.name == term.name) / len(tweet.filter_tweet))
                for model_classification in loc:
                    # print("lolol")
                    # print("cmp prob pos:{},cmp prob neg:{}, cmp prob net:{}".format(model_classification.prob_pos, model_classification.prob_neg,
                    #                                                                 model_classification.prob_net))
                    if term.name == model_classification.term_name:

                        total_prob['positif'] = total_prob['positif']+(model_classification.prob_pos * term.weight)
                        total_prob['negatif'] =  total_prob['negatif']+ (model_classification.prob_neg * term.weight)
                        total_prob['netral'] = total_prob['netral']+(model_classification.prob_net * term.weight)
            # print("prob_pos :{},prob_neg:{},prob_net:{}".format(total_prob['positif'], total_prob['negatif'], total_prob['netral']))

            tweet.predicted_sentiment = min(total_prob, key=lambda key: total_prob[key])
        return list_of_test_tweet

    def initialization_classification_model(cls,lot):
        list_of_train = copy.deepcopy(lot)
        list_of_classification = []
        for tweet in list_of_train:
            sentiment = tweet.actual_sentiment.lower()
            for term in tweet.filter_tweet:
                temp_cls = Classification()
                if cls.is_term_in_classification_empty(loc=list_of_classification,term_id = term.id):
                    temp_cls.term_name = term.name
                    temp_cls.term_id = term.id
                    cls.add_prob_by_sentiment(sentiment=sentiment,obj_cls=temp_cls, tfidf_value=term.weight)
                    list_of_classification.append(temp_cls)
                else:
                    for term_prob in list_of_classification:
                        if term_prob.term_id == term.id:
                            cls.add_prob_by_sentiment(sentiment=sentiment,obj_cls=term_prob, tfidf_value=term.weight)
        # print("Rekap----------")
        # cls.print_cls(list_of_classification)
        return list_of_classification

    def naive_bayes_complement(cls,loc,list_of_train_tokens):
        list_of_classification = copy.deepcopy(loc)
        list_of_complement = []
        laplace_smooting = 1
        vocabulary = len(list_of_train_tokens)
        #print("vocab:{}".format(vocabulary))
        total_complement_pos = 0
        total_complement_neg = 0
        total_complement_net = 0
        for term_cls in list_of_classification:
            total_complement_pos += term_cls.prob_neg + term_cls.prob_net
            total_complement_neg += term_cls.prob_pos + term_cls.prob_net
            total_complement_net += term_cls.prob_pos + term_cls.prob_neg
        #print("total_pos:{},total_neg:{},total_net:{}".format(total_complement_pos,total_complement_neg,total_complement_net))
        for term_cls in list_of_classification:
            complement = Classification()
            complement.term_name = term_cls.term_name
            complement.term_id = term_cls.term_id
            complement.prob_pos = float((term_cls.prob_neg+term_cls.prob_net+laplace_smooting)/(total_complement_pos+vocabulary))
            complement.prob_neg = float((term_cls.prob_pos+term_cls.prob_net+laplace_smooting)/(total_complement_neg+vocabulary))
            complement.prob_net = float((term_cls.prob_pos+term_cls.prob_neg+laplace_smooting)/(total_complement_net+vocabulary))
            list_of_complement.append(complement)
        # print("complement")
        # cls.print_cls(list_of_complement)
        # print("Complement---------------")
        # cls.print_cls(list_of_complement)
        return list_of_complement

    def naive_bayes_weight(cls,loc = []):
        list_of_weight = copy.deepcopy(loc)
        for term_cls in list_of_weight:
            term_cls.prob_pos = float(math.log(term_cls.prob_pos,10))
            term_cls.prob_neg = float(math.log(term_cls.prob_neg,10))
            term_cls.prob_net = float(math.log(term_cls.prob_net,10))

        # print("weight---------------")
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

        # print("Normalize--------")
        # cls.print_cls(list_of_weight_normalization)
        return list_of_weight_normalization

    def is_term_in_classification_empty(cls,loc,term_id):
        for term_prob in loc:
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

#============================= END Classification =============================
#============================= Performance Evaluation =========================
    def group_term_by_sentiment(cls):
        pass

    def k_fold_cross_validation(cls,lot,K=3,averaging='macro',multi_label = False):
        """
        :param lot: tweet data latih 
        :param K: jumlah dari K pada cross validation
        :return: 
        """

        if len(lot) >= K :
            total_accuracy = 0
            tot_precision = tot_recall = tot_f_measure = [0,0,0]
            tot_tp = tot_fp = tot_fn = 0
            random.shuffle(cls.list_of_train_tweet_pos)
            random.shuffle(cls.list_of_train_tweet_neg)
            random.shuffle(cls.list_of_train_tweet_net)
            for i in range(0,K):
                print("Processing Fold {}".format(K))
                list_of_validation_tweet = []
                list_of_validation_tweet_pos = [tweet for idx,tweet in enumerate(cls.list_of_train_tweet_pos) if idx % K == i]
                list_of_validation_tweet_neg = [tweet for idx,tweet in enumerate(cls.list_of_train_tweet_neg) if idx % K == i]
                list_of_validation_tweet_net = [tweet for idx,tweet in enumerate(cls.list_of_train_tweet_net) if idx % K == i]
                list_of_validation_tweet.extend(list_of_validation_tweet_pos)
                list_of_validation_tweet.extend(list_of_validation_tweet_neg)
                list_of_validation_tweet.extend(list_of_validation_tweet_net)
                lot_test = []
                id_test = 0
                for tweet in list_of_validation_tweet:
                    temp_test = TestTweet(
                        test_id=id_test,
                        id_str=tweet.id_str,
                        created_date=tweet.created_date,
                        tweet=tweet.tweet,
                        filter_tweet=tweet.filter_tweet,
                        sentiment=tweet.actual_sentiment
                    )
                    lot_test.append(temp_test)
                    id_test +=1
                list_of_train_tweet = [tweet for idx,tweet in enumerate(lot) if tweet not in list_of_validation_tweet]
                list_of_train_tokens = cls.initialize_train_tokens(list_of_train_tweet)
                cls.feature_extraction(list_of_train_tweet,list_of_train_tokens)
                cls.grouping_profil(list_of_train_tweet)
                model_classification = cls.naive_bayes_make_classification_model(lot=list_of_train_tweet,train_token=list_of_train_tokens)
                test_result = cls.naive_bayes_determine_classification(lot=lot_test,loc=model_classification)

                conf_matrix = cls.make_confusion_matrix(test_result)
                cls.print_conf_matrix(conf_matrix)
                accuracy = cls.calculate_accuracy(lot=test_result)
                total_accuracy += accuracy

                print("Validasi-----------{} | accuracy  {} ".format(i, str(accuracy)))
                cls.print_test_tweet(test_result)
                mean_accuracy = float(total_accuracy / (K - 1))
                if averaging.lower() == "macro":
                    precision, recall, f_measure = cls.calculate_precision_recall_f1_measure(conf_matrix, averaging=averaging)
                    tot_precision = [sum(x) for x in zip(precision, tot_precision)]
                    tot_recall = [sum(x) for x in zip(recall, tot_recall)]
                    tot_f_measure = [sum(x) for x in zip(f_measure, tot_f_measure)]
                else:
                    tp, fp, fn = cls.calculate_precision_recall_f1_measure(conf_matrix, averaging=averaging)
                    tot_tp = [sum(x) for x in zip(tp, tot_tp)]
                    tot_fp = [sum(x) for x in zip(fp, tot_fp)]
                    tot_fn = [sum(x) for x in zip(fn, tot_fn)]
            print("precision:{}".format(tot_precision))
            print("recall:{}".format(tot_recall))
            print("f-measure:{}".format(tot_f_measure))
            if averaging.lower() == "macro":
                table = cls.evaluation_performance(param_eval1=tot_precision,param_eval2=tot_recall,
                                                   param_eval3=tot_f_measure,averaging=averaging,
                                                   multi_label=multi_label,K=K)
            else:
                table = cls.evaluation_performance(param_eval1=tot_tp, param_eval2=tot_fp,
                                                   param_eval3=tot_fn, averaging=averaging,
                                                   multi_label=multi_label,K=K)
            print(table)

    def calculate_accuracy(cls,lot):
        correct = 0
        for tweet in lot:
            if tweet.predicted_sentiment.lower() == tweet.actual_sentiment.lower():
                correct+=1
        return float(((correct/(len(lot)))*100)) if len(lot) > 0 else 0

    def evaluation_performance(cls,param_eval1,param_eval2,param_eval3,K,multi_label=True,averaging="macro"):
        #print("peval1:{},peval2:{},peval3:{}".format(param_eval1,param_eval2,param_eval3))
        if multi_label:
            mean_precision  = [0, 0, 0]
            mean_recall  = [0, 0, 0]
            mean_f_measure  = [0, 0, 0]
            if averaging.lower() == "macro":
                for i in range(3):
                    mean_precision[i] = float(param_eval1[i] / (K))
                    mean_recall[i] = float(param_eval2[i] / (K))
                    mean_f_measure[i] = float(param_eval3[i] / (K))
            else:
                for i in range(3):
                    mean_precision[i] = float(param_eval1[i] / (param_eval1[i] + param_eval2[i]))
                    mean_recall[i] = float(param_eval1[i] / (param_eval2[i] + param_eval3[i]))
                    mean_f_measure[i] = float((2 * mean_precision[i] * mean_recall[i]) / (mean_precision[i] + mean_recall[i]))
            table = PrettyTable(['Kelas', 'Precision', 'Recall', 'F-Measure'])
            table.add_row(["Positif", mean_precision[0], mean_recall[0], mean_f_measure[0]])
            table.add_row(["Negatif", mean_precision[1], mean_recall[1], mean_f_measure[1]])
            table.add_row(["Netral", mean_precision[2], mean_recall[2], mean_f_measure[2]])
        else:
            sum_param_eval1 = sum(param_eval1)/K
            sum_param_eval2 = sum(param_eval2)/K
            sum_param_eval3 = sum(param_eval3)/K
            if averaging.lower() == "macro":
                mean_precision = float(sum_param_eval1/3)
                mean_recall = float(sum_param_eval2/3)
                mean_f_measure = float(sum_param_eval3/3)
            else:
                mean_precision = float(sum_param_eval1 / (sum_param_eval1 + sum_param_eval2))/3
                mean_recall = float(sum_param_eval1 / (sum_param_eval2 + sum_param_eval3))/3
                mean_f_measure = float((2 * mean_precision * mean_recall) / (mean_precision + mean_recall))/3
            table = PrettyTable(['Precision', 'Recall', 'F-Measure'])
            table.add_row([mean_precision, mean_recall, mean_f_measure])

        return table

    def make_confusion_matrix(cls,lot):
        """
        Membuat confusion matrix 3x3 dimana column berupa
        [=======================Nilai_aktual_positif    Nilai_aktual_negatif    Nilai_aktual_netral]
        [Nilai_predicted_positif    0                       0                       0
        [Nilai_predicted_negatif    0                       0                       0
        [Nilai_predicted_netral     0                       0                       0
        
        Dimana untuk True Positif adalah nilai diagonal
        False Negatif adalah nilai column
        False Positif adalah nilai row
        """
        conf_matrix = [[0,0,0],[0,0,0],[0,0,0]]
        for tweet in lot:
            actual = tweet.actual_sentiment.lower()
            predicted = tweet.predicted_sentiment.lower()
            if actual == "positif":
                if predicted == "positif":
                    conf_matrix[0][0] += 1
                elif predicted == "negatif":
                    conf_matrix[1][0] += 1
                else:
                    conf_matrix[2][0] += 1
            elif actual == "negatif":
                if predicted == "positif":
                    conf_matrix[0][1] += 1
                elif predicted == "negatif":
                    conf_matrix[1][1] += 1
                else:
                    conf_matrix[2][1] += 1
            else:
                if predicted == "positif":
                    conf_matrix[0][2] += 1
                elif predicted == "negatif":
                    conf_matrix[1][2] += 1
                else:
                    conf_matrix[2][2] += 1
        return conf_matrix

    def print_conf_matrix(cls,conf_matrix):
        t = PrettyTable(['Predicted/Actual', 'Positif', 'Negatif', 'Netral'])
        t.add_row(["Positif",conf_matrix[0][0],conf_matrix[0][1], conf_matrix[0][2]])
        t.add_row(["Negatif", conf_matrix[1][0], conf_matrix[1][1], conf_matrix[1][2]])
        t.add_row(["Netral", conf_matrix[2][0], conf_matrix[2][1], conf_matrix[2][2]])
        print(t)

    def calculate_precision_recall_f1_measure(cls,conf_matrix,averaging="macro"):
        """
        input confusion matrix dengan format:
        [=======================Nilai_aktual_positif    Nilai_aktual_negatif    Nilai_aktual_netral]
        [Nilai_predicted_positif    0                       0                       0
        [Nilai_predicted_negatif    0                       0                       0
        [Nilai_predicted_netral     0                       0                       0
        
        output Macro:
        Output berupa matrix 1x3 dengan nilai precision dengan urutan pos,neg,net
        Output berupa matrix 1x3 dengan nilai recall dengan urutan pos,neg,net
        Output berupa matrix 1x3 dengan nilai f-measure dengan urutan pos,neg,net
        
        Output Micro
        TP,FP,FN
        """
        precision_matrix = []
        recall_matrix = []
        f_measure_matrix = []
        TP_matrix = []
        FP_matrix = []
        FN_matrix = []
        TP = 0
        for i in range(3):
            TP =conf_matrix[i][i]
            #TP +=conf_matrix[i][i]
            FP = 0
            FN = 0
            for j in range(3):
                FP += conf_matrix[i][j]
                FN += conf_matrix[j][i]
               # print("i:{},j:{},FP:{},FN:{}".format(i,j,FP,FN))
            if averaging.lower() == "macro":
                precision = float(TP/FP) if FP>0 else 0
                recall = float(TP/FN) if FN>0 else 0
                print("precision:{},recall:{}".format(precision,recall))
                #print("TP:{},FP:{},FN:{}".format(TP,FP,FN))
                precision_matrix.append(precision*100)
                recall_matrix.append(recall*100)
                f_measure_matrix.append((float((2*precision*recall)/(precision+recall))*100) if precision > 0 and recall > 0 else 0)
            else :
                TP_matrix.append(TP)
                FP_matrix.append(FP)
                FN_matrix.append(FN)
        if averaging.lower() == "macro":
            return precision_matrix,recall_matrix,f_measure_matrix
        else:
            return TP_matrix,FP_matrix,FN_matrix

    def group_by_sentiment(cls,lot):
        for tweet in lot:
            if tweet.actual_sentiment.lower() == "positif":
                cls.list_of_train_tweet_pos.append(tweet)
            elif tweet.actual_sentiment.lower() == "negatif":
                cls.list_of_train_tweet_neg.append(tweet)
            else:
                cls.list_of_train_tweet_net.append(tweet)

# ============================= END Performance Evaluation =========================
    #--------------------------- ekstraksi fitur ------------------------------------------

    def initialize_train_tokens(cls,lot,base_selection = 0):
        """
        :return: membentuk token/kata yang unik dari daftar term pada data latih 
        """
        tokens = []
        list_of_train_tokens = []
        for tweet in lot:
            for term in tweet.filter_tweet:
                tokens.append(term.name)
        list_of_tokens = list(set(tokens))
        #print(list_of_tokens)
        index = 1
        for token in list_of_tokens:
            count_token = 0
            for tweet in lot:
                #count = Counter(getattr(term,'name') for term in tweet.filter_tweet)
                #count_token = count_token + tweet.filter_tweet.count(token)
                #count_token += count[token]
                count_token += sum(1 for term in tweet.filter_tweet if term.name == token)
            if count_token >= base_selection:
                list_of_train_tokens.append({"id":index, "name":token, "count":count_token , "idf": ""})
                index+=1
        print('initialize tokens done..')
        return list_of_train_tokens

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
                #print(cls.list_of_profile_trait)
            except csv.Error as e:
                sys.exit('file {}, line {}: {}'.format(fileName, reader.line_num, e))

    def grouping_profil(cls,lot):
        for tweet in lot:
            if cls.is_filter_tweet_empty(tweet.filter_tweet) == False:
                list_of_term_profil = []
                for idx,term in enumerate(tweet.filter_tweet):
                    term.profil = cls.check_profil(term.name)
                    if term.profil is not None:
                        list_of_term_profil.append(term)
                # for term in tweet.filter_tweet:
                #     print("term : {} , profil : {} , weight : {}".format(term.name, term.profil, term.weight))
                temp_term = max(list_of_term_profil, key=lambda term:term.weight ) if len(list_of_term_profil) > 0 else Term()
                tweet.profil = temp_term.profil
                #print("temp_term : {} , profil : {} , weight : {}".format(temp_term.name, temp_term.profil, temp_term.weight))

    def check_profil(cls, term):
        for i in range(1,8):
            #print(self.listOfProfilTrait[0].get("TR"+str(i)))
            for trait in cls.list_of_profile_trait[0].get('TR'+str(i)):
                if term == trait:
                   return "TR"+str(i)

    def feature_extraction(cls, lot,list_of_tokens,normalize_euclidean = True, feature_selection = True):
        """
        :return: filter_tweet yang berisi object term, dimana untuk masing-masing
        object term pada sebuah tweet sudah diberi bobot tfidf 
        """
        if feature_selection:
            for tweet in lot:
                tweet.filter_tweet = cls.feature_selection(tweet.filter_tweet, list_of_tokens)
        cls.calculate_idf(list_of_tokens,lot)
        for tweet in lot:
            for term in tweet.filter_tweet:
                tf = cls.calculate_tf(token=term.name, filterTweet=tweet.filter_tweet,isTrain=True)
                idf = 0
                id = 0
                for token in list_of_tokens:
                    if token['name'] == term.name:
                        idf = token['idf']
                        id = token['id']
                # tfidf.append({token : float(tf * idf[token])})
                term.weight = float(tf * idf )
                #print("term {} f:{} ,idf:{} , tf-idf:{}".format(term.name, tf,idf,term.weight))
                term.id = id
                if term.weight <= 0 :
                    del term
            if normalize_euclidean:
                #tweet.filter_tweet = cls.calculate_norm_euclidean(tweet.filter_tweet)[:]
                cls.calculate_norm_euclidean(tweet.filter_tweet)
        print("feature extraction done...")

    def calculate_idf(cls,list_of_tokens,lot):
        for tokens in list_of_tokens:
            count_token = 0
            for tweet in lot:
                #count = Counter(getattr(term, 'name') for term in tweet.filter_tweet)
                #if tokens['name'] in tweet.filter_tweet:
                if any(term.name == tokens['name'] for term in tweet.filter_tweet):
                    count_token+=1
            if count_token == 0:
                count_token = 1
            tokens['idf'] = float(math.log(len(lot) / count_token, 10))

    def calculate_tf(cls, token, filterTweet,isTrain=True):
        laplace_smoothing = 1
        if not cls.is_filter_tweet_empty(filterTweet):
            #return float(math.log(filterTweet.count(token)+laplace_smoothing,10))
            if isTrain:
                return float(math.log(sum(1 for term in filterTweet if term.name == token)+laplace_smoothing,10))
            else:
                return sum(1 for term in filterTweet if term.name == token)
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
        #print(tweet_stop)
        return tweet_stop

