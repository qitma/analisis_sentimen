class Tweet(object):
    def __init__(cls,id_str = None,created_date = None,tweet = None,filter_tweet = None,profil = None,sentiment = None):
        cls.id_str = id_str
        cls.created_date = created_date
        cls.tweet = tweet
        cls.filter_tweet = filter_tweet
        cls.profil = profil
        cls.actual_sentiment = sentiment