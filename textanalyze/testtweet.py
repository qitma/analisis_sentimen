from tweet import Tweet

class TestTweet(Tweet):
    def __init__(cls,test_id=0,id_str = None,created_date = None,tweet = None,filter_tweet = None,profil = None,sentiment = None,pred_sentiment = None):
        cls.test_id = test_id
        super().__init__(id_str,created_date,tweet,filter_tweet,profil,sentiment)
        cls.predicted_sentiment = pred_sentiment