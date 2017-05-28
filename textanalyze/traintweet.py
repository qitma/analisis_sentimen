from tweet import Tweet

class TrainTweet(Tweet):
   def __init__(cls,train_id=0,
                id_str = None,
                created_date = None,
                tweet = None,
                filter_tweet = None,
                profil = None,
                sentiment = None,
                status_train = None
                ):
        cls.train_id = train_id
        super().__init__(id_str,created_date,tweet,filter_tweet,profil,sentiment)
        cls.status_train = status_train
        #asd