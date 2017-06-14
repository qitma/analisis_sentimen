from tweet import Tweet

class TrainTweet(Tweet):
   def __init__(self, train_id=0,
                id_str = None,
                created_date = None,
                tweet = None,
                filter_tweet = None,
                profil = None,
                sentiment = None,
                cluster = None):
        self.train_id = train_id
        super().__init__(id_str,created_date,tweet,filter_tweet,profil,sentiment)
        self.cluster = cluster

   def __eq__(self, other):
        return self.train_id == other.train_id

   def __ne__(self, other):
        return not self.__eq__(other)