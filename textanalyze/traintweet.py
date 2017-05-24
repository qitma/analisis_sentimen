from tweet import Tweet

class TrainTweet(Tweet):
    def __init__(cls):
        cls.train_id = 0
        super().__init__()
        cls.status_train = None