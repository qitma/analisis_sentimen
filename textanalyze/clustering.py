#from tweet import  Tweet

class Clustering(object):
    def __init__(self,cluster_id = 0,tweet=None,distance=0,cluster=None):
        self.cluster_id = cluster_id
        self.tweet = tweet
        self.distance = distance
        self.cluster = cluster