#from tweet import  Tweet

class Clustering(object):
    def __init__(self,distance=0,cluster=None,dtcc = 0):
        self.distance = distance
        self.distance_to_current_cluster = dtcc
        self.cluster_name = cluster