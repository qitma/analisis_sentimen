class Classification(object):
    def __init__(cls,name=None,id=None,prob_pos = 0,prob_neg = 0,prob_net = 0):
        cls.term_name = name
        cls.term_id = id
        cls.prob_pos = prob_pos
        cls.prob_neg = prob_neg
        cls.prob_net = prob_net