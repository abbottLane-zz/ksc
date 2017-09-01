class Evaluator(object):
    def __init__(self, predictions, gold_data):
        self.predictions = predictions
        self.gold_data = gold_data
        self.label_set = set([l['category'] for l in self.gold_data])
        self.counts = self._count_tp_fp_fn()

    def _count_tp_fp_fn(self):
        count_dict=dict()
        for label in self.label_set:
            count_dict[label]={'tp':0,'fp':0,'fn':0}

        for i, prediction in enumerate(self.predictions):
            if prediction == self.gold_data[i]['category']: # its a tp for 'gold_data[i]['category']
                count_dict[self.gold_data[i]['category']]['tp']+=1
            else: # its an fp for prediction, and a fn for gold_data[i]['category]
                count_dict[prediction]['fp']+=1
                count_dict[self.gold_data[i]['category']]['fn'] +=1
        return count_dict

    def get_label_set(self):
        return self.label_set

    def get_precision(self, label):
        return float(self.counts[label]['tp'])/(self.counts[label]['tp'] + self.counts[label]['fp'])

    def get_recall(self, label):
        return float(self.counts[label]['tp'])/(self.counts[label]['tp'] + self.counts[label]['fn'])

    def get_f1(self, label):
        p = self.get_precision(label)
        r = self.get_recall(label)
        if p==0 and r==0:
            return 0
        return 2*((p*r)/(p+r))
