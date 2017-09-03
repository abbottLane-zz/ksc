import os


class Evaluator(object):
    def __init__(self, predictions, gold_data):
        self.predictions = predictions
        self.gold_data = gold_data
        self.label_set = set([l['category'] for l in self.gold_data])
        self.counts = self._count_tp_fp_fn()

    def _count_tp_fp_fn(self):
        eval_data_root_dir = os.path.join("Data","eval_data")
        count_dict=dict()
        for label in self.label_set:
            count_dict[label]={'tp':0,'fp':0,'fn':0}

        for i, prediction in enumerate(self.predictions):
            if prediction == self.gold_data[i]['category']: # its a tp for 'gold_data[i]['category']
                count_dict[self.gold_data[i]['category']]['tp']+=1
                # write tp result to file for error analysis later
                tp_dir=os.path.join(eval_data_root_dir, prediction, "tp")
                self.ensure_dir(tp_dir)
                with open(os.path.join(tp_dir, "doc"+str(i)+".txt"), "wb") as f:
                    f.write(bytes(self.gold_data[i]['title'] + " " +self.gold_data[i]['blurb'] + " " + self.gold_data[i]['full_text'], encoding="utf-8"))
            else: # its an fp for prediction, and a fn for gold_data[i]['category]
                count_dict[prediction]['fp']+=1
                # write fp result to file for error analysis later
                fp_dir = os.path.join(eval_data_root_dir, prediction, "fp")
                self.ensure_dir(fp_dir)
                with open(os.path.join(fp_dir, "doc"+str(i)+".txt"), "wb") as f:
                    f.write(bytes(self.gold_data[i]['title'] + " " +self.gold_data[i]['blurb'] + " " + self.gold_data[i]['full_text'], encoding="utf-8"))

                count_dict[self.gold_data[i]['category']]['fn'] +=1
                # write fn result to file for error analysis later
                fn_dir = os.path.join(eval_data_root_dir,self.gold_data[i]['category'], "fn")
                self.ensure_dir(fn_dir)
                with open(os.path.join(fn_dir, "doc"+str(i)+".txt"), "wb") as f:
                    f.write(bytes(self.gold_data[i]['title'] + " " +self.gold_data[i]['blurb'] + " " + self.gold_data[i]['full_text'], encoding="utf-8"))
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
        return 2*((p*r)/(p+r))

    def ensure_dir(self, file_path):
        #directory = os.path.dirname(file_path)
        if not os.path.exists(file_path):
            os.makedirs(file_path)