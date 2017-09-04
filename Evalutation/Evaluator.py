import os


class Evaluator(object):
    def __init__(self, predictions, gold_data):
        self.predictions = predictions
        self.gold_data = gold_data
        self.label_set = set([l['category'] for l in self.gold_data])
        self.counts = self._count_tp_fp_fn()

    def _count_tp_fp_fn(self):
        '''
        Counts all instances of true positives, false positives, and false negatives among the predicted instances.
        Writes tp, fp, fn text fields to directory structure in /Data/eval_data to facilitate error analysis
        :return:
        '''
        eval_data_root_dir = os.path.join("Data","eval_data")
        count_dict=dict()
        for label in self.label_set:
            count_dict[label]={'tp':0,'fp':0,'fn':0}

        for i, prediction in enumerate(self.predictions):

            ## Count TP ##
            if prediction == self.gold_data[i]['category']: # its a tp for 'gold_data[i]['category']
                count_dict[self.gold_data[i]['category']]['tp']+=1
                # write tp result to file for error analysis later
                tp_dir=os.path.join(eval_data_root_dir, prediction, "tp")
                self.ensure_dir(tp_dir)
                with open(os.path.join(tp_dir, "doc"+str(i)+".txt"), "wb") as f:
                    f.write(bytes(self.gold_data[i]['title'] +
                                  " " +self.gold_data[i]['blurb'] +
                                  " " + self.gold_data[i]['full_text'],
                                  encoding="utf-8"))

            else:
                ## Count FP ##
                count_dict[prediction]['fp']+=1
                # write fp result to file for error analysis later
                fp_dir = os.path.join(eval_data_root_dir, prediction, "fp")
                self.ensure_dir(fp_dir)
                with open(os.path.join(fp_dir, "doc"+str(i)+".txt"), "wb") as f:
                    f.write(bytes(self.gold_data[i]['title'] +
                                  " " +self.gold_data[i]['blurb'] +
                                  " " + self.gold_data[i]['full_text'],
                                  encoding="utf-8"))

                ## Count FN ##
                count_dict[self.gold_data[i]['category']]['fn'] +=1
                # write fn result to file for error analysis later
                fn_dir = os.path.join(eval_data_root_dir,self.gold_data[i]['category'], "fn")
                self.ensure_dir(fn_dir)
                with open(os.path.join(fn_dir, "doc"+str(i)+".txt"), "wb") as f:
                    f.write(bytes(self.gold_data[i]['title'] +
                                  " " +self.gold_data[i]['blurb'] +
                                  " " + self.gold_data[i]['full_text'],
                                  encoding="utf-8"))
        return count_dict

    def get_label_set(self):
        '''
        Retruns the set of all possible labels for which the Evaluator calculates Precision, Recall, and F1
        :return: set of category labels found in the gold data
        '''
        return self.label_set

    def get_precision(self, label):
        '''
        Calculates and returns precision value based on tp and fp counts
        :param label: The category for which we want to calculate precision
        :return: precision (float)
        '''
        return float(self.counts[label]['tp'])/(self.counts[label]['tp'] + self.counts[label]['fp'])

    def get_recall(self, label):
        '''
        Calculates and returns recall value based on tp and fn counts
        :param label: The category for which we want to calculate recall
        :return: recall (float)
        '''
        return float(self.counts[label]['tp'])/(self.counts[label]['tp'] + self.counts[label]['fn'])

    def get_f1(self, label):
        '''
        Calculates F-score for a given Category using the calculated precision and recall values
        :param label: The category for which we want to calculate F1
        :return: f1 score (float)
        '''
        p = self.get_precision(label)
        r = self.get_recall(label)
        return 2*((p*r)/(p+r))

    def ensure_dir(self, file_path):
        '''
        Given a directory path, check and see if the corresponding directory exists. If not, create it.
        :param file_path: String file path to check
        '''
        if not os.path.exists(file_path):
            os.makedirs(file_path)