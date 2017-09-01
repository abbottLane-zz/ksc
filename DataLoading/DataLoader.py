import json



class DataLoader:
    def __init__(self, data_dir):
        self.data=self._load_data(data_dir)
        self.train, self.test = self._partition_data(self.data)
        self.targets =self._get_target_labels(self.data)

    def _load_data(self, data_dir):
        with open(data_dir) as data_file:
            data = json.load(data_file)
            return data

    def _get_target_labels(self, data):
        targets=set()
        for d in data:
            targets.add(d['category'])
        return targets

    def _partition_data(self,data):
        '''
        Split into train/test partitions in a 80% : 20% ratio
        :return: list of training instances, list of testing instances
        '''
        test=data[int(len(data)*.8):]
        train=data[:int(len(data)*.8)]
        return train, test


    def get_full_dataset(self):
        return self.data

    def get_test_dataset(self):
        return self.test

    def get_train_data_set(self):
        return self.train

    def get_target_labels(self):
        return self.targets
