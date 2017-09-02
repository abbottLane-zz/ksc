import re
from sklearn.externals import joblib
from sklearn.feature_extraction import DictVectorizer
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.feature_selection import SelectKBest, chi2
from sklearn.svm import LinearSVC
import spacy

from Regex_Entities.regex_patterns import sub_journalism_terms, sub_food_terms, sub_photography_terms, sub_art_terms, \
    sub_design_terms, sub_craft_terms


class TextClassificationModel(object):
    def __init__(self):
        self.classifier = None
        self.vectorizer = TfidfVectorizer(stop_words="english", max_df=.25, max_features=15000, ngram_range=(1,2))
        self.segmentor = spacy.load('en')


    def train(self, text):
        '''
        Trains a document classification model based on the detected target labels and a list of Kickstarter JSON objects containing harvest-able text
        :param text: a JSON object with 'blurb', 'full_text', 'category' fields
        :return: True if model trained successfully, False otherwise
        '''
        #dict_features = self._vectorize(text)
        full_text = self.preprocess_text([x['title'] + " " + x['blurb'] + " " + x['full_text'] for x in text])
        x_train = self.vectorizer.fit_transform(full_text)
        y_train = self._get_labels(text)

        # Refine feature set
        #x_train = self._chi2_training_features(x_train, y_train, k=400, vectorizer=self.vectorizer)

        self.classifier = LinearSVC()

        self.classifier.fit(x_train, y_train)

    def predict(self, text):
        if self.classifier:

            #dict_features = self._vectorize(text)
            full_text = self.preprocess_text([x['title'] + " " + x['blurb'] + " " + x['full_text'] for x in text])
            x_test = self.vectorizer.transform(full_text)

            # refine feature set
            #x_test = self._chi2_testing_features(x_test)

            return self.classifier.predict(x_test)

        else:
            raise Exception("Cannot predict without a model. Load a stored model using: .load_model(model_dir)")


    def load_model(self, model_dir):
        '''
        Loads dumped model from desired filepath
        :param model_dir: string
        :return: a sklearn classifier object
        '''
        self.classifier = joblib.load(model_dir)

    def dump_model(self, out_dir):
        '''
        Writes TextClassification model to disk for persistance
        :return: True if write to disk success, false otherwise
        '''
        joblib.dump(self.classifier, out_dir)

    # def _vectorize(self, training_instances):
    #     '''
    #     Transform raw tweet objects to vector representations
    #     :param tweets: Tweet objects
    #     :return: list of feature vectors x
    #     '''
    #     feature_dicts = list()
    #     for t in training_instances:
    #         feature_dict={}
    #         feature_dict.update(self._get_ngram_feats(t['full_text']))
    #         feature_dict.update(self._get_ngram_feats(t['blurb']))
    #         feature_dicts.append(feature_dict)
    #     return feature_dicts

    # def _get_ngram_feats(self, text, n=1):
    #     text=text.lower()
    #     text = re.sub('\d', '0', text)  # replace all numbers with 0s
    #     text = re.sub('[.,?\'$%&:;!()\"#@]', "", text)
    #     toks = text.split()
    #     ngrams = zip(*[toks[i:] for i in range(n)])
    #     new_data = ['_'.join(w).strip('.;\'\"=,') for w in ngrams]
    #     return dict.fromkeys(new_data, True)

    def _get_labels(self, docs):
        return [d['category'] for d in docs]

    # def _chi2_training_features(self, X_train, y_train, k, vectorizer=None):
    #     self.ch2 = SelectKBest(chi2, k=k)
    #     X_train = self.ch2.fit_transform(X_train, y_train)
    #     if vectorizer:
    #         feature_names = vectorizer.get_feature_names()
    #         # keep selected feature names
    #         feature_names = [feature_names[i] for i in self.ch2.get_support(indices=True)]
    #     return X_train

    # def _chi2_testing_features(self, x_test):
    #     X_test = self.ch2.transform(x_test)
    #     return X_test

    def dump_vectorizer(self, out_dir):
        joblib.dump(self.vectorizer, out_dir)
        pass

    def load_vectorizer(self, vectorizer_path):
        self.vectorizer = joblib.load(vectorizer_path)

    def dump_ch2(self, out_dir):
        joblib.dump(self.ch2, out_dir)
        pass

    def load_ch2(self, ch2_path):
        self.ch2 = joblib.load(ch2_path)

    def preprocess_text(self, texts):
        subbed_texts= list()
        for t in texts:
            lower_t = t.lower()
            # zerod_t = re.sub('\d', '0', lower_t)
            stripped = re.sub('[.,?\'$%&:;!()\"#@]', "",lower_t)
            segmented = self.segmentor(stripped)
            #todo: chunk ner info, reconstitute text

            subbed_texts.append(stripped)
        return subbed_texts