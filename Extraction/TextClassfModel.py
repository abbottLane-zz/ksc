import re
from sklearn.externals import joblib
from sklearn.feature_extraction import DictVectorizer
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.feature_selection import SelectKBest, chi2
from sklearn.svm import LinearSVC
import spacy

from Extraction.PhraseExtractor import PhraseExtractor
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
        full_text = self.preprocess_text([x['title'] + " " + x['blurb'] + " " + x['full_text'] for x in text])
        x_train = self.vectorizer.fit_transform(full_text)
        y_train = self._get_labels(text)
        self.classifier = LinearSVC()
        self.classifier.fit(x_train, y_train)

    def predict(self, text):
        if self.classifier:

            #dict_features = self._vectorize(text)
            full_text = self.preprocess_text([x['title'] + " " + x['blurb'] + " " + x['full_text'] for x in text])
            x_test = self.vectorizer.transform(full_text)
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


    def _get_labels(self, docs):
        return [d['category'] for d in docs]

    def dump_vectorizer(self, out_dir):
        joblib.dump(self.vectorizer, out_dir)
        pass

    def load_vectorizer(self, vectorizer_path):
        self.vectorizer = joblib.load(vectorizer_path)


    def preprocess_text(self, texts):
        subbed_texts= list()
        pe = PhraseExtractor()
        for t in texts:
            lower_t = t.lower()
            stripped = re.sub('[.,?\'$%&:;!()\"#@]', "",lower_t)
            #subbed = pe.get_ner_tags(stripped)
            subbed_texts.append(stripped)
        return subbed_texts