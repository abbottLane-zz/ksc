import re
from sklearn.externals import joblib
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.svm import LinearSVC, SVC
import spacy
from Extraction.PhraseExtractor import PhraseExtractor
from Regex.regex_patterns import sub_journalism_terms, sub_performing_arts_terms, sub_craft_terms


class TextClassificationModel(object):
    '''
    Class wrapper for a text classification model. Implements standard public train and predict methods.
    '''
    def __init__(self):
        self.classifier = None
        self.vectorizer = TfidfVectorizer(stop_words="english", max_df=.25, max_features=15000, ngram_range=(1,2))
        self.segmentor = spacy.load('en')


    def train(self, pitches):
        '''
        Trains a document classification model based on the detected target labels and a list of Kickstarter JSON
        objects containing harvest-able text
        :param pitch: a JSON object with (at least) 'blurb', 'full_text', 'title', and 'category' fields
        :return: True if model trained successfully, False otherwise
        '''
        full_text = self.preprocess_text([x['title'] + " " + x['blurb'] + " " + x['full_text'] for x in pitches])
        x_train = self.vectorizer.fit_transform(full_text)
        y_train = self._get_labels(pitches)
        self.classifier = LinearSVC()
        self.classifier.fit(x_train, y_train)

    def predict(self, pitches):
        '''
        Given a pitch dictionary object, extract the text, vectorize it, and use the model to make predictions
        :param pitch: a JSON object with (at least) 'blurb', 'full_text', and 'title' fields
        :return:
        '''
        if self.classifier:
            full_text = self.preprocess_text([x['title'] + " " + x['blurb'] + " " + x['full_text'] for x in pitches])
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
        Writes TextClassification model to disk for persistence
        :return: True if write to disk success, false otherwise
        '''
        joblib.dump(self.classifier, out_dir)


    def _get_labels(self, pitch_json):
        '''
        Returns all the pitch labels in a list
        :param docs: pitch labels (list)
        :return:
        '''
        return [d['category'] for d in pitch_json]

    def dump_vectorizer(self, out_dir):
        '''
        Write vectorizer to disk so it can be loaded for prediction
        :param out_dir: The filepath where the vectorizer should be written on disk
        '''
        joblib.dump(self.vectorizer, out_dir)

    def load_vectorizer(self, vectorizer_path):
        '''
        Load a pre-fit vectorizer to use for prediction
        :param vectorizer_path: where the pre-fit vectorizer object lives on disk
        '''
        self.vectorizer = joblib.load(vectorizer_path)


    def preprocess_text(self, pitches):
        '''
        The preprocessing transformation that the raw text goes through before being put into the vectorizer
        :param texts:
        :return:
        '''
        subbed_texts= list()
        pe = PhraseExtractor()
        for t in pitches:
            lower_t = t.lower()
            stripped = re.sub('[.,?\'$%&:;!()\"#@]', "",lower_t)
            subbed = sub_performing_arts_terms(stripped)
            subbed = sub_craft_terms(subbed)
            subbed = sub_journalism_terms(subbed)
            #pos_tagged = pe.tag_pos(t)
            subbed_texts.append(subbed)
        return subbed_texts