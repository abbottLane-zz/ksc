from pprint import pprint

from sklearn.model_selection import cross_validate

from DataLoading.DataLoader import DataLoader
from Evalutation.Evaluator import Evaluator
from Extraction.PhraseExtractor import PhraseExtractor
from Extraction.TextClassfModel import TextClassificationModel


def main():
    ##Set directories
    data_dir ="/home/wlane/PycharmProjects/Kickstarter_classification/Data/kickstarter_corpus.json"
    model_out_dir = "Data/models/kick_classf.pk1"
    vectorizer_out_dir = "Data/vectorizers/kick_classf.vec"
    chi2_out_dir = "Data/vectorizers/kick_classf.ch2"

    ## Load data
    dl = DataLoader(data_dir)
    train_data = dl.get_train_data_set()

    ## train model
    model = TextClassificationModel()
    model.train(train_data)
    model.dump_model(model_out_dir)
    model.dump_vectorizer(vectorizer_out_dir)
    model.dump_ch2(chi2_out_dir)

    ## test
    test_data = dl.get_test_dataset()
    model.load_model(model_out_dir)
    model.load_vectorizer(vectorizer_out_dir)
    predictions = model.predict(test_data)


    ## evaluate
    eval = Evaluator(predictions, test_data)
    labels = eval.get_label_set()
    f1s=list()
    for label in labels:
        print (label + "\tP:"+ str(eval.get_precision(label)) + "\tR:" + str(eval.get_recall(label)) +"\tF1:" + str(eval.get_f1(label)))
        f1s.append(eval.get_f1(label))
    avg_f1s= float(sum(f1s))/len(f1s)
    print(avg_f1s)










   # # init PhraseExtractor
   #  extractor = PhraseExtractor()
   #
   #  # Load text from some file
   #  with open("Data/Spotify.txt", "rb") as f:
   #      full_text = f.read()
   #
   #  # Try both methods for phrase extraction:
   #      # The NP Chunker method
   #  top_n_phrases_np_chunker = extractor.extract_top_n_phrases_METHOD1(full_text.decode('utf-8'), n=10)
   #      # The dependency parse arc method
   #  top_n_phrases_dep_parse = extractor.extract_top_n_phrases_METHOD2(full_text.decode('utf-8'), n=10)
   #
   #  # Print results
   #  print("METHOD: NP Chunker:")
   #  pprint(top_n_phrases_np_chunker)
   #  print("METHOD: Dependency Parse Subtrees:")
   #  pprint(top_n_phrases_dep_parse)


if __name__=="__main__":
    main()
