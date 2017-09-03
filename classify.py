import sys
from pprint import pprint

from DataLoading.DataLoader import DataLoader
from Extraction.TextClassfModel import TextClassificationModel


def main(data_dir):
    ##Set directories
    model_out_dir = "Data/final_model/models/kick_classf.pk1"
    vectorizer_out_dir = "Data/final_model/vectorizers/kick_classf.vec"

    ## Load data
    dl = DataLoader(data_dir)
    data = dl.get_full_dataset()

    ## predict
    model = TextClassificationModel()
    model.load_model(model_out_dir)
    model.load_vectorizer(vectorizer_out_dir)
    predictions = model.predict(data)

    ## print predictions to stdout
    for i, p in enumerate(predictions):
        print (p)

if __name__ =="__main__":
    if len(sys.argv) == 2:
        main(sys.argv[1])
    else:
        print("No data directory specified, I'm out.")