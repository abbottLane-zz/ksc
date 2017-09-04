from DataLoading.DataLoader import DataLoader
from Extraction.TextClassfModel import TextClassificationModel

def main():
    ##Set directories
    data_dir ="/home/wlane/PycharmProjects/Kickstarter_classification/Data/kickstarter_corpus.json"
    model_out_dir = "Data/final_model/models/kick_classf.pk1"
    vectorizer_out_dir = "Data/final_model/vectorizers/kick_classf.vec"

    ## Load data
    dl = DataLoader(data_dir)
    train_data = dl.get_full_dataset()

    ## train model
    model = TextClassificationModel()
    model.train(train_data)
    model.dump_model(model_out_dir)
    model.dump_vectorizer(vectorizer_out_dir)
    print("Model trained on " + str(len(train_data)) + " Kickstarter pitches.")

if __name__=="__main__":
    main()