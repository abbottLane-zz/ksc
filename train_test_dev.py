from DataLoading.DataLoader import DataLoader
from Evalutation.Evaluator import Evaluator
from Extraction.TextClassfModel import TextClassificationModel


def main():
    ##Set directories
    data_dir ="/home/wlane/PycharmProjects/Kickstarter_classification/Data/kickstarter_corpus.json"
    model_out_dir = "Data/dev_models/models/kick_classf.pk1"
    vectorizer_out_dir = "Data/dev_models/vectorizers/kick_classf.vec"

    ## Load data
    dl = DataLoader(data_dir)
    train_data = dl.get_train_data_set()

    ## train model
    model = TextClassificationModel()
    model.train(train_data)
    model.dump_model(model_out_dir)
    model.dump_vectorizer(vectorizer_out_dir)

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
    print("Avg f1: " + str(avg_f1s))


if __name__=="__main__":
    main()
