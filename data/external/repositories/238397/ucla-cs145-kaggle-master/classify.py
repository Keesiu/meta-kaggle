"""
classify.py

Main command line tool to run cuisine classifiers

Example usage:

To test algorithm accuracy:
python classify.py -t -p 0.1 --trainfile train.json -a random
python classify.py -t -p 0.1 --trainfile train.json -a nbc
python classify.py -t -p 0.1 --trainfile train.json -a decisiontree -s 200
python classify.py -t -p 0.1 --trainfile train.json -a randomforest -e 100

To generate an actual submission:
python classify.py --trainfile train.json --testfile test.json -a nbc -o output.csv
"""

import argparse
from cuisinedatabase import *
from matrixdatabase import MatrixDatabase
from randomclassifier import RandomClassifier
from nbc import NaiveBayesClassifier
from decisiontreeclassifier import DecisionTreeClassifier
from randomforestclassifier import RandomForestClassifier
from linearsvclassifier import LinearSVClassifier
from adaboostclassifier import AdaBoostClassifier
from extratreeclassifier import ExtraTreeClassifier
from labelpropagationclassifier import LabelPropagationClassifier
from probabilitycalibrationclassifier import ProbabilityCalibrationClassifier
from multinomialnbclassifier import MultinomialNBClassifier
from baggingclassifier import BaggingClassifier
from passiveagressiveclassifier import PassiveAgressiveClassifier

def main():

    parser = argparse.ArgumentParser(description="classifiy ingredients of a recipe to a cuisine type")

    parser.add_argument("-t", "--test", action="store_true",
                    help="re-assign some of the train data to test data with known class")

    parser.add_argument("-p", type=float,
                    help="for --test, probability of assigning data entry to test instead of train")

    parser.add_argument("--trainfile",
                    help="input training data file (known class)")

    parser.add_argument("--testfile",
                    help="input testing data file (unknown class)")

    parser.add_argument("-o", "--outputfile",
                    help="output file name (csv file)")

    parser.add_argument("-a", "--algorithm",
                    help="the classification algorithm to use")

    parser.add_argument("-s", "--splits", type=int,
                    help="the number of most-frequent ingredients to use as possible splits in the decision tree classifier (e.g. 200)")

    parser.add_argument("-e", "--estimators", type=int,
                    help="the number of estimators to use in the random forest classifier (e.g. 100)")

    args = parser.parse_args()

    if args.test:
        db = TestDatabase(args.trainfile, args.p)
    else:
        db = CuisineDatabase(args.trainfile, args.testfile)

    mdb = MatrixDatabase(db)

    c = Classification()

    algo = None
    if args.algorithm == "random":
        algo = RandomClassifier()
    elif args.algorithm == "nbc":
        algo = NaiveBayesClassifier()
    elif args.algorithm == "decisiontree":
        algo = DecisionTreeClassifier(args.splits)
    elif args.algorithm == "randomforest":
        algo = RandomForestClassifier(args.estimators, mdb)
    elif args.algorithm == "linearsv":
        algo = LinearSVClassifier(mdb)
    elif args.algorithm == "adaboost":
        algo = AdaBoostClassifier(args.estimators, mdb)
    elif args.algorithm == "extratree":
        algo = ExtraTreeClassifier(mdb)
    elif args.algorithm == "labelpropagation":
        algo = LabelPropagationClassifier(mdb)
    elif args.algorithm == "probabilitycalibration":
        algo = ProbabilityCalibrationClassifier(mdb)
    elif args.algorithm == "multinomialnb":
        algo = MultinomialNBClassifier(mdb)
    elif args.algorithm == "bagging":
        algo = BaggingClassifier(mdb)
    elif args.algorithm == "passiveagressive":
        algo = PassiveAgressiveClassifier(mdb)

    print "Starting " + args.algorithm + " training..."

    i = 0
    entry = db.train(i)
    while (entry != None):
        algo.learn(entry["ingredients"], entry["cuisine"])
        i += 1
        entry = db.train(i)

    print "Learned " + str(i) + " training data entries..."

    print "Starting " + args.algorithm + " classification..."

    i = 0
    entry = db.test(i)
    while (entry != None):
        c.save(entry["id"], algo.classify(entry["ingredients"]))
        i += 1
        entry = db.test(i)

    print "Classified " + str(i) + " test data entries..."

    if args.test:
        (n_correct, n_total) = db.accuracy(c)
        accuracy = float(n_correct)/float(n_total)
        print "n_correct: " + str(n_correct)
        print "n_total: " + str(n_total)
        if n_total > 0:
            print "accuracy: " + str(accuracy)

    if args.outputfile != None:
        c.save_csv(args.outputfile)

if __name__ == '__main__':
    main()
