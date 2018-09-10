# Kaggle Bike Sharing Demand
# Joey L. Maalouf
# Approach: Support Vector Regression

# -- import any necessary modules ----------------------------------------------
import csv
from sklearn.svm import SVR
# from sklearn.grid_search import GridSearchCV


# -- define our functions ------------------------------------------------------
def to_int(input):
    try:
        return int(input)
    except TypeError:
        return [int(input[0]), int(input[1])]


def read_data(filename, xy):
    datalist = []
    # read in the file
    data = open(filename)
    reader = csv.reader(data, delimiter=",")
    for row in reader:
        if (xy == "x"):
            # store just the hour and weather
            datalist.append([row[0][11:13], row[4]])
        elif (xy == "y"):
            # store just the count
            datalist.append(row[11])
        elif (xy == "xx"):
            datalist.append(row[0])
    return datalist[1:] if xy == "xx" else [to_int(i) for i in datalist[1:]]


# -- read in the data ----------------------------------------------------------
print("Let's start reading in the data...")
x_train = read_data("train.csv", "x")
y_train = read_data("train.csv", "y")
x_test = read_data("test.csv", "x")
print("Finished reading in the data!\n")

# -- fit regression model ------------------------------------------------------
print("Let's start instantiating our model...")
# parameters = \
#     [
#         {
#             "kernel": ["rbf"],
#             "C": [1e3, 1e2, 1e1],
#             "gamma": [1e0, 1e-1, 1e-2, 1e-3]
#         },
#         {
#             "kernel": ["poly"],
#             "C": [1e3, 1e2, 1e1],
#             "gamma": [1e0, 1e-1, 1e-2, 1e-3],
#             "degree": [2, 3, 4]
#         }
#     ]
# svr = GridSearchCV(SVR(), parameters)
svr = SVR(kernel="rbf", C=1000, gamma=0.1)
print("Finished instantiating our model!\n")
print("Let's start training our model...")
model = svr.fit(x_train, y_train)
print("Finished training our model!\n")
print("Let's start predicting our new data...")
y_test = model.predict(x_test)
print("Finished predicting our new data!\n")

# print("\nBest estimator:")
# print(svr.best_estimator_)
# print("\nBest parameters:")
# print(svr.best_params_)
# print("\nScorer:")
# print(svr.scorer_)
# print("\nGrid scores:")
# for s in svr.grid_scores_:
#     print(s)

# -- output the results --------------------------------------------------------
datetime = read_data("test.csv", "xx")
with open("predicted_output_sv.csv", "w") as output:
    output.write("datetime,count\n")
    for i in range(len(y_test)):
        output.write("%s,%d\n" % (datetime[i], y_test[i]))
