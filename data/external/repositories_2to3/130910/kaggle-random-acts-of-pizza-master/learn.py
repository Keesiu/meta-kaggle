import json
import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
import sys


def unistring(s):
    """Encodes our Unicode string to ASCII, then back to Unicode.
    In the translation, we ignore all unknown characters, which removes
    all of the non-ASCII characters that break our string parsing.
    """
    return str(s).encode("ascii", "ignore").decode()


def process_data(data_json, key1, renamed1, key2, renamed2, key3, renamed3,
                 key4, renamed4, key5, renamed5, key6="", renamed6=""):
    """Converts our list of dictionaries to a dictionary of lists,
    while processing some of the data points (e.g. converting number strings
    to ints and float, as well as lowercasing strings). Also uses renamed keys
    instead of the very long default ones.
    """
    val1 = []
    val2 = []
    val3 = []
    val4 = []
    val5 = []
    val6 = []
    for d in data_json:
        val1.append(d[key1])
        val2.append(float(d[key2]))
        val3.append(int(d[key3]))
        val4.append(d[key4].lower())
        val5.append(d[key5].lower())
        if key6 is not "":
            val6.append(str(d[key6]) == "true" or str(d[key6]) == "True")
    return {renamed1: val1,
            renamed2: val2,
            renamed3: val3,
            renamed4: val4,
            renamed5: val5,
            renamed6: val6}


def get_best_fit_poly_coeffs(data_dict, x, y, deg):
    """Finds the coefficients of the polynomial function for the best fit line.
    """
    coeffs = np.polyfit(x=data_dict[x], y=data_dict[y], deg=deg)
    return coeffs


def scatter_plot(data_dict):
    """Visualizes account age versus karma on the x and y axes, and whether
    or not the requester was given a pizza as the green or red point color.
    """
    for i in range(len(data_dict["age"])):
        formatting = "." + ("g" if data_dict["result"][i] else "r")
        plt.plot(data_dict["age"][i], data_dict["karma"][i], formatting)
    coeffs = get_best_fit_poly_coeffs(data_dict, "age", "karma", 1)
    poly = np.poly1d(coeffs)
    plt.plot(data_dict["age"], poly(data_dict["age"]))
    plt.show()


def will_reciprocate(data_dict, index):
    """Searches the text of the request for phrases that suggest a promise to
    give pizza to someone else. data_dict is a dictionary of lists,and any given
    index in the lists corresponds to the same post for different keys.
    """
    title = data_dict["title"][index]
    body = data_dict["body"][index]
    phrases = ["pay it forward", "return the favor", "reciprocate"]
    for p in phrases:
        if p in title or p in body:
            return True
    return False


def print_data(data_dict):
    """Prints the data for examining or debugging purposes.
    """
    print("id, age, karma, reciprocal, result")
    for i in range(len(data_dict["id"])):
        print((
            data_dict["id"][i],
            data_dict["age"][i],
            data_dict["karma"][i],
            will_reciprocate(data_dict, i),
            data_dict["result"][i]
        ))


def train_acc(data_train, coeffs, multiplier):
    """Determines the accuracy of a set of coefficients
    (with a multiplier) on the training data.
    """
    poly = np.poly1d([i*multiplier for i in coeffs])
    predictions_train = [(poly(data_train["age"][i]) < data_train["karma"][i])
                         or will_reciprocate(data_train, i)
                         for i in range(len(data_train["id"]))]
    correct = 0
    for i in range(len(predictions_train)):
        if predictions_train[i] == data_train["result"][i]:
            correct += 1
    accuracy = correct / len(predictions_train) * 100
    return accuracy


def sweep_coeff(data_train, coeffs):
    """Sweep the coefficients to see which multiplier gives the best accuracy.
    """
    best_m = 1
    best_a = 0
    for m in np.arange(0.0, 100.0, 0.1):
        acc = train_acc(data_train, coeffs, m)
        if acc > best_a:
            best_m = m
            best_a = acc
            print(("New best multiplier:", best_m, "with an accuracy of", best_a))
    return [i*best_m for i in coeffs]


def main(argv):
    """The main program itself. Imports JSON data as a list of dictionaries,
    each of which is a data point (a pizza request). Then makes predictions
    as to whether or not the request would be fulfilled based on whether or
    not they offered to reciprocate, as well as whether or not they had more
    karma than the corresponding point on the best-fit line.
    """
    with open("train.json") as open_file:
        json_train = json.loads(unistring(open_file.read()))
    print((len(json_train), "training data points"))
    data_train = process_data(json_train,
                              "request_id",
                              "id",
                              "requester_account_age_in_days_at_request",
                              "age",
                              "requester_upvotes_minus_downvotes_at_request",
                              "karma",
                              "request_title",
                              "title",
                              "request_text_edit_aware",
                              "body",
                              "requester_received_pizza",
                              "result")
    with open("test.json") as open_file:
        json_test = json.loads(unistring(open_file.read()))
    print((len(json_test), "testing data points"))
    data_test = process_data(json_test,
                             "request_id",
                             "id",
                             "requester_account_age_in_days_at_request",
                             "age",
                             "requester_upvotes_minus_downvotes_at_request",
                             "karma",
                             "request_title",
                             "title",
                             "request_text_edit_aware",
                             "body")
    # print_data(data_dict)
    # scatter_plot(data_dict)
    coeffs = get_best_fit_poly_coeffs(data_train, "age", "karma", 1)
    print(("Initial coefficients:", coeffs))
    best_coeffs = sweep_coeff(data_train, coeffs)
    poly = np.poly1d(best_coeffs)
    print(("Best coefficients:", best_coeffs))

    predictions_test = [(poly(data_test["age"][i]) < data_test["karma"][i])
                        or will_reciprocate(data_test, i)
                        for i in range(len(data_test["id"]))]

    with open("predictions.csv", "w") as output:
        output.write("request_id,requester_received_pizza\n")
        for i in range(len(data_test["id"])):
            output.write("%s,%d\n" % (data_test["id"][i], predictions_test[i]))


if __name__ == "__main__":
    main(sys.argv)
