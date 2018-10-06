import numpy as np
import json
from TrainTuple import TrainTuple
from TestTuple import TestTuple
from HashTable import HashTable

train_file_name = "./train.json"
test_file_name = "./test.json"

class_name_json = "cuisine"
attribute_name_json = "ingredients"
id_name_json = "id"


def classify():

    train_data = import_train_data()

    number_of_tuples = np.size(train_data) * 1.0

    train_tuple_list, unique_classes = build_list_of_train_tuples(train_data)

    unique_attributes = build_unique_attributes(train_tuple_list)

    hash_table = build_hash_table(unique_attributes)

    attributes_counts = build_attributes_counts(
        train_tuple_list, unique_classes, unique_attributes, hash_table)

    test_data = import_test_data()

    test_tuple_list = build_test_tuple_list(test_data)

    bayes_classifier_algorith(
        test_tuple_list,
        hash_table,
        unique_classes,
        attributes_counts,
        number_of_tuples)

    print_class_labels_from_test(test_tuple_list)


def get_class_index(class_label, unique_classes):
    return np.where(str(class_label) == unique_classes[0])[0][0]


def get_probability_of_attribute(
        class_index,
        attribute_index,
        attributes_counts,
        unique_classes):
    class_count = unique_classes[1][class_index]

    # The Laplacian correction is used here by adding 1.0
    attribute_count = attributes_counts[class_index][attribute_index] + 1.0

    return attribute_count / class_count


def get_probability_of_class_label(
        class_index,
        unique_classes,
        number_of_tuples):
    class_count = unique_classes[1][class_index] * 1.0

    return class_count / number_of_tuples


def print_class_labels_from_test(test_tuple_list):
    print(class_name_json + "," + attribute_name_json)
    for i in range(0, np.size(test_tuple_list)):
        tuple_id = str(test_tuple_list[i].id)
        tuple_class = str(test_tuple_list[i].class_label)
        comma = ","
        sequence = (tuple_id, tuple_class)
        print(comma.join(sequence))


def bayes_classifier_algorith(
        test_tuple_list,
        hash_table,
        unique_classes,
        attributes_counts,
        number_of_tuples):
    # cycle through every test tuple
    for i in range(0, np.size(test_tuple_list)):
        test_tuple = test_tuple_list[i]
        # cycle through every class label
        for class_index in range(0, np.size(unique_classes[0])):
            prod = 1.0
            # cycle through every attribute in tuple for every class label
            for k in range(0, np.size(test_tuple.attributes)):
                attribute_index = hash_table.get(test_tuple.attributes[k])
                # if attribute_index == -1 then the attribute wasn't in the
                # training data, so ignore
                if attribute_index > -1:
                    prod *= get_probability_of_attribute(
                        class_index,
                        attribute_index,
                        attributes_counts,
                        unique_classes)

            test_tuple.class_probabilities[
                class_index] = prod * get_probability_of_class_label(
                    class_index, unique_classes, number_of_tuples)

        class_index_of_max = np.argmax(test_tuple.class_probabilities)
        test_tuple.class_label = unique_classes[0][class_index_of_max]


def build_test_tuple_list(test_data):
    test_tuple_list = []

    for i in range(0, np.size(test_data)):
        test_tuple = TestTuple(
            test_data[i][id_name_json], np.array(
                test_data[i][attribute_name_json]))
        test_tuple_list.append(test_tuple)

    return test_tuple_list


def build_attributes_counts(
        train_tuple_list,
        unique_classes,
        unique_attributes,
        hash_table):
    attributes_counts = np.zeros(
        (np.size(
            unique_classes[0]),
            np.size(unique_attributes)),
        dtype=np.int)

    for i in range(0, np.size(train_tuple_list)):
        train_tuple = train_tuple_list[i]
        class_index = get_class_index(train_tuple.class_label, unique_classes)
        attributes_array = np.array(train_tuple.attributes)

        for j in range(0, np.size(attributes_array)):
            attribute_index = hash_table.get(attributes_array[j])
            attributes_counts[class_index][attribute_index] += 1

    return attributes_counts


def build_hash_table(unique_attributes):
    hash_table = HashTable(np.size(unique_attributes))

    for i in range(0, np.size(unique_attributes)):
        hash_table.put(unique_attributes[i], i)

    return hash_table


def build_list_of_train_tuples(train_data):

    train_tuple_list = []

    dt_str = np.dtype(('U', 30))

    class_numpy = np.zeros((np.size(train_data),), dtype=dt_str)

    for i in range(0, np.size(train_data)):
        r = train_data[i]
        train_tuple = TrainTuple(
            r[class_name_json],
            r[id_name_json],
            np.array(r[attribute_name_json]))

        train_tuple_list.append(train_tuple)
        class_numpy[i] = train_tuple.class_label

    unique_classes = np.unique(class_numpy)

    class_count = np.zeros((np.size(unique_classes),), dtype=np.int)

    for i in range(0, np.size(train_data)):
        temp_class = train_tuple_list[i].class_label

        index = np.where(str(temp_class) == unique_classes)[0][0]
        class_count[index] += 1

    unique_classes = [unique_classes, class_count]

    return train_tuple_list, unique_classes


def build_unique_attributes(train_tuple_list):
    attributes_total = 0

    for i in range(0, np.size(train_tuple_list)):
        attributes_total += np.size(train_tuple_list[i].attributes)

    dt_str = np.dtype(('U', 75))

    attributes_numpy = np.zeros((attributes_total,), dtype=dt_str)

    index = 0
    for i in range(0, np.size(train_tuple_list)):
        temp_array = np.array(train_tuple_list[i].attributes)
        attributes_numpy[index:(index + np.size(temp_array))] = temp_array
        index += np.size(temp_array)

    unique_attributes = np.unique(attributes_numpy)

    return unique_attributes


def import_test_data():
    json_file = test_file_name
    json_data = open(json_file)

    test_data = json.load(json_data)
    json_data.close()

    return test_data


def import_train_data():
    json_file = train_file_name
    json_data = open(json_file)

    train_data = json.load(json_data)
    json_data.close()

    return train_data


def main():
    classify()

if __name__ == '__main__':
    main()
