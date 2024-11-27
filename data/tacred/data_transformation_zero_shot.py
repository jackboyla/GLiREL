import random
import json
import math, os, argparse
from collections import OrderedDict

TACRED_NO_REL = "no_relation"


def possible_class_names(rel_name, names, with_no_relation=False):
    if with_no_relation:
        return rel_name in names or rel_name == category_to_keep
    else:
        return rel_name in names


def read_data(f):
    # for f in [train_file, dev_file, test_file]:
    with open(f, "r") as fp:
        data = json.load(fp, object_pairs_hook=OrderedDict)
    return data


def setup_flipped_data(original_data, these_relations):
    data = {category_to_keep: []}
    for k, v in original_data.items():
        if possible_class_names(k, these_relations) and k != category_to_keep:
            data[k] = v
        else:
            data[category_to_keep].extend(v)
    return data


def main(args):
    '''
    Given the three original supervised data sections
    We re-label all instances in the three sections such that each category
    label appears only in a single data section.
    For TACRED - we choose these category sizes:
    Number of train categories is 25
    Number of dev categories is 6
    Number of test categories is 10

    You can either use the fixed split we published for TACRED or create your own
    random split.
    '''

    train_categories_size = args.train_size
    dev_categories_size = args.dev_size
    test_categories_size = args.test_size

    train_file = args.train_data
    dev_file = args.dev_data
    test_file = args.test_data

    seed = args.seed
    # predefined categories split
    predefined_split = args.fixed_categories_split
    if (seed is not None) and (predefined_split is not None):
        assert False, "We split the categories either by a fixed json file or randomly." \
                      " Please don't specify both a seed and a fixed json format"

    if predefined_split is not None:
        predefined_split = json.load(open(predefined_split))

    output_dir = args.output_dir

    original_train_data = read_data(train_file)
    original_dev_data = read_data(dev_file)
    original_test_data = read_data(test_file)

    global category_to_keep
    category_to_keep = args.category_to_keep

    '''
    This script assumes the data is in the form of a dictionary where 
    the key is the class name and the value is a list of all class instances.
    As in underneath structure:
    Data form: {"class_name: [x_0,x_1,...,x_N]"}
    '''

    categories = [k for k in original_train_data.keys() if k != category_to_keep]
    number_of_tests = math.ceil(len(categories) / test_categories_size)

    if predefined_split is None:
        # random spilt of relation
        if seed is not None:
            print("Random splits, with a fixed seed value:", seed)
            random.seed(seed)
        else:
            print("Random splits, with a random seed.")
        split = build_splits_randomly(categories, train_categories_size,
                                       dev_categories_size,
                                       test_categories_size)

    else:
        # get predefined classes
        print("Fixed splits, according to this json: ", args.fixed_categories_split)
        split = build_splits_from_json(predefined_split)

    # each split is formed: train, dev, test
    [train_relations, dev_relations, test_relations] = split
    split_train_data, split_dev_data, split_test_data = relabel_data_sections(original_train_data,
                                                                                  original_dev_data, original_test_data,
                                                                                  train_relations, dev_relations,
                                                                                  test_relations)

    create_dir_if_needed(output_dir)

    # print_data(split_train_data, split_dev_data, split_test_data, current_dir)
    json.dump(split_train_data, open(os.path.join(output_dir, "_train_data.json"), "w"))
    json.dump(split_dev_data, open(os.path.join(output_dir, "_dev_data.json"), "w"))
    json.dump(split_test_data, open(os.path.join(output_dir, "_test_data.json"), "w"))


def build_splits_from_json(predefined_split):
    split = predefined_split
    train_relations, dev_relations, test_relations = split["train"], split["dev"], split["test"]
    this_split = [train_relations, dev_relations, test_relations]
    return this_split


def build_splits_randomly(categories, train_size, dev_size, test_size):
    possible_test_relation = categories
    test_relations = random.sample(possible_test_relation, test_size)
    all_other_relations = [c for c in categories if c not in test_relations]

    dev_relations = random.sample(all_other_relations, dev_size)
    train_relations = [k for k in all_other_relations if k not in dev_relations]

    assert len(train_relations) == train_size, "Amount of train categories isn't what you specified. " \
                                                       "(Train_size + Dev_size + Test_size) should be equal to the" \
                                                       " size " \
                                                       "of the predefined categories"
    this_split = [train_relations, dev_relations, test_relations]
    return this_split


def relabel_data_sections(original_train_data, original_dev_data, original_test_data, train_relations, dev_relations,
                          test_relations):
    flipped_train_data = setup_flipped_data(original_train_data, train_relations)
    flipped_dev_data = setup_flipped_data(original_dev_data, dev_relations)
    flipped_test_data = setup_flipped_data(original_test_data, test_relations)
    return flipped_train_data, flipped_dev_data, flipped_test_data


def create_dir_if_needed(dir_name):
    if dir_name[-1] == '/':
        dir_name = dir_name[:-1]
    # Create target Directory if doesn't exist
    if not os.path.exists(dir_name):
        os.mkdir(dir_name)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--train_data", type=str, required=True)
    parser.add_argument("--train_size", default=25, type=int, required=False,
                        help="The number of train data categories")
    parser.add_argument("--dev_data", type=str, required=True)
    parser.add_argument("--dev_size", default=6, type=int, required=False,
                        help="The number of dev data categories")
    parser.add_argument("--test_data", type=str, required=True)
    parser.add_argument("--test_size", default=10, type=int, required=False,
                        help="The number of test data categories")
    parser.add_argument("--fixed_categories_split", default=None, type=str, required=False,
                        help="split the categories based on a published json file that specify "
                             + "per each split the categories")

    parser.add_argument("--category_to_keep", default=TACRED_NO_REL, type=str, required=False,
                        help="If one of the category may be kept in all data section please specify the category name")

    parser.add_argument("--seed", default=None, type=int, required=False)

    parser.add_argument("--output_dir", type=str, required=False,
                        help="The output directory where the Few-shot data is stored.")

    _args = parser.parse_args()
    main(_args)
