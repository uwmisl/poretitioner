"""
================
test_classify.py
================

This module contains tests for classify.py functionality.

"""
import os
import re

import h5py
import pytest

from poretitioner.utils import classify, raw_signal_utils
from poretitioner.utils.NTERs_trained_cnn_05152019 import load_cnn


def classify_and_store_result_test():
    # take config from test_filter.py and add to it
    #
    assert True is False


def classify_fast5_file_unfiltered_test():
    # filter_name = None
    assert True is False


def classify_fast5_file_filtered_test():
    # filter_name = "test filter"
    assert True is False


def init_classifier_invalidinput_test():
    clf_name = "invalid"
    clf_path = "not_a_path"
    with pytest.raises(ValueError):
        classify.init_classifier(clf_name, clf_path)

    clf_name = "NTER_cnn"
    clf_path = "not_a_path"
    with pytest.raises(OSError):
        classify.init_classifier(clf_name, clf_path)

    clf_name = "invalid"
    clf_path = "../../poretitioner/model/NTERs_trained_cnn_05152019.statedict.pt"
    with pytest.raises(ValueError):
        classify.init_classifier(clf_name, clf_path)


def load_cnn_test():
    clf_path = "poretitioner/utils/model/NTERs_trained_cnn_05152019.statedict.pt"
    load_cnn(clf_path)


def init_classifier_cnn_test():
    clf_name = "NTER_cnn"
    clf_path = "poretitioner/utils/model/NTERs_trained_cnn_05152019.statedict.pt"
    assert os.path.exists(clf_path)
    classifier = classify.init_classifier(clf_name, clf_path)
    assert classifier
    assert classifier.conv1 is not None  # check existence of a layer


def init_classifier_rf_test():
    assert True is False


def predict_class_test():
    # Predict classification result from 9 segment file
    # Make assertions about the class & probability

    # Load a classifier
    clf_name = "NTER_cnn"
    clf_path = "poretitioner/utils/model/NTERs_trained_cnn_05152019.statedict.pt"
    classifier = classify.init_classifier(clf_name, clf_path)

    correct_results = {
        "697de4c1-1aef-41b9-ae0d-d676e983cb7e": {"label": 5, "prob": 0.426},
        "8e8181d2-d749-4735-9cab-37648b463f88": {"label": 1, "prob": 0.595},
        "97d06f2e-90e6-4ca7-91c4-084f13b693f2": {"label": 9, "prob": 0.976},
        "a0c40f5a-c685-43b9-a3b7-ca13aa90d832": {"label": 9, "prob": 0.984},
        "ab54dab5-26a7-4062-9d77-f63fc40f702c": {"label": 9, "prob": 0.979},
        "c87905e6-fd62-4ac6-bcbd-c7f17ff4af14": {"label": 9, "prob": 0.977},
        "cd6fa746-e93b-467f-a3fc-1c9af815f836": {"label": 6, "prob": 0.802},
        "df4365f4-bfe4-4d2c-8100-34c36cd11378": {"label": 9, "prob": 0.975},
        "f5d76520-c92b-4a9c-b5cb-a04414db527e": {"label": 4, "prob": 0.233},
    }

    # Load raw data & classify
    f5_fname = "tests/data/reads_fast5_dummy_9captures.fast5"
    with h5py.File(f5_fname, "r") as f5:
        for grp in f5.get("/"):
            if "read" in str(grp):
                read_id = re.findall(r"read_(.*)", str(grp))[0]
                # print(f'"{read_id}": {{"label": , "prob": }},')

                raw = raw_signal_utils.get_fractional_blockage_for_read(f5, read_id)
                pred_label, pred_prob = classify.predict_class(clf_name, classifier, raw)

                actual_label = correct_results[read_id]["label"]
                actual_prob = correct_results[read_id]["prob"]

                assert actual_label == pred_label
                assert abs(actual_prob - pred_prob) < 0.02


def write_classifier_details_test():
    # Copy sample file
    # Call fn
    # Read back the expected details
    assert True is False


def write_classifier_result_test():
    # Copy sample file
    # Call fn
    assert True is False
