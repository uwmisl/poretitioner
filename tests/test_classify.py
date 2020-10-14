"""
================
test_classify.py
================

This module contains tests for classify.py functionality.

"""
import pytest

import poretitioner.utils.classify as classify


def classify_test():
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
    clf_path = "../../poretitioner/model/NTERs_trained_cnn.pt"
    with pytest.raises(ValueError):
        classify.init_classifier(clf_name, clf_path)


def init_classifier_test():
    assert True is False


def predict_class_test():
    assert True is False


def write_classifier_results():
    assert True is False
