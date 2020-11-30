"""
================
test_classify.py
================

This module contains tests for classify.py functionality.

"""
import os
import re
from shutil import copyfile

import h5py
import pytest

from poretitioner.utils import classify, raw_signal_utils
from poretitioner.utils.NTERs_trained_cnn_05152019 import load_cnn


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
        "697de4c1-1aef-41b9-ae0d-d676e983cb7e": {"label": 5, "prob": 0.726},
        "8e8181d2-d749-4735-9cab-37648b463f88": {"label": 8, "prob": 0.808},
        "97d06f2e-90e6-4ca7-91c4-084f13b693f2": {"label": 9, "prob": 0.977},
        "a0c40f5a-c685-43b9-a3b7-ca13aa90d832": {"label": 9, "prob": 0.977},
        "ab54dab5-26a7-4062-9d77-f63fc40f702c": {"label": 9, "prob": 0.979},
        "c87905e6-fd62-4ac6-bcbd-c7f17ff4af14": {"label": 9, "prob": 0.977},
        "cd6fa746-e93b-467f-a3fc-1c9af815f836": {"label": 6, "prob": 0.484},
        "df4365f4-bfe4-4d2c-8100-34c36cd11378": {"label": 9, "prob": 0.975},
        "f5d76520-c92b-4a9c-b5cb-a04414db527e": {"label": 6, "prob": 0.254},
    }

    # Load raw data & classify
    f5_fname = "tests/data/reads_fast5_dummy_9captures.fast5"
    with h5py.File(f5_fname, "r") as f5:
        for grp in f5.get("/"):
            if "read" in str(grp):
                read_id = re.findall(r"read_(.*)", str(grp))[0]

                raw = raw_signal_utils.get_fractional_blockage_for_read(f5, read_id)
                pred_label, pred_prob = classify.predict_class(
                    clf_name, classifier, raw[100:], class_labels=None
                )
                # print(
                #     f'"{read_id}": {{"label": {pred_label}, "prob": {pred_prob:0.3f}}},'
                # )
                p = pred_prob
                # print(f"p: {p} {type(p)}")

                actual_label = correct_results[read_id]["label"]
                actual_prob = correct_results[read_id]["prob"]

                assert actual_label == pred_label
                assert abs(actual_prob - pred_prob) < 0.02


def write_classifier_details_test():
    # Copy sample file
    orig_f5_fname = "tests/data/reads_fast5_dummy_9captures.fast5"
    test_f5_fname = "write_classifier_details_test.fast5"
    copyfile(orig_f5_fname, test_f5_fname)

    # Define subset of the config dict that contains filter info
    clf_config = {
        "classifier": "NTER_cnn",
        "classifier_path": "poretitioner/utils/model/NTERs_trained_cnn_05152019.statedict.pt",
        "classifier_version": "1.0",
        "start_obs": 100,
        "end_obs": 2100,
        "min_confidence": 0.9,
    }

    # Call fn
    with h5py.File(test_f5_fname, "r+") as f5:
        classify.write_classifier_details(f5, clf_config, "/Classification/NTER_cnn")

    # Read back the expected details
    with h5py.File(test_f5_fname, "r") as f5:
        assert "/Classification" in f5
        assert "/Classification/NTER_cnn" in f5
        attrs = f5.get("/Classification/NTER_cnn").attrs
        for key in ["model", "model_version", "model_file", "classification_threshold"]:
            assert key in attrs
    os.remove(test_f5_fname)


def write_classifier_result_test():
    # Copy sample file
    orig_f5_fname = "tests/data/classifier_details_test.fast5"
    test_f5_fname = "write_classifier_result_test.fast5"
    copyfile(orig_f5_fname, test_f5_fname)

    classifier_run_name = "NTER_cnn"
    results_path = f"/Classification/{classifier_run_name}"

    # Call fn
    pred_class = 9
    prob = 0.977
    passed = True
    with h5py.File(test_f5_fname, "r+") as f5:
        classify.write_classifier_result(
            f5,
            results_path,
            "c87905e6-fd62-4ac6-bcbd-c7f17ff4af14",
            pred_class,
            prob,
            passed,
        )

    # Read back the expected info
    with h5py.File(test_f5_fname, "r") as f5:
        assert results_path in f5
        assert f"{results_path}/c87905e6-fd62-4ac6-bcbd-c7f17ff4af14" in f5
        attrs = f5.get(f"{results_path}/c87905e6-fd62-4ac6-bcbd-c7f17ff4af14").attrs
        assert attrs["best_class"] == pred_class
        assert attrs["best_score"] == prob
        assert attrs["assigned_class"] == pred_class
    os.remove(test_f5_fname)


def get_classification_for_read_test():
    f5_fname = "tests/data/classified_9captures.fast5"
    read_id = "c87905e6-fd62-4ac6-bcbd-c7f17ff4af14"
    classifier_run_name = "NTER_cnn"
    results_path = f"/Classification/{classifier_run_name}"
    with h5py.File(f5_fname, "r") as f5:
        (
            pred_class,
            prob,
            assigned_class,
            passed_classification,
        ) = classify.get_classification_for_read(f5, read_id, results_path)
    assert pred_class == 9
    assert prob == 0.9876
    assert assigned_class == 9
    assert passed_classification is True


def classify_fast5_file_unfiltered_test():
    # Predict classification result from 9 segment file
    # Make assertions about the class & probability

    # Define subset of the config dict that contains config info
    clf_config = {
        "classifier": "NTER_cnn",
        "classifier_path": "poretitioner/utils/model/NTERs_trained_cnn_05152019.statedict.pt",
        "classifier_version": "1.0",
        "start_obs": 100,
        "end_obs": 21000,
        "min_confidence": 0.9,
    }

    # Load a classifier
    clf_name = "NTER_cnn"
    clf_path = "poretitioner/utils/model/NTERs_trained_cnn_05152019.statedict.pt"
    classifier = classify.init_classifier(clf_name, clf_path)
    classifier_run_name = "NTER_cnn"
    results_path = f"/Classification/{classifier_run_name}"

    correct_results = {
        "697de4c1-1aef-41b9-ae0d-d676e983cb7e": {"label": 5, "prob": 0.726},
        "8e8181d2-d749-4735-9cab-37648b463f88": {"label": 8, "prob": 0.808},
        "97d06f2e-90e6-4ca7-91c4-084f13b693f2": {"label": 9, "prob": 0.977},
        "a0c40f5a-c685-43b9-a3b7-ca13aa90d832": {"label": 9, "prob": 0.977},
        "ab54dab5-26a7-4062-9d77-f63fc40f702c": {"label": 9, "prob": 0.979},
        "c87905e6-fd62-4ac6-bcbd-c7f17ff4af14": {"label": 9, "prob": 0.977},
        "cd6fa746-e93b-467f-a3fc-1c9af815f836": {"label": 6, "prob": 0.484},
        "df4365f4-bfe4-4d2c-8100-34c36cd11378": {"label": 9, "prob": 0.975},
        "f5d76520-c92b-4a9c-b5cb-a04414db527e": {"label": 6, "prob": 0.254},
    }

    # Prepare file for testing
    orig_f5_fname = "tests/data/reads_fast5_dummy_9captures.fast5"
    test_f5_fname = "classify_fast5_file_unfiltered_test.fast5"
    copyfile(orig_f5_fname, test_f5_fname)

    # Classify f5 file directly
    with h5py.File(test_f5_fname, "r+") as f5:
        classify.classify_fast5_file(f5, clf_config, classifier, clf_name, "/", class_labels=None)

    # Evaluate output written to file
    with h5py.File(test_f5_fname, "r") as f5:
        for grp in f5.get("/"):
            if "read" in str(grp):
                read_id = re.findall(r"read_(.*)", str(grp))[0]

                (
                    pred_label,
                    pred_prob,
                    assigned_class,
                    passed_classification,
                ) = classify.get_classification_for_read(f5, read_id, results_path)

                actual_label = correct_results[read_id]["label"]
                actual_prob = correct_results[read_id]["prob"]
                t = clf_config["min_confidence"]

                print(
                    f"read_id: {read_id}\tactual_label: {actual_label}\tpred_label: {pred_label}"
                )

                assert actual_label == pred_label
                assert abs(actual_prob - pred_prob) < 0.02
                assert passed_classification == bool(pred_prob > t)
    os.remove(test_f5_fname)


def classify_fast5_file_filtered_test():
    # Predict classification result from 9 segment file
    # Make assertions about the class & probability

    # Define subset of the config dict that contains config info
    clf_config = {
        "classifier": "NTER_cnn",
        "classifier_path": "poretitioner/utils/model/NTERs_trained_cnn_05152019.statedict.pt",
        "classifier_version": "1.0",
        "start_obs": 100,
        "end_obs": 21000,
        "min_confidence": 0.9,
    }

    # Load a classifier
    clf_name = "NTER_cnn"
    clf_path = "poretitioner/utils/model/NTERs_trained_cnn_05152019.statedict.pt"
    classifier = classify.init_classifier(clf_name, clf_path)
    classifier_run_name = "NTER_cnn"
    results_path = f"/Classification/{classifier_run_name}"

    correct_results = {
        "697de4c1-1aef-41b9-ae0d-d676e983cb7e": {"label": 5, "prob": 0.726},
        "8e8181d2-d749-4735-9cab-37648b463f88": {"label": 8, "prob": 0.808},
        "97d06f2e-90e6-4ca7-91c4-084f13b693f2": {"label": 9, "prob": 0.977},
        "a0c40f5a-c685-43b9-a3b7-ca13aa90d832": {"label": 9, "prob": 0.977},
        "ab54dab5-26a7-4062-9d77-f63fc40f702c": {"label": 9, "prob": 0.979},
        "c87905e6-fd62-4ac6-bcbd-c7f17ff4af14": {"label": 9, "prob": 0.977},
        "cd6fa746-e93b-467f-a3fc-1c9af815f836": {"label": 6, "prob": 0.484},
        "df4365f4-bfe4-4d2c-8100-34c36cd11378": {"label": 9, "prob": 0.975},
        "f5d76520-c92b-4a9c-b5cb-a04414db527e": {"label": 6, "prob": 0.254},
    }

    # Prepare file for testing
    orig_f5_fname = "tests/data/filter_and_store_result_test.fast5"
    test_f5_fname = "classify_fast5_file_filtered_test.fast5"
    copyfile(orig_f5_fname, test_f5_fname)

    # Use reads from filtered section, not root reads path
    filter_name = "standard filter"
    reads_path = f"/Filter/{filter_name}/pass"

    # Classify f5 file directly
    with h5py.File(test_f5_fname, "r+") as f5:
        classify.classify_fast5_file(
            f5, clf_config, classifier, clf_name, reads_path, class_labels=None
        )

    # Evaluate output written to file
    with h5py.File(test_f5_fname, "r") as f5:
        for grp in f5.get(reads_path):
            if "read" in str(grp):
                read_id = re.findall(r"read_(.*)", str(grp))[0]

                (
                    pred_label,
                    pred_prob,
                    assigned_class,
                    passed_classification,
                ) = classify.get_classification_for_read(f5, read_id, results_path)

                actual_label = correct_results[read_id]["label"]
                actual_prob = correct_results[read_id]["prob"]
                t = clf_config["min_confidence"]

                print(
                    f"read_id: {read_id}\tactual_label: {actual_label}\tpred_label: {pred_label}"
                )

                assert actual_label == pred_label
                assert abs(actual_prob - pred_prob) < 0.02
                assert passed_classification == bool(pred_prob > t)
    os.remove(test_f5_fname)


def filter_and_classify_test():
    # take config from test_filter.py and add to it

    orig_f5_fname = "tests/data/reads_fast5_dummy_9captures.fast5"
    test_f5_fname = "filter_and_classify_test.fast5"
    copyfile(orig_f5_fname, test_f5_fname)

    # Define config dict that contains filter info
    config = {
        "compute": {"n_workers": 4},
        "filters": {
            "base filter": {"length": (100, None)},
            "test filter": {"min": (100, None)},
        },
        "output": {"capture_f5_dir": "tests/", "captures_per_f5": 1000},
        "classify": {
            "classifier": "NTER_cnn",
            "classifier_path": "poretitioner/utils/model/NTERs_trained_cnn_05152019.statedict.pt",
            "classifier_version": "1.0",
            "start_obs": 100,
            "end_obs": 21000,
            "min_confidence": 0.9,
        },
    }

    # Predict classification result from 9 segment file
    # Make assertions about the class & probability

    correct_results = {
        "697de4c1-1aef-41b9-ae0d-d676e983cb7e": {"label": 5, "prob": 0.726},
        "8e8181d2-d749-4735-9cab-37648b463f88": {"label": 8, "prob": 0.808},
        "97d06f2e-90e6-4ca7-91c4-084f13b693f2": {"label": 9, "prob": 0.977},
        "a0c40f5a-c685-43b9-a3b7-ca13aa90d832": {"label": 9, "prob": 0.977},
        "ab54dab5-26a7-4062-9d77-f63fc40f702c": {"label": 9, "prob": 0.979},
        "c87905e6-fd62-4ac6-bcbd-c7f17ff4af14": {"label": 9, "prob": 0.977},
        "cd6fa746-e93b-467f-a3fc-1c9af815f836": {"label": 6, "prob": 0.484},
        "df4365f4-bfe4-4d2c-8100-34c36cd11378": {"label": 9, "prob": 0.975},
        "f5d76520-c92b-4a9c-b5cb-a04414db527e": {"label": 6, "prob": 0.254},
    }

    # Classify f5 file directly
    filter_name = "base filter"
    classify.filter_and_classify(
        config, [test_f5_fname], overwrite=True, filter_name="base filter"
    )

    # Use reads from filtered section, not root reads path
    reads_path = f"/Filter/{filter_name}/pass"
    results_path = f"/Classification/{config['classify']['classifier']}"

    # Evaluate output written to file
    with h5py.File(test_f5_fname, "r") as f5:
        for grp in f5.get(reads_path):
            if "read" in str(grp):
                read_id = re.findall(r"read_(.*)", str(grp))[0]

                (
                    pred_label,
                    pred_prob,
                    assigned_class,
                    passed_classification,
                ) = classify.get_classification_for_read(f5, read_id, results_path)

                actual_label = correct_results[read_id]["label"]
                actual_prob = correct_results[read_id]["prob"]
                t = config["classify"]["min_confidence"]

                print(
                    f"read_id: {read_id}\tactual_label: {actual_label}\tpred_label: {pred_label}"
                )

                assert actual_label == pred_label
                assert abs(actual_prob - pred_prob) < 0.02
                assert passed_classification == bool(pred_prob > t)
