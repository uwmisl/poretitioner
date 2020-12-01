"""
===========
classify.py
===========

This module contains functionality for classifying nanopore captures.

"""
import os
import re

import h5py
import numpy as np
import torch
import torch.nn as nn

import poretitioner.utils.NTERs_trained_cnn_05152019 as pretrained_model
from poretitioner import logger
from poretitioner.utils import filter, raw_signal_utils

use_cuda = False  # True
# TODO : Don't hardcode use of CUDA : https://github.com/uwmisl/poretitioner/issues/41


def filter_and_classify(config, fast5_fnames, overwrite=False, filter_name=None):
    local_logger = logger.getLogger()
    clf_config = config["classify"]
    classifier_name = clf_config["classifier"]
    classifier_path = clf_config["classifier_path"]

    # Load classifier
    local_logger.info(f"Loading classifier {classifier_name}.")
    assert classifier_name in ["NTER_cnn", "NTER_rf"]
    assert classifier_path is not None and len(classifier_path) > 0
    classifier = init_classifier(classifier_name, classifier_path)

    # Filter (optional)
    if filter_name is not None:
        local_logger.info("Beginning filtering.")
        filter.filter_and_store_result(config, fast5_fnames, filter_name, overwrite=overwrite)
        read_path = f"/Filter/{filter_name}/pass"
    else:
        read_path = "/"

    # Classify
    for fast5_fname in fast5_fnames:
        with h5py.File(fast5_fname, "r+") as f5:
            classify_fast5_file(f5, clf_config, classifier, classifier_name, read_path)


def classify_fast5_file(
    f5, clf_config, classifier, classifier_run_name, read_path, class_labels=None
):
    local_logger = logger.getLogger()
    local_logger.debug(f"Beginning classification for file {f5.filename}.")
    classifier_name = clf_config["classifier"]
    classify_start = clf_config["start_obs"]  # 100 in NTER paper
    classify_end = clf_config["end_obs"]  # 21000 in NTER paper
    classifier_conf = clf_config["min_confidence"]

    assert classify_start >= 0 and classify_end >= 0
    assert classifier_conf is None or (0 <= classifier_conf and classifier_conf <= 1)

    local_logger.debug(
        f"Classification parameters: name: {classifier_name}, "
        f"range of data points: ({classify_start}, {classify_end})"
        f"confidence required to pass: {classifier_conf}"
    )

    results_path = f"/Classification/{classifier_run_name}"
    write_classifier_details(f5, clf_config, results_path)

    read_h5group_names = f5.get(read_path)
    for grp in read_h5group_names:
        if "read" not in grp:
            continue
        read_id = re.findall(r"read_(.*)", str(grp))[0]
        signal = raw_signal_utils.get_fractional_blockage_for_read(
            f5, grp, start=classify_start, end=classify_end
        )
        y, p = predict_class(classifier_name, classifier, signal, class_labels=class_labels)
        if classifier_conf is not None:
            passed_classification = False if p <= classifier_conf else True
        else:
            passed_classification = None
        write_classifier_result(f5, results_path, read_id, y, p, passed_classification)


def init_classifier(classifier_name, classifier_path):
    """Initialize the classification model. Supported classifier names include
    "NTER_cnn" and "NTER_rf".

    According to documentation for original NTER code:
    Prediction classes are 1-9:
    0:Y00, 1:Y01, 2:Y02, 3:Y03, 4:Y04, 5:Y05, 6:Y06, 7:Y07, 8:Y08, 9:noise,
    -1:below conf_thesh

    Parameters
    ----------
    classifier_name : str
        The name of any supported classifier, currently "NTER_cnn" and "NTER_rf".
    classifier_path : str
        Location of the pre-trained model file.

    Returns
    -------
    model
        Classification model (type depends on the spceified model).

    Raises
    ------
    ValueError
        Raised if the classifier name is not supported.
    """
    if classifier_name == "NTER_cnn":  # CNN classifier
        if not os.path.exists(classifier_path):
            raise OSError(f"Classifier path doesn't exist: {classifier_path}")
        nanoporeTER_cnn = pretrained_model.load_cnn(classifier_path)
        return nanoporeTER_cnn
    elif classifier_name == "NTER_rf":  # Random forest classifier
        if not os.path.exists(classifier_path):
            raise OSError(f"Classifier path doesn't exist: {classifier_path}")
        # TODO : Improve model maintainability : https://github.com/uwmisl/poretitioner/issues/38
        # return joblib.load(open(classifier_path, "rb"))
        pass
    else:
        raise ValueError(f"Invalid classifier name: {classifier_name}")


def predict_class(classifier_name, classifier, raw, class_labels=None):
    """Runs the classifier using the given raw data as input. Does not apply
    any kind of confidence threshold.

    Parameters
    ----------
    classifier_name : str
        The name of any supported classifier, currently "NTER_cnn" and "NTER_rf".
    classifier : model
        Classification model returned by init_classifier.
    raw : iterable of floats
        Time series of nanopore current values (in units of fractionalized current).
    Returns
    -------
    int or string
        Class label
    float
        Model score (for NTER_cnn and NTER_rf, it's a probability)

    Raises
    ------
    NotImplementedError
        Raised if the input classifier_name is not supported.
    """
    if classifier_name == "NTER_cnn":
        X_test = np.array([raw])
        # 2D --> 3D array (each obs in a capture becomes its own array)
        X_test = X_test.reshape(len(X_test), X_test.shape[1], 1)
        if X_test.shape[1] < 19881:
            temp = np.zeros((X_test.shape[0], 19881, 1))
            temp[:, : X_test.shape[1], :] = X_test
            X_test = temp
        X_test = X_test[:, :19881]  # First 19881 obs as per NTER paper
        # Break capture into 141x141 (19881 total data points)
        X_test = X_test.reshape(len(X_test), 1, 141, 141)
        X_test = torch.from_numpy(X_test)
        if use_cuda:
            X_test = X_test.cuda()
        outputs = classifier(X_test)
        out = nn.functional.softmax(outputs, dim=1)
        prob, lab = torch.topk(out, 1)
        if use_cuda:
            lab = lab.cpu().numpy()[0][0]
        else:
            lab = lab.numpy()[0][0]
        if class_labels is not None:
            lab = class_labels[lab]
        prob = prob[0][0].data
        return lab, prob
    elif classifier_name == "NTER_rf":
        class_proba = classifier.predict_proba(
            [[np.mean(raw), np.std(raw), np.min(raw), np.max(raw), np.median(raw)]]
        )[0]
        max_proba = np.amax(class_proba)
        lab = np.where(class_proba == max_proba)[0][0]
        if class_labels is not None:
            lab = class_labels[lab]
        return lab, class_proba
    else:
        raise NotImplementedError(f"Classifier {classifier_name} not implemented.")


def get_classification_for_read(f5, read_id, results_path):
    results_path = f"{results_path}/{read_id}"
    try:
        assert results_path in f5
    except AssertionError:
        raise ValueError(
            f"Read {read_id} has not been classified yet, or result"
            f"is not stored at {results_path} in file {f5.filename}."
        )
    pred_class = f5[results_path].attrs["best_class"]
    prob = f5[results_path].attrs["best_score"]
    assigned_class = f5[results_path].attrs["assigned_class"]
    passed_classification = True if assigned_class == pred_class else False
    return pred_class, prob, assigned_class, passed_classification


def write_classifier_details(f5, classifier_config, results_path):
    """Write metadata about the classifier that doesn't need to be repeated for
    each read.

    Parameters
    ----------
    f5 : h5py.File
        Opened fast5 file in a writable mode.
    classifier_config : dict
        Subset of the configuration parameters that belong to the classifier.
    results_path : str
        Where the classification results will be stored in the f5 file.
    """
    if results_path not in f5:
        f5.create_group(results_path)
    f5[results_path].attrs["model"] = classifier_config["classifier"]
    f5[results_path].attrs["model_version"] = classifier_config["classifier_version"]
    f5[results_path].attrs["model_file"] = classifier_config["classifier_path"]
    f5[results_path].attrs["classification_threshold"] = classifier_config["min_confidence"]


def write_classifier_result(f5, results_path, read_id, pred_class, prob, passed_classification):
    results_path = f"{results_path}/{read_id}"
    if results_path not in f5:
        f5.create_group(results_path)
    f5[results_path].attrs["best_class"] = pred_class
    f5[results_path].attrs["best_score"] = prob
    f5[results_path].attrs["assigned_class"] = pred_class if passed_classification else -1
