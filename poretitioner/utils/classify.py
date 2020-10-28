"""
===========
classify.py
===========

This module contains functionality for classifying nanopore captures.

"""
import logging
import os

import h5py
import numpy as np
import torch
import torch.nn as nn

from . import filter, raw_signal_utils
from .NTERs_trained_cnn_05152019 import load_cnn

use_cuda = True
# TODO : Don't hardcode use of CUDA : https://github.com/uwmisl/poretitioner/issues/41


def classify(config):
    logger = logging.getLogger("classify")
    if logger.handlers:
        logger.handlers = []
    logger.setLevel(logging.DEBUG)
    logger.addHandler(logging.StreamHandler())

    classifier_name = config["classify"]["classifier"]
    filter_name = config["filter"]["filter_name"]  # TODO
    filter_path = f"/Filter/{filter_name}"
    classifier_conf = config["classify"]["min_confidence"]
    classifier_path = config["classify"]["classifier_path"]
    fast5_location = config["output"]["capture_f5_dir"]
    classify_start = config["classify"]["start_obs"]  # 100 in NTER paper
    classify_end = config["classify"]["end_obs"]  # 2100 in NTER paper

    # Load classifier
    assert classifier_name in ["NTER_cnn", "NTER_rf"]
    assert classifier_path is not None and len(classifier_path) > 0
    classifier = init_classifier(classifier_name, classifier_path)

    # Find files to classify
    # TODO this should be handled elsewhere. OK for this to be default behavior
    # for the pipeline overall, but just pass in a list.
    fast5_fnames = [x for x in os.listdir(fast5_location) if x.endswith("fast5")]

    # Filter
    # TODO : implement
    filter.filter_and_store_result(config, fast5_fnames, overwrite=False)

    # Classify
    for fast5_fname in fast5_fnames:
        with h5py.File(fast5_fname, "r") as f5:
            if filter:
                read_h5group_names = f5.get(filter_path)
                # use filter_name (maybe filter_path instead?)
            else:
                read_h5group_names = f5.get("/")
            for grp in read_h5group_names:
                signal = raw_signal_utils.get_fractional_blockage_for_read(
                    f5, grp, start=classify_start, end=classify_end
                )
                y, p = predict_class(classifier, signal, classifier_conf, classifier_name)
                passed_classification = False if p <= classifier_conf else True
                # This will currently write to within the filtered reads file, is that what we want?
                write_classifier_result(f5, config, grp, y, p, passed_classification)


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
        nanoporeTER_cnn = load_cnn(classifier_path)
        nanoporeTER_cnn.eval()
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
        X_test = X_test[:, :19881]  # First 19881 obs as per NTER paper
        # Break capture into 141x141 (19881 total data points)
        X_test = X_test.reshape(len(X_test), 1, 141, 141)
        X_test = torch.from_numpy(X_test)
        X_test = X_test.cuda()
        outputs = classifier(X_test)
        out = nn.functional.softmax(outputs)
        prob, lab = torch.topk(out, 1)
        lab = lab.cpu().numpy()[0][0]
        if class_labels is not None:
            lab = class_labels[lab]
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


def write_classifier_result(f5, config, read_path, pred_class, prob, passed_classification):
    classifier_run_name = config["classify"]["name for this classification result"]
    results_path = f"{read_path}/Classification/{classifier_run_name}"
    if results_path not in f5:
        f5.create_group(results_path)
    f5[results_path].attrs["model"] = config["classify"]["classifier"]
    f5[results_path].attrs["model_version"] = config["classify"]["classifier version"]
    f5[results_path].attrs["model_file"] = config["classify"]["classifier_path"]
    f5[results_path].attrs["classification_threshold"] = config["classify"]["min_confidence"]
    f5[results_path].attrs["best_class"] = pred_class
    f5[results_path].attrs["best_score"] = prob
    f5[results_path].attrs["assigned_class"] = prob if passed_classification else -1
