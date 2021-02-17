"""
===========
classify.py
===========

This module contains functionality for classifying nanopore captures.

"""
import os
import re
from abc import ABCMeta, abstractmethod
from dataclasses import dataclass
from pathlib import PurePosixPath
from typing import List

import numpy as np
import torch
import torch.nn as nn

from .. import fast5s, logger
from ..fast5s import CaptureFile
from ..logger import Logger, getLogger
from ..signals import FractionalizedSignal, RawSignal
# TODO: Pipe through filtering https://github.com/uwmisl/poretitioner/issues/43 https://github.com/uwmisl/poretitioner/issues/68
from . import NTERs_trained_cnn_05152019 as pretrained_model
from . import filter
from .configuration import ClassifierConfiguration
from .core import PathLikeOrString

use_cuda = False  # True
# TODO : Don't hardcode use of CUDA : https://github.com/uwmisl/poretitioner/issues/41


__all__ = [
    "predict_class",
    "ClassifierDetails",
    "ClassificationResult",
    "ClassifierFile",
    "NULL_CLASSIFICATION_RESULT",
]


@dataclass(frozen=True)
class ClassificationResult:
    """The result of passing the capture data to the classifier.

    Fields
    ----------
    predicted : str
        The class predicted by the classifier.
    probability : int
        The confidence that this prediction is correct.
    assigned_class: str
        TODO: Katie Q: Where does assigned class come from?

    Returns
    -------
    [ClassificationResult]
        ClassificationResult instance.
    """

    predicted: str
    probability: float
    assigned_class: str

    @property
    def passed_classification(self) -> bool:
        """Whether this result passed classification.

        Returns
        -------
        bool
            True if the predicted class matches the assigned one.
        """
        return self.predicted == self.assigned_class

    @property
    def is_null(self) -> bool:
        """Whether this is a null classification (i.e. hasn't been classified, or couldn't be).

        Returns
        -------
        bool
            Whether this is a null classification.
        """
        return NullClassificationResult.is_null(self)


@dataclass(frozen=True)
class NullClassificationResult(ClassificationResult):
    """This represents a capture that hasn't been classified yet.
    Just a basic null class. Never meant to be instantiated, just use
    the `is_null` classmethod.
    """

    predicted: str = "NULL CLASSIFICATION: NO PREDICTION"
    probability: float = 0
    assigned_class: str = "NULL CLASSIFICATION: NO ASSIGNED CLASS"

    @classmethod
    def is_null(cls, other: ClassificationResult):
        result = (
            other.predicted == cls.predicted
            and other.probability == cls.probability
            and other.assigned_class == cls.assigned_class
        )
        return result

    def passed_classification(self):
        return False


# Constant, so we don't have to make thousands of repeats of this object when classifying.
NULL_CLASSIFICATION_RESULT = NullClassificationResult()


@dataclass(frozen=True)
class ClassifierDetails:
    """Details about the classifier that produces `ClassificationResult`.

    Fields
    ----------
    model : str
        The friendly name of the classifier than predicted this result.
    model_version : str
        The version of the model that predicted this result.
    model_file: str
        Location where this model was saved.
    classification_threshold: float
        The confidence threshold.
    """

    model: str
    model_version: str
    model_file: str
    classification_threshold: float


# TODO: Finish writing Classifier plugin architecture: https://github.com/uwmisl/poretitioner/issues/91
class ClassifierPlugin(metaclass=ABCMeta):
    @abstractmethod
    def get_model_name(self) -> str:
        pass

    @abstractmethod
    def get_model_version(self) -> str:
        pass

    @abstractmethod
    def get_model_file(self) -> str:
        pass

    @abstractmethod
    def load(self, filepath: str, can_use_cuda: bool):
        """This method is where you should do any pre processing needed.
        For exammple, loading and configuring a Pytorch model, or a sci-kit learn model.

        Parameters
        ----------
        filepath : str
            [description]

        Raises
        ------
        NotImplementedError
            [description]
        """

    @abstractmethod
    def evaluate(self, capture):
        raise NotImplementedError("Evaluate hasn't been implemented for this classifier.")


class ClassifierFile(CaptureFile):
    def __init__(
        self, capture_filepath: PathLikeOrString, mode: str = "r", logger: Logger = getLogger()
    ):
        super().__init__(capture_filepath, mode, logger=logger)

        f5 = self.f5
        if self.classification_path not in f5:
            f5.create_group(self.classification_path)

    def get_classification_for_read(self, model: str, read_id: str) -> ClassificationResult:
        """Gets the classification result for the read, if it's been classified,
        or NullClassificationResult, if it hasn't.

        Parameters
        ----------
        mode: str
            Identifier for the model that did the classification.
        read_id : str
            Read ID of the read to get the classification for.

        Returns
        -------
        ClassificationResult
            Gets the result of a read's classification if it was already classified, or
            `NULL_CLASSIFICATION_RESULT` if it hasn't been read yet.
        """
        result_path = self._get_results_path_for_model(model, read_id)
        f5 = self.f5
        result = NULL_CLASSIFICATION_RESULT
        if result_path not in f5:
            self.log.info(
                f"Read {read_id} has not been classified yet, or result"
                f"is not stored at {result_path} in file {f5.filename}."
            )
            return result
        predict_class = f5[result_path].attrs["best_class"]
        probability = f5[result_path].attrs["best_score"]
        assigned_class = f5[result_path].attrs["assigned_class"]
        result = ClassificationResult(predict_class, probability, assigned_class)
        return result

    @property
    def models(self) -> List[str]:
        f5 = self.f5
        models = sorted(f5[self.classification_path].keys())
        return models

    @property
    def classification_path(self):
        classification_root = str(PurePosixPath(self.ROOT, "Classification"))
        return classification_root

    def _get_classification_path_for_model(self, model: str) -> str:
        results_path = str(PurePosixPath(self.classification_path, model))
        return results_path

    def _get_results_path_for_model(self, model: str, read_id: str) -> str:
        results_path = str(PurePosixPath(self._get_classification_path_for_model(model), read_id))
        return results_path

    def write_details(self, classifier_details: ClassifierDetails):
        """Write metadata about the classifier that doesn't need to be repeated for
        each read.

        Parameters
        ----------
        classifier_confidence_threshold : dict
            Subset of the configuration parameters that belong to the classifier.
        """
        model = classifier_details.model

        model_path = self._get_classification_path_for_model(model)
        if model_path not in self.f5:
            self.f5.create_group(model_path)
        self.f5[model_path].attrs["model"] = classifier_details.model
        self.f5[model_path].attrs["model_version"] = classifier_details.model_version
        self.f5[model_path].attrs["model_file"] = classifier_details.model_file
        self.f5[model_path].attrs[
            "classification_threshold"
        ] = classifier_details.classification_threshold

    def write_result(self, model: str, read_id: str, result: ClassificationResult):
        results_path = self._get_results_path_for_model(model, read_id)
        if results_path not in self.f5:
            self.f5.create_group(results_path)
        self.f5[results_path].attrs["best_class"] = result.predicted
        self.f5[results_path].attrs["best_score"] = result.probability
        self.f5[results_path].attrs["assigned_class"] = (
            result.assigned_class if result.passed_classification else -1
        )


# TODO: Implement Classification with the new data model: https://github.com/uwmisl/poretitioner/issues/92
def filter_and_classify(
    config, capture_filepaths: List[PathLikeOrString], overwrite=False, filter_name=None
):
    local_logger = logger.getLogger()
    clf_config = config["classify"]
    classifier_name = clf_config["classifier"]
    classification_path = clf_config["classification_path"]

    # Load classifier
    local_logger.info(f"Loading classifier {classifier_name}.")
    assert classifier_name in ["NTER_cnn", "NTER_rf"]
    assert classification_path is not None and len(classification_path) > 0
    classifier = init_classifier(classifier_name, classification_path)

    # Filter (optional) TODO: Restore filtering https://github.com/uwmisl/poretitioner/issues/43 https://github.com/uwmisl/poretitioner/issues/68
    read_path = "/"
    # if filter_name is not None:
    #     local_logger.info("Beginning filtering.")
    #     filter.filter_and_store_result(config, fast5_fnames, filter_name, overwrite=overwrite)
    #     read_path = f"/Filter/{filter_name}/pass"
    # else:
    #     read_path = "/"

    # Classify
    classify_fast5_file(f5, clf_config, classifier, classifier_name, read_path)


# def classify_file(
#     capturef5: ClassifierFile, configuration: ClassifierConfiguration, classifier: Classifier, classifier_run_name, read_path, class_labels=None):
#     for read in capturef5.reads:
#         pass

# TODO: Implement Classification with the new data model: https://github.com/uwmisl/poretitioner/issues/92


def classify_fast5_file(
    capture_filepath: PathLikeOrString,
    clf_config,
    classifier,
    classifier_run_name,
    read_path,
    class_labels=None,
):
    local_logger = logger.getLogger()
    local_logger.debug(f"Beginning classification for file {capture_filepath}.")
    classifier_name = clf_config["classifier"]
    classifier_version = clf_config["version"]
    classifier_location = clf_config["filepath"]
    classify_start = clf_config["start_obs"]  # 100 in NTER paper
    classify_end = clf_config["end_obs"]  # 21000 in NTER paper
    classifier_confidence_threshold = clf_config["min_confidence"]

    configuration = ClassifierConfiguration(
        classifier_name,
        classifier_version,
        classify_start,
        classify_end,
        classifier_confidence_threshold,
    )

    # details = ClassifierDetails(classifier_name, , , )
    # ClassifierFile(filepath, )
    details = None  # ClassifierDetails(classifier_name, )
    assert classify_start >= 0 and classify_end >= 0
    assert classifier_confidence_threshold is None or (0 <= classifier_confidence_threshold <= 1)

    local_logger.debug(
        f"Classification parameters: name: {classifier_name}, "
        f"range of data points: ({classify_start}, {classify_end})"
        f"confidence required to pass: {classifier_confidence_threshold}"
    )

    results_path = f"/Classification/{classifier_run_name}"
    write_classifier_details(f5, clf_config, results_path)

    with ClassifierFile(capture_filepath, "r+") as classifier_f5:

        details = ClassifierDetails(
            classifier_name,
            classifier_version,
            classifier_location,
            classifier_confidence_threshold,
        )
        classifier_f5.write_details(details)

        for read in classifier_f5.reads:
            signal = classifier_f5.get_fractionalized_read(
                read, start=classify_start, end=classify_end
            )
            labels, probability = predict_class(
                classifier_name, classifier, signal, class_labels=class_labels
            )
            if classifier_confidence_threshold is not None:
                passed_classification = probability > classifier_confidence_threshold
            else:
                passed_classification = None
            write_classifier_result()

    # read_h5group_names = f5.get(read_path)
    # for grp in read_h5group_names:
    #     if "read" not in grp:
    #         continue
    #     read_id = re.findall(r"read_(.*)", str(grp))[0]

    #     signal = get_fractional_blockage_for_read(
    #         f5, grp, start=classify_start, end=classify_end
    #     )
    #     y, p = predict_class(classifier_name, classifier, signal, class_labels=class_labels)
    #     if classifier_confidence_threshold is not None:
    #         passed_classification = False if p <= classifier_confidence_threshold else True
    #     else:
    #         passed_classification = None
    #     write_classifier_result(f5, results_path, read_id, y, p, passed_classification)


# TODO: Implement Classification with the new data model: https://github.com/uwmisl/poretitioner/issues/92
# TODO: This classifier initialization should be a special case of a Plugin: https://github.com/uwmisl/poretitioner/issues/91
def init_classifier(classifier_name, classification_path):
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
    classification_path : str
        Location of the pre-trained model file.

    Returns
    -------
    model
        Classification model (type depends on the spceified model).

    Raises
    ------
    ValueError
        Raised if the classifier name is not supported.
    OSError
        Raised if the classifier path does not exist.
    """
    if classifier_name == "NTER_cnn":  # CNN classifier
        if not os.path.exists(classification_path):
            raise OSError(f"Classifier path doesn't exist: {classification_path}")
        nanoporeTER_cnn = pretrained_model.load_cnn(classification_path)
        return nanoporeTER_cnn
    elif classifier_name == "NTER_rf":  # Random forest classifier
        if not os.path.exists(classification_path):
            raise OSError(f"Classifier path doesn't exist: {classification_path}")
        # TODO : Improve model maintainability : https://github.com/uwmisl/poretitioner/issues/38
        # return joblib.load(open(classification_path, "rb"))
        pass
    else:
        raise ValueError(f"Invalid classifier name: {classifier_name}")


# TODO: Implement Classification with the new data model: https://github.com/uwmisl/poretitioner/issues/92
def predict_class(classifier_name, classifier, raw, class_labels=None) -> ClassificationResult:
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
        prob, label = torch.topk(out, 1)
        if use_cuda:
            label = label.cpu().numpy()[0][0]
        else:
            label = label.numpy()[0][0]
        if class_labels is not None:
            label = class_labels[label]
        probability = prob[0][0].data
        # TODO: Implement Classification with the new data model: https://github.com/uwmisl/poretitioner/issues/92
        # TODO: Katie Q: Where does assigned class come from?
        ClassificationResult(label, probability)
        return label, probability
    elif classifier_name == "NTER_rf":
        class_proba = classifier.predict_proba(
            [[np.mean(raw), np.std(raw), np.min(raw), np.max(raw), np.median(raw)]]
        )[0]
        max_proba = np.amax(class_proba)
        label = np.where(class_proba == max_proba)[0][0]
        if class_labels is not None:
            label = class_labels[label]
        return label, class_proba
    else:
        raise NotImplementedError(f"Classifier {classifier_name} not implemented.")


# def get_classification_for_read(f5, read_id, results_path) -> ClassificationResult:
#     local_logger = logger.getLogger()
#     results_path = f"{results_path}/{read_id}"
#     result = NULL_CLASSIFICATION_RESULT
#     if results_path not in f5:
#         local_logger.info(
#             f"Read {read_id} has not been classified yet, or result"
#             f"is not stored at {results_path} in file {f5.filename}."
#         )
#     else:
#         predicted_class = f5[results_path].attrs["best_class"]
#         probability = f5[results_path].attrs["best_score"]
#         assigned_class = f5[results_path].attrs["assigned_class"]
#         result = ClassificationResult(predicted_class, probability, assigned_class)

#     return result


# def write_classifier_details(f5, classifier_confidence_thresholdig: ClassifierConfiguration, results_path):
#     """Write metadata about the classifier that doesn't need to be repeated for
#     each read.

#     Parameters
#     ----------
#     f5 : h5py.File
#         Opened fast5 file in a writable mode.
#     classifier_confidence_threshold : dict
#         Subset of the configuration parameters that belong to the classifier.
#     results_path : str
#         Where the classification results will be stored in the f5 file.
#     """

#     if results_path not in f5:
#         f5.create_group(results_path)
#     f5[results_path].attrs["model"] = classifier_confidence_thresholdig.classifier
#     f5[results_path].attrs["model_version"] = classifier_confidence_thresholdig
#     f5[results_path].attrs["model_file"] = classifier_confidence_thresholdig["classification_path"]
#     f5[results_path].attrs["classification_threshold"] = classifier_confidence_thresholdig["min_confidence"]


# def write_classifier_result(f5, results_path, read_id, predicted_class, prob, passed_classification):
#     results_path = f"{results_path}/{read_id}"
#     if results_path not in f5:
#         f5.create_group(results_path)
#     f5[results_path].attrs["best_class"] = predicted_class
#     f5[results_path].attrs["best_score"] = prob
#     f5[results_path].attrs["assigned_class"] = predicted_class if passed_classification else -1
