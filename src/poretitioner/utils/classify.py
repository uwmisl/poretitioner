"""
===========
classify.py
===========

This module contains functionality for classifying nanopore captures.

"""
import os
import re
from abc import ABC, abstractmethod
from dataclasses import dataclass
from pathlib import PosixPath
from typing import *  # I know people don't like import *, but I think it has benefits for types (doesn't impede people from being generous with typing)

import numpy as np
import torch
import torch.nn as nn

from ..logger import Logger, getLogger
from ..signals import Capture, FractionalizedSignal, RawSignal
# TODO: Pipe through filtering https://github.com/uwmisl/poretitioner/issues/43 https://github.com/uwmisl/poretitioner/issues/68
from .models import NTERs_trained_cnn_05152019 as pretrained_model
from . import filtering
from .configuration import ClassifierConfiguration
from .core import NumpyArrayLike, PathLikeOrString, ReadId

use_cuda = False  # True
# TODO : Don't hardcode use of CUDA : https://github.com/uwmisl/poretitioner/issues/41

ClassLabel = NewType("ClassLabel", str)

# Maps a numpy array like (some vector encoding that represents a label) to a the label string.
LabelForResult = Callable[[NumpyArrayLike], ClassLabel]

__all__ = [
    "predict_class",
    "ClassificationRunId",
    "ClassifierDetails",
    "ClassifierPlugin",
    "CLASSIFICATION_PATH",
    "ClassificationResult",
    "PytorchClassifierPlugin",
]

# Uniquely identifies a classification run that happened (e.g. 'NTER_2018_RandomForest_Attempt_3').
ClassificationRunId = NewType("ClassificationRunId", str)


@dataclass(frozen=True)
class ClassifierDetails:

    model: str
    model_version: str
    classification_threshold: float

    # Timestamp of when this classification occurred, in seconds from epoch (as a float).
    #
    # Q: Why not date-time?
    #
    # A: Sadly, as of 2020, h5py doesn't provide a good way of storing dates [1].
    #    Doing so would also be less precise than storing epoch time.
    #
    # Q: Why seconds (knowing it will be fractionalized)?
    #
    # A: On most modern machines, python time.time() provides micro-second precision.
    #    But this can't be guaranteed (on older machines, it might only provide second precision) [1].
    #
    # If we really wanted an int, the alternative to guarantee an int would be to store
    # the timestamp in nanoseconds [3], but that feels verbose to me.
    #
    # [1] - https://stackoverflow.com/questions/23570632/store-datetimes-in-hdf5-with-h5py
    # [2] - https://docs.python.org/3/library/time.html#time.time
    # [3] - https://docs.python.org/3/library/time.html#time.time_ns
    timestamp_ms: float
    model_file: PathLikeOrString


@dataclass(frozen=True)
class CLASSIFICATION_PATH:
    ROOT = f"/Classification/"

    @classmethod
    def for_classification_run(cls, classification_run: ClassificationRunId) -> str:
        path = str(PosixPath(CLASSIFICATION_PATH.ROOT, classification_run))
        return path

    @classmethod
    def pass_path(cls, classification_run: ClassificationRunId) -> str:
        """Path to the group that contains the readIds that passed classification during this
        classification run.

        Parameters
        ----------
        classification_run : ClassificationRunId
            A unique identifier for the classification run that generated these results (e.g. "my_classication_run_04").

        Returns
        -------
        str
            Pathlike to path. (e.g. /Classifcation/my_classication_run_04/pass)
        """
        CLASSICATION_RUN_PATH = cls.for_classification_run(classification_run)
        path = str(PosixPath(CLASSICATION_RUN_PATH, "pass"))
        return path

    @classmethod
    def fail_path(cls, classification_run: ClassificationRunId) -> str:
        """Path to the group that contains the readIds that failed classification during this
        classification run.

        Parameters
        ----------
        classification_run : ClassificationRunId
            A unique identifier for the classification run that generated these results (e.g. "my_classication_run_04").

        Returns
        -------
        str
            Pathlike to path. (e.g. /Classifcation/my_classication_run_04/fail)
        """
        CLASSICATION_RUN_PATH = cls.for_classification_run(classification_run)
        path = str(PosixPath(CLASSICATION_RUN_PATH, "fail"))
        return path

    def read_id_path(cls, classification_run: ClassificationRunId, read_id: ReadId) -> str:
        """Path to the group that contains the classification results for a given readId.

        Parameters
        ----------
        classification_run : ClassificationRunId
            A unique identifier for the classification run that generated these results (e.g. "my_classication_run_04").

        read_id : ReadId
            The readId of the read we want to know the classification results for.

        Returns
        -------
        str
            Path to the group that contains the classification results for a given readId.
        """
        CLASSICATION_RUN_PATH = cls.for_classification_run(classification_run)
        path = str(PosixPath(CLASSICATION_RUN_PATH, f"{read_id}"))
        return path


@dataclass(frozen=True)
class ClassificationResult:
    """The result of passing the capture data to the classifier.

    Fields
    ----------
    score : float
        A value representing the 'score' of a label predicted by the classifier.
        Abstractly, the score is a measure of confidence that this label is correct, as determined by the score being greater than some threshold.

        What exact values this score can take on depends on your classifier
        (e.g. if you pass the final result through a soft-max, this score will represent a probability from 0 to 1.0).

    label : ClassLabel
        The label assigned to this prediction.

    Returns
    -------
    [ClassificationResult]
        ClassificationResult instance.
    """

    label: ClassLabel
    score: float


# TODO: Finish writing Classifier plugin architecture: https://github.com/uwmisl/poretitioner/issues/91
class ClassifierPlugin(ABC):
    @abstractmethod
    def model_name(self) -> str:
        raise NotImplementedError("model_name hasn't been implemented for this classifier.")

    @abstractmethod
    def model_version(self) -> str:
        raise NotImplementedError("model_version hasn't been implemented for this classifier.")

    @abstractmethod
    def model_file(self) -> str:
        raise NotImplementedError("model_file hasn't been implemented for this classifier.")

    @abstractmethod
    def load(self, use_cuda: bool = False):
        """Loads a model for classification.

        This method is where you should do any pre processing needed.
        For exammple, loading and configuring a Pytorch model, or a sci-kit learn model.

        Parameters
        ----------

        use_cuda : bool
            Whether to use cuda.

        Raises
        ------
        NotImplementedError
            If this method hasn't been implemented.
        """
        raise NotImplementedError("load hasn't been implemented for this classifier.")

    @abstractmethod
    def evaluate(self, capture) -> ClassificationResult:
        raise NotImplementedError("Evaluate hasn't been implemented for this classifier.")


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

        details = Classifie rDetails(
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


def write_classifier_details(
    f5, classifier_confidence_thresholdig: ClassifierConfiguration, results_path
):
    """Write metadata about the classifier that doesn't need to be repeated for
    each read.

    Parameters
    ----------
    f5 : h5py.File
        Opened fast5 file in a writable mode.
    classifier_confidence_threshold : dict
        Subset of the configuration parameters that belong to the classifier.
    results_path : str
        Where the classification results will be stored in the f5 file.
    """

    if results_path not in f5:
        f5.require_group(results_path)
    f5[results_path].attrs["model"] = classifier_confidence_thresholdig.classifier
    f5[results_path].attrs["model_version"] = classifier_confidence_thresholdig
    f5[results_path].attrs["model_file"] = classifier_confidence_thresholdig["classification_path"]
    f5[results_path].attrs["classification_threshold"] = classifier_confidence_thresholdig[
        "min_confidence"
    ]


def write_classifier_result(
    f5, results_path, read_id, predicted_class, prob, passed_classification
):
    results_path = f"{results_path}/{read_id}"
    if results_path not in f5:
        f5.require_group(results_path)
    f5[results_path].attrs["best_class"] = predicted_class
    f5[results_path].attrs["best_score"] = prob
    f5[results_path].attrs["assigned_class"] = predicted_class if passed_classification else -1


class PytorchClassifierPlugin(ClassifierPlugin):
    def __init__(
        self,
        module: nn.Module,
        name: str,
        version: str,
        #class_labels
        state_dict_filepath: PathLikeOrString,
        use_cuda: bool = False,
    ):
        """An abstract class for classifier that are built from PyTorch.

        Subclass this and implement `evaluate`

        Optionally, if you'd like to do some special pre-processing on the data or load the PyTorch module in a specific way
        do so by writing `pre_process` and `load` functions as well, and call them before evaluating the module in `evalaute`.

        For an example of this in action, see the `models/NTERs_trained_cnn_05152019.py` module.

        Parameters
        ----------
        module : nn.Module
            The PyTorch module to use as a classifier. This can be either the instantiated module, or the class itself. If the class is passed, a bare module will be instantiated from it.
        name : str
            Uniquely identifying name for this module.
        version : str
            Version of the model. Useful for keeping track of differently learned parameters.
        state_dict_filepath : PathLikeOrString
            Path to the state_dict describing the module's parameters.
            For more on PyTorch state_dicts, see https://pytorch.org/tutorials/recipes/recipes/what_is_state_dict.html.
        use_cuda : bool, optional
            Whether to use CUDA for GPU processing, by default False
        """
        super().__init__()
        self.module = (
            module if isinstance(module, nn.Module) else module()
        )  # Instantiate the module, if the user passed in the class rather than an instance.
        self.name = name
        self.version = version

        #self.class_labels = class_labels
        self.state_dict_filepath = state_dict_filepath
        self.use_cuda = use_cuda

    def load(self, use_cuda: bool = False):
        """Loads the PyTorch module. This means instantiating it,
        setting its state dict [1], and setting it to evaluation mode (so we perform an inference)[2].

        [1] - https://pytorch.org/tutorials/recipes/recipes/what_is_state_dict.html
        [2] - https://pytorch.org/docs/stable/generated/torch.nn.Module.html#torch.nn.Module.eval

        Parameters
        ----------
        filepath : str
            Filepath to a PyTorch state dict.
        use_cuda : bool, optional
            Whether to use CUDA, by default False
        """
        torch_module = self.module
        # For more on Torch devices, please see:
        #   https://pytorch.org/docs/stable/tensor_attributes.html#torch-device
        device = "cpu" if not use_cuda else "cuda"

        state_dict = torch.load(str(state_dict_filepath), map_location=torch.device(device))
        torch_module.load_state_dict(state_dict, strict=True)

        # Sets the model to inference mode
        torch_module.eval()

        # Ensures subsequent uses of self.module are correctly configured.
        self.module = torch_module

    def pre_process(self, capture: Capture) -> torch.Tensor:
        """Do some pre-processing on the data, if desired.

        Otherwise, this method just converts the fractionalized
        capture to a torch Tensor.

        Parameters
        ----------
        capture : Capture
            Capture we want to classify.

        Returns
        -------
        torch.Tensor
            A tensor that resulted from pre-processing the capture data.
        """
        tensor = torch.from_numpy(capture.fractionalized())
        if self.use_cuda:
            tensor = tensor.cuda()
        return tensor

    @abstractmethod
    def evaluate(self, capture: Capture):
        raise NotImplementedError("Evaluate hasn't been implemented for this classifier.")

    def model_name(self) -> str:
        return self.name

    def model_version(self) -> str:
        return self.version

    def model_file(self) -> str:
        return self.state_dict_filepath
