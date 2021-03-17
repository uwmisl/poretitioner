import math as m

import torch
import torch.nn as nn
import torch.nn.functional as F

from ..classify import ClassLabel, ClassificationResult, LabelForResult


class CNN(nn.Module):
    def __init__(self):
        self.O_1 = 17
        self.O_2 = 18
        self.O_3 = 32
        self.O_4 = 37

        self.K_1 = 3
        self.K_2 = 1
        self.K_3 = 4
        self.K_4 = 2

        self.KP_1 = 4
        self.KP_2 = 4
        self.KP_3 = 1
        self.KP_4 = 1

        reshape = 141

        self.conv_linear_out = int(
            m.floor(
                (
                    m.floor(
                        (
                            m.floor(
                                (
                                    m.floor(
                                        (
                                            m.floor(
                                                (reshape - self.K_1 + 1) / self.KP_1
                                            )
                                            - self.K_2
                                            + 1
                                        )
                                        / self.KP_2
                                    )
                                    - self.K_3
                                    + 1
                                )
                                / self.KP_3
                            )
                            - self.K_4
                            + 1
                        )
                        / self.KP_4
                    )
                    ** 2
                )
                * self.O_4
            )
        )

        self.FN_1 = 148

        super(CNN, self).__init__()

        self.conv1 = nn.Sequential(
            nn.Conv2d(1, self.O_1, self.K_1), nn.ReLU(), nn.MaxPool2d(self.KP_1)
        )
        self.conv2 = nn.Sequential(
            nn.Conv2d(self.O_1, self.O_2, self.K_2), nn.ReLU(), nn.MaxPool2d(self.KP_2)
        )
        self.conv3 = nn.Sequential(
            nn.Conv2d(self.O_2, self.O_3, self.K_3), nn.ReLU(), nn.MaxPool2d(self.KP_3)
        )
        self.conv4 = nn.Sequential(
            nn.Conv2d(self.O_3, self.O_4, self.K_4), nn.ReLU(), nn.MaxPool2d(self.KP_4)
        )
        self.fc1 = nn.Linear(self.conv_linear_out, self.FN_1, nn.Dropout(0.2))
        self.fc2 = nn.Linear(self.FN_1, 10)

    def forward(self, x):
        x = x.float()
        x = F.leaky_relu(self.conv1(x))
        x = F.leaky_relu(self.conv2(x))
        x = F.leaky_relu(self.conv3(x))
        x = F.leaky_relu(self.conv4(x))
        x = x.view(len(x), -1)
        x = F.logsigmoid(self.fc1(x))
        x = self.fc2(x)
        return x


def load_cnn(state_dict_path, device="cpu"):
    cnn = CNN()
    state_dict = torch.load(state_dict_path, map_location=torch.device(device))
    cnn.load_state_dict(state_dict, strict=True)
    cnn.eval()
    return cnn


class NTER_2018_CNN(PytorchClassifierPlugin):
    def __init__(
        module: nn.Module,
        name: str,
        version: str,
        state_dict_filepath: PathLikeOrString,
        class_label_for_result: Optional[LabelForResult] = None,
        use_cuda: bool = False,
    ):
        super().__init__(module, name, version, state_dict_filepath, use_cuda=use_cuda)
        self.class_label_for_result = class_label_for_result

    def pre_process(self, capture: Capture) -> torch.Tensor:
        frac = capture.fractionalized
        # 2D --> 3D array (each obs in a capture becomes its own array)
        frac_3D = frac.reshape(len(frac), frac.shape[1], 1)

        # Q: Why '19881'?
        #
        # A: We only consider the first 19881 observations, as per the NTER paper [1, 2].
        #
        #    [1] - https://www.biorxiv.org/content/10.1101/837542v1
        #    [2] - https://github.com/uwmisl/NanoporeTERs/search?q=19881

        if frac_3D.shape[1] < 19881:
            temp = np.zeros((frac_3D.shape[0], 19881, 1))
            temp[:, : frac_3D.shape[1], :] = frac_3D
            frac_3D = temp
        frac_3D = frac_3D[:, :19881]  # First 19881 obs as per NTER paper
        # Break capture into 141x141 (19881 total data points)
        frac_3D = frac_3D.reshape(len(frac_3D), 1, 141, 141)
        tensor = torch.from_numpy(frac_3D)
        if use_cuda:
            tensor = tensor.cuda()
        return tensor

    def evaluate(self, capture: Capture) -> ClassificationResult:
        # Pre-process the data, make it cuda-friendly (if applicable) and reshape it for the inference.
        data = self.pre_process(capture)

        classifier = self.module

        # Ensures the model is in inference mode.
        classifier.eval()

        # Run the model.
        outputs = classifier(data)

        out = nn.functional.softmax(outputs, dim=1)
        prob, label = torch.topk(out, 1)
        if not self.use_cuda:
            label = label.cpu().numpy()[0][0]
        else:
            label = label.numpy()[0][0]
        if class_labels is not None:
            label = class_labels[label]
        probability = prob[0][0].data

        ClassificationResult(label, probability)
        return label, probability
