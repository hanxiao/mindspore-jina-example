from jina.executors.encoders.frameworks import BaseMindsporeEncoder
from mindspore import Tensor
import numpy as np

from .lenet.src.lenet import LeNet5


class LeNet5Feat(LeNet5):
    def construct(self, x):
        x = self.conv1(x)
        x = self.relu(x)
        x = self.max_pool2d(x)
        x = self.conv2(x)
        x = self.relu(x)
        x = self.max_pool2d(x)
        x = self.flatten(x)
        x = self.fc1(x)
        x = self.relu(x)
        x = self.fc2(x)
        x = self.relu(x)
        return x


class MindsporeLeNet(BaseMindsporeEncoder):
    """
    :class:`MindsporeLeNet` Encoding image into vectors using mindspore.
    """

    def encode(self, data, *args, **kwargs):
        # data is B x D, where D = 28 * 28
        # LeNet only accepts BCHW format where H=W=32
        # hence we need to do some simple transform
        data = np.pad(data.reshape([-1, 1, 28, 28]),
                      [(0, 0), (0, 0), (0, 4), (0, 4)]).astype('float32')
        return self.model(Tensor(data)).asnumpy()

    def get_model(self):
        return LeNet5Feat()
