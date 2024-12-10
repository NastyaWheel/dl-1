import numpy as np
from .base import Criterion
from .activations import LogSoftmax


class MSELoss(Criterion):
    """
    Mean squared error criterion
    """
    def compute_output(self, input: np.ndarray, target: np.ndarray) -> float:
        """
        :param input: array of size (batch_size, *)
        :param target:  array of size (batch_size, *)
        :return: loss value
        """
        assert input.shape == target.shape, 'input and target shapes not matching'

        difference = input - target     # input - target, (batch_size, n_features)
        difference_squared = np.power(difference, 2)        # (input - target) ^ 2, (batch_size, n_features)

        se_batch = difference_squared.sum(axis=1)      # error per observation, (batch_size,)

        batch_size, n_features = input.shape
        averaging_coefficient = 1 / (batch_size * n_features)
        mse = averaging_coefficient * se_batch.sum()       # error per batch, value

        return mse
    

    def compute_grad_input(self, input: np.ndarray, target: np.ndarray) -> np.ndarray:
        """
        :param input: array of size (batch_size, *)
        :param target:  array of size (batch_size, *)
        :return: array of size (batch_size, *)
        """
        assert input.shape == target.shape, 'input and target shapes not matching'
        difference = input - target     # input - target, (batch_size, n_features)

        batch_size, n_features = input.shape
        coefficient = 2 / (batch_size * n_features)
        mse_derivative = coefficient * difference

        return mse_derivative


class CrossEntropyLoss(Criterion):
    """
    Cross-entropy criterion over distribution logits
    """
    def __init__(self):
        super().__init__()
        self.log_softmax = LogSoftmax()

    def compute_output(self, input: np.ndarray, target: np.ndarray) -> float:
        """
        :param input: logits array of size (batch_size, num_classes)
        :param target: labels array of size (batch_size, )
        :return: loss value
        """
        log_probs = self.log_softmax.compute_output(input)      # (batch_size, num_classes)
        batch_size = input.shape[0]
        log_probs_true = log_probs[np.arange(batch_size), target]       # (batch_size, )

        averaging_coefficient = 1 / batch_size
        cross_entropy = -averaging_coefficient * log_probs_true.sum()

        return cross_entropy

    def compute_grad_input(self, input: np.ndarray, target: np.ndarray) -> np.ndarray:
        """
        :param input: logits array of size (batch_size, num_classes)
        :param target: labels array of size (batch_size, )
        :return: array of size (batch_size, num_classes)
        """
        log_probs = self.log_softmax.compute_output(input)
        probs = np.exp(log_probs)

        batch_size = input.shape[0]
        one_hot = np.zeros_like(probs)      # (batch_size, num_classes)
        one_hot[np.arange(batch_size), target] = 1

        grad_input = (probs - one_hot) / batch_size

        return grad_input
