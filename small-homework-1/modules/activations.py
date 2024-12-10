import numpy as np
from .base import Module


class ReLU(Module):
    """
    Applies element-wise ReLU function
    """
    def compute_output(self, input: np.ndarray) -> np.ndarray:
        """
        :param input: array of an arbitrary size
        :return: array of the same size
        """
        output = np.maximum(0, input)

        return output

    def compute_grad_input(self, input: np.ndarray, grad_output: np.ndarray) -> np.ndarray:
        """
        :param input: array of an arbitrary size
        :param grad_output: array of the same size
        :return: array of the same size
        """
        relu_derivative = (input > 0).astype(int)
        grad_input = grad_output * relu_derivative

        return grad_input


class Sigmoid(Module):
    """
    Applies element-wise sigmoid function
    """
    def compute_output(self, input: np.ndarray) -> np.ndarray:
        """
        :param input: array of an arbitrary size
        :return: array of the same size
        """
        output = 1 / (1 + np.exp(-input))

        return output

    def compute_grad_input(self, input: np.ndarray, grad_output: np.ndarray) -> np.ndarray:
        """
        :param input: array of an arbitrary size
        :param grad_output: array of the same size
        :return: array of the same size
        """
        sigmoid_output = self.compute_output(input)
        # sigmoid_output = self.compute_output(self.get_output)
        sigmoid_derivative = sigmoid_output * (1 - sigmoid_output)
        grad_input = grad_output * sigmoid_derivative

        return grad_input


class Softmax(Module):
    """
    Applies Softmax operator over the last dimension
    """
    def compute_output(self, input: np.ndarray) -> np.ndarray:
        """
        :param input: array of size (batch_size, num_classes)
        :return: array of the same size
        """
        input_exp = np.exp(input)
        row_sum = np.sum(input_exp, axis=1)
        row_sum = row_sum[:, np.newaxis]        # (batch_size,) -> (batch_size, 1)
        output = input_exp / row_sum

        return output

    def compute_grad_input(self, input: np.ndarray, grad_output: np.ndarray) -> np.ndarray:
        """
        :param input: array of size (batch_size, num_classes)
        :param grad_output: array of the same size
        :return: array of the same size
        """
        softmax_output = self.compute_output(input)
        identity = np.eye(input.shape[1])
        diag_matrices = np.einsum('ij,jk->ijk', softmax_output, identity)           # diag_matrix for every observation (string) in softmax_output
        outer_products = np.einsum('bi,bj->bij', softmax_output, softmax_output)      # outer product for every observation (string) in softmax_output
        softmax_derivative_3d = diag_matrices - outer_products
        grad_input = np.einsum('bij,bj->bi', softmax_derivative_3d, grad_output)

        return grad_input


class LogSoftmax(Module):
    """
    Applies LogSoftmax operator over the last dimension
    """
    def compute_output(self, input: np.ndarray) -> np.ndarray:
        """
        :param input: array of size (batch_size, num_classes)
        :return: array of the same size
        """
        input_exp = np.exp(input)
        row_sum = np.sum(input_exp, axis=1)
        row_sum = row_sum[:, np.newaxis]        # (batch_size,) -> (batch_size, 1)
        output = input - np.log(row_sum)

        return output

    def compute_grad_input(self, input: np.ndarray, grad_output: np.ndarray) -> np.ndarray:
        """
        :param input: array of size (batch_size, num_classes)
        :param grad_output: array of the same size
        :return: array of the same size
        """
        log_softmax_output = self.compute_output(input)
        softmax_output = np.exp(log_softmax_output)
        identity = np.eye(input.shape[1])
        identity_3d = np.tile(identity, (input.shape[0], 1, 1))
        ones = np.ones((input.shape[1]))
        outer_products = np.einsum('bi,j->bij', softmax_output, ones)
        log_softmax_derivative_3d = identity_3d - outer_products
        grad_input = np.einsum('bij,bj->bi', log_softmax_derivative_3d, grad_output)

        return grad_input
