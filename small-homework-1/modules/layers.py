import numpy as np
from typing import List
from .base import Module
# from activations import ReLU, Sigmoid, Softmax, LogSoftmax


class Linear(Module):
    """
    Applies linear (affine) transformation of data: y = x W^T + b
    """
    def __init__(self, in_features: int, out_features: int, bias: bool = True):
        """
        :param in_features: input vector features
        :param out_features: output vector features
        :param bias: whether to use additive bias
        """
        super().__init__()
        self.in_features = in_features      # (in_features,)
        self.out_features = out_features        # (out_features,)
        self.weight = np.random.uniform(-1, 1, (out_features, in_features)) / np.sqrt(in_features)      # (out_features, in_features)
        self.bias = np.random.uniform(-1, 1, out_features) / np.sqrt(in_features) if bias else None     # (out_features,)

        self.grad_weight = np.zeros_like(self.weight)       # (out_features, in_features)
        self.grad_bias = np.zeros_like(self.bias) if bias else None          # (out_features,)

    def compute_output(self, input: np.ndarray) -> np.ndarray:
        """
        :param input: array of shape (batch_size, in_features)
        :return: array of shape (batch_size, out_features)
        """
        output = np.dot(input, self.weight.T)        # (B, in_features) * (in_features, out_features) = (B, out_features)
        if self.bias is not None:
            output += self.bias

        return output

    def compute_grad_input(self, input: np.ndarray, grad_output: np.ndarray) -> np.ndarray:
        """
        :param input: array of shape (batch_size, in_features)
        :param grad_output: array of shape (batch_size, out_features)
        :return: array of shape (batch_size, in_features)
        """
        grad_input = np.dot(grad_output, self.weight)       # (B, out_features) * (out_features, in_features) = (B, in_features)

        return grad_input

    def update_grad_parameters(self, input: np.ndarray, grad_output: np.ndarray):
        """
        :param input: array of shape (batch_size, in_features)
        :param grad_output: array of shape (batch_size, out_features)
        """
        self.grad_weight += np.dot(grad_output.T, input)        # (out_features, in_features) += (B, out_features).T * (B, in_features) 

        if self.bias is not None:
            self.grad_bias += grad_output.sum(axis=0)           # (out_features,) += (B, out_features) -> (out_features,)


    def zero_grad(self):
        self.grad_weight.fill(0)
        if self.bias is not None:
            self.grad_bias.fill(0)

    def parameters(self) -> List[np.ndarray]:
        if self.bias is not None:
            return [self.weight, self.bias]

        return [self.weight]

    def parameters_grad(self) -> List[np.ndarray]:
        if self.bias is not None:
            return [self.grad_weight, self.grad_bias]

        return [self.grad_weight]

    def __repr__(self) -> str:
        out_features, in_features = self.weight.shape
        return f'Linear(in_features={in_features}, out_features={out_features}, ' \
               f'bias={not self.bias is None})'


class BatchNormalization(Module):
    """
    Applies batch normalization transformation
    """
    def __init__(self, num_features: int, eps: float = 1e-5, momentum: float = 0.1, affine: bool = True):
        """
        :param num_features:
        :param eps:
        :param momentum:
        :param affine: whether to use trainable affine parameters
        """
        super().__init__()
        self.eps = eps
        self.momentum = momentum
        self.affine = affine

        self.running_mean = np.zeros(num_features)
        self.running_var = np.ones(num_features)

        self.weight = np.ones(num_features) if affine else None
        self.bias = np.zeros(num_features) if affine else None

        self.grad_weight = np.zeros_like(self.weight) if affine else None
        self.grad_bias = np.zeros_like(self.bias) if affine else None

        # store this values during forward path and re-use during backward pass
        self.mean = None
        self.input_mean = None  # input - mean
        self.var = None
        self.sqrt_var = None
        self.inv_sqrt_var = None
        self.norm_input = None

    def compute_output(self, input: np.ndarray) -> np.ndarray:
        """
        :param input: array of shape (batch_size, num_features)
        :return: array of shape (batch_size, num_features)
        """
        if self.training:       # train

            self.mean = np.mean(input, axis=0)
            self.input_mean = input - self.mean
            self.var = np.mean(np.power(self.input_mean, 2), axis=0)
            self.sqrt_var = np.sqrt(self.var + self.eps)
            self.inv_sqrt_var = 1 / self.sqrt_var

            self.norm_input = self.input_mean * self.inv_sqrt_var

            if self.weight is not None and self.bias is not None:
                output = self.norm_input * self.weight + self.bias
            else:
                output = self.norm_input

            self.running_mean = (1 - self.momentum) * self.running_mean + self.momentum * self.mean
            batch_size = input.shape[0]
            batch_index = batch_size / (batch_size + 1)
            self.running_var = (1 - self.momentum) * self.running_var + batch_index * self.momentum * self.var

        else:        # eval

            self.mean = self.running_mean
            self.input_mean = input - self.mean
            self.var = self.running_var
            self.sqrt_var = np.sqrt(self.var + self.eps)
            self.inv_sqrt_var = 1 / self.sqrt_var

            self.norm_input = self.input_mean * self.inv_sqrt_var

            if self.affine:
                output = self.norm_input * self.weight + self.bias
            else:
                output = self.norm_input

        return output

    def compute_grad_input(self, input: np.ndarray, grad_output: np.ndarray) -> np.ndarray:
        """
        :param input: array of shape (batch_size, num_features)
        :param grad_output: array of shape (batch_size, num_features)
        :return: array of shape (batch_size, num_features)
        """
        if self.training:
            batch_size = input.shape[0]
        
            if self.affine:
                grad_norm_input = grad_output * self.weight
            else:
                grad_norm_input = grad_output

            grad_var = np.sum(grad_norm_input * self.input_mean * -0.5 * np.power(self.var + self.eps, -1.5), axis=0)
            grad_mean = np.sum(grad_norm_input * -1.0 / self.sqrt_var, axis=0) + \
                grad_var * np.sum(-2.0 * self.input_mean, axis=0) / batch_size
            grad_input = grad_norm_input / self.sqrt_var + \
                        grad_var * 2.0 * self.input_mean / batch_size + \
                        grad_mean / batch_size
            
        else:
            grad_input = grad_output * self.inv_sqrt_var
            if self.affine:
                grad_input *= self.weight

        return grad_input

    def update_grad_parameters(self, input: np.ndarray, grad_output: np.ndarray):
        """
        :param input: array of shape (batch_size, num_features)
        :param grad_output: array of shape (batch_size, num_features)
        """
        
        if self.affine:
            self.grad_weight += np.sum(grad_output * self.norm_input, axis=0)
            self.grad_bias += np.sum(grad_output, axis=0)

    def zero_grad(self):
        if self.affine:
            self.grad_weight.fill(0)
            self.grad_bias.fill(0)

    def parameters(self) -> List[np.ndarray]:
        return [self.weight, self.bias] if self.affine else []

    def parameters_grad(self) -> List[np.ndarray]:
        return [self.grad_weight, self.grad_bias] if self.affine else []

    def __repr__(self) -> str:
        return f'BatchNormalization(num_features={len(self.running_mean)}, ' \
               f'eps={self.eps}, momentum={self.momentum}, affine={self.affine})'


class Dropout(Module):
    """
    Applies dropout transformation
    """
    def __init__(self, p=0.5):
        super().__init__()
        assert 0 <= p < 1
        self.p = p
        self.mask = None
        self.dropout_coef = 1 / (1 - self.p)

    def compute_output(self, input: np.ndarray) -> np.ndarray:
        """
        :param input: array of an arbitrary size
        :return: array of the same size
        """
        if self.training:
            size = input.shape
            self.mask = np.random.binomial(1, 1 - self.p, size)
            output = input * self.mask * self.dropout_coef

        else:
            output = input

        return output

    def compute_grad_input(self, input: np.ndarray, grad_output: np.ndarray) -> np.ndarray:
        """
        :param input: array of an arbitrary size
        :param grad_output: array of the same size
        :return: array of the same size
        """
        if self.training:
            grad_input = grad_output * self.mask * self.dropout_coef

        else:
            grad_input = grad_output

        return grad_input

    def __repr__(self) -> str:
        return f'Dropout(p={self.p})'


class Sequential(Module):
    """
    Container for consecutive application of modules
    """
    def __init__(self, *args):
        super().__init__()
        self.modules = list(args)

    def compute_output(self, input: np.ndarray) -> np.ndarray:
        """
        :param input: array of size matching the input size of the first layer
        :return: array of size matching the output size of the last layer
        """
        for module in self.modules:
            # print(module)
            # print(f'INPUT: {input.shape}')

            output = module.compute_output(input)
            # print(f'OUTPUT: {output.shape}')
            module.output = input

            input = output
        return output


    def compute_grad_input(self, input: np.ndarray, grad_output: np.ndarray) -> np.ndarray:     # input = x1     self.output = выход работы сигмоиды
        """
        :param input: array of size matching the input size of the first layer
        :param grad_output: array of size matching the output size of the last layer
        :return: array of size matching the input size of the first layer
        """
        # print('-------------------')

        for module in list(reversed(self.modules)):
            # print(module)
            input = module.output
            # print(f'INPUT: {input.shape}, GRADOUTPUT: {grad_output.shape}')

            grad_input = module.compute_grad_input(input, grad_output)
            # print(f'GRADINPUT: {grad_input.shape}')

            if hasattr(module, 'update_grad_parameters'):
                module.update_grad_parameters(input, grad_output)

            grad_output = grad_input
        return grad_input


    def __getitem__(self, item):
        return self.modules[item]

    def train(self):
        for module in self.modules:
            module.train()

    def eval(self):
        for module in self.modules:
            module.eval()

    def zero_grad(self):
        for module in self.modules:
            module.zero_grad()

    def parameters(self) -> List[np.ndarray]:
        return [parameter for module in self.modules for parameter in module.parameters()]

    def parameters_grad(self) -> List[np.ndarray]:
        return [grad for module in self.modules for grad in module.parameters_grad()]

    def __repr__(self) -> str:
        repr_str = 'Sequential(\n'
        for module in self.modules:
            repr_str += ' ' * 4 + repr(module) + '\n'
        repr_str += ')'
        return repr_str
