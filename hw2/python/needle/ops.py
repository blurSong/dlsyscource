"""Operatpr table."""
# Global operator table.
from numbers import Number
from typing import Optional, List
from .autograd import NDArray
from .autograd import Op, Tensor, Value, TensorOp
from .autograd import TensorTuple, TensorTupleOp
import numpy

# NOTE: we will numpy as the array_api
# to backup our computations, this line will change in later homeworks
import numpy as array_api


class MakeTensorTuple(TensorTupleOp):
    def compute(self, *args) -> tuple:
        return tuple(args)

    def gradient(self, out_grad, node):
        assert isinstance(out_grad, TensorTuple)
        return tuple(*[out_grad[i] for i in range(len(out_grad))])


def make_tuple(*args):
    return MakeTensorTuple()(*args)


class TupleGetItem(TensorOp):
    def __init__(self, index):
        self.index = index

    def __call__(self, a: TensorTuple, fold_const=True) -> Value:
        assert isinstance(a, TensorTuple)
        # constant folding
        if fold_const and isinstance(a.op, MakeTensorTuple):
            return a.inputs[self.index]
        return Tensor.make_from_op(self, [a])

    def compute(self, a):
        return a[self.index]

    def gradient(self, out_grad, node):
        index = self.index
        in_grad = []
        for i, value in enumerate(node.inputs[0]):
            if i != index:
                in_grad.append(zeros_like(value))
            else:
                in_grad.append(out_grad)
        return MakeTensorTuple()(*in_grad)


def tuple_get_item(value, index):
    return TupleGetItem(index)(value)


class FusedAddScalars(TensorTupleOp):
    def __init__(self, c0: float, c1: float):
        self.c0 = c0
        self.c1 = c1

    def compute(self, a):
        return a + self.c0, a + self.c1

    def gradient(self, out_grad, node):
        return out_grad[0] + out_grad[1]


def fused_add_scalars(x, c0, c1):
    return FusedAddScalars(c0, c1)(x)


class EWiseAdd(TensorOp):
    def compute(self, a: NDArray, b: NDArray):
        return a + b

    def gradient(self, out_grad: Tensor, node: Tensor):
        return out_grad, out_grad


def add(a, b):
    return EWiseAdd()(a, b)


class AddScalar(TensorOp):
    def __init__(self, scalar):
        self.scalar = scalar

    def compute(self, a: NDArray):
        return a + self.scalar

    def gradient(self, out_grad: Tensor, node: Tensor):
        return out_grad


def add_scalar(a, scalar):
    return AddScalar(scalar)(a)


class EWiseMul(TensorOp):
    def compute(self, a: NDArray, b: NDArray):
        return a * b

    def gradient(self, out_grad: Tensor, node: Tensor):
        lhs, rhs = node.inputs
        return out_grad * rhs, out_grad * lhs


def multiply(a, b):
    return EWiseMul()(a, b)


class MulScalar(TensorOp):
    def __init__(self, scalar):
        self.scalar = scalar

    def compute(self, a: NDArray):
        return a * self.scalar

    def gradient(self, out_grad: Tensor, node: Tensor):
        return (out_grad * self.scalar,)


def mul_scalar(a, scalar):
    return MulScalar(scalar)(a)


class PowerScalar(TensorOp):
    """Op raise a tensor to an (integer) power."""

    def __init__(self, scalar: int):
        self.scalar = scalar

    def compute(self, a: NDArray) -> NDArray:
        ### BEGIN YOUR SOLUTION
        return a**self.scalar
        ### END YOUR SOLUTION

    def gradient(self, out_grad, node):
        ### BEGIN YOUR SOLUTION
        return out_grad * self.scalar * node.inputs[0] ** (self.scalar - 1)
        ### END YOUR SOLUTION


def power_scalar(a, scalar):
    return PowerScalar(scalar)(a)


class EWiseDiv(TensorOp):
    """Op to element-wise divide two nodes."""

    def compute(self, a, b):
        ### BEGIN YOUR SOLUTION
        return a / b
        ### END YOUR SOLUTION

    def gradient(self, out_grad, node):
        ### BEGIN YOUR SOLUTION
        lhs, rhs = node.inputs
        return out_grad / rhs, -out_grad * lhs / rhs**2
        ### END YOUR SOLUTION


def divide(a, b):
    return EWiseDiv()(a, b)


class DivScalar(TensorOp):
    def __init__(self, scalar):
        self.scalar = scalar

    def compute(self, a):
        ### BEGIN YOUR SOLUTION
        return a / self.scalar
        ### END YOUR SOLUTION

    def gradient(self, out_grad, node):
        ### BEGIN YOUR SOLUTION
        return out_grad / self.scalar
        ### END YOUR SOLUTION


def divide_scalar(a, scalar):
    return DivScalar(scalar)(a)


class Transpose(TensorOp):
    def __init__(self, axes: Optional[tuple] = None):
        self.axes = axes

    def compute(self, a):
        ### BEGIN YOUR SOLUTION
        if not self.axes:
            return a.swapaxes(-1, -2)
        return a.swapaxes(self.axes[0], self.axes[1])
        ### END YOUR SOLUTION

    def gradient(self, out_grad, node):
        ### BEGIN YOUR SOLUTION
        if not self.axes:
            return out_grad.transpose((-1, -2))
        return out_grad.transpose(self.axes)
        ### END YOUR SOLUTION


def transpose(a, axes=None):
    return Transpose(axes)(a)


class Reshape(TensorOp):
    def __init__(self, shape):
        self.shape = shape

    def compute(self, a):
        ### BEGIN YOUR SOLUTION
        return a.reshape(self.shape)
        ### END YOUR SOLUTION

    def gradient(self, out_grad, node):
        ### BEGIN YOUR SOLUTION
        return out_grad.reshape(node.inputs[0].shape)
        ### END YOUR SOLUTION


def reshape(a, shape):
    return Reshape(shape)(a)


class BroadcastTo(TensorOp):
    def __init__(self, shape):
        self.shape = shape

    def compute(self, a):
        # m*i * i*n = m*n
        return array_api.broadcast_to(a, self.shape)

    def gradient(self, out_grad, node):
        ### BEGIN YOUR SOLUTION
        broadcast_dims = []
        x = node.inputs[0]
        in_dim = len(x.shape)
        out_dim = len(out_grad.shape)

        if in_dim == out_dim:
            for i in range(in_dim):
                if x.shape[i] != out_grad.shape[i]:
                    broadcast_dims.append(i)

        else:
            for i in range(-1, -in_dim - 1, -1):
                if x.shape[i] != out_grad.shape[i]:
                    broadcast_dims.append(i)
            for i in range(-in_dim - 1, -out_dim - 1, -1):
                broadcast_dims.append(i)

        return out_grad.sum(axes=tuple(broadcast_dims)).reshape(x.shape)
        ### END YOUR SOLUTION


def broadcast_to(a, shape):
    return BroadcastTo(shape)(a)


class Summation(TensorOp):
    def __init__(self, axes: Optional[tuple] = None):
        self.axes = axes

    def compute(self, a):
        ### BEGIN YOUR SOLUTION
        return a.sum(self.axes)
        ### END YOUR SOLUTION

    def gradient(self, out_grad, node):
        ### BEGIN YOUR SOLUTION
        x = node.inputs[0]
        in_shape = x.numpy().shape

        if self.axes is not None:
            new_shape = list(in_shape)
            if isinstance(self.axes, int):
                self.axes = (self.axes,)
            for a in self.axes:
                new_shape[a] = 1
        else:
            new_shape = array_api.ones(len(in_shape), dtype=int)

        return out_grad.reshape(new_shape).broadcast_to(x.shape)
        ### END YOUR SOLUTION


def summation(a, axes=None):
    return Summation(axes)(a)


class MatMul(TensorOp):
    def compute(self, a, b):
        ### BEGIN YOUR SOLUTION
        return a @ b
        ### END YOUR SOLUTION

    def gradient(self, out_grad, node):
        ### BEGIN YOUR SOLUTION
        lhs_grad, rhs_grad = (
            out_grad @ node.inputs[1].transpose(),
            node.inputs[0].transpose() @ out_grad,
        )
        if rhs_grad.shape != node.inputs[1].shape:
            rhs_grad = rhs_grad.sum(
                axes=tuple(range(len(rhs_grad.shape) - len(node.inputs[1].shape)))
            )
        if lhs_grad.shape != node.inputs[0].shape:
            lhs_grad = lhs_grad.sum(
                axes=tuple(range(len(lhs_grad.shape) - len(node.inputs[0].shape)))
            )

        return lhs_grad, rhs_grad


def matmul(a, b):
    return MatMul()(a, b)


class Negate(TensorOp):
    def compute(self, a):
        ### BEGIN YOUR SOLUTION
        return -a
        ### END YOUR SOLUTION

    def gradient(self, out_grad, node):
        ### BEGIN YOUR SOLUTION
        return -out_grad
        ### END YOUR SOLUTION


def negate(a):
    return Negate()(a)


class Log(TensorOp):
    def compute(self, a):
        ### BEGIN YOUR SOLUTION
        return array_api.log(a)
        ### END YOUR SOLUTION

    def gradient(self, out_grad, node):
        ### BEGIN YOUR SOLUTION
        return out_grad / node.inputs[0]
        ### END YOUR SOLUTION


def log(a):
    return Log()(a)


class Exp(TensorOp):
    def compute(self, a):
        ### BEGIN YOUR SOLUTION
        return array_api.exp(a)
        ### END YOUR SOLUTION

    def gradient(self, out_grad, node):
        ### BEGIN YOUR SOLUTION
        return out_grad * array_api.exp(node.inputs[0].numpy().astype(float))
        ### END YOUR SOLUTION


def exp(a):
    return Exp()(a)


class ReLU(TensorOp):
    def compute(self, a):
        ### BEGIN YOUR SOLUTION
        return array_api.maximum(a, 0)
        ### END YOUR SOLUTION

    def gradient(self, out_grad, node):
        ### BEGIN YOUR SOLUTION
        return out_grad * (node.inputs[0].realize_cached_data() > 0)
        ### END YOUR SOLUTION


def relu(a):
    return ReLU()(a)


class LogSumExp(TensorOp):
    def __init__(self, axes: Optional[tuple] = None):
        self.axes = axes

    def compute(self, Z):
        ### BEGIN YOUR SOLUTION
        Z_max = array_api.max(Z, axis=self.axes, keepdims=False)
        Z_max_broadcast = array_api.max(Z, axis=self.axes, keepdims=True)
        return Z_max + array_api.log(
            array_api.sum(array_api.exp(Z - Z_max_broadcast), self.axes)
        )
        ### END YOUR SOLUTION

    def gradient(self, out_grad, node):
        ### BEGIN YOUR SOLUTION
        # grad = exp(Z - maxZ) / sum(exp(Z-maxZ), axis=self.axes) * summation.grad(exp(Z-maxZ))

        Z = node.inputs[0].cached_data
        Z_max = array_api.amax(Z, axis=self.axes, keepdims=True)
        exp_Z_maxZ = array_api.exp(Z - Z_max)
        sum_exp_Z_maxZ = array_api.sum(exp_Z_maxZ, self.axes)

        log_grad = out_grad.cached_data / sum_exp_Z_maxZ
        shapez = Z.shape
        if self.axes is not None:
            new_shape = list(shapez)
            if isinstance(self.axes, int):
                self.axes = (self.axes,)
            for a in self.axes:
                new_shape[a] = 1
        else:
            new_shape = list(array_api.ones(len(shapez), dtype=int))
        sum_grad = log_grad.reshape(tuple(new_shape))
        exp_grad = exp_Z_maxZ * sum_grad
        return Tensor(exp_grad)

        ### END YOUR SOLUTION


def logsumexp(a, axes=None):
    return LogSumExp(axes=axes)(a)


class Max(TensorOp):
    ## add an aditional class for convenient
    def __init__(self, axes: Optional[tuple] = None):
        self.axes = axes

    def compute(self, a):
        return array_api.max(a, self.axes)

    def gradient(self, out_grad, node):
        Z = node.inputs[0].detach()
        Z_max_grad = array_api.zeros_like(Z)
        Z_max_grad[array_api.argmax(Z, axis=self.axes, keepdims=True)] = 1
        return out_grad * Z_max_grad


def max(a, axes=None):
    return Max(axes=axes)(a)
