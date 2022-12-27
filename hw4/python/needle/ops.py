"""Operatpr table."""
# Global operator table.
from numbers import Number
from typing import Optional, List
from .autograd import NDArray
from .autograd import Op, Tensor, Value, TensorOp
from .autograd import TensorTuple, TensorTupleOp
from . import init
import numpy

from .backend_selection import array_api, NDArray

### array_api is ndarray.py


class MakeTensorTuple(TensorTupleOp):
    def compute(self, *args) -> tuple:
        return tuple(args)

    def gradient(self, out_grad, node):
        assert isinstance(out_grad, TensorTuple)
        return tuple([out_grad[i] for i in range(len(out_grad))])


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
                in_grad.append(init.zeros_like(value))
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
        return a + numpy.float32(self.scalar)

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
        return a * numpy.float32(self.scalar)

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
        axes_list = list(range(a.ndim))
        if self.axes is None:
            axes_list[-2:] = axes_list[-1:-3:-1]
        else:
            axes_list[self.axes[0]], axes_list[self.axes[1]] = (
                self.axes[1],
                self.axes[0],
            )
        return a.permute(tuple(axes_list))
        ### END YOUR SOLUTION

    def gradient(self, out_grad, node):
        ### BEGIN YOUR SOLUTION
        axes_list = list(range(out_grad.ndim))
        if self.axes is None:
            axes_list[-2:] = axes_list[-1:-3:-1]
        else:
            axes_list[self.axes[0]], axes_list[self.axes[1]] = (
                self.axes[1],
                self.axes[0],
            )
        return out_grad.permute(tuple(axes_list))
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
        return array_api.broadcast_to(a, self.shape).compact()

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
        # return reshape(summation(out_grad, axes=tuple(axes)), input.shape)
        ### END YOUR SOLUTION


def broadcast_to(a, shape):
    return BroadcastTo(shape)(a)


class Summation(TensorOp):
    def __init__(self, axes: Optional[tuple] = None):
        self.axes = axes

    def compute(self, a):
        ### BEGIN YOUR SOLUTION
        ### I won't realize multi-axes summation
        #   since yout can always call reshape and this without losing perf.
        return a.sum(self.axes)
        ### END YOUR SOLUTION

    def gradient(self, out_grad, node):
        ### BEGIN YOUR SOLUTION
        x = node.inputs[0]
        in_shape = x.shape

        new_shape = list(in_shape)
        if self.axes is not None:
            for a in (self.axes,):
                new_shape[a] = 1
        else:
            new_shape[-1] = 1

        return out_grad.reshape(new_shape).broadcast_to(in_shape)
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
        ### END YOUR SOLUTION


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
        return a.log()
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
        return a.exp()
        ### END YOUR SOLUTION

    def gradient(self, out_grad, node):
        ### BEGIN YOUR SOLUTION
        return out_grad * node.inputs[0].exp()
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
        Z_max = Z.max(axis=self.axes, keepdims=False)
        Z_max_broadcast = Z.max(axis=self.axes, keepdims=True).broadcast_to(Z.shape)
        return Z_max + array_api.log(
            array_api.exp(Z - Z_max_broadcast).sum(axis=self.axes)
        )
        ### END YOUR SOLUTION

    def gradient(self, out_grad, node):
        ### BEGIN YOUR SOLUTION
        # grad = exp(Z - maxZ) / sum(exp(Z-maxZ), axis=self.axes) * summation.grad(exp(Z-maxZ))

        Z = node.inputs[0].cached_data
        Z_max = Z.max(axis=self.axes, keepdims=True)
        exp_Z_maxZ = array_api.exp(Z - Z_max)
        sum_exp_Z_maxZ = exp_Z_maxZ.sum(self.axes)

        log_grad = out_grad.cached_data / sum_exp_Z_maxZ
        shapez = Z.shape
        if self.axes is not None:
            new_shape = list(shapez)
            if isinstance(self.axes, int):
                self.axes = (self.axes,)
            for a in self.axes:
                new_shape[a] = 1
        else:
            new_shape = list(array_api.empty(len(shapez), dtype=int))
        sum_grad = log_grad.reshape(tuple(new_shape))
        exp_grad = exp_Z_maxZ * sum_grad
        return Tensor(exp_grad)

        ### END YOUR SOLUTION


def logsumexp(a, axes=None):
    return LogSumExp(axes=axes)(a)


class Tanh(TensorOp):
    def compute(self, a):
        ### BEGIN YOUR SOLUTION
        return a.tanh()
        ### END YOUR SOLUTION

    def gradient(self, out_grad, node):
        ### BEGIN YOUR SOLUTION
        input = node.inputs[0]
        inter = tanh(input)
        return out_grad * (1 - inter * inter)
        ### END YOUR SOLUTION


def tanh(a):
    return Tanh()(a)


class Stack(TensorOp):
    def __init__(self, axis: int):
        """
        Concatenates a sequence of arrays along a new dimension.
        Parameters:
        axis - dimension to concatenate along
        All arrays need to be of the same size.
        """
        self.axis = axis

    def compute(self, args):
        ### BEGIN YOUR SOLUTION
        array_num = len(args)
        array_shape = list(args[0].shape)
        new_shape = list(args[0].shape)
        new_shape.insert(self.axis, array_num)
        new_array = array_api.empty(
            new_shape, dtype=args[0].dtype, device=args[0].device
        )

        idxes = []
        for i in range(len(array_shape)):
            idxes.append(slice(0, array_shape[i]))
        idxes.insert(self.axis, 0)
        array_shape.insert(self.axis, 1)

        for i in range(array_num):
            idxes[self.axis] = i
            new_array[tuple(idxes)] = array_api.reshape(
                args[i], array_shape
            )  # __setitem__

        return new_array
        ### END YOUR SOLUTION

    def gradient(self, out_grad, node):
        ### BEGIN YOUR SOLUTION
        return split(out_grad, self.axis)
        ### END YOUR SOLUTION


def stack(args, axis):
    return Stack(axis)(make_tuple(*args))


class Split(TensorTupleOp):
    def __init__(self, axis: int):
        """
        Splits a tensor along an axis into a tuple of tensors.
        (The "inverse" of Stack)
        Parameters:
        axis - dimension to split
        """
        self.axis = axis

    def compute(self, A):
        ### BEGIN YOUR SOLUTION
        array_num = A.shape[self.axis]
        array_shape = list(A.shape)
        array_shape.pop(self.axis)
        arg = []

        idxes = []
        for i in range(len(array_shape)):
            idxes.append(slice(0, array_shape[i]))
        idxes.insert(self.axis, 0)

        for i in range(array_num):
            idxes[self.axis] = i
            array = array_api.array(A[tuple(idxes)], dtype=A.dtype, device=A.device)
            array = array_api.reshape(array, array_shape)  # __getitem__
            arg.append(array)

        return arg
        ### END YOUR SOLUTION

    def gradient(self, out_grad, node):
        ### BEGIN YOUR SOLUTION
        return stack(out_grad, self.axis)
        ### END YOUR SOLUTION


def split(a, axis):
    return Split(axis)(a)


class Flip(TensorOp):
    def __init__(self, axes: Optional[tuple] = None):
        self.axes = axes

    def compute(self, a):
        ### BEGIN YOUR SOLUTION
        return array_api.flip(a, self.axes)
        ### END YOUR SOLUTION

    def gradient(self, out_grad, node):
        ### BEGIN YOUR SOLUTION
        return flip(out_grad, self.axes)
        ### END YOUR SOLUTION


def flip(a, axes):
    return Flip(axes)(a)


class Dilate(TensorOp):
    def __init__(self, axes: tuple, dilation: int):
        self.axes = axes
        self.dilation = dilation

    def compute(self, a):
        ### BEGIN YOUR SOLUTION
        old_shape = list(a.shape)
        new_shape = []
        n = len(old_shape)
        index = []
        for i in range(n):
            if i not in self.axes:
                new_shape.append(old_shape[i])
                index.append(slice(new_shape[-1]))
            else:
                new_shape.append(old_shape[i] * (1 + self.dilation))
                index.append(slice(0, new_shape[-1], 1 + self.dilation))

        res = array_api.full(new_shape, 0, dtype=a.dtype, device=a.device)
        res[tuple(index)] = a

        return res
        ### END YOUR SOLUTION

    def gradient(self, out_grad, node):
        ### BEGIN YOUR SOLUTION
        return undilate(out_grad, self.axes, self.dilation)
        ### END YOUR SOLUTION


def dilate(a, axes, dilation):
    return Dilate(axes, dilation)(a)


class UnDilate(TensorOp):
    def __init__(self, axes: tuple, dilation: int):
        self.axes = axes
        self.dilation = dilation

    def compute(self, a):
        ### BEGIN YOUR SOLUTION
        old_shape = list(a.shape)
        new_shape = []
        n = len(old_shape)
        index = []
        for i in range(n):
            if i not in self.axes:
                new_shape.append(old_shape[i])
                index.append(slice(new_shape[-1]))
            else:
                new_shape.append(old_shape[i] // (1 + self.dilation))
                index.append(slice(0, old_shape[i], 1 + self.dilation))

        res = array_api.full(new_shape, 0, dtype=a.dtype, device=a.device)
        res = a[tuple(index)]

        return res  ### END YOUR SOLUTION

    def gradient(self, out_grad, node):
        ### BEGIN YOUR SOLUTION
        return dilate(out_grad, self.axes, self.dilation)
        ### END YOUR SOLUTION


def undilate(a, axes, dilation):
    return UnDilate(axes, dilation)(a)


class Conv(TensorOp):
    def __init__(self, stride: Optional[int] = 1, padding: Optional[int] = 0):
        self.stride = stride
        self.padding = padding

    def compute(self, A, B):
        ### BEGIN YOUR SOLUTION
        ### im2col
        N, H, W, CIN = A.shape
        K, _, _, COUT = B.shape
        Ns, Hs, Ws, Cs = A.strides
        inner_dim = K * K * CIN

        if self.padding > 0:
            A = A.pad(
                (
                    (0, 0),
                    (self.padding, self.padding),
                    (self.padding, self.padding),
                    (0, 0),
                ),
            )

        HOUT = (H - K + 2 * self.padding) // self.stride + 1
        WOUT = (W - K + 2 * self.padding) // self.stride + 1
        new_shape = (
            N,
            HOUT,
            WOUT,
            K,
            K,
            CIN,
        )
        new_strides = (Ns, self.stride * Hs, self.stride * Ws, Hs, Ws, Cs)
        A_ = (
            A.as_strided(shape=new_shape, strides=new_strides)
            .compact()
            .reshape((-1, inner_dim))
        )
        B_ = B.reshape((-1, COUT))
        OUT = A_ @ B_
        return OUT.reshape((N, HOUT, WOUT, COUT))
        ### END YOUR SOLUTION

    def gradient(self, out_grad, node):
        ### BEGIN YOUR SOLUTION
        raise NotImplementedError()
        ### END YOUR SOLUTION


def conv(a, b, stride=1, padding=1):
    return Conv(stride, padding)(a, b)
