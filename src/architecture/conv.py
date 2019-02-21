# https://www.python.org/dev/peps/pep-3107/

import tensorflow as tf
from math import ceil

# type input_shape: (
#   0: X (rows),
#   1: Y (columns),
#   2: depth,
# )


def convolution(
    x: tf.Tensor,
    input_shape: tuple,  # @type input_shape
    output_depth: int = 6,
    patch_shape: tuple = (1, 1),  # (x, y)
    mean: float = 0,
    stddev: float = 0.1,
    stride: tuple = (1, 1),  # (x, y)
) -> (
    tf.Tensor,
    tuple,  # @type input_shape
):
    weights = tf.Variable(tf.truncated_normal(
        shape=(*patch_shape, input_shape[2], output_depth),
        mean=mean,
        stddev=stddev,
    ))
    bias = tf.Variable(tf.zeros(output_depth))

    return (
        tf.nn.conv2d(
            x,
            weights,
            strides=[1, *stride, 1],
            padding="VALID",
        ) + bias,
        # @type input_shape
        (
            ceil(float(input_shape[0] - patch_shape[0] + 1) / stride[0]),
            ceil(float(input_shape[1] - patch_shape[1] + 1) / stride[1]),
            output_depth,
        ),
    )


def max_pool(
    x: tf.Tensor,
    input_shape: tuple,  # @type input_shape
    size: tuple = (2, 2),
    stride: tuple = (2, 2),
) -> (
    tf.Tensor,
    tuple,  # @type input_shape
):
    return (
        tf.nn.max_pool(
            x,
            ksize=[1, *size, 1],
            strides=[1, *stride, 1],
            padding='VALID',
        ),
        (
            ceil(float(input_shape[0] - size[0] + 1) / stride[0]),
            ceil(float(input_shape[1] - size[1] + 1) / stride[1]),
            input_shape[2],  # depth is preserved
        ),
    )
