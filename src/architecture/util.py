import functools
import tensorflow as tf


def relu(
    x: tf.Tensor,
    input_shape: tuple,  # @type input_shape
) -> (
    tf.Tensor,
    tuple,  # @type input_shape
):
    return (
        tf.nn.relu(x),
        input_shape,  # shape is preserved
    )


def flat(
    x: tf.Tensor,
    input_shape: tuple,  # @type input_shape
) -> (
    tf.Tensor,
    int,
):
    return (
        tf.contrib.layers.flatten(x),
        functools.reduce(lambda a, b: a * b, input_shape),
    )
