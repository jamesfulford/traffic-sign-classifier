import tensorflow as tf


def nn(
    x: tf.Tensor,
    input_size: int,
    output_size: int,
    mean: float = 0,
    stddev: float = 0.1,
) -> (
    tf.Tensor,
    int,
):
    weights = tf.Variable(
        tf.truncated_normal(
            shape=(input_size, output_size),
            mean=mean,
            stddev=stddev
        )
    )
    bias = tf.Variable(
        tf.zeros(output_size),
    )
    return (
        tf.matmul(x, weights) + bias,
        output_size,
    )
