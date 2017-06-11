import tensorflow as tf
import numpy as np

batch_size=  64
input_dim=128
num_units=256

X=tf.constant(1.,shape=[batch_size,input_dim]) # [batch ,input size]
# X_lengths=tf.constant(1.,shape=[batch_size])
# print(X_lengths.shape)
# static_rnn work way like this :
# state = cell.zero_state(...)
#     outputs = []
#     for input_ in inputs:
#       output, state = cell(input_, state)
#       outputs.append(output)
#     return (outputs, state)

cell = tf.contrib.rnn.LSTMCell(num_units=num_units, state_is_tuple=True)
cell.zero_state(batch_size, dtype=tf.float32)  #batch size
encoder_cell = tf.contrib.rnn.EmbeddingWrapper(
    cell,
        embedding_classes=400,
        embedding_size=128)
outputs, last_states = tf.contrib.rnn.static_rnn(
    cell,
    [X,X,X,X],
    # sequence_length=X_lengths ,
    dtype=tf.float32)  # three rnn cell


result = tf.contrib.learn.run_n(
    {"outputs": outputs, "last_states": last_states},
    n=1,
    feed_dict=None)
# output_dict: A dict mapping string names to tensors to run. Must all be from the same graph.
# feed_dict: dict of input values to feed each run.
# restore_checkpoint_path: A string containing the path to a checkpoint to restore.
# n: Number of times to repeat.

# print(result)
print(cell.output_size)
print(cell.state_size)
print(np.shape(result[0]["outputs"]))
print(np.shape(result[0]["last_states"]))  # the `c_state` and `m_state`

