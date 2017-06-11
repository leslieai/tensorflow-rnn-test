import tensorflow as tf
import numpy as np

batch_size=  64
input_dim=128
num_units=256

X=tf.constant(1.,shape=[batch_size,input_dim]) # [batch ,input size]

X_lengths=tf.constant(1.,shape=[batch_size])
print(X_lengths.shape)

# static_rnn work way like this :
# state = cell.zero_state(...)
#     outputs = []
#     for input_ in inputs:
#       output, state = cell(input_, state)
#       outputs.append(output)
#     return (outputs, state)

# Forward direction cell
lstm_fw_cell = tf.contrib.rnn.BasicLSTMCell(num_units, state_is_tuple=True, forget_bias=1.0)
# Backward direction cell
lstm_bw_cell = tf.contrib.rnn.BasicLSTMCell(num_units, state_is_tuple=True, forget_bias=1.0)

# cell.zero_state(batch_size, dtype=tf.float32)  #batch size

outputs, output_state_fw,output_state_bw = tf.contrib.rnn.static_bidirectional_rnn(
    lstm_fw_cell,
    lstm_bw_cell,
    [X,X,X,X],
    # sequence_length=X_lengths,
    dtype=tf.float32)  # three rnn cell


result = tf.contrib.learn.run_n(
    {"outputs": outputs, "output_state_fw": output_state_fw,'output_state_bw':output_state_bw},
    n=1,
    feed_dict=None)
# output_dict: A dict mapping string names to tensors to run. Must all be from the same graph.
# feed_dict: dict of input values to feed each run.
# restore_checkpoint_path: A string containing the path to a checkpoint to restore.
# n: Number of times to repeat.

# print(result)
# print(cell.output_size)
# print(cell.state_size)
print(np.shape(result[0]["outputs"]))
print(np.shape(result[0]["output_state_fw"]))
print(np.shape(result[0]["output_state_bw"]))

