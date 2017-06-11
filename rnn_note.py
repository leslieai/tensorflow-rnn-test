
# /Users/l/Downloads/py_github/tensorflow/tensorflow/python/ops/nn.py
# tf.nn.dynamic_rnn
# tf.nn.bidirectional_dynamic_rnn
# tf.nn.raw_rnn
# /Users/l/Downloads/py_github/tensorflow/tensorflow/python/ops/rnn.py


# /Users/l/Downloads/py_github/tensorflow/tensorflow/contrib/rnn/__init__.py


# /Users/l/Downloads/py_github/tensorflow/tensorflow/contrib/legacy_seq2seq/python/ops/seq2seq.py


# /Users/l/Downloads/py_github/tensorflow/tensorflow/examples/udacity/3_regularization.ipynb
# /Users/l/Downloads/py_github/tensorflow/tensorflow/examples/udacity/6_lstm.ipynb
# /Users/l/Downloads/py_github/models/tutorials/rnn/translate/seq2seq_model.py
# /Users/l/Downloads/py_github/tensorflow/tensorflow/examples/learn/text_classification.py
# /Users/l/Downloads/py_github/models/textsum/seq2seq_attention_model.py
# /Users/l/Downloads/tf-stanford-tutorials-master/examples/11_char_rnn_gist.py
# /Users/l/Downloads/py_github/models/skip_thoughts/skip_thoughts/skip_thoughts_model.py
# /Users/l/Downloads/py_github/models/im2txt/im2txt/show_and_tell_model.py
# /Users/l/Downloads/py_github/tensorflow/tensorflow/docs_src/api_guides/python/contrib.learn.md

# /Users/l/Downloads/py_github/tensorflow/tensorflow/python/summary/summary.py
# /Users/l/Downloads/py_github/tensorflow/tensorflow/python/training/training.py





# /Users/l/Downloads/py_github/tensorflow/tensorflow/contrib/rnn/python/ops/core_rnn_cell.py
# @@EmbeddingWrapper
# @@InputProjectionWrapper
# @@OutputProjectionWrapper
#
# /Users/l/Downloads/py_github/tensorflow/tensorflow/contrib/rnn/python/ops/rnn.py
# stack_bidirectional_rnn
# """Creates a bidirectional recurrent neural network.
# stack_bidirectional_dynamic_rnn
#   """Creates a dynamic bidirectional recurrent neural network.
#
# /Users/l/Downloads/py_github/tensorflow/tensorflow/python/ops/rnn.py
# @@bidirectional_dynamic_rnn
# @@dynamic_rnn
# @@raw_rnn
# @@static_rnn
# @@static_state_saving_rnn
# @@static_bidirectional_rnn
# tf.nn.bidirectional_dynamic_rnn
# tf.nn.dynamic_rnn
# tf.contrib.rnn.static_bidirectional_rnn
#

# /Users/l/Downloads/py_github/tensorflow/tensorflow/contrib/rnn/python/ops/rnn_cell.py

# /Users/l/Downloads/py_github/tensorflow/tensorflow/python/ops/rnn_cell_impl.py
# ## Base interface for all RNN Cells
# @@RNNCell
#
# ## RNN Cells for use with TensorFlow's core RNN methods
# @@BasicRNNCell
# @@BasicLSTMCell
# @@GRUCell
# @@LSTMCell
# tf.contrib.rnn.LSTMCell
#
# ## Classes storing split `RNNCell` state
# @@LSTMStateTuple
#
# ## RNN Cell wrappers (RNNCells that wrap other RNNCells)
# @@MultiRNNCell
# @@DropoutWrapper
# @@DeviceWrapper
# @@ResidualWrapper
#
#
#   ```LSTM
# h_prev # previous hidden state
# x # input
# i # input gate
# f # forget gate
# ci #new candidate values
# o # output gate
#
#   xh = [x, h_prev]
#   [i, f, ci, o] = xh * w + b
#   f = f + forget_bias
#
#   if not use_peephole:
#     wci = wcf = wco = 0
#
#   i = sigmoid(cs_prev * wci + i)
#   f = sigmoid(cs_prev * wcf + f)
#   ci = tanh(ci) #new candidate values
#
#   cs = ci .* i + cs_prev .* f
#   cs = clip(cs, cell_clip)   #cell state
#
#   o = sigmoid(cs * wco + o) #Output
#   co = tanh(cs) #Final memory cell
#   h = co .* o  #Final hidden state
#   ```
#
# ```GRU
#   x_h_prev = [x, h_prev]
#
#   [r_bar u_bar] = x_h_prev * w_ru + b_ru
#
#   r = sigmoid(r_bar)
#   u = sigmoid(u_bar)
#
#   h_prevr = h_prev \circ r
#
#   x_h_prevr = [x h_prevr]
#
#   c_bar = x_h_prevr * w_c + b_c
#   c = tanh(c_bar)
#
#   h = (1-u) \circ c + u \circ h_prev
# ```
# softmax = exp(logits) / reduce_sum(exp(logits), dim)
# """Computes Rectified Linear 6: `min(max(features, 0), 6)`.
# softmax_cross_entropy_with_logits
# sparse_softmax_cross_entropy_with_logits
# r"""Computes softplus: `log(exp(features) + 1)`.
# For each batch `i` and class `j` we have
# softmax[i, j] = exp(logits[i, j]) / sum_j(exp(logits[i, j]))
#  r"""Computes rectified linear: `max(features, 0)`.
#  r"""Computes Quantized Rectified Linear X: `min(max(features, 0), max_value)`
#  r"""Computes Quantized Rectified Linear 6: `min(max(features, 0), 6)`
# /Library/Frameworks/Python.framework/Versions/3.5/lib/python3.5/site-packages/tensorflow/python/ops/gen_nn_ops.py
