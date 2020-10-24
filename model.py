
import tensorflow as tf
import tensorflow_probability as tfp
tfd = tfp.distributions

import math
import six

import numpy as np

def dropout(input_tensor, dropout_prob):
  """Perform dropout.

  Args:
    input_tensor: float Tensor.
    dropout_prob: Python float. The probability of dropping out a value (NOT of
      *keeping* a dimension as in `tf.nn.dropout`).

  Returns:
    A version of `input_tensor` with dropout applied.
  """
  if dropout_prob is None or dropout_prob == 0.0:
    return input_tensor

  output = tf.nn.dropout(input_tensor, 1.0 - dropout_prob)
  return output


def reshape_to_matrix(input_tensor):
  """Reshapes a >= rank 2 tensor to a rank 2 tensor (i.e., a matrix)."""
  ndims = input_tensor.shape.ndims
  if ndims < 2:
    raise ValueError("Input tensor must have at least rank 2. Shape = %s" %
                     (input_tensor.shape))
  if ndims == 2:
    return input_tensor

  width = input_tensor.shape[-1]
  output_tensor = tf.reshape(input_tensor, [-1, width])
  return output_tensor


def reshape_from_matrix(output_tensor, orig_shape_list):
  """Reshapes a rank 2 tensor back to its original rank >= 2 tensor."""
  if len(orig_shape_list) == 2:
    return output_tensor

  output_shape = get_shape_list(output_tensor)

  orig_dims = orig_shape_list[0:-1]
  width = output_shape[-1]

  return tf.reshape(output_tensor, orig_dims + [width])


def get_shape_list(tensor, expected_rank=None, name=None):
  """Returns a list of the shape of tensor, preferring static dimensions.

  Args:
    tensor: A tf.Tensor object to find the shape of.
    expected_rank: (optional) int. The expected rank of `tensor`. If this is
      specified and the `tensor` has a different rank, and exception will be
      thrown.
    name: Optional name of the tensor for the error message.

  Returns:
    A list of dimensions of the shape of tensor. All static dimensions will
    be returned as python integers, and dynamic dimensions will be returned
    as tf.Tensor scalars.
  """
  if name is None:
    name = tensor.name

  if expected_rank is not None:
    assert_rank(tensor, expected_rank, name)

  shape = tensor.shape.as_list()

  non_static_indexes = []
  for (index, dim) in enumerate(shape):
    if dim is None:
      non_static_indexes.append(index)

  if not non_static_indexes:
    return shape

  dyn_shape = tf.shape(tensor)
  for index in non_static_indexes:
    shape[index] = dyn_shape[index]
  return shape

def assert_rank(tensor, expected_rank, name=None):
  """Raises an exception if the tensor rank is not of the expected rank.

  Args:
    tensor: A tf.Tensor to check the rank of.
    expected_rank: Python integer or list of integers, expected rank.
    name: Optional name of the tensor for the error message.

  Raises:
    ValueError: If the expected shape doesn't match the actual shape.
  """
  if name is None:
    name = tensor.name

  expected_rank_dict = {}
  if isinstance(expected_rank, six.integer_types):
    expected_rank_dict[expected_rank] = True
  else:
    for x in expected_rank:
      expected_rank_dict[x] = True

  actual_rank = tensor.shape.ndims
  if actual_rank not in expected_rank_dict:
    scope_name = tf.get_variable_scope().name
    raise ValueError(
        "For the tensor `%s` in scope `%s`, the actual rank "
        "`%d` (shape = %s) is not equal to the expected rank `%s`" %
        (name, scope_name, actual_rank, str(tensor.shape), str(expected_rank)))

def create_attention_mask_from_input_mask(from_tensor, to_mask):
  """Create 3D attention mask from a 2D tensor mask.

  Args:
    from_tensor: 2D or 3D Tensor of shape [batch_size, from_seq_length, ...].
    to_mask: int32 Tensor of shape [batch_size, to_seq_length].

  Returns:
    float Tensor of shape [batch_size, from_seq_length, to_seq_length].
  """
  from_shape = get_shape_list(from_tensor, expected_rank=[2, 3])
  batch_size = from_shape[0]
  from_seq_length = from_shape[1]

  to_shape = get_shape_list(to_mask, expected_rank=2)
  to_seq_length = to_shape[1]

  to_mask = tf.cast(
      tf.reshape(to_mask, [batch_size, 1, to_seq_length]), tf.float32)

  # We don't assume that `from_tensor` is a mask (although it could be). We
  # don't actually care if we attend *from* padding tokens (only *to* padding)
  # tokens so we create a tensor of all ones.
  #
  # `broadcast_ones` = [batch_size, from_seq_length, 1]
  broadcast_ones = tf.ones(
      shape=[batch_size, from_seq_length, 1], dtype=tf.float32)

  # Here we broadcast along two dimensions to create the mask.
  mask = broadcast_ones * to_mask

  return mask

def attention_layer(from_tensor,
                    to_tensor,
                    attention_mask=None,
                    num_attention_heads=1,
                    size_per_head=512,
                    query_act=None,
                    key_act=None,
                    value_act=None,
                    attention_probs_dropout_prob=0.0,
                    initializer_range=0.02,
                    do_return_2d_tensor=False,
                    batch_size=None,
                    from_seq_length=None,
                    to_seq_length=None):
  """Performs multi-headed attention from `from_tensor` to `to_tensor`.

  This is an implementation of multi-headed attention based on "Attention
  is all you Need". If `from_tensor` and `to_tensor` are the same, then
  this is self-attention. Each timestep in `from_tensor` attends to the
  corresponding sequence in `to_tensor`, and returns a fixed-with vector.

  This function first projects `from_tensor` into a "query" tensor and
  `to_tensor` into "key" and "value" tensors. These are (effectively) a list
  of tensors of length `num_attention_heads`, where each tensor is of shape
  [batch_size, seq_length, size_per_head].

  Then, the query and key tensors are dot-producted and scaled. These are
  softmaxed to obtain attention probabilities. The value tensors are then
  interpolated by these probabilities, then concatenated back to a single
  tensor and returned.

  In practice, the multi-headed attention are done with transposes and
  reshapes rather than actual separate tensors.

  Args:
    from_tensor: float Tensor of shape [batch_size, from_seq_length,
      from_width].
    to_tensor: float Tensor of shape [batch_size, to_seq_length, to_width].
    attention_mask: (optional) int32 Tensor of shape [batch_size,
      from_seq_length, to_seq_length]. The values should be 1 or 0. The
      attention scores will effectively be set to -infinity for any positions in
      the mask that are 0, and will be unchanged for positions that are 1.
    num_attention_heads: int. Number of attention heads.
    size_per_head: int. Size of each attention head.
    query_act: (optional) Activation function for the query transform.
    key_act: (optional) Activation function for the key transform.
    value_act: (optional) Activation function for the value transform.
    attention_probs_dropout_prob: (optional) float. Dropout probability of the
      attention probabilities.
    initializer_range: float. Range of the weight initializer.
    do_return_2d_tensor: bool. If True, the output will be of shape [batch_size
      * from_seq_length, num_attention_heads * size_per_head]. If False, the
      output will be of shape [batch_size, from_seq_length, num_attention_heads
      * size_per_head].
    batch_size: (Optional) int. If the input is 2D, this might be the batch size
      of the 3D version of the `from_tensor` and `to_tensor`.
    from_seq_length: (Optional) If the input is 2D, this might be the seq length
      of the 3D version of the `from_tensor`.
    to_seq_length: (Optional) If the input is 2D, this might be the seq length
      of the 3D version of the `to_tensor`.

  Returns:
    float Tensor of shape [batch_size, from_seq_length,
      num_attention_heads * size_per_head]. (If `do_return_2d_tensor` is
      true, this will be of shape [batch_size * from_seq_length,
      num_attention_heads * size_per_head]).

  Raises:
    ValueError: Any of the arguments or tensor shapes are invalid.
  """

  def transpose_for_scores(input_tensor, batch_size, num_attention_heads,
                           seq_length, width):
    output_tensor = tf.reshape(
        input_tensor, [batch_size, seq_length, num_attention_heads, width])

    output_tensor = tf.transpose(output_tensor, [0, 2, 1, 3])
    return output_tensor

  from_shape = get_shape_list(from_tensor, expected_rank=[2, 3])
  to_shape = get_shape_list(to_tensor, expected_rank=[2, 3])

  if len(from_shape) != len(to_shape):
    raise ValueError(
        "The rank of `from_tensor` must match the rank of `to_tensor`.")

  if len(from_shape) == 3:
    batch_size = from_shape[0]
    from_seq_length = from_shape[1]
    to_seq_length = to_shape[1]
  elif len(from_shape) == 2:
    if (batch_size is None or from_seq_length is None or to_seq_length is None):
      raise ValueError(
          "When passing in rank 2 tensors to attention_layer, the values "
          "for `batch_size`, `from_seq_length`, and `to_seq_length` "
          "must all be specified.")

  # Scalar dimensions referenced here:
  #   B = batch size (number of sequences)
  #   F = `from_tensor` sequence length
  #   T = `to_tensor` sequence length
  #   N = `num_attention_heads`
  #   H = `size_per_head`

  from_tensor_2d = reshape_to_matrix(from_tensor)
  to_tensor_2d = reshape_to_matrix(to_tensor)

  # `query_layer` = [B*F, N*H]
  query_layer = tf.layers.dense(
      from_tensor_2d,
      num_attention_heads * size_per_head,
      activation=query_act,
      name="query",
      kernel_initializer=create_initializer(initializer_range))

  # `key_layer` = [B*T, N*H]
  key_layer = tf.layers.dense(
      to_tensor_2d,
      num_attention_heads * size_per_head,
      activation=key_act,
      name="key",
      kernel_initializer=create_initializer(initializer_range))

  # `value_layer` = [B*T, N*H]
  value_layer = tf.layers.dense(
      to_tensor_2d,
      num_attention_heads * size_per_head,
      activation=value_act,
      name="value",
      kernel_initializer=create_initializer(initializer_range))

  # `query_layer` = [B, N, F, H]
  query_layer = transpose_for_scores(query_layer, batch_size,
                                     num_attention_heads, from_seq_length,
                                     size_per_head)

  # `key_layer` = [B, N, T, H]
  key_layer = transpose_for_scores(key_layer, batch_size, num_attention_heads,
                                   to_seq_length, size_per_head)

  # Take the dot product between "query" and "key" to get the raw
  # attention scores.
  # `attention_scores` = [B, N, F, T]
  attention_scores = tf.matmul(query_layer, key_layer, transpose_b=True)
  attention_scores = tf.multiply(attention_scores,
                                 1.0 / math.sqrt(float(size_per_head)))

  if attention_mask is not None:
    # `attention_mask` = [B, 1, F, T]
    attention_mask = tf.expand_dims(attention_mask, axis=[1])

    # Since attention_mask is 1.0 for positions we want to attend and 0.0 for
    # masked positions, this operation will create a tensor which is 0.0 for
    # positions we want to attend and -10000.0 for masked positions.
    adder = (1.0 - tf.cast(attention_mask, tf.float32)) * -10000.0

    # Since we are adding it to the raw scores before the softmax, this is
    # effectively the same as removing these entirely.
    attention_scores += adder

  # Normalize the attention scores to probabilities.
  # `attention_probs` = [B, N, F, T]
  attention_probs = tf.nn.softmax(attention_scores)

  # This is actually dropping out entire tokens to attend to, which might
  # seem a bit unusual, but is taken from the original Transformer paper.
  attention_probs = dropout(attention_probs, attention_probs_dropout_prob)

  # `value_layer` = [B, T, N, H]
  value_layer = tf.reshape(
      value_layer,
      [batch_size, to_seq_length, num_attention_heads, size_per_head])

  # `value_layer` = [B, N, T, H]
  value_layer = tf.transpose(value_layer, [0, 2, 1, 3])

  # `context_layer` = [B, N, F, H]
  context_layer = tf.matmul(attention_probs, value_layer)

  # `context_layer` = [B, F, N, H]
  context_layer = tf.transpose(context_layer, [0, 2, 1, 3])

  if do_return_2d_tensor:
    # `context_layer` = [B*F, N*H]
    context_layer = tf.reshape(
        context_layer,
        [batch_size * from_seq_length, num_attention_heads * size_per_head])
  else:
    # `context_layer` = [B, F, N*H]
    context_layer = tf.reshape(
        context_layer,
        [batch_size, from_seq_length, num_attention_heads * size_per_head])

  return context_layer

def create_initializer(initializer_range=0.02):
  """Creates a `truncated_normal_initializer` with the given range."""
  return tf.compat.v1.truncated_normal_initializer(stddev=initializer_range)

def conv1d_layer(inputs, filter_width, in_channels, out_channels, padding, activation, initializer, trainable=True, name="conv"):
  with tf.compat.v1.variable_scope(name):
    filter = tf.compat.v1.get_variable(initializer=initializer, shape=[filter_width, in_channels, out_channels], trainable=trainable, name='filter')
    conv = tf.nn.conv1d(inputs, filter, [1], padding=padding, name="conv")
    bias = tf.compat.v1.get_variable(initializer=tf.zeros_initializer, shape=[out_channels], trainable=trainable, name='bias')
    conv_bias = tf.nn.bias_add(conv, bias, name='conv_bias')
    conv_bias_relu = activation(conv_bias, name='conv_bias_relu')
    return conv_bias_relu

def dense_layer(input_tensor, hidden_size, activation, initializer, name="dense"):
  with tf.compat.v1.variable_scope(name):
    input_shape = get_shape_list(input_tensor, expected_rank=3)
    batch_size = input_shape[0]
    seq_length = input_shape[1]
    input_width = input_shape[2]

    x = tf.reshape(input_tensor, [-1, input_width])
    w = tf.compat.v1.get_variable(initializer=initializer, shape=[input_width, hidden_size], name="w")
    z = tf.matmul(x, w, transpose_b=False)
    b = tf.compat.v1.get_variable(initializer=tf.zeros_initializer, shape=[hidden_size], name="b")
    y = tf.nn.bias_add(z, b)
    if (activation):
      y = activation(y)
    return tf.reshape(y, [batch_size, seq_length, hidden_size])

def layer_norm(input_tensor, trainable=True, name=None):
  """Run layer normalization on the last dimension of the tensor."""
  #return tf.keras.layers.LayerNormalization(name=name,trainable=trainable,axis=-1,epsilon=1e-14,dtype=tf.float32)(input_tensor)
  return tf.contrib.layers.layer_norm(
      inputs=input_tensor, begin_norm_axis=-1, begin_params_axis=-1, trainable=trainable, scope=name)

def mix_units(input_tensor, num_units, hidden_size, initializer, dropout_prob=0.1):
  #the mix density network is composed of multi-ple stacked linear layers, each followed by the layer normal-ization,  ReLU activation and the dropout layer but the last.

  layer_output = input_tensor
  for i in range(num_units):
    with tf.variable_scope("unit_index_%d" %i):
      layer_output = dense_layer(layer_output, hidden_size, activation=None, initializer=initializer)
      layer_output = layer_norm(layer_output)
      layer_output = tf.nn.relu(layer_output)
      layer_output = dropout(layer_output, dropout_prob)
  
  return layer_output 

def pdf(data, mu, var):
  import math

  pi = tf.constant(math.pi, dtype=tf.float32)
  epsilon = tf.constant(1e-14, dtype=tf.float32)

  #p get sometimes 1e-42 (>max float32)
  p = tf.math.exp(-tf.math.pow(data - mu, 2, name='p2') / (2.0 * (var + epsilon))) / tf.math.sqrt(2.0 * pi * (var + epsilon), name='p1')

  p2 = tf.reduce_sum(tf.math.log(p + tf.constant(1e-35, dtype=tf.float32)), axis=-1, keepdims=False)
  p2 = tf.clip_by_value(p2, tf.constant(math.log(1e-35), dtype=tf.float32), tf.constant(math.log(1e+35), dtype=tf.float32))
  p2 = tf.math.exp(p2)

  return tf.clip_by_value(p2, tf.constant(1e-35, dtype=tf.float32), tf.constant(1e+35, dtype=tf.float32))

def sentence_probabilities(mu, var, m, y, n, b, size_m, size_n, max_mel_length):
  with tf.compat.v1.variable_scope("sentence_probabilities"):
    #calculate all probabilities for sentence
    t = tf.constant(0) #first t row calculated already
    probabilities = tf.TensorArray(size=n[b], dtype=tf.float32, name="probabilities")
    _, probabilities_last = tf.while_loop(lambda t, probabilities: tf.less(t, n[b]), lambda t, probabilities: [t + 1, probabilities.write(t, pdf(tf.tile(tf.expand_dims(y[b,t], 0), [m[b], 1]), mu[b,:m[b]], var[b,:m[b]]))], [t, probabilities], maximum_iterations=max_mel_length, name="mel_loop")

  extended_n = tf.cond(size_n > n[b], 
    lambda: tf.concat([tf.reshape(probabilities_last.stack(), [n[b], m[b]]), tf.zeros([size_n - n[b], m[b]], tf.float32)], axis=0),
    lambda: tf.reshape(probabilities_last.stack(), [n[b], m[b]]))

  return tf.cond(size_m > m[b], 
    lambda: tf.concat([extended_n, tf.zeros([size_n, size_m - m[b]], tf.float32)], axis=1),
    lambda: extended_n)

def calculate_alpha(mu, var, m, y, n):
  #mu/var = mean/covariance created by neural model with m actual items
  #m - actual input sentence length tensor (var is s in the document). mu and var are at max size
  #y - mel spectrogram generated from audio for the sentence
  #n - actual mel spectrogram length tensor (var is t in the document), y is at max mel spectrogram size

  mel_shape = get_shape_list(y, expected_rank=3)
  batch_size = mel_shape[0]
  max_mel_length = mel_shape[1]
  mel_width = mel_shape[2]

  mix_shape = get_shape_list(mu, expected_rank=3)
  max_mix_length = mix_shape[1]
  mix_width = mix_shape[2]

  assert mel_width == mix_width

  #mu: [B, F, 80]
  #var: [B, F=max_mix_length, 80]
  #m: [B]
  #y: [B, max_mel_length, 80]
  #n: [B]

  with tf.compat.v1.variable_scope("alignment_loss"):
    v_b = tf.constant(0)
    v_batch_log_alpha = tf.TensorArray(size=batch_size, dtype=tf.float32, name="batch_log_alpha")
    v_losses = tf.TensorArray(size=batch_size, dtype=tf.float32, name="losses")

    def batch_body(b, batch_log_alpha, losses):
      probabilities = tf.cast(sentence_probabilities(mu, var, m, y, n, b, m[b], n[b], max_mel_length), tf.float64)

      pa = tf.concat([[probabilities[0, 0]], tf.zeros([m[b]-1], tf.float64)], axis=0)

      scaler = 1 / (tf.reduce_sum(pa) + tf.constant(1e-300, dtype=tf.float64))
      prev_alpha = pa * scaler
      c = tf.log(scaler)

      prev_alpha = tf.clip_by_value(prev_alpha, tf.constant(1e-300, dtype=tf.float64), tf.constant(1e+300, dtype=tf.float64))

      sentence_alpha = tf.TensorArray(size=n[b], dtype=tf.float64, name="sentence_alpha")
      sentence_alpha = sentence_alpha.write(0, prev_alpha)
      sentence_c = tf.TensorArray(size=n[b], dtype=tf.float64, name="sentence_c")
      sentence_c = sentence_c.write(0, c)

      v_t = tf.constant(1) #first t row calculated already

      #range: up to 1024 (we have 1024 frame in y spectrograms)
      def mel_body(t, c, prev_alpha, sentence_alpha, sentence_c): #go by mel spectrogram (t,n,y)

        n_a = (prev_alpha[1:]+prev_alpha[:-1])*probabilities[t, 1:]
        new_a = tf.concat([[prev_alpha[0]*(probabilities[t, 0])], n_a], axis=0)

        scaler = 1 / (tf.reduce_sum(new_a) + tf.constant(1e-300, dtype=tf.float64))
        new_a = new_a * scaler
        c = c + tf.log(scaler)

        new_a = tf.clip_by_value(new_a, tf.constant(1e-300, dtype=tf.float64), tf.constant(1e+300, dtype=tf.float64))

        return [t + 1, c, tf.reshape(new_a, [m[b]]), sentence_alpha.write(t, new_a), sentence_c.write(t, c)]

      t_last, c_out, alpha_l, sentence_alpha_out, sentence_c_out = tf.while_loop(lambda v_t, c, prev_alpha, sentence_alpha, sentence_c: tf.less(v_t, n[b]), mel_body, [v_t, c, prev_alpha, sentence_alpha, sentence_c], name="mel_loop")

      sentence_log_alpha_tensor = tf.cast(tf.math.log(sentence_alpha_out.stack()), dtype=tf.float32) - tf.reshape(tf.cast(sentence_c_out.stack(), dtype=tf.float32), [-1, 1])

      extended_log_alpha_n = tf.cond(max_mel_length > n[b], 
        lambda: tf.concat([tf.reshape(sentence_log_alpha_tensor, [n[b], m[b]]), tf.zeros([max_mel_length - n[b], m[b]], tf.float32)], axis=0),
        lambda: tf.reshape(sentence_log_alpha_tensor, [n[b], m[b]]))

      extended_log_alpha_nm = tf.cond(max_mix_length > m[b], 
        lambda: tf.concat([extended_log_alpha_n, tf.zeros([max_mel_length, max_mix_length - m[b]], tf.float32)], axis=1),
        lambda: extended_log_alpha_n)

      return [b + 1, batch_log_alpha.write(b, extended_log_alpha_nm), losses.write(b, -extended_log_alpha_nm[n[b]-1,m[b]-1])]

    _, v_batch_log_alpha_out, v_losses_out = tf.while_loop(lambda v_b, v_batch_log_alpha, v_losses: tf.less(v_b, batch_size), batch_body, [v_b, v_batch_log_alpha, v_losses], maximum_iterations=batch_size, name="batch_loop")

  return tf.reshape(v_batch_log_alpha_out.stack(), [batch_size, max_mel_length, max_mix_length]), tf.reshape(v_losses_out.stack(), [batch_size, -1])

def calculate_durations(alpha, m, n):
  alpha_shape = get_shape_list(alpha, expected_rank=3)
  batch_size = alpha_shape[0]
  max_mel_length = alpha_shape[1]
  max_mix_length = alpha_shape[2]

  with tf.compat.v1.variable_scope("alpha_durations"):
    v_b = tf.constant(0)
    v_batch_durations = tf.TensorArray(size=batch_size, dtype=tf.int32, name="batch_durations")

    def batch_body(b, batch_durations):
      best = tf.TensorArray(size=n[b], dtype=tf.int32, name="best")
      position = m[b] - 1
      best_vector = tf.sparse.SparseTensor(indices=[[position]], values=[1], dense_shape=[max_mix_length])
      best = best.write(n[b] - 1, tf.sparse.to_dense(best_vector, default_value=0, validate_indices=True))
      v_t = n[b] - 2

      def mel_body(t, prev_position, best):
        #position = tf.cond(prev_position == 0, lambda: 0, lambda: tf.cond(alpha[b, t, prev_position - 1] > alpha[b, t, prev_postion], lambda: prev_position - 1, lambda: prev_position))

        position = tf.case([(tf.equal(prev_position, 0), lambda: tf.constant(0)), (tf.greater(alpha[b, t, prev_position - 1], alpha[b, t, prev_position]), lambda: prev_position - 1)], default=lambda: prev_position)

        best_vector = tf.sparse.SparseTensor(indices=[[position]], values=[1], dense_shape=[max_mix_length])

        return [t - 1, position, best.write(t, tf.sparse.to_dense(best_vector, default_value=0, validate_indices=True))]

      _, _, best_out = tf.while_loop(lambda v_t, position, best: tf.greater(v_t, -1), mel_body, [v_t, position, best], name="mel_loop")

      d = tf.reduce_sum(best_out.stack(), axis=0, keepdims=False)

      return [b + 1, batch_durations.write(b, d)]

    _, v_batch_durations_out = tf.while_loop(lambda v_b, v_batch_durations: tf.less(v_b, batch_size), batch_body, [v_b, v_batch_durations], maximum_iterations=batch_size, name="batch_loop")

  return tf.reshape(v_batch_durations_out.stack(), [batch_size, max_mix_length])

def transformer_model(input_tensor,
                      attention_mask=None,
                      hidden_size=768,
                      num_hidden_layers=12,
                      num_attention_heads=12,
                      intermediate_size=768,
                      intermediate_act_fn=tf.nn.relu,
                      hidden_dropout_prob=0.1,
                      attention_probs_dropout_prob=0.1,
                      initializer_range=0.02,
                      do_return_all_layers=False):
  """Multi-headed, multi-layer Transformer from "Attention is All You Need".

  This is almost an exact implementation of the original Transformer encoder.

  See the original paper:
  https://arxiv.org/abs/1706.03762

  Also see:
  https://github.com/tensorflow/tensor2tensor/blob/master/tensor2tensor/models/transformer.py

  Args:
    input_tensor: float Tensor of shape [batch_size, seq_length, hidden_size].
    attention_mask: (optional) int32 Tensor of shape [batch_size, seq_length,
      seq_length], with 1 for positions that can be attended to and 0 in
      positions that should not be.
    hidden_size: int. Hidden size of the Transformer.
    num_hidden_layers: int. Number of layers (blocks) in the Transformer.
    num_attention_heads: int. Number of attention heads in the Transformer.
    intermediate_size: int. The size of the "intermediate" (a.k.a., feed
      forward) layer.
    intermediate_act_fn: function. The non-linear activation function to apply
      to the output of the intermediate/feed-forward layer.
    hidden_dropout_prob: float. Dropout probability for the hidden layers.
    attention_probs_dropout_prob: float. Dropout probability of the attention
      probabilities.
    initializer_range: float. Range of the initializer (stddev of truncated
      normal).
    do_return_all_layers: Whether to also return all layers or just the final
      layer.

  Returns:
    float Tensor of shape [batch_size, seq_length, hidden_size], the final
    hidden layer of the Transformer.

  Raises:
    ValueError: A Tensor shape or parameter is invalid.
  """
  if hidden_size % num_attention_heads != 0:
    raise ValueError(
        "The hidden size (%d) is not a multiple of the number of attention "
        "heads (%d)" % (hidden_size, num_attention_heads))

  attention_head_size = int(hidden_size / num_attention_heads)
  input_shape = get_shape_list(input_tensor, expected_rank=3)
  batch_size = input_shape[0]
  seq_length = input_shape[1]
  input_width = input_shape[2]

  # The Transformer performs sum residuals on all layers so the input needs
  # to be the same as the hidden size.
  if input_width != hidden_size:
    raise ValueError("The width of the input tensor (%d) != hidden size (%d)" %
                     (input_width, hidden_size))

  # We keep the representation as a 2D tensor to avoid re-shaping it back and
  # forth from a 3D tensor to a 2D tensor. Re-shapes are normally free on
  # the GPU/CPU but may not be free on the TPU, so we want to minimize them to
  # help the optimizer.
  prev_output = reshape_to_matrix(input_tensor)

  all_layer_outputs = []
  for layer_idx in range(num_hidden_layers):
    with tf.variable_scope("layer_%d" % layer_idx):
      layer_input = prev_output

      with tf.variable_scope("attention"):
        attention_heads = []
        with tf.variable_scope("self"):
          attention_head = attention_layer(
              from_tensor=layer_input,
              to_tensor=layer_input,
              attention_mask=attention_mask,
              num_attention_heads=num_attention_heads,
              size_per_head=attention_head_size,
              attention_probs_dropout_prob=attention_probs_dropout_prob,
              initializer_range=initializer_range,
              do_return_2d_tensor=True,
              batch_size=batch_size,
              from_seq_length=seq_length,
              to_seq_length=seq_length)
          attention_heads.append(attention_head)

        attention_output = None
        if len(attention_heads) == 1:
          attention_output = attention_heads[0]
        else:
          # In the case where we have other sequences, we just concatenate
          # them to the self-attention head before the projection.
          attention_output = tf.concat(attention_heads, axis=-1)

        # Run a linear projection of `hidden_size` then add a residual
        # with `layer_input`.
        with tf.variable_scope("output"):
          attention_output = tf.layers.dense(
              attention_output,
              hidden_size,
              kernel_initializer=create_initializer(initializer_range))
          attention_output = dropout(attention_output, hidden_dropout_prob)
          attention_output = layer_norm(attention_output + layer_input)

      # The activation is only applied to the "intermediate" hidden layer.
      #B, F, H
      #ss2 = get_shape_list(attention_output, expected_rank=2)
      #print ("ss2: ", ss2[0], ss2[1])

      intermediate_input = tf.reshape(attention_output, [batch_size, seq_length, input_width])
      with tf.variable_scope("intermediate"):
        # `context_layer` = [B, F, N*H]
        layer1_output = conv1d_layer(intermediate_input, 3, hidden_size, intermediate_size, "SAME", 
            intermediate_act_fn, create_initializer(initializer_range), name="conv_1")
        layer1_with_dropout = dropout(layer1_output, hidden_dropout_prob)
        intermediate_output = conv1d_layer(layer1_with_dropout, 3, intermediate_size, hidden_size, "SAME", 
            intermediate_act_fn, create_initializer(initializer_range), name="conv_2")
        intermediate_with_dropout = dropout(intermediate_output, hidden_dropout_prob)
      intermediate_output = tf.reshape(intermediate_with_dropout , [-1, input_width])

      # Down-project back to `hidden_size` then add the residual.
      with tf.variable_scope("output"):
        layer_output = tf.layers.dense(
            intermediate_output,
            hidden_size,
            kernel_initializer=create_initializer(initializer_range))
        layer_output = dropout(layer_output, hidden_dropout_prob)
        layer_output = layer_norm(layer_output + attention_output)
        prev_output = layer_output
        all_layer_outputs.append(layer_output)

  if do_return_all_layers:
    final_outputs = []
    for layer_output in all_layer_outputs:
      final_output = reshape_from_matrix(layer_output, input_shape)
      final_outputs.append(final_output)
    return final_outputs
  else:
    final_output = reshape_from_matrix(prev_output, input_shape)
    return final_output


class AlignttsModel(object):

  #   F = `from_tensor` sequence length in characters
  #   T = `to_tensor` sequence length in frames
  #   V - alphabet size
  #   E - embeddings, hidden size
  #   D - duration embeddings, hidden size
  #   M - number of mel buckets in final spectrogram
  def __init__(self,
               input_tensor,
               input_lengths,
               input_masks,
               input_durations,
               mel_tensor,
               mel_lengths,
               hidden_size=768,
               num_hidden_layers=6,
               num_attention_heads=2,
               filter_width=3,
               duration_predictor_hidden_layers=2,
               duration_predictor_attention_heads=2,
               duration_predictor_hidden_size=128,
               num_mix_density_hidden_layers=4, #as in DEEP MIXTURE DENSITY NETWORKS GOOGLE Paper
               mix_density_hidden_size=256,
               alphabet_size=29,
               initializer_range=0.02,
               activation_fn=tf.nn.relu,
               alpha=1.0,
               dropout_prob=0.1,
               use_durations=2, #use duration predictor by default
               is_trainable=True):
    input_shape = get_shape_list(input_tensor, expected_rank=2)
    batch_size = input_shape[0]
    max_input_length = input_shape[1]
  
    mel_shape = get_shape_list(mel_tensor, expected_rank=3)
    max_mel_length = mel_shape[1]
    self._num_mels = mel_shape[2]

    if is_trainable == False:
       dropout_prob = 0.0
   
    #1). embedding table: [V, E] so like this [alphabet_siz, hidden_size]
    #lookup in embeddings table to find entry for each character
    with tf.compat.v1.variable_scope("input_embeddings"):
      #[A, E]
      self._encoder_embedding_table = tf.compat.v1.get_variable(initializer=create_initializer(initializer_range), 
            shape=[alphabet_size, hidden_size], name='encoder_embedding_table')
  
      encoder_embedding_expanded = tf.expand_dims(self._encoder_embedding_table, 0)
      encoder_embedding_expanded = tf.tile(encoder_embedding_expanded, [batch_size, 1, 1])
  
      #[B, F] --> [B, F, E]
      self._encoder_embedding = tf.gather(encoder_embedding_expanded, input_tensor, axis=1, batch_dims=1, name="encoder_embedding")

    #2) make positional encoding
    #[B, F, E] --> [B, F, E]
    with tf.compat.v1.variable_scope("input_positions"):
      self._encoder_position_table = tf.compat.v1.get_variable(initializer=create_initializer(initializer_range), 
            shape=[max_input_length, hidden_size], name='encoder_position_table')
  
      self._encoder_embedding_with_positions = self._encoder_embedding + tf.expand_dims(self._encoder_position_table, 0)

    #3) create 3D mask from 2D to mask attention with shorter sentenses then max_input_length sentence
    attention_mask = create_attention_mask_from_input_mask(
              input_tensor, input_masks)
  
    #4). encoder FFT block
    with tf.compat.v1.variable_scope("encoder_ttf"):
      encoder_dropout_prob = dropout_prob
      
      self._encoder_tensor = transformer_model(self._encoder_embedding_with_positions,
                        attention_mask=attention_mask,
                        hidden_size=hidden_size,
                        num_hidden_layers=num_hidden_layers,
                        num_attention_heads=num_attention_heads,
                        intermediate_size=hidden_size,
                        intermediate_act_fn=activation_fn,
                        hidden_dropout_prob=encoder_dropout_prob,
                        attention_probs_dropout_prob=encoder_dropout_prob ,
                        initializer_range=0.02,
                        do_return_all_layers=False)
  
    #5.1) Mix density network
    with tf.compat.v1.variable_scope("mix_density_network"):
      #[B, F, E] --> [B, F, 256]
      self._mix_density_tensor = mix_units(self._encoder_tensor, num_mix_density_hidden_layers, mix_density_hidden_size, create_initializer(initializer_range), dropout_prob)

      #[B, F, E] --> [B, F, M*2]
      self._mu_and_variance = dense_layer(self._mix_density_tensor, self._num_mels*2, activation=tf.math.softplus, initializer=create_initializer(initializer_range), name="mu_and_variance")

      #The  last  linear  layer  outputs  the  mean  and  variance  vectorof multi-dimensional gaussian distributions, which representsthe mel-spectrum distribution of each character.
      #The hiddensize of the linear layer in the mix network is set to 256 and thedimension of the output is 160 (80 dimensions for the meanand 80 dimensions for variance of the gaussian distribution).

      self._log_alpha, self._per_example_alignment_loss = calculate_alpha(self._mu_and_variance[:, :, :self._num_mels], self._mu_and_variance[:, :, self._num_mels:], input_lengths, mel_tensor, mel_lengths)

      self._mix_durations = calculate_durations(self._log_alpha, input_lengths, mel_lengths)
      #self._mix_durations  = tf.Print(_mix_durations , [_mix_durations], "Mix Durations", summarize=50)

    #5.2). Length regulator
    #increase hidden lengths as per length predictor values and alpha speech speed coefficient
    #5.2.1). embedding table: [V, D] so like this [alphabet_siz, hidden_size]
    #lookup in embeddings table to find entry for each character
    with tf.compat.v1.variable_scope("duration_embeddings"):
      #[A, D]
      duration_embedding_table = tf.compat.v1.get_variable(initializer=create_initializer(initializer_range), 
            shape=[alphabet_size, duration_predictor_hidden_size], name='duration_embedding_table')
  
      duration_embedding_expanded = tf.expand_dims(duration_embedding_table, 0)
      duration_embedding_expanded = tf.tile(duration_embedding_expanded, [batch_size, 1, 1])
  
      #[B, F] --> [B, F, D]
      duration_embedding = tf.gather(duration_embedding_expanded, input_tensor, axis=1, batch_dims=1, name="duration_embedding")
  
    #5.2.2) make positional encoding
    #[B, F, D] --> [B, F, D]
    with tf.compat.v1.variable_scope("duration_positions"):
      duration_position_table = tf.compat.v1.get_variable(initializer=create_initializer(initializer_range), 
            shape=[max_input_length, duration_predictor_hidden_size], name='duration_position_table')
  
      duration_embedding_with_positions = duration_embedding + tf.expand_dims(duration_position_table, 0)
  
    #5.2.3). duration FFT block
    #[B, F, D] --> [B, F, D]
    with tf.compat.v1.variable_scope("duration_ttf"):
      duration_prdictor_dropout_prob = dropout_prob

      duration_tensor = transformer_model(duration_embedding_with_positions,
                      attention_mask=attention_mask,
                      hidden_size=duration_predictor_hidden_size,
                      num_hidden_layers=duration_predictor_hidden_layers,
                      num_attention_heads=duration_predictor_attention_heads,
                      intermediate_size=duration_predictor_hidden_size,
                      intermediate_act_fn=activation_fn,
                      hidden_dropout_prob=duration_prdictor_dropout_prob,
                      attention_probs_dropout_prob=duration_prdictor_dropout_prob,
                      initializer_range=0.02,
                      do_return_all_layers=False)
    #nominal_durations is an output to train duration predictor
    #[B, F, D] --> [B, F, D]
    nominal_durations = dense_layer(duration_tensor, 1, activation=activation_fn, initializer=create_initializer(initializer_range), name="nominal_durations")
    #[B, F, D] --> [B, F]
    self._nominal_durations = tf.squeeze(nominal_durations, axis=-1) 

    #scale back durations so the sum less or equal to max
    mel_length = tf.reduce_sum(tf.cast(tf.math.multiply(self._nominal_durations + 0.5, alpha), tf.int32), axis=-1, keep_dims=True)
    scaling_factor = tf.clip_by_value(tf.cast(mel_length, dtype=tf.float32) / tf.cast(max_mel_length, dtype=tf.float32), tf.constant(1, dtype=tf.float32), tf.cast(tf.reduce_max(mel_length), dtype=tf.float32))

    scaled_durations = self._nominal_durations / scaling_factor

    #[B, F] --> [B, F]
    self._mel_durations = tf.cast(tf.math.multiply(scaled_durations + 0.5, alpha), tf.int32)

    #Use duration from: 0 - input, 1 - mix network, 2 - duration predictor
    durations = tf.case([(tf.equal(use_durations, 0), lambda: input_durations), (tf.equal(use_durations, 1), lambda: self._mix_durations)], default=lambda: self._mel_durations)

    #32, 200 -> 32, 1
    lengths = tf.fill([batch_size, 1], max_mel_length) - tf.reduce_sum(durations, axis=1, keep_dims=True)
    #32, 200 -> 32, 201
    durations_with_extra_lengths = tf.concat([durations, lengths], axis=1)

    #32, 200, 768 -> 32, 201, 768
    encoder_with_extra_zero = tf.concat([self._encoder_tensor, tf.zeros([batch_size, 1, hidden_size], tf.float32)], axis=1) 

    flatten_durations = tf.reshape(durations_with_extra_lengths, [-1])
    flatten_encoder = tf.reshape(encoder_with_extra_zero, [-1, hidden_size])

    #32*201, 768 -> 32*1024, 768
    #OOM 3216,914,768
    encoder_with_flatten_durations = tf.repeat(flatten_encoder, flatten_durations, axis=0, name="encoder_with_flatten_durations")

    encoder_with_durations = tf.reshape(encoder_with_flatten_durations, [batch_size, max_mel_length, hidden_size], name="encoder_with_durations")

    #6.1). Add positional encoding
    #[B, T, E] --> [B, T, E]
    with tf.compat.v1.variable_scope("mel_positions"):
      self._decoder_position_table = tf.compat.v1.get_variable(initializer=create_initializer(initializer_range), 
        shape=[max_mel_length, hidden_size], name='decoder_position_table')
      decoder_embedding_with_positions = encoder_with_durations  + tf.expand_dims(self._decoder_position_table, 0)

    #6.2). Decoder mask
    #2 --> 32,2
    mask_template = tf.tile(tf.constant([[1,0]], tf.int32), [batch_size, 1])
    mel_length = tf.reduce_sum(durations, axis=1, keep_dims=True)
    #32,2
    mask_durations = tf.concat([mel_length, tf.fill([batch_size, 1], max_mel_length) - mel_length], axis=1)

    #32,2 -> 32*2
    flatten_mask_template = tf.reshape(mask_template, [-1])
    #32,2 -> 32*2
    flatten_mask_durations = tf.reshape(mask_durations, [-1])

    #32*2 -> 32*1024
    decoder_flatten_mask = tf.repeat(flatten_mask_template, flatten_mask_durations, axis=0, name="decoder_flatten_mask")
    #32*1024 -> 32,1024
    decoder_mask = tf.reshape(decoder_flatten_mask, [batch_size, max_mel_length], name="decoder_mask")

    # create 3D mask from 2D to mask attention with shorter mel duration then max_mel_length
    decoder_attention_mask = create_attention_mask_from_input_mask(
              decoder_embedding_with_positions, decoder_mask)
  
    #7). decoder FFT block
    with tf.compat.v1.variable_scope("decoder_ttf"):
      decoder_dropout_prob = dropout_prob

      #attention_mask=decoder_mask,
      self._decoder_tensor = transformer_model(decoder_embedding_with_positions,
                       attention_mask=decoder_attention_mask,
                       hidden_size=hidden_size,
                       num_hidden_layers=num_hidden_layers,
                       num_attention_heads=num_attention_heads,
                       intermediate_size=hidden_size,
                       intermediate_act_fn=activation_fn,
                       hidden_dropout_prob=decoder_dropout_prob,
                       attention_probs_dropout_prob=decoder_dropout_prob,
                       initializer_range=0.02,
                       do_return_all_layers=False)
  
    #8). Linear layer, returns mel: [B, T, E] --> [B, T, M]
    self._mel_spectrograms = dense_layer(self._decoder_tensor, self._num_mels, activation=None, initializer=create_initializer(initializer_range), name="mel_spectrograms")
  
  @property
  def per_example_alignment_loss(self):
    return self._per_example_alignment_loss 

  @property
  def log_alpha(self):
    return self._log_alpha

  @property
  def encoder_embedding(self):
    return self._encoder_embedding

  @property
  def encoder_tensor(self):
    return self._encoder_tensor

  @property
  def mix_density_tensor(self):
    return self._mix_density_tensor

  @property
  def mu_and_variance(self):
    return self._mu_and_variance

  @property
  def mu(self):
    return self._mu_and_variance[:, :, :self._num_mels]

  @property
  def var(self):
    return self._mu_and_variance[:, :, self._num_mels:] 

  @property
  def mix_durations(self):
    return self._mix_durations

  @property
  def nominal_durations(self):
    return self._nominal_durations

  @property
  def mel_durations(self):
    return self._mel_durations

  @property
  def mel_spectrograms (self):
    return self._mel_spectrograms 
