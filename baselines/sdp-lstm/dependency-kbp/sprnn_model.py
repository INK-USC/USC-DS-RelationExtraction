import sys
import tensorflow as tf

import data_utils

tf.app.flags.DEFINE_integer('num_layers', 2, 'Number of cell layers')

tf.app.flags.DEFINE_integer('hidden_size', 300, 'Size of word embeddings and hidden layers')
tf.app.flags.DEFINE_integer('pos_size', 32, 'Size of POS embeddings')
tf.app.flags.DEFINE_integer('ner_size', 0, 'Size of NER embeddings')
tf.app.flags.DEFINE_integer('deprel_size', 32, 'Size of DepRel embeddings')

# tf.app.flags.DEFINE_integer('vocab_size', 11893, 'Vocabulary size')
tf.app.flags.DEFINE_integer('num_class', 42, 'Number of class to consider')

tf.app.flags.DEFINE_integer('sent_len', 100, 'Input sentence length. This is after the padding is performed.')
tf.app.flags.DEFINE_float('max_grad_norm', 5.0, 'The maximum norm used to clip the gradients')
tf.app.flags.DEFINE_float('dropout', 0.5, 'Dropout rate that applies to the LSTM. 0 is no dropout.')

tf.app.flags.DEFINE_boolean('pool', False, 'Add a max pooling layer at the end')

tf.app.flags.DEFINE_boolean('attn', False, 'Whether to use an attention layer')
tf.app.flags.DEFINE_integer('attn_size', 256, 'Size of attention layer')
tf.app.flags.DEFINE_float('attn_stddev', 0.001, 'The attention weights are initialized as normal(0, attn_stddev)')

tf.app.flags.DEFINE_boolean('bi', False, 'Whether to use a bi-directional lstm')

FLAGS = tf.app.flags.FLAGS
# Vocab size for different fields (other than 'word')
NUM_POS = len(data_utils.POS_TO_ID)
NUM_NER = len(data_utils.NER_TO_ID)
NUM_DEPREL = len(data_utils.DEPREL_TO_ID)

def _variable_on_cpu(name, shape, initializer):
	with tf.device('/cpu:0'):
		var = tf.get_variable(name, shape, initializer=initializer)
	return var

def _get_lstm_graph_info():
	total_size = FLAGS.hidden_size + FLAGS.pos_size + FLAGS.ner_size + FLAGS.deprel_size
	if FLAGS.bi:
		model_name = 'Bi-SPRNN'
	else:
		model_name = 'SPRNN'
	info = 'Building %s graph with [%d layers, %d hidden_size (%d word, %d pos, %d ner, %d deprel)]' % \
		(model_name, FLAGS.num_layers, total_size, FLAGS.hidden_size, FLAGS.pos_size, FLAGS.ner_size, FLAGS.deprel_size)
	if FLAGS.pool:
		info += ' and a max-pooling layer'
	if FLAGS.attn:
		info += ' and an attention layer [%d attn size]' % FLAGS.attn_size
	info += ' ...'
	return info

def _create_embedding_layer(name, vocab_size, dim, inputs, is_train):
	W_emb = _variable_on_cpu(name, shape=[vocab_size, dim], initializer=tf.random_uniform_initializer(minval=-1.0, maxval=1.0))
	sent_batch = tf.nn.embedding_lookup(params=W_emb, ids=inputs)
	if is_train:
		sent_batch = tf.nn.dropout(sent_batch, 1-FLAGS.dropout)
	return W_emb, sent_batch

def _get_rnn_cell(hidden_size, num_layers, is_train, dropout):
	'''
		Return a LSTM cell. Useful to get it modularized.
	'''
	cell = tf.nn.rnn_cell.BasicLSTMCell(num_units=hidden_size)
	if is_train:
		cell = tf.nn.rnn_cell.DropoutWrapper(cell, output_keep_prob=1-dropout)
	cell = tf.nn.rnn_cell.MultiRNNCell([cell] * num_layers)
	return cell

def max_over_time(inputs, index, seq_lens):
	''' Use with tf.map_fn()
		Args:
			inputs: [batch_size, sent_len, dim]
			seq_lens: [batch_size]
		Return:
			output: [dim]
	'''
	l = seq_lens[index]
	valid_inputs = inputs[index, :l] # [l, dim]
	output = tf.reduce_max(valid_inputs, reduction_indices=[0])
	return output

def attention_over_time(inputs, hidden, params, index, seq_lens):
	''' Use with tf.map_fn()
		Args:
			inputs: [batch_size, sent_len, dim]
			hidden: [batch_size, dim]
			params: must be [attn_W_h, attn_W_q, attn_b, attn_V, attn_V_b]
			seq_lens: [batch_size]
		Return:
			attention: attention distribution (after softmax) with shape [batch_size, sent_len]
	'''
	attn_W_h, attn_W_q, attn_b, attn_V, attn_V_b = params
	l = seq_lens[index]
	valid_inputs = inputs[index, :l] # [l, dim]
	current_hidden = tf.expand_dims(hidden[index, :], 0) # [1, dim]
	x1 = tf.matmul(valid_inputs, tf.transpose(attn_W_h)) # input transformation [l, attn]
	x2 = tf.matmul(current_hidden, tf.transpose(attn_W_q)) # query transformation [1, attn]
	y = tf.tanh(tf.nn.bias_add(x1 + x2, attn_b)) # [l, attn_size] # non-linear
	e = tf.transpose(tf.nn.bias_add(tf.matmul(y, attn_V), attn_V_b)) # [1, l]
	attention = tf.nn.softmax(e) # [1, l]
	attention = tf.pad(attention, paddings=[[0,0],[0,FLAGS.sent_len-l]], mode='CONSTANT') # pad to [1, sent_len]
	return attention

def _create_attention_layer(rnn_outputs, rnn_final_hidden, seq_lens, dim, batch_size):
	# initialization: weights ~ n(0, stddev), biases = 0, V = zero vector
	attn_bias_init = 0.0 # the original attention paper uses 0.0001 for weights initialization
	attn_size = FLAGS.attn_size
	attn_W_h = tf.get_variable('attn_W_h', shape=[attn_size, dim], initializer=tf.random_normal_initializer(stddev=FLAGS.attn_stddev))
	attn_W_q = tf.get_variable('attn_W_q', shape=[attn_size, dim], initializer=tf.random_normal_initializer(stddev=FLAGS.attn_stddev))
	attn_b = tf.get_variable('attn_b', shape=[attn_size], initializer=tf.constant_initializer(attn_bias_init))
	attn_V = tf.get_variable('attn_V', shape=[attn_size, 1], initializer=tf.constant_initializer(0.0))
	attn_V_b = tf.get_variable('attn_V_b', shape=[1], initializer=tf.constant_initializer(attn_bias_init))

	params = [attn_W_h, attn_W_q, attn_b, attn_V, attn_V_b]
	attention = tf.map_fn(lambda idx: attention_over_time(rnn_outputs, rnn_final_hidden, params, idx, seq_lens), tf.range(0, batch_size), dtype=tf.float32)
	attn_final_state = tf.reduce_sum(tf.reshape(attention, [-1, FLAGS.sent_len, 1]) * rnn_outputs, [1]) # shape: [batch_size, dim]
	return attention, attn_final_state

def _create_rnn_along_subpath(subpath_sent_batch, seq_lens, dim, batch_size, is_train):
	"""
		Build a rnn along the given path and return the resulting path vector.
	"""
	# rnn cell
	if FLAGS.bi:
		cell_fw, cell_bw = _get_rnn_cell(dim, FLAGS.num_layers, is_train, FLAGS.dropout), \
				_get_rnn_cell(dim, FLAGS.num_layers, is_train, FLAGS.dropout)
	else:
		cell = _get_rnn_cell(dim, FLAGS.num_layers, is_train, FLAGS.dropout)
	# rnn layer
	# NOTE that rnn is doing dynamic batching now so outputs will be tailed by zero vectors in a batch
	if FLAGS.bi:
		rnn_outputs, rnn_final_states = tf.nn.bidirectional_dynamic_rnn(cell_fw, cell_bw, subpath_sent_batch, dtype=tf.float32, sequence_length=seq_lens)
		rnn_outputs = tf.concat(2, rnn_outputs) # the original is a tuple, we need to concatenate them
		rnn_final_states_fw, rnn_final_states_bw = rnn_final_states
		rnn_final_state = tf.concat(1, [rnn_final_states_fw[-1][1], rnn_final_states_bw[-1][1]])
		dim = 2*dim
	else:
		rnn_outputs, rnn_final_states = tf.nn.dynamic_rnn(cell, subpath_sent_batch, dtype=tf.float32, sequence_length=seq_lens)
		# rnn_final_states is tuple of tuple for LSTM, outputs is a tensor [batch_size, sent_len, dim]
		rnn_final_state = rnn_final_states[-1][1]

	# [optional] add pooling or attention layer
	if FLAGS.pool:
		final_hidden = tf.map_fn(lambda idx: max_over_time(rnn_outputs, idx, seq_lens), tf.range(0, batch_size), dtype=tf.float32)
	elif FLAGS.attn:
		# attention layer
		attention, attn_final_state = _create_attention_layer(rnn_outputs, rnn_final_state, seq_lens, dim, batch_size)
		final_hidden = attn_final_state
	else:
		final_hidden = rnn_final_state
	return final_hidden, dim

class SPRNNModel(object):
	"""
	A RNN model that runs over sub dependency paths.
	Left path will be from SUBJ to ROOT, and right path will be from OBJ to ROOT.
	"""

	def __init__(self, vocab_size, is_train=True):
		self.is_train = is_train
		self.vocab_size = vocab_size
		self.build_graph()

	def build_graph(self):
		# sanity check
		if FLAGS.pool and FLAGS.attn:
			raise Exception("Max-pooling layer and attention layer cannot be added at the same time.")

		# print graph info
		print _get_lstm_graph_info()

		# note that for each tensor here, the left part and right part are stacked together along first dimension
		self.word_inputs = tf.placeholder(dtype=tf.int32, shape=[None, FLAGS.sent_len], name='input_word')
		self.pos_inputs = tf.placeholder(dtype=tf.int32, shape=[None, FLAGS.sent_len], name='input_pos')
		# self.ner_inputs = tf.placeholder(dtype=tf.int32, shape=[None, FLAGS.sent_len], name='input_ner')
		self.deprel_inputs = tf.placeholder(dtype=tf.int32, shape=[None, FLAGS.sent_len], name='input_deprel')
		self.seq_lens = tf.placeholder(dtype=tf.int32, shape=[None], name='input_seq_len')

		self.labels = tf.placeholder(dtype=tf.int32, shape=[None], name='input_label') # labels have the normal batch size

		# self.batch_size = tf.shape(self.word_inputs)[0]/2 # real batch size is half
		self.batch_size = tf.shape(self.word_inputs)[0]
		dim = FLAGS.hidden_size

		# embedding layers
		self.w_emb, word_sent_batch = _create_embedding_layer('word_embedding', self.vocab_size, FLAGS.hidden_size, self.word_inputs, self.is_train)
		sent_batch_list = [word_sent_batch]
		if FLAGS.pos_size > 0:
			self.pos_emb, pos_sent_batch = _create_embedding_layer('pos_embedding', NUM_POS, FLAGS.pos_size, self.pos_inputs, self.is_train)
			dim += FLAGS.pos_size
			sent_batch_list.append(pos_sent_batch)
		# if FLAGS.ner_size > 0:
		# 	self.ner_emb, ner_sent_batch = _create_embedding_layer('ner_embedding', NUM_NER, FLAGS.ner_size, self.ner_inputs, self.is_train)
		# 	dim += FLAGS.ner_size
		# 	sent_batch_list.append(ner_sent_batch)
		if FLAGS.deprel_size > 0:
			self.deprel_emb, deprel_sent_batch = _create_embedding_layer('deprel_embedding', NUM_DEPREL, FLAGS.deprel_size, self.deprel_inputs, self.is_train)
			dim += FLAGS.deprel_size
			sent_batch_list.append(deprel_sent_batch)

		# concatenate all embeddings to form complete sentence batch
		# each element of sent_batch_list is of shape [batch_size, sent_len, hidden_size]
		# sent_batch = tf.concat(2, sent_batch_list, 'sent_batch')
		sent_batch = tf.concat(sent_batch_list, 2, 'sent_batch')

		left_sent_batch, right_sent_batch = tf.split(axis=0, num_or_size_splits=2, value=sent_batch)
		left_seq_lens, right_seq_lens = tf.split(axis=0, num_or_size_splits=2, value=self.seq_lens)

		with tf.variable_scope('left_path') as scope:
			left_final_hidden, left_dim = _create_rnn_along_subpath(left_sent_batch, left_seq_lens, dim, self.batch_size, self.is_train)
		with tf.variable_scope('right_path') as scope:
			right_final_hidden, right_dim = _create_rnn_along_subpath(right_sent_batch, right_seq_lens, dim, self.batch_size, self.is_train)
		#
		final_hidden = tf.concat([left_final_hidden, right_final_hidden], 1, 'final_hidden')

		# softmax layer
		self.softmax_w = tf.get_variable('softmax_w', shape=[left_dim + right_dim, FLAGS.num_class])
		self.softmax_b = tf.get_variable('softmax_b', shape=[FLAGS.num_class])
		self.logits = tf.nn.bias_add(tf.matmul(final_hidden, self.softmax_w), self.softmax_b)

		# loss and accuracy
		cross_entropy = tf.nn.sparse_softmax_cross_entropy_with_logits(logits=self.logits, labels=self.labels, name='cross_entropy_per_batch')
		self.loss = tf.reduce_mean(cross_entropy, name='cross_entropy_loss')

		correct_prediction = tf.to_int32(tf.nn.in_top_k(self.logits, self.labels, 1))
		self.true_count_op = tf.reduce_sum(correct_prediction)

		# get predictions and probs, shape [batch_size] tensors
		self.probs = tf.nn.softmax(self.logits)
		self.confidence, self.prediction = tf.nn.top_k(self.probs, k=1)
		self.confidence = tf.squeeze(self.confidence)
		self.prediction = tf.squeeze(self.prediction)

		# train on a batch
		self.lr = tf.Variable(1.0, trainable=False)
		if self.is_train:
			opt = tf.train.AdagradOptimizer(self.lr)
			tvars = tf.trainable_variables()
			grads, _ = tf.clip_by_global_norm(tf.gradients(self.loss, tvars), FLAGS.max_grad_norm)
			self.train_op = opt.apply_gradients(zip(grads, tvars))
		else:
			self.train_op = tf.no_op()
		return

	def assign_lr(self, session, lr_value):
		session.run(tf.assign(self.lr, lr_value))

	def assign_embedding(self, session, pretrained):
		session.run(tf.assign(self.w_emb, pretrained))

