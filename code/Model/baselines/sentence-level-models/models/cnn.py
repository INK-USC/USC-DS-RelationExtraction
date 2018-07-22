__author__ = 'Maosen'
import torch
import torch.nn as nn
import torch.nn.functional as F
import utils
from utils import pos2id, ner2id
import sys
from tqdm import tqdm

class CNN(nn.Module):
	def __init__(self, args, rel2id, word_emb=None):
		super(CNN, self).__init__()
		# arguments
		hidden, vocab_size, emb_dim, pos_dim, ner_dim, position_dim, attn_dim, window_size, num_layers, dropout = \
			args.hidden, args.vocab_size, args.emb_dim, args.pos_dim, args.ner_dim, \
			args.position_dim, args.attn_dim, args.window_size, args.num_layers, args.dropout

		# embeddings
		if word_emb is not None:
			assert vocab_size, emb_dim == word_emb.shape
			self.word_emb = nn.Embedding(vocab_size, emb_dim, padding_idx=utils.PAD_ID, _weight=torch.from_numpy(word_emb).float())
			# self.word_emb.weight.data.copy_(torch.from_numpy(word_emb))
			# self.word_emb.weight.requires_grad = False
		else:
			self.word_emb = nn.Embedding(vocab_size, emb_dim, padding_idx=utils.PAD_ID)
			self.word_emb.weight.data[1:, :].uniform_(-1.0, 1.0)

		self.pos_dim = pos_dim
		self.ner_dim = ner_dim
		self.hidden = hidden
		if pos_dim > 0:
			self.pos_emb = nn.Embedding(len(pos2id), pos_dim, padding_idx=utils.PAD_ID)
			self.pos_emb.weight.data[1:, :].uniform_(-1.0, 1.0)
		if ner_dim > 0:
			self.ner_emb = nn.Embedding(len(ner2id), ner_dim, padding_idx=utils.PAD_ID)
			self.ner_emb.weight.data[1:, :].uniform_(-1.0, 1.0)

		if position_dim > 0:
			self.position_emb = nn.Embedding(utils.MAXLEN*2, position_dim)
			self.position_emb.weight.data.uniform_(-1.0, 1.0)

		# CNN
		input_size = emb_dim + position_dim*2
		self.cnn = torch.nn.Conv1d(in_channels=1, out_channels=hidden, kernel_size=(input_size, window_size), stride=1)
		self.pooling = torch.nn.AdaptiveMaxPool1d(1)
		self.dropout = nn.Dropout(dropout)

		# linear parameters of Position-aware attention
		# feat_dim = hidden*2 + position_dim*2
		# self.attn_dim = attn_dim
		# self.feat_dim = feat_dim

		self.flinear = nn.Linear(hidden, len(rel2id))
		self.flinear.weight.data.normal_(std=0.001)



	def forward(self, inputs):
		words, pos, ner, subj_pos, obj_pos = inputs
		# pos_subj and pos_obj are relative position to subject/object
		batch, maxlen = words.size()

		masks = torch.eq(words, utils.PAD_ID)
		seq_lens = masks.eq(utils.PAD_ID).long().sum(1).squeeze().tolist()

		emb_words = self.word_emb(words)
		# emb_pos = self.pos_emb(pos)
		# emb_ner = self.ner_emb(ner)

		emb_subj_pos = self.position_emb(subj_pos + utils.MAXLEN)
		emb_obj_pos = self.position_emb(obj_pos + utils.MAXLEN)

		input = torch.cat([emb_words, emb_subj_pos, emb_obj_pos], dim=2).contiguous()
		input = torch.transpose(input, 1, 2)  # [batch, input_size, seq_len]
		input = torch.unsqueeze(input, dim=1)  # [batch, 1, input_size, seq_len ]

		output = self.cnn(input)  # [batch, hidden, 1, time']
		output = torch.squeeze(output, 2)  # [batch, hidden, time']
		output = self.pooling(output)  # [batch, hidden, 1]
		output = torch.squeeze(output, 2)  # [batch, hidden]
		output = torch.nn.functional.tanh(output)

		output = self.dropout(output)

		logits = self.flinear(output)

		return logits





