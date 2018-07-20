import data_utils

def _get_feed_dict_for_others(model, x_batch, y_batch, x_lens, use_pos=True, use_ner=True, use_deprel=True):
    feed = {model.word_inputs:x_batch[data_utils.WORD_FIELD], model.labels:y_batch, model.seq_lens:x_lens}
    if use_pos:
        feed[model.pos_inputs] = x_batch[data_utils.POS_FIELD]
    if use_ner:
        feed[model.ner_inputs] = x_batch[data_utils.NER_FIELD]
    if use_deprel:
        feed[model.deprel_inputs] = x_batch[data_utils.DEPREL_FIELD]
    return feed

def _get_feed_dict_for_sprnn(model, x_batch, y_batch, x_lens, use_pos=True, use_ner=True, use_deprel=True):
    all_fields = [data_utils.WORD_FIELD, data_utils.POS_FIELD, data_utils.NER_FIELD, data_utils.DEPREL_FIELD, data_utils.ROOT_FIELD]
    # convert root sequence into a list of ROOT index's
    root_seq = x_batch[data_utils.ROOT_FIELD]
    max_len = len(root_seq[0])
    root_index_seq = []
    for s in root_seq:
        root_index_seq.append(s.index('ROOT')) # find the root position
    # for each sequence type, divide the sequence based on the index sequence and pad to max length
    # left batch: from subject to root; right_batch: from object to root
    left_batch = {}
    right_batch = {}
    for k in all_fields: # each value is a batch of different sequence
        batch = x_batch[k]
        left_batch_seq, right_batch_seq = [], []
        assert(len(batch) == len(root_index_seq))
        for s, idx, length in zip(batch, root_index_seq, x_lens):
            l = s[:idx+1]
            r = s[idx:length][::-1] # remember to inverse the right batch so that ROOT is at the end
            left_batch_seq.append(l + [data_utils.PAD_ID] * (max_len-len(l))) # pad
            right_batch_seq.append(r + [data_utils.PAD_ID] * (max_len-len(r)))
        left_batch[k], right_batch[k] = left_batch_seq, right_batch_seq
    # calculate left and right seq lengths
    left_lens = [idx + 1 for idx in root_index_seq]
    right_lens = [(l - idx) for idx, l in zip(root_index_seq, x_lens)]
    # now create the feed dict
    feed = {model.word_inputs: left_batch[data_utils.WORD_FIELD] + right_batch[data_utils.WORD_FIELD], \
        model.seq_lens: left_lens + right_lens, model.labels: y_batch}
    if use_pos:
        feed[model.pos_inputs] = left_batch[data_utils.POS_FIELD] + right_batch[data_utils.POS_FIELD]
    if use_ner:
        feed[model.ner_inputs] = left_batch[data_utils.NER_FIELD] + right_batch[data_utils.NER_FIELD]
    if use_deprel:
        feed[model.deprel_inputs] = left_batch[data_utils.DEPREL_FIELD] + right_batch[data_utils.DEPREL_FIELD]
    return feed
