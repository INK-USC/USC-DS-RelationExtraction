import json


def is_overlap(a, b):
    if set(a) & set(b):
        return True
    else:
        return False


def process_file(in_file, out_file, rel_types, training=True):
    for line in in_file:
        data = json.loads(line.strip())
        sent_tokens_lower = [t.lower() for t in data['tokens']]
        rms = data['relationMentions']
        for rm in rms:
            relations = rm['labels']
            em1_start = rm['em1Start']
            em1_end = rm['em1End']
            em2_start = rm['em2Start']
            em2_end = rm['em2End']
            if is_overlap(list(range(em1_start, em1_end)), list(range(em2_start, em2_end))):
                continue
            new_sent_str = ""
            for idx, token in enumerate(sent_tokens_lower):
                if em1_start <= idx < em1_end - 1 or em2_start <= idx < em2_end - 1:
                    new_sent_str += token + '_'
                else:
                    new_sent_str += token + ' '
            em1 = '_'.join(sent_tokens_lower[em1_start:em1_end])
            em2 = '_'.join(sent_tokens_lower[em2_start:em2_end])
            for r in relations:
                if training:
                    if r not in rel_types:
                        rel_types[r] = len(rel_types)
                out_file.write('_\t_\t' + em1 + '\t' + em2 + '\t' + r + '\t' + new_sent_str + '\n')
    return rel_types

def process(fin1, fin2, fout):
    rl = {}
    for line in fin2:
        wl = line.strip().split(' ')
        rl[wl[0]] = int(wl[1])

    for line in fin1:
        wl = line.strip().split('\t')
        if wl[4] not in rl:
            wl[4] = 'None'
        sent = ' '.join(wl[5].split())
        fout.write('0\t'+sent+'\t'+wl[2]+'\t'+wl[3]+'\t'+str(rl[wl[4]])+'\n')

if __name__ == '__main__':
    dataset = 'TACRED'
    if dataset == 'TACRED':
        USE_PROVIDED_DEV = True
    else:
        USE_PROVIDED_DEV = False

    with open("data/" + dataset + "/train_new.json", encoding='utf-8') as train_file, \
            open("data/" + dataset + "/test_new.json", encoding='utf-8') as test_file, \
            open("data/" + dataset + "/relation2id.txt", 'w', encoding='utf-8') as relation_file, \
            open("data/" + dataset + "/train_original.txt", 'w', encoding='utf-8') as train_out_file, \
            open("data/" + dataset + "/test_original.txt", 'w', encoding='utf-8') as test_out_file:
        rel_types = {}
        rel_types = process_file(train_file, train_out_file, rel_types, training=True)
        process_file(test_file, test_out_file, rel_types, training=False)
        if USE_PROVIDED_DEV:
            with open("data/" + dataset + "/dev_new.json", encoding='utf-8') as dev_file,\
                    open("data/" + dataset + "/dev_original.txt","w", encoding='utf-8') as dev_out_file:
                process_file(dev_file, dev_out_file, rel_types, training=False)
        for relation in rel_types:
            relation_file.write(relation + ' ' + str(rel_types[relation]) + '\n')

    fin1 = open('data/' + dataset + '/train_original.txt', encoding='utf-8')
    fin2 = open('data/' + dataset + '/relation2id.txt', encoding='utf-8')
    fout = open('data/' + dataset + '/train.txt', 'w', encoding='utf-8')

    process(fin1, fin2, fout)

    fin1 = open('data/' + dataset + '/test_original.txt', encoding='utf-8')
    fin2 = open('data/' + dataset + '/relation2id.txt', encoding='utf-8')
    fout = open('data/' + dataset + '/test.txt', 'w', encoding='utf-8')

    process(fin1, fin2, fout)

    if USE_PROVIDED_DEV:
        fin1 = open('data/' + dataset + '/dev_original.txt', encoding='utf-8')
        fin2 = open('data/' + dataset + '/relation2id.txt', encoding='utf-8')
        fout = open('data/' + dataset + '/dev.txt', 'w', encoding='utf-8')
        
        process(fin1, fin2, fout)



