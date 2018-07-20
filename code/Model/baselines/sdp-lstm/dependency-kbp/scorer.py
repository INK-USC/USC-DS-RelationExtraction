#!/usr/bin/env python

import sys
from collections import Counter

def score(key_file, pred_files, f_measure=1, verbose=False):
    f_measure = float(f_measure)
    key = [str(line).rstrip('\n') for line in open(key_file)]
    predictions = [[str(line).rstrip('\n').split('\t') for line in open(str(prediction))] for prediction in pred_files]

    # Check that the lengths match
    for prediction in predictions:
        if len(prediction) != len(key):
            print("Key and prediction file must have same number of elements: %d in key vs %d in prediction" % (len(key), len(prediction)))
            quit(1)

    # The sufficient statistics for computing accuracy scores
    correct_by_relation = Counter()
    guessed_by_relation = Counter()
    gold_by_relation    = Counter()

    # Loop over the data to compute a score
    for row in range(len(key)):
        gold = key[row]
        guess = 'no_relation'
        guess_score = -1.0;
        for system in predictions:
            system_guess = system[row][0]
            system_score = float(system[row][1])
            if system_guess != 'no_relation' and system_score > guess_score:
                guess = system_guess
                guess_score = system_score

        if gold == 'no_relation' and guess == 'no_relation':
            pass
        elif gold == 'no_relation' and guess != 'no_relation':
            guessed_by_relation[guess] += 1
        elif gold != 'no_relation' and guess == 'no_relation':
            gold_by_relation[gold] += 1
        elif gold != 'no_relation' and guess != 'no_relation':
            guessed_by_relation[guess] += 1
            gold_by_relation[gold] += 1
            if gold == guess:
                correct_by_relation[guess] += 1

    # Print verbose information
    if verbose:
        print "Per-relation statistics:"
        relations = gold_by_relation.keys()
        longest_relation = 0
        for relation in sorted(relations):
            longest_relation = max(len(relation), longest_relation)
        for relation in sorted(relations):
            # (compute the score)
            correct = correct_by_relation[relation]
            guessed = guessed_by_relation[relation]
            gold    = gold_by_relation[relation]
            prec = 1.0
            if guessed > 0:
                prec = float(correct) / float(guessed)
            recall = 0.0
            if gold > 0:
                recall = float(correct) / float(gold)
            f = 0.0
            if prec + recall > 0:
                f = (1.0+f_measure**2) * prec * recall / (f_measure**2 * prec + recall)
            # (print the score)
            sys.stdout.write(("{:<" + str(longest_relation) + "}").format(relation))
            sys.stdout.write("  P: ")
            if prec < 0.1: sys.stdout.write(' ')
            if prec < 1.0: sys.stdout.write(' ')
            sys.stdout.write("{:.2%}".format(prec))
            sys.stdout.write("  R: ")
            if recall < 0.1: sys.stdout.write(' ')
            if recall < 1.0: sys.stdout.write(' ')
            sys.stdout.write("{:.2%}".format(recall))
            sys.stdout.write("  F-%g: " % f_measure)
            if f < 0.1: sys.stdout.write(' ')
            if f < 1.0: sys.stdout.write(' ')
            sys.stdout.write("{:.2%}".format(f))
            sys.stdout.write("  #: %d" % gold)
            sys.stdout.write("\n")
        print ""


    # Print the aggregate score
    if verbose:
        print "Final Score (micro P, R, F-%g):" % f_measure
    prec_micro = 1.0
    if sum(guessed_by_relation.values()) > 0:
        prec_micro   = float(sum(correct_by_relation.values())) / float(sum(guessed_by_relation.values()))
    recall_micro = 0.0
    if sum(gold_by_relation.values()) > 0:
        recall_micro = float(sum(correct_by_relation.values())) / float(sum(gold_by_relation.values()))
    f_micro = 0.0
    if prec_micro + recall_micro > 0.0:
        f_micro = (1.0 + f_measure**2) * prec_micro * recall_micro / (f_measure**2 * prec_micro + recall_micro)
    # print "\t%.3f\t%.3f\t%.3f" % (prec_micro*100, recall_micro*100, f_micro*100)
    return prec_micro, recall_micro, f_micro
