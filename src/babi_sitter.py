from __future__ import print_function
from functools import reduce
import re
import numpy as np

from keras.preprocessing.sequence import pad_sequences


def tokenize(sent):
    '''Return the tokens of a sentence including punctuation.
    >>> tokenize('Bob dropped the apple. Where is the apple?')
    ['Bob', 'dropped', 'the', 'apple', '.', 'Where', 'is', 'the', 'apple', '?']
    '''
    return [x.strip() for x in re.split('(\W+)?', sent) if x.strip()]


def parse_stories(lines, only_supporting=False):
    '''Parse stories provided in the bAbi tasks format
    If only_supporting is true, only the sentences that support the answer are kept.
    '''
    data = []
    story = []
    for line in lines:
        line = line.decode('utf-8').strip()
        nid, line = line.split(' ', 1)
        nid = int(nid)
        if nid == 1:
            story = []
        if '\t' in line:
            q, a, supporting = line.split('\t')
            q = tokenize(q)
            substory = None
            if only_supporting:
                # Only select the related substory
                supporting = map(int, supporting.split())
                substory = [story[i - 1] for i in supporting]
            else:
                # Provide all the substories
                substory = [x for x in story if x]
            data.append((substory, q, a))
            story.append('')
        else:
            sent = tokenize(line)
            story.append(sent)
    return data


def get_stories(f, only_supporting=False, max_length=None):
    '''Given a file name, read the file, retrieve the stories, and then convert the sentences into a single story.
    If max_length is supplied, any stories longer than max_length tokens will be discarded.
    '''
    data = parse_stories(f.readlines(), only_supporting=only_supporting)
    flatten = lambda data: reduce(lambda x, y: x + y, data)
    data = [(flatten(story), q, answer) for story, q, answer in data if not max_length or len(flatten(story)) < max_length]
    return data


def vectorize_stories(data, word_idx, story_maxlen, query_maxlen):
    X = []
    Xq = []
    Y = []
    for story, query, answer in data:
        x = [word_idx[w] for w in story]
        xq = [word_idx[w] for w in query]
        y = np.zeros(len(word_idx) + 1)  # let's not forget that index 0 is reserved
        y[word_idx[answer]] = 1
        X.append(x)
        Xq.append(xq)
        Y.append(y)
    return pad_sequences(X, maxlen=story_maxlen), pad_sequences(Xq, maxlen=query_maxlen), np.array(Y)


QFILE = {1: 'tasks_1-20_v1-2/en-10k/qa1_single-supporting-fact_train.txt',
         2: 'tasks_1-20_v1-2/en-10k/qa2_two-supporting-facts_train.txt',
         3: 'tasks_1-20_v1-2/en-10k/qa3_three-supporting-facts_train.txt',
         4: 'tasks_1-20_v1-2/en-10k/qa4_two-arg-relations_train.txt',
         5: 'tasks_1-20_v1-2/en-10k/qa5_three-arg-relations_train.txt',
         6: 'tasks_1-20_v1-2/en-10k/qa6_yes-no-questions_train.txt',
         7: 'tasks_1-20_v1-2/en-10k/qa7_counting_train.txt',
         8: 'tasks_1-20_v1-2/en-10k/qa8_lists-sets_train.txt',
         9: 'tasks_1-20_v1-2/en-10k/qa9_simple-negation_train.txt',
         10: 'tasks_1-20_v1-2/en-10k/qa10_indefinite-knowledge_train.txt',
         11: 'tasks_1-20_v1-2/en-10k/qa11_basic-coreference_train.txt',
         12: 'tasks_1-20_v1-2/en-10k/qa12_conjunction_train.txt',
         13: 'tasks_1-20_v1-2/en-10k/qa13_compound-coreference_train.txt',
         14: 'tasks_1-20_v1-2/en-10k/qa14_time-reasoning_train.txt',
         15: 'tasks_1-20_v1-2/en-10k/qa15_basic-deduction_train.txt',
         16: 'tasks_1-20_v1-2/en-10k/qa16_basic-induction_train.txt',
         17: 'tasks_1-20_v1-2/en-10k/qa17_positional-reasoning_train.txt',
         18: 'tasks_1-20_v1-2/en-10k/qa18_size-reasoning_train.txt',
         19: 'tasks_1-20_v1-2/en-10k/qa19_path-finding_train.txt',
         20: 'tasks_1-20_v1-2/en-10k/qa20_agents-motivations_train.txt'}
