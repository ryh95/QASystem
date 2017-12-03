import logging
import numpy as np
import os
from os.path import join as pjoin

logger = logging.getLogger(__name__)
logger.setLevel(logging.DEBUG)
logging.basicConfig(format='%(levelname)s:%(message)s', level=logging.DEBUG)

# class Config(object):
#     def __init__(self, data_dir, small_dir=None, small_val = None, sorted_data=True):
#             # self.val_answer_file = pjoin(data_dir, 'val.answer')
#         if sorted_data:
#             self.train_answer_span_file = pjoin(data_dir, 'train.span_sorted')
#             self.train_question_file = pjoin(data_dir, 'train.ids.question_sorted')
#             self.train_context_file = pjoin(data_dir, 'train.ids.context_sorted')
#         else:
#             if small_dir is None:
#                 # self.train_answer_file = pjoin(data_dir, 'train.answer')
#                 self.train_answer_span_file = pjoin(data_dir, 'train.span')
#                 self.train_question_file = pjoin(data_dir, 'train.ids.question')
#                 self.train_context_file = pjoin(data_dir, 'train.ids.context')
#             else:
#                 self.train_answer_span_file = pjoin(data_dir, 'train.span_' + str(small_dir))
#                 self.train_question_file = pjoin(data_dir, 'train.ids.question_' + str(small_dir))
#                 self.train_context_file = pjoin(data_dir, 'train.ids.context_' + str(small_dir))
#
#         if small_val is None:
#             self.val_answer_span_file = pjoin(data_dir, 'val.span')
#             self.val_question_file = pjoin(data_dir, 'val.ids.question')
#             self.val_context_file = pjoin(data_dir, 'val.ids.context')
#         else:
#             self.val_answer_span_file = pjoin(data_dir, 'val.span_')+str(small_val)
#             self.val_question_file = pjoin(data_dir, 'val.ids.question_')+str(small_val)
#             self.val_context_file = pjoin(data_dir, 'val.ids.context_')+str(small_val)

def load_glove_embeddings(embed_path):
    logger.info("Loading glove embedding...")
    glove = np.load(embed_path)['glove']
    logger.info("Dimension: {}".format(glove.shape[1]))
    logger.info("Vocabulary: {}" .format(glove.shape[0]))
    return glove

def add_paddings(sentence, max_length, n_features=1):
    mask = [True] * len(sentence)
    pad_len = max_length - len(sentence)
    if pad_len > 0:
        padded_sentence = sentence + [0] * pad_len
        mask += [False] * pad_len
    else:
        padded_sentence = sentence[:max_length]
        mask = mask[:max_length]
    return padded_sentence, mask

def preprocess_dataset(dataset, question_maxlen, context_maxlen):
    processed = []
    for q, q_len, c, c_len, ans in dataset:
        # add padding:
        q_padded, q_mask = add_paddings(q, question_maxlen)
        c_padded, c_mask = add_paddings(c, context_maxlen)
        processed.append([q_padded, q_mask, c_padded, c_mask, ans])
    return processed

def strip(x):
    return list(map(int, x.strip().split(" ")))

def read_data(config):
    train = []
    max_q_len = 0
    max_c_len = 0
    max_ans_end = 0
    logger.info("Loading training data from %s ...", config.train_question_file)
    # debug info
    # if os.path.exists(config.train_question_file):
    #     print('hello')

    with open(config.train_question_file, mode="r") as q_file, \
         open(config.train_context_file, mode="r") as c_file, \
         open(config.train_answer_span_file, mode="r") as a_file:
            for (q, c, a) in zip(q_file, c_file, a_file):
                question = strip(q)
                context = strip(c)
                answer = strip(a)
                max_ans_end = max(max_ans_end, answer[1])
                # ignore examples that have answers outside context_maxlen
                # if context_maxlen is not None and answer[1] >= context_maxlen:
                #     continue
                sample = [question, len(question), context, len(context), answer]
                train.append(sample)
                max_q_len = max(max_q_len, len(question))
                max_c_len = max(max_c_len, len(context))
    logger.info("Finish loading %d train data." % len(train))
    logger.info("Max question length %d" % max_q_len)
    logger.info("Max context length %d" % max_c_len)
    logger.info("Max answer end %d" % max_ans_end)

    val = []
    max_q_len_v = 0
    max_c_len_v = 0
    max_ans_end_v = 0
    logger.info("Loading validation data...")
    with open(config.val_question_file, mode="r") as q_file, \
         open(config.val_context_file, mode="r") as c_file, \
         open(config.val_answer_span_file, mode="r") as a_file:
            for (q, c, a) in zip(q_file, c_file, a_file):
                question = strip(q)
                context = strip(c)
                answer = strip(a)
                max_ans_end_v = max(max_ans_end_v, answer[1])
                # ignore examples that have answers outside context_maxlen
                # if context_maxlen is not None and answer[1] >= context_maxlen:
                #     continue
                sample = [question, len(question), context, len(context), answer]
                val.append(sample)
                max_q_len_v = max(max_q_len_v, len(question))
                max_c_len_v = max(max_c_len_v, len(context))
    logger.info("Finish loading %d validation data." % len(val))
    logger.info("Max question length %d" % max_q_len_v)
    logger.info("Max context length %d" % max_c_len_v)
    logger.info("Max answer end %d" % max_ans_end_v)

    # train = preprocess_dataset(train, question_maxlen, context_maxlen)
    # val = preprocess_dataset(val, question_maxlen, context_maxlen)
    # print(train)

    return {"training": train, "validation": val, "question_maxlen": max_q_len, "context_maxlen": max_c_len}


# if __name__ == '__main__':
#     read_data('../../data/squad', 100)