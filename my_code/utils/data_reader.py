import logging
import numpy as np
from os.path import join as pjoin
from tensorflow.python.platform import gfile

logger = logging.getLogger(__name__)
logger.setLevel(logging.DEBUG)
logging.basicConfig(format='%(levelname)s:%(message)s', level=logging.DEBUG)

class Config(object):
    def __init__(self, data_dir, small_dir=None, small_val = None, sorted_data=True):
            # self.val_answer_file = pjoin(data_dir, 'val.answer')
        if sorted_data:
            self.train_answer_span_file = pjoin(data_dir, 'train.span_sorted')
            self.train_question_file = pjoin(data_dir, 'train.ids.question_sorted')
            self.train_context_file = pjoin(data_dir, 'train.ids.context_sorted')
        else:
            if small_dir is None:
                # self.train_answer_file = pjoin(data_dir, 'train.answer')
                self.train_answer_span_file = pjoin(data_dir, 'train.span')
                self.train_question_file = pjoin(data_dir, 'train.ids.question')
                self.train_context_file = pjoin(data_dir, 'train.ids.context')
            else:
                self.train_answer_span_file = pjoin(data_dir, 'train.span_' + str(small_dir))
                self.train_question_file = pjoin(data_dir, 'train.ids.question_' + str(small_dir))
                self.train_context_file = pjoin(data_dir, 'train.ids.context_' + str(small_dir))

        if small_val is None:
            self.val_answer_span_file = pjoin(data_dir, 'val.span')
            self.val_question_file = pjoin(data_dir, 'val.ids.question')
            self.val_context_file = pjoin(data_dir, 'val.ids.context')
        else:
            self.val_answer_span_file = pjoin(data_dir, 'val.span_')+str(small_val)
            self.val_question_file = pjoin(data_dir, 'val.ids.question_')+str(small_val)
            self.val_context_file = pjoin(data_dir, 'val.ids.context_')+str(small_val)

def load_glove_embeddings(embed_path):
    logger.info("Loading glove embedding...")
    glove = np.load(embed_path)['glove']
    logger.info("Dimension: {}".format(glove.shape[1]))
    logger.info("Vocabulary: {}" .format(glove.shape[0]))
    return glove



def strip(x):
    return map(int, x.strip().split(" "))

def read_data(data_dir, small_dir=None, small_val = None, question_maxlen=None, context_maxlen=None, debug_train_samples=None, debug_val_samples=None):
    config = Config(data_dir, small_dir=small_dir, small_val = small_val)

    train = []
    max_q_len = 0
    max_c_len = 0
    max_ans_end = 0
    logger.info("Loading training data from %s ...", config.train_question_file)
    with gfile.GFile(config.train_question_file, mode="rb") as q_file, \
         gfile.GFile(config.train_context_file, mode="rb") as c_file, \
         gfile.GFile(config.train_answer_span_file, mode="rb") as a_file:
            # Todo: why use sorted files ? ids.question_sorted ids.context_sorted span_sorted
            # Reason : This is because of padding mechanism according to create_feed_dict function
            for (q, c, a) in zip(q_file, c_file, a_file):
                question = strip(q)
                context = strip(c)
                answer = strip(a)
                max_ans_end = max(max_ans_end, answer[1])
                # ignore examples that have answers outside context_maxlen
                if context_maxlen is not None and answer[1] >= context_maxlen:
                    continue
                sample = [question, len(question), context, len(context), answer]
                train.append(sample)
                max_q_len = max(max_q_len, len(question))
                max_c_len = max(max_c_len, len(context))
                if debug_train_samples is not None and len(train) == debug_train_samples:
                    break
    logger.info("Finish loading %d train data." % len(train))
    logger.info("Max question length %d" % max_q_len)
    logger.info("Max context length %d" % max_c_len)
    logger.info("Max answer end %d" % max_ans_end)

    val = []
    max_q_len_v = 0
    max_c_len_v = 0
    max_ans_end_v = 0
    logger.info("Loading validation data...")
    with gfile.GFile(config.val_question_file, mode="rb") as q_file, \
         gfile.GFile(config.val_context_file, mode="rb") as c_file, \
         gfile.GFile(config.val_answer_span_file, mode="rb") as a_file:
            for (q, c, a) in zip(q_file, c_file, a_file):
                question = strip(q)
                context = strip(c)
                answer = strip(a)
                max_ans_end_v = max(max_ans_end_v, answer[1])
                # ignore examples that have answers outside context_maxlen
                if context_maxlen is not None and answer[1] >= context_maxlen:
                    continue
                sample = [question, len(question), context, len(context), answer]
                val.append(sample)
                max_q_len_v = max(max_q_len_v, len(question))
                max_c_len_v = max(max_c_len_v, len(context))
                if debug_val_samples is not None and len(val) == debug_val_samples:
                    break
    logger.info("Finish loading %d validation data." % len(val))
    logger.info("Max question length %d" % max_q_len_v)
    logger.info("Max context length %d" % max_c_len_v)
    logger.info("Max answer end %d" % max_ans_end_v)

    if question_maxlen is None:
        question_maxlen = max(max_q_len_v, max_q_len)
    if context_maxlen is None:
        context_maxlen = max(max_c_len_v, max_c_len)

    # train = preprocess_dataset(train, question_maxlen, context_maxlen)
    # val = preprocess_dataset(val, question_maxlen, context_maxlen)
    # print(train)

    return {"training": train, "validation": val, "question_maxlen": max_q_len, "context_maxlen": max_c_len}


if __name__ == '__main__':
    read_data('../../data/squad', 100)