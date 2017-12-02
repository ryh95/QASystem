from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import os
import json

import torch

from .config import parse_args
from .qa_model import Encoder, QASystem, Decoder
from os.path import join as pjoin
from os.path import exists as pexists

from .utils.data_reader import read_data, load_glove_embeddings

import logging

logging.basicConfig(level=logging.INFO)


def initialize_model(session, model, train_dir):
    # ckpt = tf.train.get_checkpoint_state(train_dir)
    ckpt = torch.load(train_dir)
    v2_path = ckpt.model_checkpoint_path + ".index" if ckpt else ""
    if ckpt and (pexists(ckpt.model_checkpoint_path) or pexists(v2_path)):
        logging.info("Reading model parameters from %s" % ckpt.model_checkpoint_path)
        # saver = tf.train.Saver()
        # saver.restore(session, ckpt.model_checkpoint_path)
        # TODO: load model

    else:
        logging.info("Created model with fresh parameters.")
        # session.run(tf.global_variables_initializer())
        # logging.info('Num params: %d' % sum(v.get_shape().num_elements() for v in tf.trainable_variables()))
        # TODO: how to calculate num parameters in a model?
    return model


def initialize_vocab(vocab_path):
    if pexists(vocab_path):
        rev_vocab = []
        # with tf.gfile.GFile(vocab_path, mode="rb") as f:
        with open(vocab_path,'r') as f:
            rev_vocab.extend(f.readlines())
        rev_vocab = [line.strip('\n') for line in rev_vocab]
        vocab = dict([(x, y) for (y, x) in enumerate(rev_vocab)])
        return vocab, rev_vocab
    else:
        raise ValueError("Vocabulary file %s not found.", vocab_path)


def get_normalized_train_dir(train_dir):
    """
    Adds symlink to {train_dir} from /tmp/cs224n-squad-train to canonicalize the
    file paths saved in the checkpoint. This allows the model to be reloaded even
    if the location of the checkpoint files has moved, allowing usage with CodaLab.
    This must be done on both train.py and qa_answer.py in order to work.
    """
    global_train_dir = train_dir
    #global_train_dir = '/tmp/cs224n-squad-train'
    #if os.path.exists(global_train_dir):
    #    os.unlink(global_train_dir)
    if not pexists(train_dir):
        os.makedirs(train_dir)
    #print('source: ',os.path.abspath(train_dir))
    #print('dst: ', global_train_dir)
    #os.symlink(os.path.abspath(train_dir), global_train_dir)
    return global_train_dir


def main():


    #dataset = read_data(FLAGS.data_dir, small_dir=None, small_val=None, \
    #    debug_train_samples=FLAGS.debug_train_samples, debug_val_samples=100, context_maxlen=FLAGS.context_maxlen)
    args = parse_args()
    dataset = read_data(args.data_dir)
    if args.context_maxlen is None:
        args.context_maxlen = dataset['context_maxlen']
    if args.question_maxlen is None:
        args.question_maxlen = dataset['question_maxlen']

    embed_path = args.embed_path or pjoin("data", "squad", "glove.trimmed.{}.npz".format(args.embedding_size))
    embeddings = load_glove_embeddings(embed_path)

    vocab_path = args.vocab_path or pjoin(args.data_dir, "vocab.dat")
    vocab, rev_vocab = initialize_vocab(vocab_path)
    args.vocab_size = len(vocab)

    qa = QASystem(embeddings, args)

    if not os.path.exists(args.log_dir):
        os.makedirs(args.log_dir)
    file_handler = logging.FileHandler(pjoin(args.log_dir, "log.txt"))
    logging.getLogger().addHandler(file_handler)

    print(vars(args))
    with open(os.path.join(args.log_dir, "flags.json"), 'w') as fout:
        json.dump(args.__flags, fout)

    gpu_options = torch.cuda.is_available()
    #gpu_options.allow_growth=True

    # with tf.Session(config=tf.ConfigProto(gpu_options=gpu_options)) as sess:
    load_train_dir = get_normalized_train_dir(args.load_train_dir or args.train_dir)
    initialize_model(qa, load_train_dir)

    save_train_dir = get_normalized_train_dir(args.train_dir)
    qa.train(dataset, save_train_dir, rev_vocab)


if __name__ == "__main__":
    main()
