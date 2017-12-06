from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import os
import json

import torch
from torch import nn
from torch import optim
# from config import parse_args
# from .qa_model import Encoder, QASystem, Decoder

from os.path import join as pjoin
from os.path import exists as pexists

# from .utils.data_reader import read_data, load_glove_embeddings

import logging

from config import parse_args
# from my_code.qa_model import QASystem
from my_code.model import QASystem
from my_code.trainer import Trainer
from my_code.utils.data_reader import read_data, load_glove_embeddings

logging.basicConfig(level=logging.INFO)


def load_ckpt(model, args):
    ckpt = pjoin(args.ckpt_dir,args.expname)
    if pexists(args.ckpt_dir):
        if pexists(ckpt):
            model_para = torch.load(ckpt)
            logging.info("Reading model parameters from %s" % ckpt)
            # saver = tf.train.Saver()
            # saver.restore(session, ckpt.model_checkpoint_path)
            # TODO: load model
        else:
            logging.info("Try to load ckpt but didn't found, start with fresh paras")
    else:
        os.makedirs(args.ckpt_dir)
        logging.info("Checkpoint directory doesn't exist, create it")

    return model

def print_model_detail(model):
    logging.info("Print model")
    logging.info(model)
    all_paras = 0
    for para in model.parameters():
        all_paras += torch.numel(para.data)
    logging.info("Num paras: {}".format(all_paras))

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


if __name__ == "__main__":
    # dataset = read_data(FLAGS.data_dir, small_dir=None, small_val=None, \
    #    debug_train_samples=FLAGS.debug_train_samples, debug_val_samples=100, context_maxlen=FLAGS.context_maxlen)
    args = parse_args()
    dataset = read_data(args)
    # if args.context_maxlen is None:
    #     args.context_maxlen = dataset['context_maxlen']
    # if args.question_maxlen is None:
    #     args.question_maxlen = dataset['question_maxlen']
    #
    embed_path = args.embed_path or pjoin(args.data_dir, "glove.trimmed.{}.npz".format(args.embedding_size))
    # embeddings = load_glove_embeddings(embed_path)
    embeddings = None
    #
    vocab_path = args.vocab_path or pjoin(args.data_dir, "vocab.dat")
    vocab, rev_vocab = initialize_vocab(vocab_path)
    args.vocab_size = len(vocab)

    qa = QASystem(embeddings, args)
    if not args.RE_TRAIN_EMBED:
        qa.emb_layer.weight = nn.Parameter(torch.from_numpy(embeddings))
        qa.emb_layer.weight.requires_grad = False
    qa_parameters = filter(lambda p: p.requires_grad, qa.parameters())

    logging.info('model initialized!')
    #
    # if not os.path.exists(args.log_dir):
    #     os.makedirs(args.log_dir)
    # file_handler = logging.FileHandler(pjoin(args.log_dir, "log.txt"))
    # logging.getLogger().addHandler(file_handler)
    #
    # print(vars(args))
    # with open(os.path.join(args.log_dir, "flags.json"), 'w') as fout:
    #     json.dump(args.__flags, fout)
    #
    # gpu_options = torch.cuda.is_available()
    # # gpu_options.allow_growth=True
    #
    # # with tf.Session(config=tf.ConfigProto(gpu_options=gpu_options)) as sess:

    qa = load_ckpt(qa, args)
    print_model_detail(qa)

    criterion = nn.NLLLoss()

    if args.optimizer=='adam':
        optimizer   = optim.Adam(qa_parameters, lr=args.lr)
    elif args.optimizer=='adagrad':
        optimizer   = optim.Adagrad(qa_parameters, lr=args.lr)
    elif args.optimizer=='sgd':
        optimizer   = optim.SGD(qa_parameters, lr=args.lr)

    trainer = Trainer(args, qa, criterion, optimizer)
    trainer.train(dataset,rev_vocab)
