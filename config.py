import argparse
from os.path import join as pjoin
import os

base_dir = os.path.dirname(os.path.realpath(__file__))
data_dir = pjoin(base_dir,'data/squad')
train_answer_span_file = pjoin(data_dir,'train.span_sorted')
train_question_file = pjoin(data_dir, 'train.ids.question_sorted')
train_context_file = pjoin(data_dir, 'train.ids.context_sorted')
val_answer_span_file = pjoin(data_dir,'val.span')
val_question_file = pjoin(data_dir, 'val.ids.question')
val_context_file = pjoin(data_dir, 'val.ids.context')

def parse_args():
    parser = argparse.ArgumentParser(description='PyTorch TreeLSTM for Sentence Similarity on Dependency Trees')
    parser.add_argument("--lr",default=0.0005,help='learning rate')
    parser.add_argument("--max_gradient_norm",default=10.0,help='Clip gradients to this norm.')

    parser.add_argument("--dropout", default=0.20, help="Fraction of units randomly dropped on non-recurrent connections.")
    parser.add_argument("--batch_size", default=24, help="Batch size to use during training.")
    parser.add_argument("--epochs", default=25, help="Number of epochs to train.")
    parser.add_argument("--encoder_state_size", default=100, help="Size of each encoder model layer.")
    parser.add_argument("--decoder_state_size", default=100, help="Size of each decoder model layer.")
    parser.add_argument("--output_size", default=750, help="The output size of your model.")
    parser.add_argument("--embedding_size", default=100, help="Size of the pretrained vocabulary.")

    parser.add_argument("--base_dir",default=base_dir,help="base directory")
    parser.add_argument("--data_dir", default=data_dir,help= "SQuAD directory (default ./data/squad)")
    parser.add_argument("--train_answer_span_file",default=train_answer_span_file)
    parser.add_argument("--train_question_file",default=train_question_file)
    parser.add_argument("--train_context_file",default=train_context_file)
    parser.add_argument("--train_dir", default="train",
                               help="Training directory to save the model parameters (default: ./train).")
    parser.add_argument("--val_answer_span_file", default=val_answer_span_file)
    parser.add_argument("--val_question_file", default=val_question_file)
    parser.add_argument("--val_context_file", default=val_context_file)

    parser.add_argument("--load_train_dir", default="",
                               help="Training directory to load model parameters from to resume training (default: {train_dir}).")
    parser.add_argument("--log_dir", default="log", help="Path to store log and flag files (default: ./log)")
    parser.add_argument("--optimizer", default="adam", help="adam / sgd")
    parser.add_argument("--print_every", default=1, help="How many iterations to do per print.")
    parser.add_argument("--keep", default=0, help="How many checkpoints to keep, 0 indicates keep all.")
    parser.add_argument("--vocab_path", default="",
                               help="Path to vocab file (default: ./data/squad/vocab.dat)")
    parser.add_argument("--embed_path", default="",
                               help="Path to the trimmed GLoVe embedding (default: ./data/squad/glove.trimmed.{embedding_size}.npz)")

    parser.add_argument("--question_maxlen", default=None, help="Max length of question (default: 30")
    parser.add_argument("--context_maxlen",default= None, help="Max length of the context (default: 400)")
    parser.add_argument("--n_features", default=1, help="Number of features for each position in the sentence.")
    parser.add_argument("--log_batch_num", default=100,help= "Number of batches to write logs on tensorboard.")
    parser.add_argument("--decoder_hidden_size", default=100, help="Number of decoder_hidden_size.")
    parser.add_argument("--QA_ENCODER_SHARE", default=True,help= "QA_ENCODER_SHARE weights.")
    parser.add_argument("--tensorboard", default=False, help="Write tensorboard log or not.")
    parser.add_argument("--RE_TRAIN_EMBED", default=False, help="Max length of the context (default: 400)")
    parser.add_argument("--debug_train_samples", default=None, help="number of samples for debug (default: None)")
    parser.add_argument("--ema_weight_decay", default=0.999, help="exponential decay for moving averages ")
    parser.add_argument("--evaluate_sample_size", default=400, help="number of samples for evaluation (default: 400)")
    parser.add_argument("--model_selection_sample_size", default=1000,
                               help="number of samples for selecting best model (default: 1000)")
    parser.add_argument("--window_batch", default=3, help="window size / batch size")


    cuda_parser = parser.add_mutually_exclusive_group(required=False)
    cuda_parser.add_argument('--cuda', dest='cuda', action='store_true')
    cuda_parser.add_argument('--no-cuda', dest='cuda', action='store_false')
    parser.set_defaults(cuda=True)

    args = parser.parse_args()
    return args