import argparse

def parse_args():
    parser = argparse.ArgumentParser(description='PyTorch TreeLSTM for Sentence Similarity on Dependency Trees')
    parser.add_argument("--lr",default=0.0005,help='learning rate')
    parser.add_argument("--max_gradient_norm",default=10.0,help='Clip gradients to this norm.')

    parser.add_argument("--dropout", 0.20, "Fraction of units randomly dropped on non-recurrent connections.")
    parser.add_argument("--batch_size", 24, "Batch size to use during training.")
    parser.add_argument("--epochs", 25, "Number of epochs to train.")
    parser.add_argument("--encoder_state_size", 100, "Size of each encoder model layer.")
    parser.add_argument("--decoder_state_size", 100, "Size of each decoder model layer.")
    parser.add_argument("--output_size", 750, "The output size of your model.")
    parser.add_argument("--embedding_size", 100, "Size of the pretrained vocabulary.")
    parser.add_argument("--data_dir", "data/squad", "SQuAD directory (default ./data/squad)")
    parser.add_argument("--train_dir", "train",
                               "Training directory to save the model parameters (default: ./train).")
    parser.add_argument("--load_train_dir", "",
                               "Training directory to load model parameters from to resume training (default: {train_dir}).")
    parser.add_argument("--log_dir", "log", "Path to store log and flag files (default: ./log)")
    parser.add_argument("--optimizer", "adam", "adam / sgd")
    parser.add_argument("--print_every", 1, "How many iterations to do per print.")
    parser.add_argument("--keep", 0, "How many checkpoints to keep, 0 indicates keep all.")
    parser.add_argument("--vocab_path", "data/squad/vocab.dat",
                               "Path to vocab file (default: ./data/squad/vocab.dat)")
    parser.add_argument("--embed_path", "",
                               "Path to the trimmed GLoVe embedding (default: ./data/squad/glove.trimmed.{embedding_size}.npz)")

    parser.add_argument("--question_maxlen", None, "Max length of question (default: 30")
    parser.add_argument("--context_maxlen", None, "Max length of the context (default: 400)")
    parser.add_argument("--n_features", 1, "Number of features for each position in the sentence.")
    parser.add_argument("--log_batch_num", 100, "Number of batches to write logs on tensorboard.")
    parser.add_argument("--decoder_hidden_size", 100, "Number of decoder_hidden_size.")
    parser.add_argument("--QA_ENCODER_SHARE", True, "QA_ENCODER_SHARE weights.")
    parser.add_argument("--tensorboard", False, "Write tensorboard log or not.")
    parser.add_argument("--RE_TRAIN_EMBED", False, "Max length of the context (default: 400)")
    parser.add_argument("--debug_train_samples", None, "number of samples for debug (default: None)")
    parser.add_argument("--ema_weight_decay", 0.999, "exponential decay for moving averages ")
    parser.add_argument("--evaluate_sample_size", 400, "number of samples for evaluation (default: 400)")
    parser.add_argument("--model_selection_sample_size", 1000,
                               "number of samples for selecting best model (default: 1000)")
    parser.add_argument("--window_batch", 3, "window size / batch size")


    cuda_parser = parser.add_mutually_exclusive_group(required=False)
    cuda_parser.add_argument('--cuda', dest='cuda', action='store_true')
    cuda_parser.add_argument('--no-cuda', dest='cuda', action='store_false')
    parser.set_defaults(cuda=True)

    args = parser.parse_args()
    return args