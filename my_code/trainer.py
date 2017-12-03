import numpy as np
from tqdm import tqdm
import torch
from torch.autograd import Variable as Var
# from utils import map_label_to_target

class Trainer(object):
    def __init__(self, args, model, criterion, optimizer):
        super(Trainer, self).__init__()
        self.args       = args
        self.model      = model
        self.criterion  = criterion
        self.optimizer  = optimizer
        self.epoch      = 0

    # helper function for training
    def train(self, dataset, train_dir, vocab):
        # tic = time.time()
        # TODO:pytorch get trainable paras
        # params = tf.trainable_variables()
        # num_params = sum(map(lambda t: np.prod(tf.shape(t.value()).eval()), params))
        # toc = time.time()
        # logging.info("Number of params: %d (retreival took %f secs)" % (num_params, toc - tic))

        training_set = dataset['training']  # [question, len(question), context, len(context), answer]
        validation_set = dataset['validation']
        f1_best = 0
        # TODO: finish tensorboard in pytorch
        # if self.config.tensorboard:
        #     train_writer_dir = self.config.log_dir + '/train/' # + datetime.datetime.now().strftime('%m-%d_%H-%M-%S')
        #     self.train_writer = tf.summary.FileWriter(train_writer_dir, session.graph)
        # for epoch in range(self.config.epochs):
            # logging.info("=" * 10 + " Epoch %d out of %d " + "=" * 10, epoch + 1, self.config.epochs)

            # TODO run training
            # score = self.run_epoch(session, epoch, training_set, vocab, validation_set,
            #                        sample_size=self.config.evaluate_sample_size)
            # logging.info("-- validation --")
            # TODO validate answer
            # self.validate(session, validation_set)

            # TODO get f1 and em
            # f1, em = self.evaluate_answer(session, validation_set, vocab,
            #                               sample=self.config.model_selection_sample_size, log=True)

            # TODO Saving the model
            # if f1 > f1_best:
            #     f1_best = f1
            #     saver = tf.train.Saver()
            #     # TODO torch save model
            #     # saver.save(session, train_dir + '/fancier_model')
            #     logging.info('New best f1 in val set')
            #     logging.info('')
            # saver = tf.train.Saver()
            # TODO torch save model
            # saver.save(session, train_dir + '/fancier_model_' + str(epoch))

    # helper function for testing
    # def test(self, dataset):
    #     self.model.eval()
    #     loss = 0
    #     predictions = torch.zeros(len(dataset))
    #     indices = torch.arange(1,dataset.num_classes+1)
    #     for idx in tqdm(range(len(dataset)),desc='Testing epoch  '+str(self.epoch)+''):
    #         ltree,lsent,rtree,rsent,label = dataset[idx]
    #         linput, rinput = Var(lsent, volatile=True), Var(rsent, volatile=True)
    #         target = Var(map_label_to_target(label,dataset.num_classes), volatile=True)
    #         if self.args.cuda:
    #             linput, rinput = linput.cuda(), rinput.cuda()
    #             target = target.cuda()
    #         output = self.model(ltree,linput,rtree,rinput)
    #         err = self.criterion(output, target)
    #         loss += err.data[0]
    #         output = output.data.squeeze().cpu()
    #         predictions[idx] = torch.dot(indices, torch.exp(output))
    #     return loss/len(dataset), predictions


    def create_feed_dict(self, question_batch, question_len_batch, context_batch, context_len_batch, JX=10, JQ=10, answer_batch=None, is_train = True):
        data_dict = {}
        JQ = np.max(question_len_batch)
        JX = np.max(context_len_batch)
        # print('This batch len: JX = %d, JQ = %d', JX, JQ)
        def add_paddings(sentence, max_length):
            mask = [True] * len(sentence)
            pad_len = max_length - len(sentence)
            if pad_len > 0:
                padded_sentence = sentence + [0] * pad_len
                mask += [False] * pad_len
            else:
                padded_sentence = sentence[:max_length]
                mask = mask[:max_length]
            return padded_sentence, mask

        def padding_batch(data, max_len):
            padded_data = []
            padded_mask = []
            for sentence in data:
                d, m = add_paddings(sentence, max_len)
                padded_data.append(d)
                padded_mask.append(m)
            return (padded_data, padded_mask)

        question, question_mask = padding_batch(question_batch, JQ)
        context, context_mask = padding_batch(context_batch, JX)

        data_dict['question_placeholder'] = question
        data_dict['question_mask_placeholder'] = question_mask
        data_dict['context_placeholder'] = context
        data_dict['context_mask_placeholder'] = context_mask
        data_dict['JQ'] = JQ
        data_dict['JX'] = JX

        if answer_batch is not None:
            start = answer_batch[:,0]
            end = answer_batch[:,1]
            data_dict['answer_start_placeholders'] = start
            data_dict['answer_end_placeholders'] = end
        if is_train:
            data_dict['dropout_placeholder'] = 0.8
        else:
            data_dict['dropout_placeholder'] = 1.0

        return data_dict

    def run_epoch(self, session, epoch_num, training_set, vocab, validation_set, sample_size=400):
        set_num = len(training_set)
        batch_size = self.config.batch_size
        batch_num = int(np.ceil(set_num * 1.0 / batch_size))
        sample_size = 400

        prog = Progbar(target=batch_num)
        avg_loss = 0
        for i, batch in enumerate(minibatches(training_set, self.config.batch_size, window_batch = self.config.window_batch)):
            global_batch_num = batch_num * epoch_num + i
            _, summary, loss = self.optimize(session, batch)
            prog.update(i + 1, [("training loss", loss)])
            if self.config.tensorboard and global_batch_num % self.config.log_batch_num == 0:
                self.train_writer.add_summary(summary, global_batch_num)
            if (i+1) % self.config.log_batch_num == 0:
                logging.info('')
                self.evaluate_answer(session, training_set, vocab, sample=sample_size, log=True)
                self.evaluate_answer(session, validation_set, vocab, sample=sample_size, log=True)
            avg_loss += loss
        avg_loss /= batch_num
        logging.info("Average training loss: {}".format(avg_loss))
        return avg_loss

    def evaluate_answer(self, session, dataset, vocab, sample=400, log=False):
        f1 = 0.
        em = 0.

        N = len(dataset)
        sampleIndices = np.random.choice(N, sample, replace=False)
        evaluate_set = [dataset[i] for i in sampleIndices]
        predicts = self.predict_on_batch(session, evaluate_set)

        for example, (start, end) in zip(evaluate_set, predicts):
            q, _, c, _, (true_s, true_e) = example
            # print (start, end, true_s, true_e)
            context_words = [vocab[w] for w in c]

            true_answer = ' '.join(context_words[true_s : true_e + 1])
            if start <= end:
                predict_answer = ' '.join(context_words[start : end + 1])
            else:
                predict_answer = ''
            f1 += f1_score(predict_answer, true_answer)
            em += exact_match_score(predict_answer, true_answer)


        f1 = 100 * f1 / sample
        em = 100 * em / sample

        if log:
            logging.info("F1: {}, EM: {}, for {} samples".format(f1, em, sample))

        return f1, em

    def test(self, session, validation_set):
        """
        in here you should compute a cost for your validation set
        and tune your hyperparameters according to the validation set performance
        :return:
        """
        question_batch, question_len_batch, context_batch, context_len_batch, answer_batch = validation_set
        input_feed = self.create_feed_dict(question_batch, question_len_batch, context_batch, context_len_batch, answer_batch=answer_batch, is_train = False)

        output_feed = [self.loss]
        outputs = session.run(output_feed, input_feed)
        return outputs

    def answer(self, session, test_batch):
        """
        Returns the probability distribution over different positions in the paragraph
        so that other methods like self.answer() will be able to work properly
        :return:
        """

        # fill in this feed_dictionary like:
        # input_feed['test_x'] = test_x

        question_batch, question_len_batch, context_batch, context_len_batch, answer_batch = test_batch
        input_feed =  self.create_feed_dict(question_batch, question_len_batch, context_batch, context_len_batch, answer_batch=None, is_train = False)
        output_feed = [self.preds[0], self.preds[1]]
        outputs = session.run(output_feed, input_feed)

        s, e = outputs

        best_spans, scores = zip(*[get_best_span(si, ei, ci) for si, ei, ci in zip(s, e, context_batch)])
        return best_spans

    def predict_on_batch(self, session, dataset):
        batch_num = int(np.ceil(len(dataset) * 1.0 / self.config.batch_size))
        # prog = Progbar(target=batch_num)
        predicts = []
        for i, batch in tqdm(enumerate(minibatches(dataset, self.config.batch_size, shuffle=False))):
            pred = self.answer(session, batch)
            # prog.update(i + 1)
            predicts.extend(pred)
        return predicts

    def predict_on_batch(self, session, dataset):
        batch_num = int(np.ceil(len(dataset) * 1.0 / self.config.batch_size))
        # prog = Progbar(target=batch_num)
        predicts = []
        for i, batch in tqdm(enumerate(minibatches(dataset, self.config.batch_size, shuffle=False))):
            pred = self.answer(session, batch)
            # prog.update(i + 1)
            predicts.extend(pred)
        return predicts

    def validate(self, sess, valid_dataset):
        """
        Iterate through the validation dataset and determine what
        the validation cost is.

        This method calls self.test() which explicitly calculates validation cost.

        How you implement this function is dependent on how you design
        your data iteration function

        :return:
        """
        batch_num = int(np.ceil(len(valid_dataset) * 1.0 / self.config.batch_size))
        prog = Progbar(target=batch_num)
        avg_loss = 0
        for i, batch in enumerate(minibatches(valid_dataset, self.config.batch_size)):
            loss = self.test(sess, batch)[0]
            prog.update(i + 1, [("validation loss", loss)])
            avg_loss += loss
        avg_loss /= batch_num
        logging.info("Average validation loss: {}".format(avg_loss))
        return avg_loss

    def optimize(self, session, training_set):
        """
        Takes in actual data to optimize your model
        This method is equivalent to a step() function
        :return:
        """
        question_batch, question_len_batch, context_batch, context_len_batch, answer_batch = training_set
        input_feed = self.create_feed_dict(question_batch, question_len_batch, context_batch, context_len_batch, answer_batch=answer_batch, is_train = True)

        output_feed = [self.train_op, self.merged, self.loss]

        outputs = session.run(output_feed, input_feed)
        return outputs

    def setup_loss(self, preds):
        with vs.variable_scope("loss"):
            s, e = preds # [None, JX]*2
            assert s.get_shape().ndims == 2
            assert e.get_shape().ndims == 2
            loss1 = tf.reduce_sum(tf.nn.sparse_softmax_cross_entropy_with_logits(logits=s, labels=self.answer_start_placeholders),)
            loss2 = tf.reduce_sum(tf.nn.sparse_softmax_cross_entropy_with_logits(logits=e, labels=self.answer_end_placeholders),)
            # loss1 = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(logits=s, labels=self.answer_start_placeholders),)
            # loss2 = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(logits=e, labels=self.answer_end_placeholders),)
        loss = loss1 + loss2
        tf.summary.scalar('loss', loss)
        return loss

    def setup_loss_pytorch(self,preds):
        s, e = preds  # [None, JX]*2
        # loss1 = tf.reduce_sum(tf.nn.sparse_softmax_cross_entropy_with_logits(logits=s, labels=self.answer_start_placeholders),)
        # loss2 = tf.reduce_sum(tf.nn.sparse_softmax_cross_entropy_with_logits(logits=e, labels=self.answer_end_placeholders),)
        # TODO: get NLL loss
        loss1 = torch.sum()
        loss2 = torch.sum()
        loss = loss1 + loss2
        # TODO: add loss to tf-board to visualize
        return loss

    def get_optimizer_pytorch(opt):
        if opt == "adam":
            # TODO: add para
            optfn = torch.optim.Adam()
        elif opt == "sgd":
            optfn = torch.optim.SGD()
        else:
            assert (False)
        return optfn

    # with gradient clipping
    def get_optimizer_pytorch(opt, loss, max_grad_norm, learning_rate):
        if opt == "adam":
            optfn = torch.optim.Adam(lr=learning_rate)
        elif opt == "sgd":
            optfn = torch.optim.SGD(lr=learning_rate)
        else:
            assert (False)

        grads_and_vars = optfn.compute_gradients(loss)
        variables = [output[1] for output in grads_and_vars]
        gradients = [output[0] for output in grads_and_vars]

        # gradients = tf.clip_by_global_norm(gradients, clip_norm=max_grad_norm)[0]
        # TODO: clip norm

        grads_and_vars = [(gradients[i], variables[i]) for i in range(len(gradients))]
        train_op = optfn.apply_gradients(grads_and_vars)

        return train_op
