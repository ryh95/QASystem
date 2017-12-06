from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import time, datetime
import logging
import torch
from torch.nn.utils.rnn import pack_padded_sequence, pad_packed_sequence
from torch.autograd import Variable as Var
from tqdm import tqdm
import numpy as np
from six.moves import xrange  # pylint: disable=redefined-builtin
from torch import nn
from operator import mul, itemgetter
from .utils.util import ConfusionMatrix, Progbar, minibatches, one_hot, minibatch, get_best_span


logging.basicConfig(level=logging.INFO)

# def variable_summaries(var):
#   """Attach a lot of summaries to a Tensor (for TensorBoard visualization)."""
#   with tf.name_scope('summaries'):
#     mean = tf.reduce_mean(var)
#     tf.summary.scalar('mean', mean)
#     with tf.name_scope('stddev'):
#       stddev = tf.sqrt(tf.reduce_mean(tf.square(var - mean)))
#     tf.summary.scalar('stddev', stddev)
#     tf.summary.scalar('max', tf.reduce_max(var))
#     tf.summary.scalar('min', tf.reduce_min(var))
#     tf.summary.histogram('histogram', var)

# No gradient clipping:
# def get_optimizer(opt):
#     if opt == "adam":
#         optfn = tf.train.AdamOptimizer
#     elif opt == "sgd":
#         optfn = tf.train.GradientDescentOptimizer
#     else:
#         assert (False)
#     return optfn

# With gradient clipping:
# def get_optimizer(opt, loss, max_grad_norm, learning_rate):
#     if opt == "adam":
#         optfn = tf.train.AdamOptimizer(learning_rate=learning_rate)
#     elif opt == "sgd":
#         optfn = tf.train.GradientDescentOptimizer(learning_rate=learning_rate)
#     else:
#         assert (False)
#
#     grads_and_vars = optfn.compute_gradients(loss)
#     variables = [output[1] for output in grads_and_vars]
#     gradients = [output[0] for output in grads_and_vars]
#
#     gradients = tf.clip_by_global_norm(gradients, clip_norm=max_grad_norm)[0]
#     grads_and_vars = [(gradients[i], variables[i]) for i in range(len(gradients))]
#     train_op = optfn.apply_gradients(grads_and_vars)
#
#     return train_op

# def softmax_mask_prepro(tensor, mask): # set huge neg number(-1e10) in padding area
#     assert tensor.get_shape().ndims == mask.get_shape().ndims
#     m0 = tf.subtract(tf.constant(1.0), tf.cast(mask, 'float32'))
#     paddings = tf.multiply(m0,tf.constant(-1e10))
#     tensor = tf.select(mask, tensor, paddings)
#     return tensor

def softmax_mask_prepro_pytorch(tensor,mask):
    m0 = mask - 1.0
    paddings = m0 * (1e5)
    # TODO use plus instead of select may have bugs, if then need to fix after
    # tensor = torch.select(mask, tensor, paddings)
    tensor += paddings.float()
    return tensor

class Attention(nn.Module):
    def __init__(self,config):
        super(Attention,self).__init__()
        self.a_t_softmax = nn.Softmax(dim=2)
        self.b_softmax = nn.Softmax(dim=1)
        self.config = config

    # def calculate(self, h, u, h_mask, u_mask, JX, JQ, dropout = 1.0):
        # compare the question representation with all the context hidden states.
        #         e.g. S = h.T * u
        #              a_x = softmax(S)
        #              a_q = softmax(S.T)
        #              u_a = sum(a_x*U)
        #              h_a = sum(a_q*H)
        # """
        # :param h: [N, JX, d_en]
        # :param u: [N, JQ, d_en]
        # :param h_mask:  [N, JX]
        # :param u_mask:  [N, JQ]
        #
        # :return: [N, JX, d_com]
        # """
        # logging.debug('-'*5 + 'attention' + '-'*5)
        # logging.debug('Context representation: %s' % str(h))
        # logging.debug('Question representation: %s' % str(u))
        # d_en = h.get_shape().as_list()[-1]
        # # h [None, JX, d_en]
        # # u [None, JQ, d_en]
        #
        # # get similarity
        # h_aug = tf.tile(tf.reshape(h, shape = [-1, JX, 1, d_en]),[1, 1, JQ, 1])
        # u_aug = tf.tile(tf.reshape(u, shape = [-1, 1, JQ, d_en]),[1, JX, 1, 1])
        # h_mask_aug = tf.tile(tf.expand_dims(h_mask, -1), [1, 1, JQ]) # [N, JX] -(expend)-> [N, JX, 1] -(tile)-> [N, JX, JQ]
        # u_mask_aug = tf.tile(tf.expand_dims(u_mask, -2), [1, JX, 1]) # [N, JQ] -(expend)-> [N, 1, JQ] -(tile)-> [N, JX, JQ]
        # # s = tf.reduce_sum(tf.multiply(h_aug, u_aug), axis = -1) # h * u: [N, JX, d_en] * [N, JQ, d_en] -> [N, JX, JQ]
        # s = self.get_logits([h_aug, u_aug], None, True, is_train=(dropout<1.0), func='tri_linear', input_keep_prob=dropout)  # [N, M, JX, JQ]
        # hu_mask_aug = h_mask_aug & u_mask_aug
        # s = softmax_mask_prepro(s, hu_mask_aug)
        #
        # # get a_x
        # a_x = tf.nn.softmax(s, dim=-1) # softmax -> [N, JX, softmax(JQ)]
        #
        # #     use a_x to get u_a
        # a_x = tf.reshape(a_x, shape = [-1, JX, JQ, 1])
        # u_aug = tf.reshape(u, shape = [-1, 1, JQ, d_en])
        # u_a = tf.reduce_sum(tf.multiply(a_x, u_aug), axis = -2)# a_x * u: [N, JX, JQ](weight) * [N, JQ, d_en] -> [N, JX, d_en]
        # logging.debug('Context with attention: %s' % str(u_a))
        #
        # # get a_q
        # a_q = tf.reduce_max(s, axis=-1) # max -> [N, JX]
        # a_q = tf.nn.softmax(a_q, dim=-1) # softmax -> [N, softmax(JX)]
        # #     use a_q to get h_a
        # a_q = tf.reshape(a_q, shape = [-1, JX, 1])
        # h_aug = tf.reshape(h, shape = [-1, JX, d_en])
        #
        # h_a = tf.reduce_sum(tf.multiply(a_q, h_aug), axis = -2)# a_q * h: [N, JX](weight) * [N, JX, d_en] -> [N, d_en]
        # assert h_a.get_shape().as_list() == [None, d_en]
        # h_a = tf.tile(tf.expand_dims(h_a, -2), [1, JX, 1]) # [None, JX, d_en]
        #
        # h_0_u_a = h*u_a #[None, JX, d_en]
        # h_0_h_a = h*h_a #[None, JX, d_en]
        # return tf.concat(2,[h, u_a, h_0_u_a, h_0_h_a])

    def calculate_pytorch(self, h, u, h_mask, u_mask, JX, JQ):
        # compare the question representation with all the context hidden states.
        #         e.g. S = h.T * u
        #              a_x = softmax(S)
        #              a_q = softmax(S.T)
        #              u_a = sum(a_x*U)
        #              h_a = sum(a_q*H)
        """
        :param h: [N, JX, d_en]
        :param u: [N, JQ, d_en]
        :param h_mask:  [N, JX]
        :param u_mask:  [N, JQ]

        :return: [N, JX, d_com]
        """
        # d_en = h.get_shape().as_list()[-1]
        # # get similarity
        # h_aug = h.view(-1, JX, 1, d_en).expand(-1, -1, JQ, -1)
        # u_aug = u.view(-1, 1, JQ, d_en).expand(-1, JX, -1, -1)
        h_mask_aug = h_mask.unsqueeze(-1).expand(-1,JX,JQ)
        u_mask_aug = u_mask.unsqueeze(1).expand(-1, JX,JQ)
        hu_mask_aug = h_mask_aug & u_mask_aug

        # TODO : first use vanilla cos similarity, then can use some NN like the Bidaf paper
        # s = self.get_logits([h_aug, u_aug], None, True, is_train=(dropout < 1.0), func='tri_linear',
        #                     input_keep_prob=dropout)  # [N, M, JX, JQ]

        s = torch.bmm(h,torch.transpose(u,1,2))

        s = softmax_mask_prepro_pytorch(s, hu_mask_aug)

        # test for softmax
        # print(self.softmax(Var(torch.tensor([[55,1],[55,1]]))))

        # get a_t
        # should be row softmax
        a_t = self.a_t_softmax(s)

        # use a_x to get u_a
        if self.config.encoder_bidirectional == True:
            d_en = 2*self.config.encoder_state_size
        else:
            d_en = self.config.encoder_state_size

        a_t = a_t.unsqueeze(-1).expand(-1, JX, JQ, d_en)
        u_expand = u.unsqueeze(1).expand(-1, JX, JQ, d_en)

        u_tilde = torch.sum(a_t * u_expand, 2)

        # get b
        b,_ = torch.max(s, -1,keepdim=True)
        b = self.b_softmax(b)
        #     use a_q to get h_a

        h_tilde = torch.sum(b * h,1,keepdim=True).expand(-1,JX,d_en)

        return torch.cat([h, u_tilde, h*u_tilde, h*h_tilde],2)

    # this function is from https://github.com/allenai/bi-att-flow/tree/master/my/tensorflow
    # def get_logits(args, size, bias, bias_start=0.0, scope=None, mask=None, wd=0.0, input_keep_prob=1.0, is_train=None, func=None):
    #
    #     def linear_logits(args, bias, bias_start=0.0, scope=None, mask=None, wd=0.0, input_keep_prob=1.0, is_train=None):
    #
    #         def linear(args, output_size, bias, bias_start=0.0, scope=None, squeeze=False, wd=0.0, input_keep_prob=1.0,
    #                    is_train=None):
    #             if args is None or (nest.is_sequence(args) and not args):
    #                 raise ValueError("`args` must be specified")
    #             if not nest.is_sequence(args):
    #                 args = [args]
    #
    #             def flatten(tensor, keep):
    #                 fixed_shape = tensor.get_shape().as_list()
    #                 start = len(fixed_shape) - keep
    #                 left = reduce(mul, [fixed_shape[i] or tf.shape(tensor)[i] for i in range(start)])
    #                 out_shape = [left] + [fixed_shape[i] or tf.shape(tensor)[i] for i in range(start, len(fixed_shape))]
    #                 flat = tf.reshape(tensor, out_shape)
    #                 return flat
    #
    #             flat_args = [flatten(arg, 1) for arg in args]
    #             #if input_keep_prob < 1.0:
    #             #   assert is_train is not None
    #             flat_args = [tf.cond(is_train, lambda: tf.nn.dropout(arg, input_keep_prob), lambda: arg)
    #                              for arg in flat_args]
    #             flat_out = _linear(flat_args, output_size, bias, bias_start=bias_start, scope=scope)
    #
    #             def reconstruct(tensor, ref, keep):
    #                 ref_shape = ref.get_shape().as_list()
    #                 tensor_shape = tensor.get_shape().as_list()
    #                 ref_stop = len(ref_shape) - keep
    #                 tensor_start = len(tensor_shape) - keep
    #                 pre_shape = [ref_shape[i] or tf.shape(ref)[i] for i in range(ref_stop)]
    #                 keep_shape = [tensor_shape[i] or tf.shape(tensor)[i] for i in range(tensor_start, len(tensor_shape))]
    #                 # pre_shape = [tf.shape(ref)[i] for i in range(len(ref.get_shape().as_list()[:-keep]))]
    #                 # keep_shape = tensor.get_shape().as_list()[-keep:]
    #                 target_shape = pre_shape + keep_shape
    #                 out = tf.reshape(tensor, target_shape)
    #                 return out
    #
    #             out = reconstruct(flat_out, args[0], 1)
    #             if squeeze:
    #                 out = tf.squeeze(out, [len(args[0].get_shape().as_list())-1])
    #
    #             return out
    #
    #         with tf.variable_scope(scope or "Linear_Logits"):
    #             logits = linear(args, 1, bias, bias_start=bias_start, squeeze=True, scope='first',
    #                             wd=wd, input_keep_prob=input_keep_prob, is_train=is_train)
    #             return logits
    #
    #     assert len(args) == 2
    #     new_arg = args[0] * args[1]
    #     return linear_logits([args[0], args[1], new_arg], bias, bias_start=bias_start, scope=scope, mask=mask, wd=wd, input_keep_prob=input_keep_prob,
    #                          is_train=is_train)

    def get_logits_pytorch(self):
        # TODO finish this function
        pass

    def forward(self, h, u, h_mask, u_mask, JX, JQ):
        return self.calculate_pytorch(h, u, h_mask, u_mask, JX, JQ)

class Encoder(nn.Module):
    def __init__(self, input_size, state_size,bidirectional ,dropout = 0):
        super(Encoder,self).__init__()
        self.input_size = input_size
        self.state_size = state_size
        self.bidirectional = bidirectional
        #self.dropout = dropout
        #logging.info("Dropout rate for encoder: {}".format(self.dropout))

        # debug forward: set input_size to 1
        self.lstm = nn.LSTM(self.input_size,self.state_size,bidirectional=bidirectional,dropout=dropout)

    # def encode(self, inputs, mask, encoder_state_input, dropout = 1.0):
    #     """
    #     In a generalized encode function, you pass in your inputs,
    #     sequence_length, and an initial hidden state input into this function.
    #
    #     :param inputs: Symbolic representations of your input (padded all to the same length)
    #     :param mask: mask of the sequence
    #     :param encoder_state_input: (Optional) pass this as initial hidden state
    #                                 to tf.nn.dynamic_rnn to build conditional representations
    #     :return: an encoded representation of your input.
    #              It can be context-level representation, word-level representation,
    #              or both.
    #     """
    #
    #     logging.debug('-'*5 + 'encode' + '-'*5)
    #     # Forward direction cell
    #     lstm_fw_cell = tf.nn.rnn_cell.LSTMCell(self.state_size, state_is_tuple=True)
    #     # Backward direction cell
    #     lstm_bw_cell = tf.nn.rnn_cell.LSTMCell(self.state_size, state_is_tuple=True)
    #
    #
    #     lstm_fw_cell = tf.nn.rnn_cell.DropoutWrapper(lstm_fw_cell, input_keep_prob = dropout)
    #     lstm_bw_cell = tf.nn.rnn_cell.DropoutWrapper(lstm_bw_cell, input_keep_prob = dropout)
    #
    #     initial_state_fw = None
    #     initial_state_bw = None
    #     if encoder_state_input is not None:
    #         initial_state_fw, initial_state_bw = encoder_state_input
    #
    #     logging.debug('Inputs: %s' % str(inputs))
    #     sequence_length = tf.reduce_sum(tf.cast(mask, 'int32'), axis=1)
    #     sequence_length = tf.reshape(sequence_length, [-1,])
    #     # Get lstm cell output
    #     (outputs_fw, outputs_bw), (final_state_fw, final_state_bw) = tf.nn.bidirectional_dynamic_rnn(cell_fw=lstm_fw_cell,\
    #                                                   cell_bw=lstm_bw_cell,\
    #                                                   inputs=inputs,\
    #                                                   sequence_length=sequence_length,
    #                                                   initial_state_fw=initial_state_fw,\
    #                                                   initial_state_bw=initial_state_bw,
    #                                                   dtype=tf.float32)
    #
    #     # Concatinate forward and backword hidden output vectors.
    #     # each vector is of size [batch_size, sequence_length, cell_state_size]
    #
    #     logging.debug('fw hidden state: %s' % str(outputs_fw))
    #     hidden_state = tf.concat(2, [outputs_fw, outputs_bw])
    #     logging.debug('Concatenated bi-LSTM hidden state: %s' % str(hidden_state))
    #     # final_state_fw and final_state_bw are the final states of the forwards/backwards LSTM
    #     concat_final_state = tf.concat(1, [final_state_fw[1], final_state_bw[1]])
    #     logging.debug('Concatenated bi-LSTM final hidden state: %s' % str(concat_final_state))
    #     return hidden_state, concat_final_state, (final_state_fw, final_state_bw)

    def forward(self, inputs, seq_len, init_hidden_state):

        batch_size = inputs.size()[0]
        if self.bidirectional == True:
            # TODO not hard code num layers
            h_0 = Var(torch.zeros(2,batch_size,self.state_size))
            c_0 = Var(torch.zeros(2,batch_size,self.state_size))
        else:
            h_0 = Var(torch.zeros(1,batch_size,self.state_size))
            c_0 = Var(torch.zeros(1,batch_size,self.state_size))

        if init_hidden_state is not None:
            h_0, c_0 = init_hidden_state


        indices = np.argsort(-seq_len)
        sorted_seq_len = seq_len[indices]
        indices = Var(torch.LongTensor(indices))

        # debug forward set view(-1,1)
        # TODO use index_select to do sort and recover
        expand_indices = indices.view(-1,1,1).expand(inputs.size())
        sorted_inputs = torch.gather(inputs,0,expand_indices)

        # debug for forward
        # sorted_inputs = sorted_inputs.unsqueeze(-1).float()

        packed = pack_padded_sequence(sorted_inputs, sorted_seq_len.tolist(),batch_first=True)
        sorted_outputs, sorted_hidden = self.lstm(packed,(h_0,c_0))
        sorted_outputs, output_lengths = pad_packed_sequence(sorted_outputs,batch_first=True)

        # debug for forward
        # expand_indices = expand_indices.unsqueeze(-1).expand(sorted_outputs.size())

        # recover to unsorted outputs
        # debug forward set view(-1,1)
        expand_indices = indices.view(-1,1,1).expand_as(sorted_outputs)
        outputs = Var(torch.zeros(sorted_outputs.size())).scatter_(0,expand_indices,sorted_outputs)

        return outputs

class Decoder(nn.Module):
    def __init__(self, output_size, state_size,dropout):
        super(Decoder,self).__init__()
        self.output_size = output_size
        self.state_size = state_size
        self.m_lstm = nn.LSTM(state_size*8,state_size,num_layers=2,bidirectional=True,dropout=dropout)
        self.m2_lstm = nn.LSTM(state_size*2,state_size,bidirectional=True,dropout=dropout)
        self.wp_1 = nn.Linear(state_size*10,1,bias=False)
        self.wp_2 = nn.Linear(state_size*10,1,bias=False)
        self.softmax = nn.LogSoftmax(dim=1)

    # def decode(self, g, context_mask, JX, dropout = 1.0):
    #     """
    #     takes in a knowledge representation
    #     and output a probability estimation over
    #     all paragraph tokens on which token should be
    #     the start of the answer span, and which should be
    #     the end of the answer span.
    #     m_2 = bi_LSTM*2(g)
    #     """
        # d_de = self.state_size*2
        # with tf.variable_scope('g'):
        #     m, m_repr, m_state = \
        #          self.decode_LSTM(inputs=g, mask=context_mask, encoder_state_input=None, dropout = dropout)
        # with tf.variable_scope('m'):
        #     m_2, m_2_repr, m_2_state = \
        #          self.decode_LSTM(inputs=m, mask=context_mask, encoder_state_input=None, dropout = dropout)
        # # assert m_2.get_shape().as_list() == [None, JX, d_en2]
        #
        # with tf.variable_scope('start'):
        #     s = self.get_logit(m_2, JX) #[N, JX]*2
        # # or s, e = self.get_logit_start_end(m_2) #[N, JX]*2
        # s = softmax_mask_prepro(s, context_mask)
        #
        # print(s.get_shape())
        #
        # s_prob = tf.nn.softmax(s)
        #
        # print(s_prob.get_shape())
        #
        # s_prob = tf.tile(tf.expand_dims(s_prob, 2), [1,1,d_de])
        #
        # e_input = tf.concat(2, [m_2, m_2 * s_prob, s_prob])
        # with tf.variable_scope('end'):
        #     e = self.get_logit(e_input, JX) #[N, JX]*2
        #
        # e = softmax_mask_prepro(e, context_mask)
        # return (s, e)

    def decode_pytorch(self, g, c_len,c_mask):
        """
        takes in a knowledge representation
        and output a probability estimation over
        all paragraph tokens on which token should be
        the start of the answer span, and which should be
        the end of the answer span.
        m_2 = bi_LSTM*2(g)
        """
        indices = np.argsort(-c_len)
        sorted_seq_len = c_len[indices]
        indices = Var(torch.LongTensor(indices))
        # Get lstm cell output

        expand_indices = indices.view(-1, 1, 1).expand(g.size())
        sorted_g = torch.gather(g, 0, expand_indices)

        packed = pack_padded_sequence(sorted_g, sorted_seq_len.tolist(), batch_first=True)
        sorted_m, _ = self.m_lstm(packed)
        sorted_m, _ = pad_packed_sequence(sorted_m, batch_first=True)

        # recover to unsorted outputs
        expand_indices = indices.view(-1, 1, 1).expand_as(sorted_m)
        m = Var(torch.zeros(sorted_m.size())).scatter_(0, expand_indices, sorted_m)

        packed = pack_padded_sequence(sorted_m,sorted_seq_len.tolist(), batch_first=True)
        sorted_m2,_ = self.m2_lstm(packed)
        sorted_m2,_ = pad_packed_sequence(sorted_m2,batch_first=True)

        expand_indices = indices.view(-1, 1, 1).expand_as(sorted_m2)
        m2 = Var(torch.zeros(sorted_m2.size())).scatter_(0, expand_indices, sorted_m2)

        # or s, e = self.get_logit_start_end(m_2) #[N, JX]*2
        p1_logits = softmax_mask_prepro_pytorch(self.wp_1(torch.cat([g,m],2)).squeeze(-1), c_mask)
        p1 = self.softmax(p1_logits)

        p2_logits = softmax_mask_prepro_pytorch(self.wp_2(torch.cat([g,m2],2)).squeeze(-1), c_mask)
        p2 = self.softmax(p2_logits)

        return (p1, p2)

    # def decode_LSTM(self, inputs, mask, encoder_state_input, dropout = 1.0, output_dropout = False):
    #     logging.debug('-'*5 + 'decode_LSTM' + '-'*5)
    #     # Forward direction cell
    #     lstm_fw_cell = tf.nn.rnn_cell.LSTMCell(self.state_size, state_is_tuple=True)
    #     # Backward direction cell
    #     lstm_bw_cell = tf.nn.rnn_cell.LSTMCell(self.state_size, state_is_tuple=True)
    #
    #     # add dropout
    #
    #     if output_dropout:
    #         lstm_fw_cell = tf.nn.rnn_cell.DropoutWrapper(lstm_fw_cell, input_keep_prob = dropout, output_keep_prob = dropout)
    #         lstm_bw_cell = tf.nn.rnn_cell.DropoutWrapper(lstm_bw_cell, input_keep_prob = dropout, output_keep_prob = dropout)
    #     else:
    #         lstm_fw_cell = tf.nn.rnn_cell.DropoutWrapper(lstm_fw_cell, input_keep_prob = dropout)
    #         lstm_bw_cell = tf.nn.rnn_cell.DropoutWrapper(lstm_bw_cell, input_keep_prob = dropout)
    #
    #     initial_state_fw = None
    #     initial_state_bw = None
    #     if encoder_state_input is not None:
    #         initial_state_fw, initial_state_bw = encoder_state_input
    #
    #     logging.debug('Inputs: %s' % str(inputs))
    #     sequence_length = tf.reduce_sum(tf.cast(mask, 'int32'), axis=1)
    #     sequence_length = tf.reshape(sequence_length, [-1,])
    #     # Get lstm cell output
    #     (outputs_fw, outputs_bw), (final_state_fw, final_state_bw) = tf.nn.bidirectional_dynamic_rnn(cell_fw=lstm_fw_cell,\
    #                                                   cell_bw=lstm_bw_cell,\
    #                                                   inputs=inputs,\
    #                                                   sequence_length=sequence_length,
    #                                                   initial_state_fw=initial_state_fw,\
    #                                                   initial_state_bw=initial_state_bw,
    #                                                   dtype=tf.float32)
    #
    #     logging.debug('fw hidden state: %s' % str(outputs_fw))
    #     hidden_state = tf.concat(2, [outputs_fw, outputs_bw])
    #     logging.debug('Concatenated bi-LSTM hidden state: %s' % str(hidden_state))
    #     # final_state_fw and final_state_bw are the final states of the forwards/backwards LSTM
    #     concat_final_state = tf.concat(1, [final_state_fw[1], final_state_bw[1]])
    #     logging.debug('Concatenated bi-LSTM final hidden state: %s' % str(concat_final_state))
    #     return hidden_state, concat_final_state, (final_state_fw, final_state_bw)

    # def get_logit(self, X, JX):
    #     d = X.get_shape().as_list()[-1]
    #     assert X.get_shape().ndims == 3
    #     X = tf.reshape(X, shape = [-1, d])
    #     W1 = tf.get_variable('W1', initializer=tf.contrib.layers.xavier_initializer(), shape=(d, 1), dtype=tf.float32)
    #     pred1 = tf.matmul(X, W1)
    #     pred1 = tf.reshape(pred1, shape = [-1, JX])
    #     tf.summary.histogram('logit_start', pred1)
    #     return pred1
    
    # def get_logit_start_end(self, X, JX):
    #     d = X.get_shape().as_list()[-1]
    #     X = tf.reshape(X, shape = [-1, d])
    #     X = tf.reshape(X, shape = [-1, d])
    #     W1 = tf.get_variable('W1', initializer=tf.contrib.layers.xavier_initializer(), shape=(d, 1), dtype=tf.float32)
    #     W2 = tf.get_variable('W2', initializer=tf.contrib.layers.xavier_initializer(), shape=(d, 1), dtype=tf.float32)
    #     pred1 = tf.matmul(X, W1)
    #     pred2_0 = tf.matmul(X, W2)
    #     pred1 = tf.reshape(pred1, shape = [-1, JX])
    #     pred2_0 = tf.reshape(pred2_0, shape = [-1, JX])
    #
    #     pred2_1 = tf.concat(1,[pred1, pred2_0])
    #     W_se = tf.get_variable('W_se', initializer=tf.contrib.layers.xavier_initializer(), shape=(2*JX, JX), dtype=tf.float32)
    #     b_se = tf.get_variable('b_se', initializer=tf.contrib.layers.xavier_initializer(), shape=(JX,), dtype=tf.float32)
    #     pred2 = tf.matmul(pred2_1, W_se)+b_se
    #     return pred1, pred2

    def forward(self, g, c_len, c_mask):
        return self.decode_pytorch(g, c_len, c_mask)

class QASystem(nn.Module):
    def __init__(self, pretrained_embeddings, config):
        """
        Initializes your System

        :param encoder: an encoder that you constructed in train.py
        :param decoder: a decoder that you constructed in train.py
        :param args: pass in more arguments as needed
        """
        super(QASystem, self).__init__()
        self.pretrained_embeddings = pretrained_embeddings
        self.emb_layer = nn.Embedding(config.vocab_size, config.embedding_size)
        self.q_encoder = Encoder(config.embedding_size, config.encoder_state_size,config.encoder_bidirectional,config.dropout)
        self.c_encoder = Encoder(config.embedding_size, config.encoder_state_size,config.encoder_bidirectional,config.dropout)

        self.decoder = Decoder(config.output_size, config.decoder_state_size,config.dropout)
        self.attention = Attention(config)
        self.config = config

        # ==== set up placeholder tokens ====
        # self.question_placeholder = tf.placeholder(dtype=tf.int32, name="q", shape=(None, None))
        # self.question_mask_placeholder = tf.placeholder(dtype=tf.bool, name="q_mask", shape=(None, None))
        # self.context_placeholder = tf.placeholder(dtype=tf.int32, name="c", shape=(None, None))
        # self.context_mask_placeholder = tf.placeholder(dtype=tf.bool, name="c_mask", shape=(None, None))
        # # self.answer_placeholders = tf.placeholder(dtype=tf.int32, name="a", shape=(None, config.answer_size))
        # self.answer_start_placeholders = tf.placeholder(dtype=tf.int32, name="a_s", shape=(None,))
        # self.answer_end_placeholders = tf.placeholder(dtype=tf.int32, name="a_e", shape=(None,))
        # self.dropout_placeholder = tf.placeholder(dtype=tf.float32, name="dropout", shape=())
        # self.JX = tf.placeholder(dtype=tf.int32, name='JX', shape=())
        # self.JQ = tf.placeholder(dtype=tf.int32, name='JQ', shape=())


        # ==== assemble pieces ====
        # with tf.variable_scope("qa", initializer=tf.uniform_unit_scaling_initializer(1.0)):
        #     self.q, self.x = self.setup_embeddings()
        #     self.preds = self.setup_system(self.x, self.q)
        #     self.loss = self.setup_loss(self.preds)



        # ==== set up training/updating procedure ====
        # No gradient clipping:
        # get_op = get_optimizer(self.config.optimizer)
        # self.train_op = get_op(self.config.learning_rate).minimize(self.loss)

        # With gradient clipping:
        # opt_op = get_optimizer_pytorch("adam", self.loss, config.max_gradient_norm, config.learning_rate)
        #
        # if config.ema_weight_decay is not None:
        #     self.train_op = self.build_ema(opt_op)
        # else:
        #     self.train_op = opt_op
        # TODO : tensorboard
        # self.merged = tf.summary.merge_all()

    def forward(self, q,c,q_len,c_len,q_mask,c_mask,JQ,JX):
        # debug encoder forward : annotate next line
        q, c = self.setup_embeddings_pytorch(q,c)
        preds = self.setup_system_pytorch(c, q,c_len,q_len,c_mask,q_mask,JX,JQ)

        return preds

    # def build_ema(self, opt_op):
    #     self.ema = tf.train.ExponentialMovingAverage(self.config.ema_weight_decay)
    #     ema_op = self.ema.apply(tf.trainable_variables())
    #     with tf.control_dependencies([opt_op]):
    #         train_op = tf.group(ema_op)
    #     return train_op

    # def setup_system(self, x, q):
    #     d = x.get_shape().as_list()[-1] # self.config.embedding_size
    #         #   x: [None, JX, d]
    #         #   q: [None, JQ, d]
    #     assert x.get_shape().ndims == 3
    #     assert q.get_shape().ndims == 3
    #
    #     # Step 1: encode x and q, respectively, with independent weights
    #     #         e.g. u = encode_question(q)  # get U (2d*J) as representation of q
    #     #         e.g. h = encode_context(x, u_state)   # get H (2d*T) as representation of x
    #     with tf.variable_scope('q'):
    #         u, question_repr, u_state = \
    #              self.encoder.encode(inputs=q, mask=self.question_mask_placeholder, encoder_state_input=None, dropout = self.dropout_placeholder)
    #         if self.config.QA_ENCODER_SHARE:
    #             tf.get_variable_scope().reuse_variables()
    #             h, context_repr, context_state =\
    #                  self.encoder.encode(inputs=x, mask=self.context_mask_placeholder, encoder_state_input=None, dropout = self.dropout_placeholder)
    #     if not self.config.QA_ENCODER_SHARE:
    #         with tf.variable_scope('c'):
    #             h, context_repr, context_state =\
    #                  self.encoder.encode(inputs=x, mask=self.context_mask_placeholder, encoder_state_input=None, dropout = self.dropout_placeholder)
    #              # self.encoder.encode(inputs=x, mask=self.context_mask_placeholder, encoder_state_input=None)
    #     d_en = self.config.encoder_state_size*2
    #     # assert h.get_shape().as_list() == [None, None, d_en], "Expected {}, got {}".format([None, JX, d_en], h.get_shape().as_list())
    #     # assert u.get_shape().as_list() == [None, None, d_en], "Expected {}, got {}".format([None, JQ, d_en], u.get_shape().as_list())
    #
    #
    #     # Step 2: combine H and U using "Attention"
    #     #         e.g. s = h.T * u
    #     #              a_x = softmax(s)
    #     #              a_q = softmax(s.T)
    #     #              u_hat = sum(a_x*u)
    #     #              h_hat = sum(a_q*h)
    #     #              g = combine(u, h, u_hat, h_hat)
    #     # --------op1--------------
    #     g = self.attention.calculate(h, u, self.context_mask_placeholder, self.question_mask_placeholder, JX = self.JX, JQ = self.JQ, dropout = self.dropout_placeholder) # concat[h, u_a, h*u_a, h*h_a]
    #     d_com = d_en*4
    #     # assert g.get_shape().as_list() == [None, None, d_com], "Expected {}, got {}".format([None, JX, d_com], g.get_shape().as_list())
    #
    #     # Step 3:
    #     # 2 LSTM layers
    #     # logistic regressions
    #     pred1, pred2 = self.decoder.decode(g, self.context_mask_placeholder, dropout = self.dropout_placeholder, JX = self.JX)
    #     return pred1, pred2

    def setup_system_pytorch(self, c, q,c_len,q_len,c_mask,q_mask,JX,JQ):
        #   x: [None, JX, d]
        #   q: [None, JQ, d]

        # Step 1: encode x and q, respectively, with independent weights
        #         e.g. u = encode_question(q)  # get U (2d*J) as representation of q
        #         e.g. h = encode_context(x, u_state)   # get H (2d*T) as representation of x

        u = self.q_encoder(inputs=q, seq_len=q_len, init_hidden_state=None)
        if self.config.QA_ENCODER_SHARE:
            h = self.q_encoder(inputs=c, seq_len=c_len, init_hidden_state=None)
        if not self.config.QA_ENCODER_SHARE:
            h = self.c_encoder(inputs=c, seq_len=c_len, init_hidden_state=None)

        # Step 2: combine H and U using "Attention"
        #         e.g. s = h.T * u
        #              a_x = softmax(s)
        #              a_q = softmax(s.T)
        #              u_hat = sum(a_x*u)
        #              h_hat = sum(a_q*h)
        #              g = combine(u, h, u_hat, h_hat)
        # --------op1--------------
        g = self.attention(h, u, c_mask, q_mask, JX=JX,JQ=JQ)  # concat[h, u_a, h*u_a, h*h_a]

        # Step 3:
        # 2 LSTM layers
        # logistic regressions
        pred1, pred2 = self.decoder(g, c_len, c_mask)
        return pred1, pred2

    # def setup_embeddings(self):
    #     with vs.variable_scope("embeddings"):
    #         if self.config.RE_TRAIN_EMBED:
    #             pretrained_embeddings = tf.Variable(self.pretrained_embeddings, name="Emb", dtype=tf.float32)
    #         else:
    #             pretrained_embeddings = tf.cast(self.pretrained_embeddings, tf.float32)
    #
    #         question_embeddings = tf.nn.embedding_lookup(pretrained_embeddings, self.question_placeholder)
    #         question_embeddings = tf.reshape(question_embeddings, shape = [-1, self.JQ, self.config.embedding_size])
    #
    #         context_embeddings = tf.nn.embedding_lookup(pretrained_embeddings, self.context_placeholder)
    #         context_embeddings = tf.reshape(context_embeddings, shape = [-1, self.JX, self.config.embedding_size])
    #
    #     return question_embeddings, context_embeddings

    def setup_embeddings_pytorch(self,q,c):

        q_emb = self.emb_layer(q).float()

        c_emb = self.emb_layer(c).float()

        return q_emb, c_emb

