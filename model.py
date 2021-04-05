import torch
import torch.autograd as autograd
from torch.autograd import Variable
from utils import *
from crf import CRF

START_TAG = '<START>'
STOP_TAG = '<STOP>'


def to_scalar(var):
    return var.view(-1).data.tolist()[0]


def argmax(vec):
    _, idx = torch.max(vec, 1)
    return to_scalar(idx)


def prepare_sequence(seq, to_ix):
    idxs = [to_ix[w] for w in seq]
    tensor = torch.LongTensor(idxs)
    return Variable(tensor)


def log_sum_exp(vec):
    # vec 2D: 1 * tagset_size
    max_score = vec[0, argmax(vec)]
    max_score_broadcast = max_score.view(1, -1).expand(1, vec.size()[1])
    return max_score + \
        torch.log(torch.sum(torch.exp(vec - max_score_broadcast)))


class BiLSTM_CRF(nn.Module):

    def __init__(self, vocab_size, tag_to_ix, embedding_dim, hidden_dim, char_lstm_dim=25,
                 char_to_ix=None, pre_word_embeds=None, char_embedding_dim=25, use_gpu=False,
                 n_cap=None, cap_embedding_dim=None, use_crf=True, char_mode='CNN',alpha=0):
        super(BiLSTM_CRF, self).__init__()
        self.use_gpu = use_gpu
        self.embedding_dim = embedding_dim
        self.hidden_dim = hidden_dim
        self.vocab_size = vocab_size
        self.tag_to_ix = tag_to_ix
        self.n_cap = n_cap
        self.cap_embedding_dim = cap_embedding_dim
        self.use_crf = use_crf
        self.tagset_size = len(tag_to_ix)
        self.out_channels = char_lstm_dim
        self.char_mode = char_mode
        self.alpha=alpha

        print('char_mode: %s, out_channels: %d, hidden_dim: %d, ' % (char_mode, char_lstm_dim, hidden_dim))

        if self.n_cap and self.cap_embedding_dim:
            self.cap_embeds = nn.Embedding(self.n_cap, self.cap_embedding_dim)
            init_embedding(self.cap_embeds.weight)

        if char_embedding_dim is not None:
            self.char_lstm_dim = char_lstm_dim
            self.char_embeds = nn.Embedding(len(char_to_ix), char_embedding_dim)
            init_embedding(self.char_embeds.weight)
            if self.char_mode == 'LSTM':
                #self.char_lstm = nn.LSTM(char_embedding_dim, char_lstm_dim, num_layers=1, bidirectional=True)
                self.char_lstm=nn.LSTM(char_embedding_dim, char_lstm_dim, num_layers=1, bidirectional=True, batch_first=True)
                init_lstm(self.char_lstm)
            if self.char_mode == 'CNN':
                self.char_cnn3 = nn.Conv2d(in_channels=1, out_channels=self.out_channels, kernel_size=(3, char_embedding_dim), padding=(2,0))

        self.word_embeds = nn.Embedding(vocab_size, embedding_dim)
        if pre_word_embeds is not None:
            self.pre_word_embeds = True
            self.word_embeds.weight = nn.Parameter(torch.FloatTensor(pre_word_embeds))
        else:
            self.pre_word_embeds = False

        self.dropout = nn.Dropout(0.5)
        if self.n_cap and self.cap_embedding_dim:
            if self.char_mode == 'LSTM':
                #self.lstm = nn.LSTM(embedding_dim+char_lstm_dim*2+cap_embedding_dim, hidden_dim, bidirectional=True)
                self.lstm = nn.LSTM(embedding_dim+char_lstm_dim*2+cap_embedding_dim, hidden_dim, bidirectional=True, batch_first=True)
            if self.char_mode == 'CNN':
                self.lstm = nn.LSTM(embedding_dim+self.out_channels+cap_embedding_dim, hidden_dim, bidirectional=True, batch_first=True)
        else:
            if self.char_mode == 'LSTM':
                #self.lstm = nn.LSTM(embedding_dim+char_lstm_dim*2, hidden_dim, bidirectional=True)
                self.lstm = nn.LSTM(embedding_dim+char_lstm_dim*2, hidden_dim, bidirectional=True, batch_first=True)
            if self.char_mode == 'CNN':
                self.lstm = nn.LSTM(embedding_dim+self.out_channels, hidden_dim, bidirectional=True)
                
        init_lstm(self.lstm)
        self.hw_trans = nn.Linear(self.out_channels, self.out_channels)
        self.hw_gate = nn.Linear(self.out_channels, self.out_channels)
        self.h2_h1 = nn.Linear(hidden_dim*2, hidden_dim)
        self.tanh = nn.Tanh()
        self.hidden2tag = nn.Linear(hidden_dim*2, self.tagset_size)
        init_linear(self.h2_h1)
        init_linear(self.hidden2tag)
        init_linear(self.hw_gate)
        init_linear(self.hw_trans)

        if self.use_crf:
            self.crf = CRF(self.hidden_dim*2, self.tagset_size-2)


    def _get_lstm_features(self, sentence, chars2, caps, chars2_length, word_length,adv=False,grads=None):

        word_grads=[]
        char_grads=[]
        if adv:
            char_grads=grads[0]
            char_grads=_scale_unit_l2(char_grads)
            #char_grads=char_grads/(torch.norm(char_grads,dim=2).unsqueeze(2)+1e-8)
            word_grads=grads[1]
            word_grads=_scale_unit_l2(word_grads.unsqueeze(0)).squeeze(0)
            #word_grads=word_grads/(torch.norm(word_grads,dim=1).unsqueeze(1)+1e-8)

        chars_embeds = self.char_embeds(chars2)
        if adv:
            chars_embeds=chars_embeds+self.alpha*char_grads*(sum(chars2_length)*chars_embeds.shape[-1])**0.5


        if self.char_mode == 'LSTM':
            # self.char_lstm_hidden = self.init_lstm_hidden(dim=self.char_lstm_dim, bidirection=True, batchsize=chars2.size(0))
            #chars_embeds = self.char_embeds(chars2).transpose(0, 1)
            #chars_embeds=chars_embeds.transpose(0, 1)
            chars_embeds = self.char_embeds(chars2)
            #packed = torch.nn.utils.rnn.pack_padded_sequence(chars_embeds, chars2_length)
            packed = torch.nn.utils.rnn.pack_padded_sequence(chars_embeds, chars2_length, batch_first=True, enforce_sorted=False)
            lstm_out_char, _ = self.char_lstm(packed)
            outputs, output_lengths = torch.nn.utils.rnn.pad_packed_sequence(lstm_out_char, batch_first=True, total_length=chars_embeds.size(1))
            
            #outputs = outputs.transpose(0, 1)
            
            '''
            masking
            '''
            output_forward, output_backward = torch.chunk(outputs, 2, 2)
            index=torch.LongTensor(chars2_length)-1
            index=index.cuda()
            masks=index.view(-1,1,1).expand(output_forward.shape)
            output_forward=output_forward.gather(1,masks)[:,0,:]
            output_backward=output_backward[:,0,:]
            chars_embeds=torch.cat([output_forward,output_backward],dim=1)
            
            '''
            chars_embeds_temp = Variable(torch.FloatTensor(torch.zeros((outputs.size(0), outputs.size(2)))))
            if self.use_gpu:
                chars_embeds_temp = chars_embeds_temp.cuda()
            for i, index in enumerate(output_lengths):
                chars_embeds_temp[i] = torch.cat((outputs[i, index-1, :self.char_lstm_dim], outputs[i, 0, self.char_lstm_dim:]))
            chars_embeds = chars_embeds_temp.clone()
            for i in range(chars_embeds.size(0)):
                chars_embeds[matching_char[i]] = chars_embeds_temp[i]
            '''

        if self.char_mode == 'CNN':
            chars_embeds = self.char_embeds(chars2).unsqueeze(1)
            chars_cnn_out3 = self.char_cnn3(chars_embeds)
            chars_embeds = nn.functional.max_pool2d(chars_cnn_out3,
                                                 kernel_size=(chars_cnn_out3.size(2), 1)).view(chars_cnn_out3.size(0), self.out_channels)

        # t = self.hw_gate(chars_embeds)
        # g = nn.functional.sigmoid(t)
        # h = nn.functional.relu(self.hw_trans(chars_embeds))
        # chars_embeds = g * h + (1 - g) * chars_embeds
        
        word_mask=sentence.gt(0)
        seq_length = masks.sum(1)
        curr=0
        l=[]
        for i in word_length:
            l.append(chars_embeds[curr:curr+i,:])
            curr=curr+i

        chars_embeds_padd=torch.nn.utils.rnn.pad_sequence(l,batch_first=True)

        embeds = self.word_embeds(sentence)
        if self.n_cap and self.cap_embedding_dim:
            cap_embedding = self.cap_embeds(caps)

        if self.n_cap and self.cap_embedding_dim:
            embeds = torch.cat((embeds, chars_embeds, cap_embedding), 1)
        else:
            #embeds = torch.cat((embeds, chars_embeds), 1)
            embeds = torch.cat((embeds, chars_embeds_padd), 2)

        #embeds = embeds.unsqueeze(1)
        embeds = self.dropout(embeds)
        pack_sequence = torch.nn.utils.rnn.pack_padded_sequence(embeds, word_length, batch_first=True, enforce_sorted=False)
        packed_output, _ = self.lstm(pack_sequence)
        lstm_out, _ = torch.nn.utils.rnn.pad_packed_sequence(packed_output, batch_first=True)
        lstm_out = self.dropout(lstm_out)
        '''
        lstm_out = self.dropout(lstm_out)
        lstm_out, _ = self.lstm(embeds)
        lstm_out = lstm_out.view(len(sentence), self.hidden_dim*2)
        lstm_out = self.dropout(lstm_out)
        lstm_feats = self.hidden2tag(lstm_out)
        '''
        return lstm_out, word_mask
    
    
    def forward(self, sentence, chars, caps, chars2_length, word_length):
        feats, masks = self._get_lstm_features(sentence, chars, caps, chars2_length, word_length)
        # viterbi to get tag_seq
        if self.use_crf:
            #score, tag_seq = self.viterbi_decode(feats)
            scores, tag_seq = self.crf(feats, masks)
        else:
            scores, tag_seq = torch.max(feats, 1)
            tag_seq = list(tag_seq.cpu().data)
           
        return scores, tag_seq

    def loss(self, sentence, chars, caps, chars2_length, tags, word_length, avg=True):
        features, masks = self._get_lstm_features(sentence, chars, caps, chars2_length, word_length)
        loss = self.crf.loss(features, tags, masks=masks, avg=avg)
        return loss
