import optparse
from collections import OrderedDict
import torch
import os

def add_data_args(parser):
    parser.add_option("-T", "--train", default="dataset/eng.train",help="Train set location")
    parser.add_option("-d", "--dev", default="dataset/eng.testa", help="Dev set location")
    parser.add_option("-t", "--test", default="dataset/eng.testb", help="Test set location")
    parser.add_option('--test_train', default='dataset/eng.train54019', help='test train')
    
    parser.add_option("-s", "--tag_scheme", default="iobes",help="Tagging scheme (IOB or IOBES)")
    parser.add_option("-l", "--lower", default="1",type='int', help="Lowercase words (this will not affect character inputs)")
    parser.add_option("-z", "--zeros", default="0",type='int', help="Replace digits with 0")
    
    return parser

def add_load_args(parser):
    parser.add_option("-p", "--pre_emb", default="models/glove.6B.100d.txt",help="Location of pretrained embeddings")
    parser.add_option("-A", "--all_emb", default="1",type='int', help="Load all embeddings")
    
    parser.add_option("-r", "--reload", default="",type='string', help="Reload the saved model via name ")
    return parser

    
def add_save_args(parser):
    parser.add_option('--name', default='test',help='model name')
    parser.add_option('--loss', default='loss.txt',help='loss file location')
    return parser


def add_model_args(parser):
    parser.add_option("-c", "--char_dim", default="25",type='int', help="Char embedding dimension")
    parser.add_option("-C", "--char_lstm_dim", default="25",type='int', help="Char LSTM hidden layer size")
    parser.add_option("-b", "--char_bidirect", default="1",type='int', help="Use a bidirectional LSTM for chars")
    parser.add_option("-w", "--word_dim", default="100",type='int', help="Token embedding dimension")
    parser.add_option("-W", "--word_lstm_dim", default="200",type='int', help="Token LSTM hidden layer size")
    parser.add_option("-B", "--word_bidirect", default="1",type='int', help="Use a bidirectional LSTM for words")

    parser.add_option("-a", "--cap_dim", default="0",type='int', help="Capitalization feature dimension (0 to disable)")
    parser.add_option("-f", "--crf", default="1",type='int', help="Use CRF (0 to disable)")
    parser.add_option("-D", "--dropout", default="0.5",type='float', help="Droupout on the input (0 = no dropout)")
    parser.add_option("-g", '--use_gpu', default='1',type='int', help='whether or not to ues gpu')


    parser.add_option('--char_mode', choices=['CNN', 'LSTM'], default='CNN',help='char_CNN or char_LSTM')

    parser.add_option("--epochs", default='150',type='int', help='training epochs')
    parser.add_option("--batch_size", default='64',type='int', help='training epochs')
    parser.add_option("--early_stop", action='store_true',help='apply early stop')
    return parser


def add_adv_args(parser):
    parser.add_option("--adv", action='store_true',help='use adversarial training or not')
    parser.add_option("--norm", action='store_true', help='normalize input embeddings or not')
    parser.add_option("--alpha", default='0.0',type='float', help='alpha')
    return parser


def add_non_adv_args(parser):
    parser.add_option("--non_gradient", action='store_true',help='using non-gradient adv examples or not')
    parser.add_option('--warmup', type=float, default="0.01",help='percentage of data to warmup on (.01 = 1% of all training iters). Default 0.01')
    parser.add_option('--warmup_style', choices=['constant','linear','exponential'], default='linear',help='learning rate decay function')
    parser.add_option("--exp_weight", default='1.0',type='float', help='weight for exp')
    parser.add_option("--launch_epoch", default='0',type='int', help='introducing paraphrases after certain epoch')
    parser.add_option("--per_adv", default='1',type='int', help='number of generated adv examples for one training sample ')
    parser.add_option("--adv_path", default="",help="Path to adv examples")
    parser.add_option("--dynamic_inference", action='store_true' ,help="inference adv example before feed to the model ")
    
    
    return parser



def check_args(opts, parameters):
    assert os.path.isfile(opts.train)
    assert os.path.isfile(opts.dev)
    assert os.path.isfile(opts.test)
    assert parameters['char_dim'] > 0 or parameters['word_dim'] > 0
    assert 0. <= parameters['dropout'] < 1.0
    assert parameters['tag_scheme'] in ['iob', 'iobes']
    assert not parameters['all_emb'] or parameters['pre_emb']
    assert not parameters['pre_emb'] or parameters['word_dim'] > 0
    assert not parameters['pre_emb'] or os.path.isfile(parameters['pre_emb'])

    
def get_args():
    
    optparser = optparse.OptionParser()
    
    optparser=add_data_args(optparser)
    optparser=add_load_args(optparser)
    optparser=add_save_args(optparser)
    optparser=add_model_args(optparser)
    optparser=add_non_adv_args(optparser)
    optparser=add_adv_args(optparser)

    
    opts = optparser.parse_args()[0]
    parameters = OrderedDict()
    parameters['tag_scheme'] = opts.tag_scheme
    parameters['lower'] = opts.lower == 1
    parameters['zeros'] = opts.zeros == 1
    parameters['char_dim'] = opts.char_dim
    parameters['char_lstm_dim'] = opts.char_lstm_dim
    parameters['char_bidirect'] = opts.char_bidirect == 1
    parameters['word_dim'] = opts.word_dim
    parameters['word_lstm_dim'] = opts.word_lstm_dim
    parameters['word_bidirect'] = opts.word_bidirect == 1
    parameters['pre_emb'] = opts.pre_emb
    parameters['all_emb'] = opts.all_emb == 1
    parameters['cap_dim'] = opts.cap_dim
    parameters['crf'] = opts.crf == 1
    parameters['dropout'] = opts.dropout
    parameters['reload'] = opts.reload
    parameters['name'] = opts.name
    parameters['char_mode'] = opts.char_mode
    parameters['epochs']=opts.epochs

    parameters['norm']=opts.norm
    parameters['adv']=opts.adv
    parameters['alpha']=opts.alpha
    
    
    parameters['non_gradient']=opts.non_gradient
    parameters['exp_weight']=opts.exp_weight
    parameters['warmup']=opts.warmup
    parameters['warmup_style']=opts.warmup_style
    parameters['launch_epoch']=opts.launch_epoch
    parameters['per_adv']=opts.per_adv
    parameters['adv_path']=opts.adv_path
    parameters['dynamic_inference']=opts.dynamic_inference
    parameters['batch_size']=opts.batch_size
    parameters['early_stop']=opts.early_stop

    
    
    parameters['use_gpu'] = opts.use_gpu == 1 and torch.cuda.is_available()

    
    check_args(opts,parameters)
    
    return opts, parameters

