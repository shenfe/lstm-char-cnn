
import os
import argparse

import torch
import torch.optim as optim

import data as data_loader
from model.model import CNLM
from train import train_eval


FILE_NAME_LIST = ['train.txt', 'valid.txt', 'test.txt']


def parse_args():
    parser = argparse.ArgumentParser(
        formatter_class=argparse.RawDescriptionHelpFormatter)

    def _(*args, **kwargs):
        parser.add_argument(*args, **kwargs)

    # data params
    _('--data_dir', '-data', metavar='PATH', dest='ddir', default='./data_dir/', help='todo')
    _('--train_dir', '-train', metavar='PATH', dest='tdir', default='./train_dir/', help='todo')
    _('--model_dir', '-model', metavar='PATH', dest='mdir', default='./model_dir/', help='todo')

    # model params
    _('--rnn_hidden', '-hdim', type=int, dest='hdim', default=650, help='todo')
    _('--char_embed_size', '-cedim', type=int, dest='cedim', default=15, help='todo')
    _('--word_embed_size', '-wedim', type=int, dest='wedim', default=128, help='todo')
    _('--high_layers', '-hlayers', type=int, dest='hlayers', default=2, help='todo')
    _('--rnn_layers', '-rlayers', type=int, dest='rlayers', default=2, help='todo')
    _('--dropout', '-drop', type=float, dest='drop', default=0.5, help='todo')
    _('--kernels_width', '-ker_wid', dest='ker_wid', default='[1, 2, 3, 4, 5, 6, 7]', help='todo')
    _('--kernels_nums', '-ker_num', dest='ker_num', default='[50, 100, 150, 200, 200, 200, 200]', help='todo')
    _('--param_init', '-pinit', type=float, dest='param_init', default=0.05, help='todo')

    # optimize params
    _('--learning_rate_decay', '-lr_decay', type=float, dest='lr_decay', default=0.5, help='todo')
    _('--decay_when', '-decay_when', type=float, dest='decay_when', default=1.0, help='todo')
    _('--learning_rate', '-lr', type=float, dest='lr', default=1.0, help='todo')
    _('--batch_size', '-batch', type=int, dest='batch', default=20, help='todo')
    _('--max_epoch', '-mepoch', type=int, dest='mepoch', default=25, help='todo')
    _('--max_steps', '-msteps', type=int, dest='msteps', default=10000, help='todo')
    _('--max_sent_len', '-maxlen', type=int, dest='maxlen', default=35, help='todo')
    _('--max_word_len', '-maxwdlen', type=int, dest='maxwdlen', default=30, help='todo')
    _('--clip', '-clip', type=float, dest='clip', default=5.0, help='todo')

    # book keeping
    _('--seed', '-seed', type=int, dest='seed', default=3435, help='todo')
    _('--display_frequency', '-dfreq', type=int, dest='dfreq', default=100, help='todo')
    _('--eval_frequency', '-efreq', type=int, dest='efreq', default=1000, help='todo')
    _('--use_cuda', '-cuda', dest='cuda', action='store_true', help='todo')
    _('--save_model', '-save', dest='save', action='store_true', help='todo')

    args = parser.parse_args()
    print(vars(args))
    return args


def preview_data(reader, vocab):
    cnt = 0
    for x, y in reader.iter():
        if cnt > 0:
            break
        cnt += 1
        # x <batch, maxlen>
        for s, t in zip(x, y):
            # per batch
            sln, tln = [], []
            for si, ti in zip(s, t):
                sln.append(vocab.token(si))
                tln.append(vocab.token(ti))
            print("%40s\t-\t%40s\n" % (" ".join(sln), " ".join(tln)))
        print("\t\t------batch--------\n")


def main(args):
    if not os.path.exists(args.tdir):
        os.mkdir(args.tdir)
        print('[Create] %s' % args.tdir)

    if args.cuda:
        assert torch.cuda.is_available()

    word_vocab, char_vocab, word_tensor, char_tensor, actual_maxwdlen = data_loader.load_data(
        args.ddir, 65, FILE_NAME_LIST)

    args.wvsize = word_vocab.size

    train_reader = data_loader.DataReader(word_tensor[FILE_NAME_LIST[0]], char_tensor[FILE_NAME_LIST[0]],
                                          args.batch, args.maxlen)
    valid_reader = data_loader.DataReader(word_tensor[FILE_NAME_LIST[1]], char_tensor[FILE_NAME_LIST[1]],
                                          args.batch, args.maxlen)
    test_reader = data_loader.DataReader(word_tensor[FILE_NAME_LIST[2]], char_tensor[FILE_NAME_LIST[2]],
                                         args.batch, args.maxlen)

    args.cvsize = char_vocab.size
    args.wvsize = word_vocab.size
    # preview_data(train_reader, word_vocab)

    if args.seed:
        torch.manual_seed(args.seed)

    ker_sizes = [int(item.strip()) for item in args.ker_wid[1: -1].split(',')]
    ker_feats = [int(item.strip()) for item in args.ker_num[1: -1].split(',')]
    c2w = sum(ker_feats)
    nlm = CNLM(cvsize=args.cvsize,
               cedim=args.cedim,
               wvsize=args.wvsize,
               wedim=args.wedim,
               cnn_size=c2w,
               hdim=args.hdim,
               kernel_sizes=ker_sizes,
               kernel_features=ker_feats,
               nhlayers=args.hlayers,
               nrlayers=args.rlayers,
               tie_weight=False,
               droprate=args.drop)

    print(nlm)

    if args.cuda:
        nlm.cuda()

    opt = optim.SGD(nlm.parameters(), lr=args.lr)

    train_eval(nlm, opt, train_reader, valid_reader, test_reader, args)


if __name__ == '__main__':

    args = parse_args()

    main(args)
