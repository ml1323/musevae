import argparse
import numpy as np
import torch

from lg_cvae import Solver as lg_solver
from sg_net import Solver as sg_solver
from micro import Solver as micro_solver
from eval import Solver as eval_solver


# set the random seed manually for reproducibility
SEED = 1
np.random.seed(SEED)
torch.manual_seed(SEED)
if torch.cuda.is_available():
    torch.cuda.manual_seed(SEED)


###############################################################################

def print_opts(opts):
    '''
    Print the values of all command-line arguments
    '''

    print('=' * 80)
    print('Opts'.center(80))
    print('-' * 80)
    for key in opts.__dict__:
        if opts.__dict__[key]:
            print('{:>30}: {:<30}'.format(key, opts.__dict__[key]).center(80))
    print('=' * 80)


# -----------------------------------------------------------------------------#

def create_parser():
    '''
    Create a parser for command-line arguments
    '''

    parser = argparse.ArgumentParser()

    parser.add_argument('--run_id', default=0, type=int,
                        help='run id')
    parser.add_argument('--model_name', default='micro', type=str,
                        help='model name: one of [lg_ae, lg_cvae, sg_net, micro]')
    parser.add_argument('--device', default='cpu', type=str,
                        help='cpu/cuda')

    # training hyperparameters
    parser.add_argument('--batch_size', default=8, type=int,
                        help='batch size')
    parser.add_argument('--lr', default=1e-3, type=float,
                        help='learning rate')

    # saving directories and checkpoint/sample iterations
    parser.add_argument('--ckpt_load_iter', default=0, type=int,
                        help='iter# to load the previously saved model ' +
                             '(default=0 to start from the scratch)')
    parser.add_argument('--max_iter', default=10000, type=float,
                        help='maximum number of batch iterations')
    parser.add_argument('--ckpt_dir', default='ckpts', type=str)

    # Dataset options
    parser.add_argument('--delim', default=',', type=str)
    parser.add_argument('--loader_num_workers', default=0, type=int)
    parser.add_argument('--dt', default=0.4, type=float)
    parser.add_argument('--obs_len', default=8, type=int)
    parser.add_argument('--pred_len', default=12, type=int)
    parser.add_argument('--skip', default=1, type=int)
    parser.add_argument('--dataset_dir', default='./datasets/pfsd', type=str, help='dataset directory')
    parser.add_argument('--dataset_name', default='pfsd', type=str,
                        help='dataset name')
    parser.add_argument('--scale', default=1.0, type=float)
    parser.add_argument('--heatmap_size', default=160, type=int)
    parser.add_argument('--anneal_epoch', default=10, type=int)

    # Macro
    parser.add_argument('--pretrained_lg_path', default='ckpts/pretrained_models_pfsd/lg_cvae.pt', type=str)
    parser.add_argument('--w_dim', default=10, type=int)
    parser.add_argument('--fcomb', default=2, type=int)
    parser.add_argument('--fb', default=1, type=float)
    parser.add_argument('--num_goal', default=3, type=int)


    # Micro
    parser.add_argument('--kl_weight', default=50.0, type=float,
                        help='kl weight')
    parser.add_argument('--ll_prior_w', default=1.0, type=float)
    parser.add_argument('--z_dim', default=20, type=int,
                        help='dimension of the shared latent representation')
    parser.add_argument('--encoder_h_dim', default=64, type=int)
    parser.add_argument('--decoder_h_dim', default=128, type=int)
    parser.add_argument('--map_feat_dim', default=32, type=int)
    parser.add_argument('--dropout_mlp', default=0.3, type=float)
    parser.add_argument('--dropout_rnn', default=0.25, type=float)
    parser.add_argument('--mlp_dim', default=256, type=int)
    parser.add_argument('--map_mlp_dim', default=256, type=int)

    # Evaluation
    parser.add_argument('--n_w', default=5, type=int)
    parser.add_argument('--n_z', default=1, type=int)
    return parser


# -----------------------------------------------------------------------------#

def main(args):
    assert args.dataset_name in ['pfsd', 'sdd', 'nuScenes'], ('For dataset_name, only one of [pfsd, sdd, nuScenes] are supported.')

    if args.ckpt_load_iter == args.max_iter:

        solver = eval_solver(args)
        solver.load_checkpoint()

        ade_min, fde_min = solver.all_evaluation()
        print('------------------------------------------')
        print('dataset name: ', args.dataset_name)
        print('lg_sampling, micro_sampling: ', args.n_w, ', ', args.n_z)
        print('ade min: ', ade_min)
        print('fde min: ', fde_min)
        print('------------------------------------------')

    else:
        if args.model_name =='lg_ae' or args.model_name =='lg_cvae':
            solver = lg_solver(args)
        elif args.model_name =='sg_net':
            solver = sg_solver(args)
        elif args.model_name =='micro':
            solver = micro_solver(args)
        else:
            raise ValueError('model_name should be one of [lg_ae, lg_cvae, sg_net, micro]')
        solver.train()


###############################################################################

if __name__ == "__main__":
    parser = create_parser()
    args = parser.parse_args()
    print_opts(args)
    main(args)
