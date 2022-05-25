import argparse
import utils

parser = argparse.ArgumentParser("cifar")
parser.add_argument('--data', type=str, default='../data', help='location of the data corpus')
parser.add_argument('--batch_size', type=int, default=128, help='batch size')  # もともとは64
parser.add_argument('--learning_rate', type=float, default=0.05, help='init learning rate')  # もともとは0.025
parser.add_argument('--learning_rate_min', type=float, default=0.001, help='min learning rate')
parser.add_argument('--momentum', type=float, default=0.9, help='momentum')
parser.add_argument('--weight_decay', type=float, default=3e-4, help='weight decay')
parser.add_argument('--report_freq', type=float, default=50, help='report frequency')
parser.add_argument('--gpu', type=int, default=0, help='gpu device id')
parser.add_argument('--epochs', type=int, default=50, help='num of training epochs')
parser.add_argument('--init_channels', type=int, default=16, help='num of init channels')
parser.add_argument('--layers', type=int, default=8, help='total number of layers')
parser.add_argument('--model_path', type=str, default='saved_models', help='path to save the model')
parser.add_argument('--cutout', action='store_true', default=False, help='use cutout')
parser.add_argument('--cutout_length', type=int, default=16, help='cutout length')
parser.add_argument('--drop_path_prob', type=float, default=0.3, help='drop path probability')
parser.add_argument('--save', type=str, default='EXP', help='experiment name')
parser.add_argument('--seed', type=int, default=2, help='random seed')
parser.add_argument('--grad_clip', type=float, default=5, help='gradient clipping')
parser.add_argument('--train_portion', type=float, default=0.5, help='portion of training data')
parser.add_argument('--unrolled', action='store_true', default=False, help='use one-step unrolled validation loss')
parser.add_argument('--arch_learning_rate', type=float, default=3e-4, help='learning rate for arch encoding')
parser.add_argument('--arch_weight_decay', type=float, default=1e-3, help='weight decay for arch encoding')
parser.add_argument('--cifar100', action='store_true', default=False, help='search with cifar100 dataset')

# for auxiliary skip connect
parser.add_argument('--auxiliary_skip', action='store_true', default=False, help='add an auxiliary skip')
parser.add_argument('--skip_beta', type=float, default=1.0, help='ratio to overshoot or discount auxiliary skip')
parser.add_argument('--decay', default='cosine', choices=[None, 'cosine', 'slow_cosine','linear'], help='select scheduler decay on epochs')
parser.add_argument('--decay_start_epoch', type=int, default=0, help='epoch to start decay')
parser.add_argument('--decay_stop_epoch', type=int, default=50, help='epoch to stop decay')
parser.add_argument('--decay_max_epoch', type=int, default=50, help='max epochs to decay')

args = parser.parse_args()

beta_decay_scheduler = utils.DecayScheduler(base_lr=args.skip_beta, 
                                        T_max=args.decay_max_epoch, 
                                        T_start=args.decay_start_epoch, 
                                        T_stop=args.decay_stop_epoch, 
                                        decay_type=args.decay)
