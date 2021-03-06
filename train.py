"""Training GCMC model on the MovieLens data set.

The script loads the full graph to the training device.
"""
import os
import time
import argparse
import logging
import random
import string
import numpy as np
import torch
import torch as th
import torch.nn as nn
from opacus import PrivacyEngine
from data import MovieLens
from model import BiDecoder, GCMCLayer
from utils import get_activation, get_optimizer, torch_total_param_num, torch_net_info, MetricLogger, GraphNorm
from discriminators import Discriminator

from opacus.accountants import RDPAccountant
#from .dputil import get_noise_multiplier
from opacus.accountants.utils import get_noise_multiplier

from opacus.grad_sample import GradSampleModule
from opacus.optimizers import DPOptimizer
from opacus.validators import ModuleValidator


class Net(nn.Module):
    def __init__(self, args):
        super(Net, self).__init__()
        self._act = get_activation(args.model_activation)
        self.encoder = GCMCLayer(args.rating_vals,
                                 args.src_in_units,
                                 args.dst_in_units,
                                 args.gcn_agg_units,
                                 args.gcn_out_units,
                                 args.gcn_dropout,
                                 args.gcn_agg_accum,
                                 agg_act=self._act,
                                 share_user_item_param=args.share_param,
                                 device=args.device)
        self.decoder = BiDecoder(in_units=args.gcn_out_units,
                                 num_classes=len(args.rating_vals),
                                 num_basis=args.gen_r_num_basis_func)

        self.gn = GraphNorm('gn',  args.gcn_out_units)

    def forward(self, enc_graph, dec_graph, ufeat, ifeat):
        user_out, movie_out = self.encoder(
            enc_graph,
            ufeat,
            ifeat)
        # graphnorm on both user and movie embeddings (viewed as one graph)
        padded_graph_out = torch.cat([user_out, movie_out])
        normalized_graph_out = self.gn(dec_graph, padded_graph_out)
        normalized_user_out, normalized_movie_out = normalized_graph_out[:user_out.shape[0]], normalized_graph_out[user_out.shape[0]:]
        # end of adding graphnorm
        pred_ratings = self.decoder(dec_graph, normalized_user_out, normalized_movie_out)
        return pred_ratings

    def encode(self, enc_graph, ufeat, ifeat):
        user_out, movie_out = self.encoder(
            enc_graph,
            ufeat,
            ifeat)
        return user_out, movie_out


def evaluate(args, net, dataset, segment='valid'):
    possible_rating_values = dataset.possible_rating_values
    nd_possible_rating_values = th.FloatTensor(
        possible_rating_values).to(args.device)

    if segment == "valid":
        rating_values = dataset.valid_truths
        enc_graph = dataset.valid_enc_graph
        dec_graph = dataset.valid_dec_graph
    elif segment == "test":
        rating_values = dataset.test_truths
        enc_graph = dataset.test_enc_graph
        dec_graph = dataset.test_dec_graph
    else:
        raise NotImplementedError

    # Evaluate RMSE
    net.eval()
    with th.no_grad():
        pred_ratings = net(enc_graph, dec_graph,
                           dataset.user_feature, dataset.movie_feature)
    real_pred_ratings = (th.softmax(pred_ratings, dim=1) *
                         nd_possible_rating_values.view(1, -1)).sum(dim=1)
    rmse = ((real_pred_ratings - rating_values) ** 2.).mean().item()
    rmse = np.sqrt(rmse)
    return rmse

def evaluate_dist(dis_net, usr_embds, dis_labels):
    dis_net.eval()
    correct = 0
    total = 0
    criterion = nn.CrossEntropyLoss()
    loss = 0
    with th.no_grad():
        outputs = dis_net.predict(usr_embds)['output']
        predicted = _, predicted = th.max(outputs.data, 1)
        total += dis_labels.size(0)
        loss += criterion(outputs, dis_labels)
        correct += (predicted == dis_labels).sum().item()

    print('Accuracy of the Discriminator on the %d users: %f %%' % (total, 100.0 * correct / total), f'loss:{loss}')
    return correct / total
    
def evaluate_fair(args, usr_embds, dis_labels, max_epoch = 20):
    print('End of Main Training. Begin FairEvaluation:')
    # Build a discriminator
    eval_dis_net = Discriminator(embed_dim=args.gcn_out_units, out_dim=2, random_seed=2020, dropout=0.3, neg_slope=0.2,
                            model_dir_path='../model/', model_name='', name='gender_eval').to(args.device)
    # Optimizer for discriminator
    eval_dis_opt = get_optimizer(args.train_optimizer)(
        eval_dis_net.parameters(), lr=args.dis_lr)
    eval_dis_net.train()
    for _ in range(max_epoch):
        dis_loss = eval_dis_net(usr_embds, dis_labels)
        eval_dis_opt.zero_grad()
        dis_loss.backward()
        eval_dis_opt.step()
        evaluate_dist(eval_dis_net, usr_embds, dis_labels)
    eval_dis_net.eval()
    return evaluate_dist(eval_dis_net, usr_embds, dis_labels)

def train(args):
    print(args)
    dataset = MovieLens(args.data_name, args.device, use_one_hot_fea=args.use_one_hot_fea, symm=args.gcn_agg_norm_symm,
                        test_ratio=args.data_test_ratio, valid_ratio=args.data_valid_ratio)
    print("Loading data finished ...\n")

    args.src_in_units = dataset.user_feature_shape[1]
    args.dst_in_units = dataset.movie_feature_shape[1]
    args.rating_vals = dataset.possible_rating_values

    # build the net
    net = Net(args=args)
    net = net.to(args.device)
    # print(net)
    nd_possible_rating_values = th.FloatTensor(
        dataset.possible_rating_values).to(args.device)
    rating_loss_net = nn.CrossEntropyLoss()
    learning_rate = args.train_lr

    optimizer = get_optimizer(args.train_optimizer)(
        net.parameters(), lr=learning_rate)
    dp_opt = get_optimizer(args.train_optimizer)(
        net.decoder.combine_basis.parameters(), lr=learning_rate)

    if not args.disable_dp:
        # initialize privacy accountant
        accountant = RDPAccountant()

        noise_multiplier = get_noise_multiplier(
            target_epsilon=args.epsilon,
            target_delta=args.delta,
            sample_rate=1,
            epochs=2000,
            accountant=accountant.mechanism()
        )

        noise_multiplier = float(noise_multiplier)
        print('Epsilon', args.epsilon, 'Noise Multiplier', noise_multiplier)
        net.decoder.combine_basis = GradSampleModule(net.decoder.combine_basis)

        dp_opt = DPOptimizer(
            optimizer=dp_opt,
            noise_multiplier=noise_multiplier,  # same as make_private arguments
            max_grad_norm=args.train_grad_clip,  # same as make_private arguments
            # if you're averaging your gradients, you need to know the denominator
            expected_batch_size=20000
        )

        dp_opt.attach_step_hook(
            accountant.get_optimizer_hook_fn(
                sample_rate=1
            )
        )

    if not args.disable_fair:
        # Build a discriminator
        dis_net = Discriminator(embed_dim=args.gcn_out_units, out_dim=2, random_seed=2020, dropout=0.3, neg_slope=0.2,
                                model_dir_path='../model/', model_name='', name='gender').to(args.device)
        # Pick sensitive attribute, here we only consider "gender" for ml-100k
        dis_labels = th.FloatTensor(dataset._process_user_fea())[
            :, 1].type(th.LongTensor).to(args.device)
        # Optimizer for discriminator
        dis_opt = get_optimizer(args.train_optimizer)(
            dis_net.parameters(), lr=args.dis_lr)

    print("Loading network finished ...\n")

    # perpare training data
    train_gt_labels = dataset.train_labels
    train_gt_ratings = dataset.train_truths

    # prepare the logger
    train_loss_logger = MetricLogger(['iter', 'loss', 'rmse'], ['%d', '%.4f', '%.4f'],
                                     os.path.join(args.save_dir, 'train_loss%d.csv' % args.save_id))
    valid_loss_logger = MetricLogger(['iter', 'rmse'], ['%d', '%.4f'],
                                     os.path.join(args.save_dir, 'valid_loss%d.csv' % args.save_id))
    test_loss_logger = MetricLogger(['iter', 'rmse'], ['%d', '%.4f'],
                                    os.path.join(args.save_dir, 'test_loss%d.csv' % args.save_id))

    # declare the loss information
    best_valid_rmse = np.inf
    no_better_valid = 0
    best_iter = -1
    count_rmse = 0
    count_num = 0
    count_loss = 0

    dataset.train_enc_graph = dataset.train_enc_graph.int().to(args.device)
    dataset.train_dec_graph = dataset.train_dec_graph.int().to(args.device)
    dataset.valid_enc_graph = dataset.train_enc_graph
    dataset.valid_dec_graph = dataset.valid_dec_graph.int().to(args.device)
    dataset.test_enc_graph = dataset.test_enc_graph.int().to(args.device)
    dataset.test_dec_graph = dataset.test_dec_graph.int().to(args.device)

    print("Start training ...")
    dur = []
    for iter_idx in range(1, args.train_max_iter):
        if iter_idx > 3:
            t0 = time.time()

        if not args.disable_fair:
            # Train Discriminator
            net.eval()
            dis_net.train()
            with th.no_grad():
                usr_embds, _ = net.encode(dataset.train_enc_graph, dataset.user_feature, dataset.movie_feature)
            for _ in range(args.dis_step):
                dis_loss = dis_net(usr_embds, dis_labels)
                dis_opt.zero_grad()
                dis_loss.backward()
                dis_opt.step()
            dis_net.eval()
            evaluate_dist(dis_net, usr_embds, dis_labels)

        net.train()
        pred_ratings = net(dataset.train_enc_graph, dataset.train_dec_graph,
                           dataset.user_feature, dataset.movie_feature)

        rs_loss = rating_loss_net(pred_ratings, train_gt_labels).mean()
        count_loss += rs_loss.item()

        if not args.disable_fair:
            usr_embds, _ = net.encode(
                dataset.train_enc_graph, dataset.user_feature, dataset.movie_feature)
            dis_loss = dis_net(usr_embds, dis_labels)
            loss = rs_loss - args.lam*dis_loss
            dis_opt.zero_grad()
        else:
            loss = rs_loss

        count_loss += loss.item()
        optimizer.zero_grad()
        dp_opt.zero_grad()
        loss.backward()
        nn.utils.clip_grad_norm_(net.parameters(), args.train_grad_clip)

        for p in net.decoder.combine_basis.parameters():
            p.requires_grad = False
        optimizer.step()

        for p in net.decoder.combine_basis.parameters():
            p.requires_grad = True
        dp_opt.step()

        if iter_idx > 3:
            dur.append(time.time() - t0)

        if iter_idx == 1:
            print("Total #Param of net: %d" % (torch_total_param_num(net)))
            print(torch_net_info(net, save_path=os.path.join(
                args.save_dir, 'net%d.txt' % args.save_id)))

        real_pred_ratings = (th.softmax(pred_ratings, dim=1) *
                             nd_possible_rating_values.view(1, -1)).sum(dim=1)
        rmse = ((real_pred_ratings - train_gt_ratings) ** 2).sum()
        count_rmse += rmse.item()
        count_num += pred_ratings.shape[0]

        if iter_idx % args.train_log_interval == 0:
            train_loss_logger.log(iter=iter_idx,
                                  loss=count_loss/(iter_idx+1), rmse=count_rmse/count_num)
            logging_str = "Iter={}, loss={:.4f}, rmse={:.4f}, time={:.4f}".format(
                iter_idx, count_loss/iter_idx, count_rmse/count_num,
                np.average(dur))
            count_rmse = 0
            count_num = 0
            if not args.disable_dp:
                epsilon, best_alpha = accountant.get_privacy_spent(
                    delta=args.delta
                )
                print(
                    f"(?? = {epsilon:.2f}, ?? = {args.delta}) for ?? = {best_alpha}"
                )

        if iter_idx % args.train_valid_interval == 0:
            valid_rmse = evaluate(args=args, net=net,
                                  dataset=dataset, segment='valid')
            valid_loss_logger.log(iter=iter_idx, rmse=valid_rmse)
            logging_str += ',\tVal RMSE={:.4f}'.format(valid_rmse)

            if valid_rmse < best_valid_rmse:
                best_valid_rmse = valid_rmse
                no_better_valid = 0
                best_iter = iter_idx
                test_rmse = evaluate(args=args, net=net,
                                     dataset=dataset, segment='test')
                best_test_rmse = test_rmse
                test_loss_logger.log(iter=iter_idx, rmse=test_rmse)
                logging_str += ', Test RMSE={:.4f}'.format(test_rmse)
            else:
                no_better_valid += 1
                if no_better_valid > args.train_early_stopping_patience\
                        and learning_rate <= args.train_min_lr:
                    logging.info(
                        "Early stopping threshold reached. Stop training.")
                    break
                if no_better_valid > args.train_decay_patience:
                    new_lr = max(learning_rate *
                                 args.train_lr_decay_factor, args.train_min_lr)
                    if new_lr < learning_rate:
                        learning_rate = new_lr
                        logging.info("\tChange the LR to %g" % new_lr)
                        for p in optimizer.param_groups:
                            p['lr'] = learning_rate
                        no_better_valid = 0
        if iter_idx % args.train_log_interval == 0:
            print(logging_str)
            
    # Evaluate Fairness
    if not args.disable_fair:
        with th.no_grad():
            usr_embds, _ = net.encode(dataset.train_enc_graph, dataset.user_feature, dataset.movie_feature)
        evaluate_fair(args, usr_embds, dis_labels, max_epoch = args.train_max_iter)
        
    print('Best Iter Idx={}, Best Valid RMSE={:.4f}, Best Test RMSE={:.4f}'.format(
        best_iter, best_valid_rmse, best_test_rmse))
    train_loss_logger.close()
    valid_loss_logger.close()
    test_loss_logger.close()


def config():
    parser = argparse.ArgumentParser(description='GCMC')
    parser.add_argument('--seed', default=2, type=int)
    parser.add_argument('--device', default='0', type=int,
                        help='Running device. E.g `--device 0`, if using cpu, set `--device -1`')
    parser.add_argument('--save_dir', type=str, help='The saving directory')
    parser.add_argument('--save_id', type=int, help='The saving log id')
    parser.add_argument('--silent', action='store_true')
    parser.add_argument('--data_name', default='ml-100k', type=str,
                        help='The dataset name: ml-100k, ml-1m, ml-10m')
    # for ml-100k the test ration is 0.2
    parser.add_argument('--data_test_ratio', type=float, default=0.1)
    parser.add_argument('--data_valid_ratio', type=float, default=0.1)
    parser.add_argument('--use_one_hot_fea', action='store_true', default=True)
    parser.add_argument('--model_activation', type=str, default="leaky")
    parser.add_argument('--gcn_dropout', type=float, default=0.7)
    parser.add_argument('--gcn_agg_norm_symm', type=bool, default=True)
    parser.add_argument('--gcn_agg_units', type=int, default=500)
    parser.add_argument('--gcn_agg_accum', type=str, default="sum")
    parser.add_argument('--gcn_out_units', type=int, default=75)
    parser.add_argument('--gen_r_num_basis_func', type=int, default=2)
    parser.add_argument('--train_max_iter', type=int, default=2000)
    parser.add_argument('--train_log_interval', type=int, default=1)
    parser.add_argument('--train_valid_interval', type=int, default=1)
    parser.add_argument('--train_optimizer', type=str, default="adam")
    parser.add_argument('--train_grad_clip', type=float, default=1.0)
    parser.add_argument('--train_lr', type=float, default=0.01)
    parser.add_argument('--train_min_lr', type=float, default=0.001)
    parser.add_argument('--train_lr_decay_factor', type=float, default=0.5)
    parser.add_argument('--train_decay_patience', type=int, default=50)
    parser.add_argument('--train_early_stopping_patience',
                        type=int, default=100)
    parser.add_argument('--share_param', default=False, action='store_true')
    parser.add_argument('--max-per-sample-grad_norm', type=float,
                        default=10.0, help="Clip per-sample gradients to this norm", )
    parser.add_argument(
        "--delta",
        type=float,
        default=1e-5,
        metavar="D",
        help="Target delta (default: 1e-5)",
    )
    parser.add_argument(
        "--epsilon",
        type=float,
        default=1,
        metavar="D",
        help="Target epsilon",
    )
    parser.add_argument(
        "--disable-dp",
        action="store_true",
        default=False,
        help="Disable privacy training and just train with vanilla optimizer",
    )
    parser.add_argument(
        "--secure-rng",
        action="store_true",
        default=False,
        help="Enable Secure RNG to have trustworthy privacy guarantees. Comes at a performance cost",
    )
    parser.add_argument(
        "--disable-fair",
        action="store_true",
        default=False,
        help="Disable fair training and train without discriminator",
    )
    parser.add_argument(
        "--dis_lr",
        type=float,
        default=1e-2,
        help="Learning rate for discriminator.",
    )
    parser.add_argument(
        "--lam",
        type=float,
        default=1,
        help="A factor for the tradeoff between fairness and performance (loss=rs_loss-lam*dis_loss). (default: 10)",
    )
    parser.add_argument(
        "--dis_step",
        type=int,
        default=100,
        help="Number of steps for training discriminator per epoch. (default: 10)",
    )

    args = parser.parse_args()
    args.device = th.device(
        args.device) if args.device >= 0 else th.device('cpu')

    # configure save_fir to save all the info
    if args.save_dir is None:
        args.save_dir = args.data_name+"_" + \
            ''.join(random.choices(string.ascii_uppercase + string.digits, k=2))
    if args.save_id is None:
        args.save_id = np.random.randint(20)
    args.save_dir = os.path.join("log", args.save_dir)
    if not os.path.isdir(args.save_dir):
        os.makedirs(args.save_dir)

    return args


if __name__ == '__main__':
    args = config()
    np.random.seed(args.seed)
    th.manual_seed(args.seed)
    if th.cuda.is_available():
        th.cuda.manual_seed_all(args.seed)
    train(args)
