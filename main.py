import os
import sys
import csv
import glob
import logging
import shutil
import argparse
import numpy as np
import matplotlib.pyplot as plt
import torch as T
import torch.nn as nn
import torch.optim as optim
import torch.utils.data as data

from torch.autograd import Function
from datetime import datetime
from pypianoroll import Multitrack, Track

# os.environ['CUDA_VISIBLE_DEVICES'] = '1'
DEVICE = T.device('cuda' if T.cuda.is_available() else 'cpu')


# ------- Arguments -------
class DataParams:  # `lpd_5_cleansed`
    track_names = ['drums', 'piano', 'guitar', 'bass', 'strings']
    program_nums = [118, 0, 24, 33, 49]
    is_drums = [True, False, False, False, False]
    tempo = 80.0
    beat_resolution = 24
    track_pair = [(1, 2), (1, 3), (1, 4), (2, 3), (2, 4), (3, 4)]
    track_pair_names = ['P-G', 'P-B', 'P-S', 'G-B', 'G-S', 'B-S']


dp = DataParams()


def get_args():
    parser = argparse.ArgumentParser()

    # experiment args
    parser.add_argument('--mode', default='infer', help="Support: train, infer")
    parser.add_argument('--exp_name', default="lpd5pne400_CVAE_FreeBits256_BinaryRegularizer_Coupled")

    # data args
    parser.add_argument('--data_dir', default='./data/lpd_5_cleansed-piano_non_empty400bool')
    parser.add_argument('--num_track', type=int, default=5)
    parser.add_argument('--num_timestep', type=int, default=400)
    parser.add_argument('--num_pitch', type=int, default=84)
    parser.add_argument('--c_track_idx', type=int, default=1)

    # hyper-parameters
    parser.add_argument('--bz', default=64)
    parser.add_argument('--lr', default=1e-3)
    parser.add_argument('--n_epoch', default=30)
    parser.add_argument('--free_bits', default=256)

    # model args
    parser.add_argument('--z_dim', default=16)

    # restore
    parser.add_argument('--restore_epoch', default=20, help="which epoch to restore checkpoint from")

    return parser.parse_args()


# ------- Log -------
def log(args):
    def check_path(path):
        if not os.path.exists(path):
            os.makedirs(path)

    def get_logger(log_file):
        logging.basicConfig(level=logging.INFO, format='%(message)s')
        logger = logging.getLogger()
        logger.addHandler(logging.FileHandler(log_file, mode='w'))
        return logger

    # create log paths
    log_paths = dict()
    log_paths['root'] = os.path.join('./log', args.exp_name)
    log_paths['src'] = os.path.join(log_paths['root'], 'src')
    log_paths['gen'] = os.path.join(log_paths['root'], 'gen')
    log_paths['model'] = os.path.join(log_paths['root'], 'model')
    for _, v in log_paths.items():
        check_path(v)

    # backup source codes
    shutil.copyfile('./main.py', os.path.join(log_paths['src'], 'main.py'))

    # create logger
    lgr = get_logger(os.path.join(log_paths['root'],   # args, model architecture, progress, etc.
                                  'exp_{}.log'.format(datetime.now().strftime("%Y%m%d_%H%M%S"))))
    lgr.info("\n\n{:-^70}".format(" Configuration "))
    for k, v in vars(args).items():
        lgr.info('%s = %s' % (str(k), str(v)))

    # create analysis data file
    analysis_file = os.path.join(log_paths['root'], 'analysis.csv')
    csv_f = open(analysis_file, 'a', newline='')
    csv_writer = csv.writer(csv_f)
    if not os.path.getsize(analysis_file):  # if file is empty, write header list
        header_list = ["loss", "reconstruct_loss", "latent_loss"]
        csv_writer.writerow(header_list)

    return log_paths, lgr, csv_writer


# ------- Dataset -------
class Piano400Subset(data.Dataset):
    """
    Segment each song into 400-timestep piano roll from lpd_5_cleansed. Remove those whose piano track is empty.

    Args:
      root - Where to put the samples
    """

    def __init__(self, root, c_track_idx):
        self.sample_names = glob.glob(os.path.join(root, "*.npy"))
        self.c_track_idx = c_track_idx

    def __getitem__(self, index):
        fname = self.sample_names[index]
        sample = np.load(fname)
        sample = np.transpose(sample, (2, 0, 1))  # set channel/track first    # (5, 400, 84)
        y = np.expand_dims(sample[self.c_track_idx, :, :], axis=0)             # (1, 400, 84)
        x = np.delete(sample, self.c_track_idx, axis=0)                        # (4, 400, 84)
        return x.astype(np.float32), y.astype(np.float32)

    def __len__(self):
        return len(self.sample_names)


# ------- Utils -------
def save_midi(pr, filename):
    # (num_time_step, 84, 5) > (num_time_step, 128, 5)
    pr = np.concatenate((np.zeros((pr.shape[0], 24, pr.shape[2])), pr,
                         np.zeros((pr.shape[0], 20, pr.shape[2]))), axis=1)
    pr = np.around(pr).astype(np.bool_)  # binary

    tracks = list()
    for idx in range(pr.shape[2]):
        track = Track(pr[..., idx], dp.program_nums[idx], dp.is_drums[idx], dp.track_names[idx])
        tracks.append(track)
    mt = Multitrack(tracks=tracks, tempo=dp.tempo, beat_resolution=dp.beat_resolution)

    mt.plot(filename=filename + '.png')
    plt.close('all')
    mt.write(filename + '.mid')


# ------- Metrics -------
def eval_tonal_distances(pr_batch):
    """ Evaluate tonal distance of one batch of piano rolls. """

    def to_chroma(piano_roll):
        chroma = piano_roll.reshape(piano_roll.shape[0], 12, -1).sum(axis=2)
        return chroma  # (num_time_step, 12)

    def metrics_harmonicity(chroma1, chroma2):

        def get_tonal_matrix(r1=1.0, r2=1.0, r3=0.5):
            tm = np.empty((6, 12), dtype=np.float32)
            tm[0, :] = r1 * np.sin(np.arange(12) * (7. / 6.) * np.pi)
            tm[1, :] = r1 * np.cos(np.arange(12) * (7. / 6.) * np.pi)
            tm[2, :] = r2 * np.sin(np.arange(12) * (3. / 2.) * np.pi)
            tm[3, :] = r2 * np.cos(np.arange(12) * (3. / 2.) * np.pi)
            tm[4, :] = r3 * np.sin(np.arange(12) * (2. / 3.) * np.pi)
            tm[5, :] = r3 * np.cos(np.arange(12) * (2. / 3.) * np.pi)
            return tm

        tonal_matrix = get_tonal_matrix()

        def tonal_distance(beat_chroma1, beat_chroma2):
            beat_chroma1 = beat_chroma1 / np.sum(beat_chroma1)
            c1 = np.matmul(tonal_matrix, beat_chroma1)
            beat_chroma2 = beat_chroma2 / np.sum(beat_chroma2)
            c2 = np.matmul(tonal_matrix, beat_chroma2)
            return np.linalg.norm(c1 - c2)

        score_list = []
        for r in range(chroma1.shape[0] // dp.beat_resolution):
            chr1 = np.sum(chroma1[dp.beat_resolution * r: dp.beat_resolution * (r + 1)], axis=0)
            chr2 = np.sum(chroma2[dp.beat_resolution * r: dp.beat_resolution * (r + 1)], axis=0)
            score_list.append(tonal_distance(chr1, chr2))
        return np.mean(score_list)

    num_batch = len(pr_batch)
    pair_num = len(dp.track_pair)
    score_pair_matrix = np.zeros([pair_num, num_batch]) * np.nan

    for idx in range(num_batch):
        pr = pr_batch[idx]

        # compute eval pair
        for p in range(pair_num):
            pair = dp.track_pair[p]
            score_pair_matrix[p, idx] = metrics_harmonicity(to_chroma(pr[:, :, pair[0]]),
                                                            to_chroma(pr[:, :, pair[1]]))
    score_pair_matrix_mean = np.nanmean(score_pair_matrix, axis=1)
    return score_pair_matrix_mean


# ------- Model -------
class Net(nn.Module):
    def __init__(self, args):
        super(Net, self).__init__()
        emb_y_dim = 256
        self.args = args
        self.en_out_dim = 512

        self.encoder_private = nn.ModuleList(
            [nn.Sequential(
                nn.Conv2d(1, 8, (7, 1), (5, 1), (1, 0)),   # 8x80x84
                nn.BatchNorm2d(8),
                nn.LeakyReLU(),
                nn.Conv2d(8, 16, (1, 4), (1, 2), (0, 1)),  # 16x80x42
                nn.BatchNorm2d(16),
                nn.LeakyReLU(),
                nn.Conv2d(16, 16, (4, 1), (2, 1), (1, 0)),  # 16x40x42
                nn.BatchNorm2d(16),
                nn.LeakyReLU(),
            ) for _ in range(self.args.num_track)]
        )

        self.encoder_shared = nn.Sequential(
            nn.Conv2d(16*5, 128, (1, 4), (1, 2), (0, 1)),  # 128x40x21
            nn.BatchNorm2d(128),
            nn.LeakyReLU(),
            nn.Conv2d(128, 256, (4, 1), (2, 1), (1, 0)),  # 256x20x21
            nn.BatchNorm2d(256),
            nn.LeakyReLU(),
            nn.Conv2d(256, 256, (1, 5), (1, 3), (0, 1)),  # 256x20x7
            nn.BatchNorm2d(256),
            nn.LeakyReLU(),
            nn.Conv2d(256, 512, (4, 1), (5, 1), (2, 0)),  # 512x5x7
            nn.BatchNorm2d(512),
            nn.LeakyReLU(),
            nn.Conv2d(512, self.en_out_dim, (5, 7), (1, 1), (0, 0)),  # self.en_out_dimx1x1
            nn.BatchNorm2d(self.en_out_dim),
            nn.LeakyReLU(),
        )
        self.compute_mu = nn.Linear(self.en_out_dim, self.args.z_dim)
        self.compute_sigma = nn.Linear(self.en_out_dim, self.args.z_dim)

        self.emb_y = nn.Sequential(
            nn.Conv2d(1, 64, (7, 1), (5, 1), (1, 0)),  # 64x80x84
            nn.BatchNorm2d(64),
            nn.LeakyReLU(),
            nn.Conv2d(64, 64, (4, 4), (2, 2), (1, 1)),  # 64x40x42
            nn.BatchNorm2d(64),
            nn.LeakyReLU(),
            nn.Conv2d(64, 128, (4, 4), (2, 2), (1, 1)),  # 128x20x21
            nn.BatchNorm2d(128),
            nn.LeakyReLU(),
            nn.Conv2d(128, 128, (4, 5), (5, 3), (2, 1)),  # 128x5x7
            nn.BatchNorm2d(128),
            nn.LeakyReLU(),
            nn.Conv2d(128, emb_y_dim, (5, 7), (1, 1), (0, 0)),  # emb_y_dimx1x1
            nn.BatchNorm2d(emb_y_dim),
            nn.LeakyReLU(),
        )

        self.decoder_shared = nn.Sequential(
            nn.ConvTranspose2d(self.args.z_dim+emb_y_dim, 512, (5, 7), (1, 1), (0, 0)),   # 512x5x7
            nn.BatchNorm2d(512),
            nn.ReLU(),
            nn.ConvTranspose2d(512, 256, (4, 1), (5, 1), (2, 0)),  # 256x20x7
            nn.BatchNorm2d(256),
            nn.ReLU(),
            nn.ConvTranspose2d(256, 256, (1, 5), (1, 3), (0, 1)),  # 256x20x21
            nn.BatchNorm2d(256),
            nn.ReLU(),
            nn.ConvTranspose2d(256, 128, (4, 1), (2, 1), (1, 0)),  # 128x40x21
            nn.BatchNorm2d(128),
            nn.ReLU(),
            nn.ConvTranspose2d(128, 128, (1, 4), (1, 2), (0, 1)),  # 128x40x42
            nn.BatchNorm2d(128),
            nn.ReLU(),
        )
        self.decoder_private = nn.ModuleList(
            [nn.Sequential(
                nn.ConvTranspose2d(128, 64, (4, 1), (2, 1), (1, 0)),  # 64x80x42
                nn.BatchNorm2d(64),
                nn.ReLU(),
                nn.ConvTranspose2d(64, 64, (1, 4), (1, 2), (0, 1)),  # 64x80x84
                nn.BatchNorm2d(64),
                nn.ReLU(),
                nn.ConvTranspose2d(64, 1, (7, 1), (5, 1), (1, 0)),  # 1x400x84
                nn.Sigmoid(),
            ) for _ in range(self.args.num_track-1)]
        )

        self.to(DEVICE)

    def forward_to_z(self, x, y):
        xy = T.cat((x, y), 1)

        out_list = []
        for i in range(self.args.num_track):
            en_private_out = self.encoder_private[i](xy[:, i, :, :].unsqueeze(1))
            out_list.append(en_private_out)
        private_out = T.cat(out_list, 1)     # (16*5, 40, 42)
        en_out = self.encoder_shared(private_out)
        en_out = en_out.view(-1, self.en_out_dim)

        mu = self.compute_mu(en_out)
        sigma = self.compute_sigma(en_out)
        return mu, sigma

    def forward(self, z, y):
        z = z.view(-1, self.args.z_dim, 1, 1)
        y = self.emb_y(y)

        zy = T.cat((z, y), 1)     # (z_dim+emb_y_dim, 1, 1)
        de_shared_out = self.decoder_shared(zy)
        out_list = []
        for i in range(self.args.num_track - 1):
            de_private_out = self.decoder_private[i](de_shared_out)
            out_list.append(de_private_out)
        x_ = T.cat(out_list, 1)     # 4x400x84
        return x_

    def summary(self, logger):
        model_parameters = filter(lambda p: p.requires_grad, self.parameters())
        params = sum([np.prod(p.size()) for p in model_parameters])
        logger.info(self.__class__.__name__)
        logger.info(self)
        logger.info("Total number of parameters: %d" % params)


def sample_z(mu, sigma):
    eps = T.randn(sigma.size())
    z = mu + T.exp(0.5 * sigma) * eps.to(DEVICE)
    return z


class Regularizer(Function):
    @staticmethod
    def forward(ctx, in_tensor):
        ctx.save_for_backward(in_tensor)
        return T.abs(T.abs(2 * in_tensor - 1) - 1)

    @staticmethod
    def backward(ctx, grad_output):
        in_tensor, = ctx.saved_tensors
        condition1 = T.ge(in_tensor, 1.)                              # x >= 1
        condition2 = T.ge(in_tensor, 0.) * T.le(in_tensor, 0.5)       # 0 <= x <= 0.5
        current_grad = 1 - (1 - condition1) * (1 - condition2)        # grad=1 if condi1|| condi2, grad=0 otherwise
        current_grad = current_grad.float()
        current_grad = (current_grad * 2 - 1) * 2                     # 1 -> 2, 0 -> -2
        return grad_output * current_grad


Reg = Regularizer.apply


class Track2Tracks:
    def __init__(self, args):
        # args
        self.args = args

        # network
        self.net = Net(args)

        # self.SigmoidCE_criterion = nn.BCEWithLogitsLoss(reduction='sum')
        self.ce_criterion = nn.BCELoss(reduction='sum')

        # log
        self.log_paths, self.logger, self.csv_writer = log(self.args)
        self.logger.info("\n\n{:-^70}".format(" Network Architecture "))
        self.net.summary(self.logger)

    def train(self):
        train_set = Piano400Subset(root=os.path.join(self.args.data_dir, 'train'), c_track_idx=self.args.c_track_idx)
        train_data_loader = T.utils.data.DataLoader(
            dataset=train_set,
            batch_size=self.args.bz,
            shuffle=True,
            num_workers=2,
            drop_last=True,
        )

        develop_set = Piano400Subset(root=os.path.join(self.args.data_dir, 'develop'),
                                     c_track_idx=self.args.c_track_idx)
        develop_data_loader = T.utils.data.DataLoader(
            dataset=develop_set,
            batch_size=self.args.bz,
            shuffle=True,
            num_workers=2,
            drop_last=True,
        )

        self.logger.info("\n\nTraining starts ...")
        opt = optim.Adam(self.net.parameters(), lr=self.args.lr, betas=[0.5, 0.999])

        try:
            for epoch in range(1, self.args.n_epoch + 1):
                self.net.train()

                for batch, (x, y) in enumerate(train_data_loader):
                    x = x.to(DEVICE)
                    y = y.to(DEVICE)

                    mu, sigma = self.net.forward_to_z(x, y)
                    z = sample_z(mu, sigma)  # (bz, z_dim)
                    x_ = self.net(z, y)

                    # E[log P(X|z)]
                    # reconstruct_loss = self.SigmoidCE_criterion(x_, x) / self.args.bz
                    reconstruct_loss = self.ce_criterion(x_, x) / self.args.bz
                    # D_KL(Q(z|X) || P(z|X))
                    kl_div = 0.5 * (T.exp(sigma) + mu ** 2 - 1 - sigma).sum() / self.args.bz  # sigma~=log_sigma
                    # add free bits to KL
                    free_nats = self.args.free_bits * T.log(T.tensor(2.)).to(DEVICE)
                    latent_loss = T.max(kl_div - free_nats, T.tensor(0.).to(DEVICE))
                    # binary regularizer
                    binary_reg = Reg(x_).sum() / self.args.bz
                    loss = reconstruct_loss + latent_loss + binary_reg

                    self.net.zero_grad()
                    loss.backward()
                    opt.step()

                    # save loss value per step
                    value_list = [loss.item(), reconstruct_loss.item(), latent_loss.item()]
                    self.csv_writer.writerow(value_list)

                    if batch % 50 == 0:
                        self.logger.info('Epoch %d Batch %d | loss: %.4f rec_loss: %.4f kl_loss: %.4f |'
                                         'binary_reg: %.4f' %
                                         (epoch, batch, loss.item(), reconstruct_loss.item(), latent_loss.item(),
                                          binary_reg))

                # eval
                if epoch % 5 == 0:
                    with T.no_grad():
                        self.net.eval()

                        rnd_batch_idx = np.random.randint(0, len(develop_set) // self.args.bz)
                        pair_list = []
                        for batch, (x, y) in enumerate(develop_data_loader):
                            pr_real = np.insert(x, [self.args.c_track_idx], y, 1)

                            y = y.to(DEVICE)
                            z = T.randn(self.args.bz, self.args.z_dim).to(DEVICE)

                            x_ = self.net(z, y)
                            pr = np.insert(x_.cpu(), [self.args.c_track_idx], y.cpu(), 1)

                            pr_real = np.transpose(pr_real, (0, 2, 3, 1))
                            pr = np.transpose(pr, (0, 2, 3, 1))

                            if batch == rnd_batch_idx:  #
                                for i in range(2):
                                    # set channel/track last, then save
                                    save_midi(pr_real[i],
                                              self.log_paths['gen'] + '/eval_epoch{}_{}_real'.format(epoch, i))
                                    save_midi(pr[i],
                                              self.log_paths['gen'] + '/eval_epoch{}_{}'.format(epoch, i))

                            pair = eval_tonal_distances(pr.numpy())
                            pair_list.append(pair)

                        list2arr = np.asarray(pair_list)
                        mean_pair = np.mean(list2arr, 0)
                        self.logger.info("{:-^70}".format(" Metrics Evaluated on Whole DevelopSet "))
                        self.logger.info("[Tonal distance between two tracks]")
                        self.logger.info(['{:^6}'.format(s) for s in dp.track_pair_names])
                        self.logger.info(['{:^6.2f}'.format(p) for p in mean_pair])

                # save model
                if epoch % 10 == 0:
                    T.save(self.net.state_dict(), self.log_paths['model'] + '/net_params_{}.pth'.format(epoch))

        except KeyboardInterrupt:
            self.logger.info("\nTraining stops due to keyboard interrupt.")
            sys.exit()

    def infer(self):
        test_set = Piano400Subset(root=os.path.join(self.args.data_dir, 'test'), c_track_idx=self.args.c_track_idx)
        test_data_loader = T.utils.data.DataLoader(
            dataset=test_set,
            batch_size=self.args.bz,
            shuffle=True,
            num_workers=2,
            drop_last=True,
        )

        with T.no_grad():
            restore_ckpt = self.log_paths['model'] + '/net_params_{}.pth'.format(self.args.restore_epoch)
            self.logger.info("\n\nLoading checkpoint: {} ...".format(restore_ckpt))
            self.net.load_state_dict(T.load(restore_ckpt))
            self.net.eval()

            rnd_batch_idx = np.random.randint(0, len(test_set) // self.args.bz)
            pair_list = []
            for batch, (x, y) in enumerate(test_data_loader):
                self.logger.info("Infering on Batch {}".format(batch + 1))
                pr_real = np.insert(x, [self.args.c_track_idx], y, 1)

                y = y.to(DEVICE)
                z = T.randn(self.args.bz, self.args.z_dim).to(DEVICE)

                x_ = self.net(z, y)
                pr = np.insert(x_.cpu(), [self.args.c_track_idx], y.cpu(), 1)

                pr_real = np.transpose(pr_real, (0, 2, 3, 1))
                pr = np.transpose(pr, (0, 2, 3, 1))

                if batch == rnd_batch_idx:  #
                    for i in range(2):
                        # set channel/track last, then save
                        save_midi(pr_real[i], self.log_paths['gen'] + '/generate_real-{}-{}'.format(batch, i))
                        save_midi(pr[i], self.log_paths['gen'] + '/generate_sampling-{}-{}'.format(batch, i))

                pair = eval_tonal_distances(pr.numpy())
                pair_list.append(pair)

            list2arr = np.asarray(pair_list)
            mean_pair = np.mean(list2arr, 0)
            self.logger.info("{:-^70}".format(" Metrics Evaluated on Whole Testset "))
            self.logger.info("[Tonal distance between two tracks]")
            self.logger.info(['{:^6}'.format(s) for s in dp.track_pair_names])
            self.logger.info(['{:^6.2f}'.format(p) for p in mean_pair])


if __name__ == '__main__':
    a = get_args()
    t2t = Track2Tracks(a)

    if a.mode == 'train':
        t2t.train()
    elif a.mode == 'infer':
        t2t.infer()
