import numpy as np
import pandas as pd

import chainer
import wavenet

from chainer import training
from chainer.training import extensions

import base
import matplotlib.pyplot as plt
import os
import argparse


class VisitData(object):

    def __init__(self, x_len=64, y_len=39, cut_peak=False, use_3over=False):
        air_visit_data = pd.read_csv(base.INPUT_DIR + 'air_visit_data.csv', parse_dates=[1])
        air_visit_data = air_visit_data.pivot(index='air_store_id', columns='visit_date', values='visitors')

        x_orig = air_visit_data.values
        self.n, self.dim = x_orig.shape
        self.store_id = air_visit_data.index

        self.x = np.nan_to_num(x_orig)
        self.orig_log_x = np.log1p(self.x)
        self.orig_is_finite = np.isfinite(x_orig)

        if use_3over:
            self.is_finite = self.orig_is_finite & (self.x > 3 + 1e-6)
            self.log_x = self.orig_log_x.copy()
            self.log_x[~self.is_finite] = 0
        else:
            self.is_finite = self.orig_is_finite.copy()
            self.log_x = self.orig_log_x.copy()

        if cut_peak:
            for i, top4 in enumerate(self.log_x.argsort(axis=1)[:, -4:]):
                self.log_x[i, top4[1:]] = self.log_x[i, top4[0]]

        self.ts_min = -1 * np.ones(self.n, dtype=int)
        done = np.zeros(self.n, dtype=bool)
        for j in range(self.dim):
            self.ts_min[(~done) & (self.is_finite[:, j])] = j
            done |= self.is_finite[:, j]

        self.ts_max = -1 * np.ones(self.n, dtype=int)
        done = np.zeros(self.n, dtype=bool)
        for j in range(self.dim)[::-1]:
            self.ts_max[(~done) & (self.is_finite[:, j])] = j
            done |= self.is_finite[:, j]

        self.test_1y_ago = 113
        self.july = 182
        self.y_len = y_len
        self.x_len = x_len

        date_info = pd.read_csv(base.INPUT_DIR + 'date_info.csv', parse_dates=[0])
        self.date = date_info['calendar_date'].values
        self.dayofmonth = (date_info['calendar_date'].dt.day.values - 16) / 15
        self.dayofweek = date_info['calendar_date'].dt.dayofweek.values
        self.holiday_flg = ((date_info['calendar_date'].dt.dayofweek >= 5) | date_info['holiday_flg']).values

        air_store_info = pd.read_csv(base.INPUT_DIR + 'air_store_info.csv')
        air_store_info = air_store_info.set_index('air_store_id').loc[air_visit_data.index, :]
        air_store_info['large_area'] = air_store_info['air_area_name'].map(lambda x: x.split()[0])
        air_store_info['in_tokyo'] = (air_store_info['large_area'] == 'Tōkyō-to')

        genres = ['Izakaya', 'Cafe/Sweets', 'Dining bar', 'Italian/French', 'Bar/Cocktail', 'Japanese food']
        air_store_info.loc[~air_store_info['air_genre_name'].isin(genres), 'air_genre_name'] = 'Other'
        air_store_info['air_genre_num'], genres = pd.factorize(air_store_info['air_genre_name'])
        assert len(genres) == 7

        self.air_store_info = air_store_info
        self.in_tokyo = (air_store_info['large_area'] == 'Tōkyō-to').values
        self.genre = air_store_info['air_genre_num'].values

        hpg_reserve = pd.read_csv(base.INPUT_DIR + 'hpg_reserve.csv', parse_dates=[1, 2])
        hpg_reserve['visit_date'] = pd.to_datetime(hpg_reserve['visit_datetime'].dt.date)
        hpg_reserve['reserve_date'] = pd.to_datetime(hpg_reserve['reserve_datetime'].dt.date)
        self.hpg_reserve = hpg_reserve.groupby(['hpg_store_id', 'visit_date', 'reserve_date'])['reserve_visitors'].sum().reset_index()

        store_id_relation = pd.read_csv(base.INPUT_DIR + 'store_id_relation.csv')
        hpg_reserve = self.hpg_reserve.merge(store_id_relation, on=['hpg_store_id'], how='inner')
        hpg_reserve = hpg_reserve[['air_store_id', 'visit_date', 'reserve_date', 'reserve_visitors']]

        air_reserve = pd.read_csv(base.INPUT_DIR + 'air_reserve.csv', parse_dates=[1, 2])
        air_reserve['visit_date'] = pd.to_datetime(air_reserve['visit_datetime'].dt.date)
        air_reserve['reserve_date'] = pd.to_datetime(air_reserve['reserve_datetime'].dt.date)
        air_reserve = air_reserve[['air_store_id', 'visit_date', 'reserve_date', 'reserve_visitors']]
        air_reserve = pd.concat([air_reserve, hpg_reserve], axis=0)
        self.air_reserve = air_reserve.groupby(['air_store_id', 'visit_date', 'reserve_date'])['reserve_visitors'].sum().reset_index()

        same_geopos_lists = self.air_store_info.reset_index(drop=True).reset_index().groupby(['latitude', 'longitude'])['index'].apply(lambda x: x.tolist()).values
        same_geopos_with = [None for _ in range(self.n)]
        for same_list in same_geopos_lists:
            for i in same_list:
                # TODO : should we exclude i from the list?
                same_geopos_with[i] = same_list
        self.same_geopos_with = same_geopos_with

        precipitation = base.load_df(base.WORKING_DIR + 'precipitation.h5').values
        self.is_havy_rain = (precipitation >= 10).astype(np.float32)
        self.log_clipped_precipitation = np.log1p(precipitation.clip(0, 10))

        self.small_occ = []
        for j in range(4):
            self.small_occ.append((np.abs(self.x - (j + 1)) < 1e-6).astype(np.float32))

    def sample_target(self, batch_size, y_start=None):
        if isinstance(y_start, list):
            y_start = np.random.choice(y_start)
        cand_indices, = np.where((self.ts_min <= y_start - self.x_len) & (y_start + self.y_len - 1 <= self.ts_max))
        indices = cand_indices[np.random.permutation(len(cand_indices))[:batch_size]]
        return indices, y_start

    def get_air_reserve(self, indices, start, size):
        reserved = []
        air_reserve = self.air_reserve[self.air_reserve['reserve_date'] < self.date[start]]
        for i in indices:
            y_reserve = np.zeros(size, dtype=np.float32)
            store_reserve = air_reserve[air_reserve['air_store_id'] == self.store_id[i]]
            if len(store_reserve) > 0:
                store_reserve = store_reserve.groupby('visit_date')['reserve_visitors'].sum()
                for j in range(size):
                    if self.date[start + j] in store_reserve:
                        y_reserve[j] = store_reserve[self.date[start + j]]
            reserved.append(y_reserve)
        return np.asarray(reserved, dtype=np.float32)

    def get_same_geopos_x(self, indices, start):
        xs = []
        for i in indices:
            log_x = self.log_x[self.same_geopos_with[i], start: start + self.x_len]
            is_finite = self.is_finite[self.same_geopos_with[i], start: start + self.x_len]
            log_x_mean = log_x.sum(axis=1) / (is_finite.sum(axis=1) + 1e-6)

            # TODO
            agg_x = (log_x - log_x_mean[:, None]).mean(axis=0)
            xs.append(agg_x)
        return np.asarray(xs, dtype=np.float32)

    def get_batches(self, indices, y_start, return_y=False):
        n = len(indices)

        hpg_use = (self.hpg_reserve['reserve_date'] < self.date[y_start])
        hpg_use = hpg_use & (self.date[y_start - self.x_len] <= self.hpg_reserve['visit_date'])
        hpg_use = hpg_use & (self.hpg_reserve['visit_date'] <= self.date[y_start + self.y_len - 1])
        reserved = self.hpg_reserve[hpg_use].groupby('visit_date')['reserve_visitors'].sum()
        reserved = np.log1p(reserved)

        # make encoder features
        enc_slice = slice(y_start - self.x_len, y_start)
        store_offset = self.log_x[indices, enc_slice].sum(axis=1)
        store_offset /= (self.is_finite[indices, enc_slice].sum(axis=1) + 1e-6)
        store_std = []
        for i in indices:
            x_non_zero = self.log_x[i, enc_slice][self.is_finite[i, enc_slice]]
            store_std.append(x_non_zero.std() if len(x_non_zero) > 0 else 0)
        store_std = np.array(store_std, dtype=np.float32)

        x_enc = np.zeros((n, 30, self.x_len), dtype=np.float32)

        # store infomation
        x_enc[:, 0, :] = store_offset[:, None]
        x_enc[:, 1, :] = store_std[:, None]
        x_enc[:, 2, :] = self.in_tokyo[indices, None]
        for j in range(7):
            x_enc[:, 3 + j, :] = (self.genre[indices, None] == j)

        # store visit infomation
        x_enc[:, 10, :] = self.log_x[indices, enc_slice] - x_enc[:, 0, :]
        x_enc[:, 10, :][~self.is_finite[indices, enc_slice]] = 0
        x_enc[:, 11, :] = ~self.is_finite[indices, enc_slice]
        x_enc[:, 12, :] = np.log1p(self.get_air_reserve(indices, y_start - self.x_len, self.x_len)) - x_enc[:, 0, :]

        # other store information
        global_hpg_reserve_offset = reserved[self.date[enc_slice]].mean()
        x_enc[:, 13, :] = reserved[self.date[enc_slice]].values - global_hpg_reserve_offset
        air_offsets = self.log_x[:, enc_slice].sum(axis=1) / (self.is_finite[:, enc_slice].sum(axis=1) + 1e-6)
        x_enc[:, 14, :] = (self.log_x[:, enc_slice] - air_offsets[:, None]).mean(axis=0)[None, :]

        # weather infomation
        x_enc[:, 15, :] = self.is_havy_rain[indices, enc_slice]
        x_enc[:, 16, :] = self.log_clipped_precipitation[indices, enc_slice]

        # date infomation
        x_enc[:, 17, :] = self.holiday_flg[None, enc_slice]
        x_enc[:, 18, :] = self.dayofmonth[None, enc_slice]
        for j in range(7):
            x_enc[:, 19 + j, :] = (self.dayofweek[enc_slice] == j)

        # occurrence infomation of small values
        for j in range(4):
            x_enc[:, 26, :] = self.small_occ[j][indices, enc_slice]

        # make decoder features
        dec_slice = slice(y_start, y_start + self.y_len)
        x_dec = np.zeros((n, 23, self.y_len), dtype=np.float32)

        # store infomation
        x_dec[:, 0, :] = store_offset[:, None]
        x_dec[:, 1, :] = store_std[:, None]
        x_dec[:, 2, :] = self.in_tokyo[indices, None]
        for j in range(7):
            x_dec[:, 3 + j, :] = (self.genre[indices, None] == j)

        # reservation infomation
        x_dec[:, 10, :] = np.log1p(self.get_air_reserve(indices, y_start, self.y_len)) - x_dec[:, 0, :]
        x_dec[:, 11, :] = reserved[self.date[dec_slice]].values - global_hpg_reserve_offset

        # weather infomation
        x_dec[:, 12, :] = self.is_havy_rain[indices, dec_slice]
        x_dec[:, 13, :] = self.log_clipped_precipitation[indices, dec_slice]

        # date infomation
        x_dec[:, 14, :] = self.holiday_flg[dec_slice]
        x_dec[:, 15, :] = self.dayofmonth[None, dec_slice]
        for j in range(7):
            x_dec[:, 16 + j, :] = (self.dayofweek[dec_slice] == j)

        batches = (x_enc, x_dec)
        if return_y:
            y = self.log_x[indices, None, dec_slice].astype(np.float32)
            mask = self.is_finite[indices, None, dec_slice]
            batches += (y, mask)

        return batches

    def __len__(self):
        return len(self.x)


class VisitDataIterator(chainer.iterators.SerialIterator):

    def __init__(self, data, batch_size, y_start, shuffle=True, repeat=True, return_y=False):
        assert isinstance(y_start, list) or isinstance(y_start, int)
        self.data = data
        self.y_start = y_start
        self.batch_size = batch_size
        self.return_y = return_y
        self._repeat = repeat
        self._shuffle = shuffle

        self._N = len(data)
        self._indices = list(range(self._N))

        self.reset()

    @property
    def epoch_detail(self):
        return self.epoch + self.current_position / self._N

    def reset(self):
        self.current_position = 0
        self.epoch = 0
        self.is_new_epoch = False

        # use -1 instead of None internally.
        self._previous_epoch_detail = -1.

    def __next__(self):
        if not self._repeat and self.epoch > 0:
            raise StopIteration

        self._previous_epoch_detail = self.epoch_detail

        i = self.current_position
        i_end = i + self.batch_size

        if self._shuffle:
            indices, y_start = self.data.sample_target(self.batch_size, self.y_start)
        else:
            indices, y_start = self._indices[i:i_end], self.y_start
        indices = list(indices)

        if i_end >= self._N:
            if self._repeat:
                rest = i_end - self._N
                if rest > 0:
                    if self._shuffle:
                        new_indices, _ = self.data.sample_target(rest, y_start)
                    else:
                        new_indices = self._indices[:rest]
                    indices += list(new_indices)
                self.current_position = rest
            else:
                self.current_position = 0

            self.epoch += 1
            self.is_new_epoch = True
        else:
            self.is_new_epoch = False
            self.current_position = i_end

        batches = self.data.get_batches(indices, y_start, self.return_y)
        return [tuple([batch[i] for batch in batches]) for i in range(len(batches[0]))]

    next = __next__


def fit(model, train_iter, args, valid_iter=None):

    optimizer = chainer.optimizers.Adam(alpha=args.learning_rate)
    optimizer.setup(model)

    updater = training.StandardUpdater(train_iter, optimizer, device=args.gpu)
    trainer = training.Trainer(updater, (args.nb_epoch, 'epoch'), out=args.save_path)

    trainer.extend(extensions.dump_graph('main/loss'))
    trainer.extend(extensions.LogReport())
    trainer.extend(extensions.ProgressBar())

    if valid_iter is not None:
        evaluator = extensions.Evaluator(
            valid_iter, model, device=args.gpu)
        trainer.extend(evaluator, trigger=(args.valid_interval, 'epoch'))
        if args.model_name.endswith('clf'):
            trainer.extend(extensions.PrintReport(
                ['epoch', 'main/loss', 'main/accuracy', 'validation/main/loss', 'validation/main/accuracy', 'elapsed_time']))
        else:
            trainer.extend(extensions.PrintReport(
                ['epoch', 'main/loss', 'validation/main/loss', 'elapsed_time']))

        # implement early stopping
        update_best_trigger = training.triggers.MinValueTrigger(
            'validation/main/loss', trigger=(args.valid_interval, 'epoch'))

        @training.make_extension(trigger=update_best_trigger, priority=-100)
        def save_base_model(trainer):
            trainer.best_epoch = trainer.updater.epoch
            chainer.serializers.save_npz(args.save_path + 'best.model', model)

        @training.make_extension(trigger=(args.valid_interval, 'epoch'), priority=-101)
        def early_stopping(trainer):
            epoch = trainer.updater.epoch
            if epoch - trainer.best_epoch >= args.early_stopping_rounds:
                trainer.stop_trigger.period = epoch

        trainer.best_epoch = 0
        trainer.extend(save_base_model)
        trainer.extend(early_stopping)
    else:
        trainer.extend(extensions.PrintReport(
            ['epoch', 'main/loss', 'elapsed_time']))

    trainer.run()

    if valid_iter is not None:
        om.print('best_epoch: ' + str(trainer.best_epoch))
        chainer.serializers.load_npz(args.save_path + 'best.model', model)

    return model


def predict(model, iterator, args):
    iterator.reset()
    with chainer.using_config('train', False), chainer.no_backprop_mode():
        y_preds = []
        for batch in iterator:
            x_enc, x_dec = chainer.dataset.concat_examples(batch, args.gpu)[:2]
            y_pred = model.predict(x_enc, x_dec)
            y_pred = chainer.cuda.to_cpu(y_pred.data)
            y_preds.append(y_pred)

        y_preds = np.concatenate(y_preds, axis=0)
    return y_preds


def validation(model, data, y_start, args, om):

    def rmse(y_true, y_pred, mask):
        se = (y_true - y_pred) ** 2
        mse = np.sum(se * mask) / np.sum(mask)
        return np.sqrt(mse)

    it = VisitDataIterator(data, args.batch_size, y_start, shuffle=False, repeat=False, return_y=True)
    with chainer.using_config('train', False), chainer.no_backprop_mode():
        y_pred = []
        offset = []
        for batch in it:
            x_enc, x_dec = chainer.dataset.concat_examples(batch, args.gpu)[:2]
            y_hat = model.predict(x_enc, x_dec)
            y_pred.append(chainer.cuda.to_cpu(y_hat.data))
            offset.append(chainer.cuda.to_cpu(x_dec[:, 0, 0]))

        y_pred = np.concatenate(y_pred, axis=0)
        offset = np.concatenate(offset, axis=0)

    y_pred = y_pred[:, 0, :]

    mask = data.orig_is_finite[:, y_start:y_start + data.y_len]
    y_true = data.orig_log_x[:, y_start:y_start + data.y_len]
    om.print('validation score (y_start = {})'.format(y_start))
    om.print('total score:   {:.8f}'.format(rmse(y_true, y_pred, mask)))
    om.print('public score:  {:.8f}'.format(rmse(y_true[:, :6], y_pred[:, :6], mask[:, :6])))
    om.print('private score: {:.8f}'.format(rmse(y_true[:, 6:], y_pred[:, 6:], mask[:, 6:])))

    if args.with_fig:
        dir_name = args.save_path + 'y_start{}'.format(y_start)
        if not os.path.exists(dir_name):
            os.mkdir(dir_name)
        dir_name += '/'

        date_slice = slice(y_start - data.x_len, y_start + data.y_len)
        store_label = data.air_store_info['air_area_name'] + ' // ' + data.air_store_info['air_genre_name']
        date = data.date[:data.dim]
        for i in np.where((mask.sum(axis=1) > 0))[0]:
            loss = np.sqrt(np.sum(((y_true[i] - y_pred[i]) ** 2) * mask[i]) / np.sum(mask[i]))

            plt.title(store_label[i])
            plt.xlabel('loss : {:.6f}'.format(loss))
            temp_mask = np.zeros(data.dim, dtype=bool)
            temp_mask[date_slice] = data.orig_is_finite[i, date_slice]
            plt.plot(date[temp_mask], data.orig_log_x[i, temp_mask])
            plt.hlines([offset[i]], date[temp_mask][0], date[temp_mask][-1])
            temp_mask[date_slice][:-data.y_len] = False
            plt.plot(date[temp_mask], y_pred[i, temp_mask[y_start:y_start + data.y_len]])
            plt.savefig(dir_name + '{:0>3d}.png'.format(i))
            plt.close('all')

            plt.title(store_label[i])
            plt.xlabel('loss : {:.6f}'.format(loss))
            temp_mask = np.zeros(data.dim, dtype=bool)
            temp_mask[date_slice] = data.orig_is_finite[i, date_slice]
            plt.plot(date[temp_mask], data.orig_log_x[i, temp_mask] - offset[i])
            temp_mask[date_slice][:-data.y_len] = False
            plt.plot(date[temp_mask], y_pred[i, temp_mask[y_start:y_start + data.y_len]] - offset[i])
            plt.savefig(dir_name + 'offset_{:0>3d}.png'.format(i))
            plt.close('all')

    return y_pred


parser = argparse.ArgumentParser()
parser.add_argument('--x_len', type=int, default=56)
parser.add_argument('--gpu', type=int, default=0)
parser.add_argument('--batch_size', type=int, default=128)
parser.add_argument('--nb_epoch', type=int, default=400)
parser.add_argument('--early_stopping_rounds', type=int, default=1000)
parser.add_argument('--learning_rate', type=float, default=1e-3)
parser.add_argument('--valid_interval', type=int, default=2)
parser.add_argument('--seed', type=int, default=1024)
parser.add_argument('--wavenet_residual_channels', type=int, default=16)
parser.add_argument('--wavenet_skip_channels', type=int, default=16)
parser.add_argument('--wavenet_steps', type=int, default=5)
parser.add_argument('--wavenet_repeats', type=int, default=3)
parser.add_argument('--gru_n_layers', type=int, default=1)
parser.add_argument('--gru_hidden_units', type=int, default=256)
parser.add_argument('--gru_dropout', type=float, default=0.1)
parser.add_argument('--twin_clf_target_num', type=int, default=4)
parser.add_argument('--model_name', type=str, default='wavenet')
parser.add_argument('--without_valid', action='store_true')
parser.add_argument('--with_fig', action='store_true')
parser.add_argument('--description', type=str, default='')
parser.add_argument('--cut_peak', action='store_true')
parser.add_argument('--use_3over', action='store_true')
args = parser.parse_args()


om = base.OutputManager(vars(args))
args.save_path = om.get_path()
np.random.seed(args.seed)

data = VisitData(x_len=args.x_len, cut_peak=args.cut_peak, use_3over=args.use_3over)

x_enc, x_dec = data.get_batches([0], 300, False)
if args.model_name == 'wavenet':
    model = wavenet.WaveNetEncoderDecoder(
        x_enc.shape[1],
        x_dec.shape[1],
        args.wavenet_residual_channels,
        args.wavenet_skip_channels,
        [2 ** i for i in range(args.wavenet_steps)] * args.wavenet_repeats,
        [2 for i in range(args.wavenet_steps)] * args.wavenet_repeats,
    )
elif args.model_name == 'wavenet_twin':
    model = wavenet.WaveNetEncoderDecoderTwin(
        x_enc.shape[1],
        x_dec.shape[1],
        args.wavenet_residual_channels,
        args.wavenet_skip_channels,
        [2 ** i for i in range(args.wavenet_steps)] * args.wavenet_repeats,
        [2 for i in range(args.wavenet_steps)] * args.wavenet_repeats,
        args.twin_clf_target_num
    )
elif args.model_name == 'wavenet_clf':
    model = wavenet.WaveNetEncoderDecoderCLF(
        x_enc.shape[1],
        x_dec.shape[1],
        args.wavenet_residual_channels,
        args.wavenet_skip_channels,
        [2 ** i for i in range(args.wavenet_steps)] * args.wavenet_repeats,
        [2 for i in range(args.wavenet_steps)] * args.wavenet_repeats,
        args.twin_clf_target_num
    )
elif args.model_name == 'gru':
    model = wavenet.GRUEncoderDecoder(
        x_enc.shape[1],
        x_dec.shape[1],
        args.gru_n_layers,
        args.gru_hidden_units,
        args.gru_dropout
    )
elif args.model_name == 'gru_twin':
    model = wavenet.GRUEncoderDecoderTwin(
        x_enc.shape[1],
        x_dec.shape[1],
        args.gru_n_layers,
        args.gru_hidden_units,
        args.gru_dropout,
        args.twin_clf_target_num
    )

if args.gpu >= 0:
    chainer.cuda.get_device_from_id(args.gpu).use()
    model.to_gpu(args.gpu)

if not args.without_valid:
    train_iter = VisitDataIterator(
        data,
        args.batch_size,
        list(range(data.x_len, data.dim - 2 * data.y_len + 1)),
        shuffle=True,
        repeat=True,
        return_y=True
    )
    valid_iter = VisitDataIterator(
        data,
        args.batch_size,
        data.dim - data.y_len,
        shuffle=False,
        repeat=False,
        return_y=True
    )
    fit(model, train_iter, args, valid_iter)

    if args.model_name.endswith('clf'):
        pred_valid_1yago = predict(model, VisitDataIterator(data, args.batch_size, data.test_1y_ago, shuffle=False, repeat=False, return_y=True), args)
        pred_valid_latest = predict(model, VisitDataIterator(data, args.batch_size, data.dim - data.y_len, shuffle=False, repeat=False, return_y=True), args)
        base.save_npy(pred_valid_1yago, args.save_path + 'pred_valid_1yago.npy')
        base.save_npy(pred_valid_latest, args.save_path + 'pred_valid_latest.npy')
    else:
        pred_valid_1yago = validation(model, data, data.test_1y_ago, args, om)
        pred_valid_latest = validation(model, data, data.dim - data.y_len, args, om)
        base.save_df(pred_valid_1yago, args.save_path + 'pred_valid_1yago.h5')
        base.save_df(pred_valid_latest, args.save_path + 'pred_valid_latest.h5')
else:
    train_iter = VisitDataIterator(
        data,
        args.batch_size,
        list(range(data.x_len, data.dim - data.y_len + 1)),
        shuffle=True,
        repeat=True,
        return_y=True
    )
    fit(model, train_iter, args)

test_iter = VisitDataIterator(
    data,
    args.batch_size,
    data.dim,
    shuffle=False,
    repeat=False,
    return_y=False
)
pred = predict(model, test_iter, args)

if args.model_name.endswith('clf'):
    base.save_npy(pred, args.save_path + 'pred.npy')
else:
    pred = pred[:, 0, :]
    base.save_df(pred, args.save_path + 'pred.h5')

    id_test = pd.read_csv(base.INPUT_DIR + 'sample_submission.csv')['id']
    pred = np.expm1(pred)
    pred = pd.DataFrame(pred, index=data.store_id, columns=data.date[data.dim:]).reset_index()
    pred = pred.melt(id_vars='air_store_id', var_name='date', value_name='visitors')
    pred['id'] = pred['air_store_id'] + pred['date'].map(lambda x: x.strftime('_%Y-%m-%d'))
    pred.loc[pred['visitors'] < 1, 'visitors'] = 1
    pred.set_index('id').loc[id_test, 'visitors'].reset_index().to_csv(args.save_path + 'submission.csv', index=False)


# run this once
base.save_npy(data.orig_log_x[:, data.test_1y_ago:data.test_1y_ago + data.y_len], base.WORKING_DIR + 'log_x_1yago.npy')
base.save_npy(data.orig_is_finite[:, data.test_1y_ago:data.test_1y_ago + data.y_len], base.WORKING_DIR + 'mask_1yago.npy')
base.save_npy(data.orig_log_x[:, data.dim - data.y_len:], base.WORKING_DIR + 'log_x_latest.npy')
base.save_npy(data.orig_is_finite[:, data.dim - data.y_len:], base.WORKING_DIR + 'mask_latest.npy')
