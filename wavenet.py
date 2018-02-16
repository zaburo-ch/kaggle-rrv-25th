import numpy as np
import pandas as pd

import chainer
import chainer.links as L
import chainer.functions as F


class DilatedCausalConvolution1D(chainer.Chain):

    def __init__(self, in_channels, out_channels, filter_width, dilation, zero_pad=True):
        super(DilatedCausalConvolution1D, self).__init__()

        self.zero_pad = zero_pad
        self.shift = (filter_width - 1) * dilation

        pad = self.shift if zero_pad else 0
        with self.init_scope():
            self.conv = L.DilatedConvolution2D(
                in_channels, out_channels, ksize=(1, filter_width),
                dilate=(1, dilation), pad=(0, pad)
            )

    def __call__(self, x, pad=None):
        if not self.zero_pad:
            assert pad is not None
            if self.shift <= pad.shape[-1]:
                x = F.concat([pad[..., -self.shift:], x], axis=2)
            else:
                xp = chainer.cuda.get_array_module(x)
                zero = xp.zeros(pad.shape[:-1] + (pad.shape[-1] - self.shift,))
                x = F.concat([zero, pad, x], axis=2)

        x = x[..., None, :]
        y = self.conv(x)

        if self.zero_pad:
            y = y[..., 0, :-self.shift]
        else:
            y = y[..., 0, :]

        return y


class TimeDistributedDense(chainer.Chain):

    def __init__(self, in_channels, out_channels):
        super(TimeDistributedDense, self).__init__()
        with self.init_scope():
            self.dense = L.Convolution2D(in_channels, out_channels, ksize=(1, 1))

    def __call__(self, x):
        x = x[..., None, :]
        y = self.dense(x)
        return y[..., 0, :]


class WaveNet(chainer.Chain):

    def __init__(
        self,
        input_channels=5,
        residual_channels=32,
        skip_channels=32,
        dilations=[2 ** i for i in range(8)] * 3,
        filter_widths=[2 for i in range(8)] * 3,
        zero_pad=True,
        output_channels=1,
    ):
        super(WaveNet, self).__init__()
        self.residual_channels = residual_channels
        self.skip_channels = skip_channels
        self.num_convs = len(dilations)

        with self.init_scope():
            self.in_dense = TimeDistributedDense(input_channels, residual_channels)

            self.res_convs = []
            self.res_denses = []
            for i, (dilation, filter_width) in enumerate(zip(dilations, filter_widths)):
                conv = DilatedCausalConvolution1D(residual_channels, 2 * residual_channels, filter_width, dilation, zero_pad)
                dense = TimeDistributedDense(residual_channels, skip_channels + residual_channels)

                self.res_convs.append(conv)
                self.res_denses.append(dense)

                self.add_link('conv{}'.format(i), conv)
                self.add_link('dense{}'.format(i), dense)

            self.out_dense1 = TimeDistributedDense(self.num_convs * skip_channels, 128)
            self.out_dense2 = TimeDistributedDense(128, output_channels)

    def __call__(self, x, pads=None, return_ci=True, return_h=False):

        if pads is None:
            pads = [None] * self.num_convs

        x = F.tanh(self.in_dense(x))

        conv_inputs, skip_outputs = [], []
        for conv, dense, pad in zip(self.res_convs, self.res_denses, pads):
            conv_inputs.append(x)

            conv_filter, conv_gate = F.split_axis(conv(x, pad), 2, 1)
            conv_output = F.tanh(conv_filter) * F.sigmoid(conv_gate)

            skip_output, res_output = F.split_axis(dense(conv_output), [self.skip_channels], 1)

            x += res_output
            skip_outputs.append(skip_output)

        skip_outputs = F.relu(F.concat(skip_outputs, axis=1))
        h = F.relu(self.out_dense1(skip_outputs))
        y_hat = self.out_dense2(h)

        ret = (y_hat,)
        if return_ci:
            ret += (conv_inputs,)
        if return_h:
            ret += (h,)

        return ret


def masked_rmse(y_true, y_pred, mask):
    se = (y_true - y_pred) ** 2
    mse = F.sum(se * mask) / F.sum(mask.astype(np.float32))
    return F.sqrt(mse)


class WaveNetEncoderDecoder(chainer.Chain):

    def __init__(
        self,
        encoder_input_channels=5,
        decoder_input_channels=5,
        residual_channels=32,
        skip_channels=32,
        dilations=[2 ** i for i in range(8)] * 3,
        filter_widths=[2 for i in range(8)] * 3
    ):
        super(WaveNetEncoderDecoder, self).__init__()

        with self.init_scope():
            self.encoder = WaveNet(encoder_input_channels, residual_channels, skip_channels, dilations, filter_widths)
            self.decoder = WaveNet(decoder_input_channels + 1, residual_channels, skip_channels, dilations, filter_widths, False)

    def predict(self, x_encoder, x_decoder):
        y_encoder, conv_inputs = self.encoder(x_encoder)

        y_encoder = F.tile(y_encoder[..., -1:], (1, 1, x_decoder.shape[-1]))
        x_decoder = F.concat([x_decoder, y_encoder], axis=1)
        y_decoder, _ = self.decoder(x_decoder, pads=conv_inputs)

        # TODO : pass x_encoder[:1, ...] to this method as separate variable
        y_hat = y_decoder + x_decoder[:, :1, :]
        return y_hat

    def __call__(self, x_encoder, x_decoder, y, y_mask):
        y_hat = self.predict(x_encoder, x_decoder)
        loss = masked_rmse(y, y_hat, y_mask)
        chainer.report({'loss': loss}, self)
        return loss


class GRUEncoderDecoder(chainer.Chain):

    def __init__(
        self,
        encoder_input_channels=5,
        decoder_input_channels=5,
        n_layers=3,
        hidden_units=1024,
        dropout=0.1
    ):
        super(GRUEncoderDecoder, self).__init__()
        with self.init_scope():
            self.encoder = L.NStepGRU(n_layers, encoder_input_channels, hidden_units, dropout)
            self.decoder = L.NStepGRU(n_layers, decoder_input_channels, hidden_units, dropout)
            self.decoder_dense = TimeDistributedDense(hidden_units, 1)

    def predict(self, x_encoder, x_decoder):
        offset = x_decoder[:, :1, :]
        x_encoder = chainer.Variable(x_encoder)
        x_decoder = chainer.Variable(x_decoder)

        x_encoder = [x.T for x in x_encoder]
        x_decoder = [x.T for x in x_decoder]
        hy, _ = self.encoder(None, x_encoder)
        _, ys = self.decoder(hy, x_decoder)
        ys = F.concat([y.T[None, ...] for y in ys], axis=0)
        ys = self.decoder_dense(ys)

        # TODO : pass x_encoder[:1, ...] to this method as separate variable
        ys = ys + offset
        return ys

    def __call__(self, x_encoder, x_decoder, y, y_mask):
        y_hat = self.predict(x_encoder, x_decoder)
        loss = masked_rmse(y, y_hat, y_mask)
        chainer.report({'loss': loss}, self)
        return loss


class GRUEncoderDecoderTwin(chainer.Chain):

    def __init__(
        self,
        encoder_input_channels=5,
        decoder_input_channels=5,
        n_layers=3,
        hidden_units=1024,
        dropout=0.1,
        clf_target_num=3
    ):
        super(GRUEncoderDecoderTwin, self).__init__()
        self.clf_target_num = clf_target_num
        self.clf_targets = np.log1p(range(1, clf_target_num + 1)).reshape(1, clf_target_num, 1).astype(np.float32)
        with self.init_scope():
            self.encoder = L.NStepGRU(n_layers, encoder_input_channels, hidden_units, dropout)
            self.decoder = L.NStepGRU(n_layers, decoder_input_channels, hidden_units, dropout)
            self.decoder_reg_dense = TimeDistributedDense(hidden_units, 1)
            self.decoder_clf_dense = TimeDistributedDense(hidden_units, clf_target_num + 1)

    def predict(self, x_encoder, x_decoder, with_prob=False):
        offset = x_decoder[:, :1, :]
        x_encoder = chainer.Variable(x_encoder)
        x_decoder = chainer.Variable(x_decoder)

        x_encoder = [x.T for x in x_encoder]
        x_decoder = [x.T for x in x_decoder]
        hy, _ = self.encoder(None, x_encoder)
        _, ys = self.decoder(hy, x_decoder)
        ys = F.concat([y.T[None, ...] for y in ys], axis=0)

        ys_reg = self.decoder_reg_dense(ys)
        ys_reg += offset
        ys_clf = self.decoder_clf_dense(ys)

        if not with_prob:
            # NOTE : this code is only for prediction in test time.
            # It breaks the computational graph chain of `ys_reg`
            assert not chainer.config.train
            assert not chainer.config.enable_backprop
            ys_reg = ys_reg.data[:, 0, :]
            xp = chainer.cuda.get_array_module(ys_reg)

            clf_result = ys_clf.data.argmax(axis=1)
            assert clf_result.shape == ys_reg.shape
            for j in range(1, self.clf_target_num + 1):
                ys_reg[clf_result == j] = xp.log1p(j)

            return chainer.Variable(ys_reg[:, None, :])
        else:
            ys_clf = F.softmax(ys_clf, axis=1)
            return ys_reg, ys_clf

    def __call__(self, x_encoder, x_decoder, y, mask):
        if chainer.config.train:
            ys_reg, ys_clf = self.predict(x_encoder, x_decoder, with_prob=True)

            assert y.ndim == 3 and y.shape[1] == 1
            clf_targets = np.tile(self.clf_targets, y.shape)
            if chainer.cuda.get_array_module(y) != np:
                clf_targets = chainer.cuda.to_gpu(clf_targets)

            ys_all = F.concat([ys_reg, clf_targets], axis=1)

            # calculate rooted mean expected squared error
            se = (F.broadcast_to(y, ys_all.shape) - ys_all) ** 2
            assert ys_clf.shape == se.shape
            ese = F.sum(se * ys_clf, axis=1, keepdims=True)
            assert ese.shape == mask.shape
            mese = F.sum(ese * mask) / F.sum(mask.astype(np.float32))
            loss = F.sqrt(mese)
        else:
            y_hat = self.predict(x_encoder, x_decoder, with_prob=False)
            loss = masked_rmse(y, y_hat, mask)

        chainer.report({'loss': loss}, self)
        return loss


class WaveNetEncoderDecoderTwin(chainer.Chain):

    def __init__(
        self,
        encoder_input_channels=5,
        decoder_input_channels=5,
        residual_channels=32,
        skip_channels=32,
        dilations=[2 ** i for i in range(8)] * 3,
        filter_widths=[2 for i in range(8)] * 3,
        clf_target_num=3
    ):
        super(WaveNetEncoderDecoderTwin, self).__init__()
        self.clf_target_num = clf_target_num
        self.clf_targets = np.log1p(range(1, clf_target_num + 1)).reshape(1, clf_target_num, 1).astype(np.float32)
        with self.init_scope():
            self.encoder = WaveNet(encoder_input_channels, residual_channels, skip_channels, dilations, filter_widths)
            self.decoder = WaveNet(decoder_input_channels + 1, residual_channels, skip_channels, dilations, filter_widths, False)
            self.decoder_clf_dense = TimeDistributedDense(128, clf_target_num + 1)

    def predict(self, x_encoder, x_decoder, with_prob=False):
        offset = x_decoder[:, :1, :]
        y_encoder, conv_inputs = self.encoder(x_encoder, return_ci=True, return_h=False)

        y_encoder = F.tile(y_encoder[..., -1:], (1, 1, x_decoder.shape[-1]))
        x_decoder = F.concat([x_decoder, y_encoder], axis=1)

        ys_reg, h = self.decoder(x_decoder, pads=conv_inputs, return_ci=False, return_h=True)
        ys_reg += offset
        ys_clf = self.decoder_clf_dense(h)

        if not with_prob:
            # NOTE : this code is only for prediction in test time.
            # It breaks the computational graph chain of `ys_reg`
            assert not chainer.config.train
            assert not chainer.config.enable_backprop
            ys_reg = ys_reg.data[:, 0, :]
            xp = chainer.cuda.get_array_module(ys_reg)

            clf_result = ys_clf.data.argmax(axis=1)
            assert clf_result.shape == ys_reg.shape
            for j in range(1, self.clf_target_num + 1):
                ys_reg[clf_result == j] = xp.log1p(j)

            return chainer.Variable(ys_reg[:, None, :])
        else:
            ys_clf = F.softmax(ys_clf, axis=1)
            return ys_reg, ys_clf

    def __call__(self, x_encoder, x_decoder, y, mask):
        if chainer.config.train:
            ys_reg, ys_clf = self.predict(x_encoder, x_decoder, with_prob=True)

            assert y.ndim == 3 and y.shape[1] == 1
            clf_targets = np.tile(self.clf_targets, y.shape)
            if chainer.cuda.get_array_module(y) != np:
                clf_targets = chainer.cuda.to_gpu(clf_targets)

            ys_all = F.concat([ys_reg, clf_targets], axis=1)

            # calculate rooted mean expected squared error
            se = (F.broadcast_to(y, ys_all.shape) - ys_all) ** 2
            assert ys_clf.shape == se.shape
            ese = F.sum(se * ys_clf, axis=1, keepdims=True)
            assert ese.shape == mask.shape
            mese = F.sum(ese * mask) / F.sum(mask.astype(np.float32))
            loss = F.sqrt(mese)
        else:
            y_hat = self.predict(x_encoder, x_decoder, with_prob=False)
            loss = masked_rmse(y, y_hat, mask)

        chainer.report({'loss': loss}, self)
        return loss


class WaveNetEncoderDecoderCLF(chainer.Chain):

    def __init__(
        self,
        encoder_input_channels=5,
        decoder_input_channels=5,
        residual_channels=32,
        skip_channels=32,
        dilations=[2 ** i for i in range(8)] * 3,
        filter_widths=[2 for i in range(8)] * 3,
        clf_target_num=3,
    ):
        super(WaveNetEncoderDecoderCLF, self).__init__()
        self.clf_target_num = clf_target_num
        with self.init_scope():
            self.encoder = WaveNet(encoder_input_channels, residual_channels, skip_channels, dilations, filter_widths)
            self.decoder = WaveNet(decoder_input_channels + 1, residual_channels, skip_channels, dilations, filter_widths, False, clf_target_num + 1)

    def predict(self, x_encoder, x_decoder, normed=True):
        y_encoder, conv_inputs = self.encoder(x_encoder)

        y_encoder = F.tile(y_encoder[..., -1:], (1, 1, x_decoder.shape[-1]))
        x_decoder = F.concat([x_decoder, y_encoder], axis=1)
        y_decoder, _ = self.decoder(x_decoder, pads=conv_inputs)

        if normed:
            y_decoder = F.softmax(y_decoder, axis=1)

        return y_decoder

    def __call__(self, x_encoder, x_decoder, y, mask):
        y_hat = self.predict(x_encoder, x_decoder, normed=False)

        y_hat = F.transpose(y_hat, (0, 2, 1))
        assert y.ndim == 3 and y.shape[1] == 1
        y = y.reshape(-1)
        mask = mask.reshape(-1)
        y_hat = y_hat.reshape(-1, y_hat.shape[2])

        xp = chainer.cuda.get_array_module(y)
        t = xp.zeros(y.shape[0], dtype=np.int32)
        for j in range(self.clf_target_num):
            t[xp.abs(y - xp.log1p(j + 1)) < 1e-6] = j + 1
        t[~mask] = -1
        loss = F.softmax_cross_entropy(y_hat, t, ignore_label=-1)
        acc = F.accuracy(y_hat, t, ignore_label=-1)

        chainer.report({'loss': loss, 'accuracy': acc}, self)
        return loss
