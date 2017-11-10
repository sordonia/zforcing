import numpy as np
import numpy.random as npr
from scipy.io import loadmat
import os
import json
from collections import defaultdict, OrderedDict
import pickle


def chunk(sequence, n):
    """ Yield successive n-sized chunks from sequence. """
    for i in range(0, len(sequence), n):
        yield sequence[i:i + n]


class TimitData():
    def __init__(self, fn, batch_size):
        data = np.load(fn)

        ####
        # IMPORTANT: u_train is the input and x_train is the target.
        ##
        u_train, x_train = data['u_train'], data['x_train']
        u_valid, x_valid = data['u_valid'], data['x_valid']
        (u_test, x_test, mask_test) = data['u_test'],  data['x_test'], data['mask_test']

        # assert u_test.shape[0] == 1680
        # assert x_test.shape[0] == 1680
        # assert mask_test.shape[0] == 1680

        self.u_train = u_train
        self.x_train = x_train
        self.u_valid = u_valid
        self.x_valid = x_valid

        # make multiple of batchsize
        n_test_padded = ((u_test.shape[0] // batch_size) + 1)*batch_size
        assert n_test_padded > u_test.shape[0]
        pad = n_test_padded - u_test.shape[0]
        u_test = np.pad(u_test, ((0, pad), (0, 0), (0, 0)), mode='constant')
        x_test = np.pad(x_test, ((0, pad), (0, 0), (0, 0)), mode='constant')
        mask_test = np.pad(mask_test, ((0, pad), (0, 0)), mode='constant')
        self.u_test = u_test
        self.x_test = x_test
        self.mask_test = mask_test

        self.n_train = u_train.shape[0]
        self.n_valid = u_valid.shape[0]
        self.n_test = u_test.shape[0]
        self.batch_size = batch_size

        print("TRAINING SAMPLES LOADED", self.u_train.shape)
        print("TEST SAMPLES LOADED", self.u_test.shape)
        print("VALID SAMPLES LOADED", self.u_valid.shape)
        print("TEST AVG LEN        ", np.mean(self.mask_test.sum(axis=1)) * 200)
        # test that x and u are correctly shifted
        assert np.sum(self.u_train[:, 1:] - self.x_train[:, :-1]) == 0.0
        assert np.sum(self.u_valid[:, 1:] - self.x_valid[:, :-1]) == 0.0
        for row in range(self.u_test.shape[0]):
            l = int(self.mask_test[row].sum())
            if l > 0:  # if l is zero the sequence is fully padded.
                assert np.sum(self.u_test[row, 1:l] -
                              self.x_test[row, :l-1]) == 0.0, row

    def _iter_data(self, u, x, mask=None):
        # u refers to the input whereas x, to the target.
        indices = range(len(u))
        for idx in chunk(indices, n=self.batch_size):
            u_batch, x_batch = u[idx], x[idx]
            if mask is None:
                mask_batch = np.ones((x_batch.shape[0], x_batch.shape[1]), dtype='float32')
            else:
                mask_batch = mask[idx]
            yield u_batch.transpose(1, 0, 2), x_batch.transpose(1, 0, 2), mask_batch.T

    def get_train_batch(self):
        return iter(self._iter_data(self.u_train, self.x_train))

    def get_valid_batch(self):
        return iter(self._iter_data(self.u_valid, self.x_valid))

    def get_test_batch(self):
        return iter(self._iter_data(self.u_test, self.x_test,
                                    mask=self.mask_test))


class BlizzardIterator(object):
    def __init__(self, data, batch_size=None, nbatch=None,
		 start=0, end=None, shuffle=False, infinite_data=0,
                 pseudo_n=1000000):
        if (batch_size or nbatch) is None:
            raise ValueError("Either batch_size or nbatch should be given.")
        if (batch_size and nbatch) is not None:
            raise ValueError("Provide either batch_size or nbatch.")
        self.infinite_data = infinite_data
        if not infinite_data:
            self.start = start
            self.end = data.num_examples() if end is None else end
            if self.start >= self.end or self.start < 0:
                raise ValueError("Got wrong value for start %d." % self.start)
            self.nexp = self.end - self.start
            if nbatch is not None:
                self.batch_size = int(np.float(self.nexp / float(nbatch)))
                self.nbatch = nbatch
            elif batch_size is not None:
                self.batch_size = batch_size
                self.nbatch = int(np.float(self.nexp / float(batch_size)))
            self.shuffle = shuffle
        else:
            self.pseudo_n = pseudo_n
        self.data = data
        self.name = self.data.name

    def __iter__(self):
        if self.infinite_data:
            for i in range(self.pseudo_n):
                yield self.data.slices()
        else:
            start = self.start
            end = self.end - self.end % self.batch_size
            for idx in range(start, end, self.batch_size):
                x_batch = self.data.slices(idx, idx + self.batch_size)[0]
                y_batch = self.data.slices(idx + 1, idx + self.batch_size + 1)[0]
                mask_batch = np.ones((x_batch.shape[0], x_batch.shape[1]), dtype=x_batch.dtype)
                yield x_batch, y_batch, mask_batch
