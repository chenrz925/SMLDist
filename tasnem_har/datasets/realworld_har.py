from collections import OrderedDict
from functools import reduce
from itertools import product
from multiprocessing import Pool
from pathlib import Path
from typing import List
from os import listdir, path
from zipfile import ZipFile
from logging import getLogger as get_logger

import pandas as pd
import numpy as np
from tasker import Definition
from tasker.mixin import ProfileMixin, value
from torch.utils.data import Dataset


class RealWorldHAR(Dataset, ProfileMixin):
    logger = get_logger('datasets.realworld_har.RealWorldHAR')

    @classmethod
    def define(cls) -> List[Definition]:
        return [
            value('root_dir', str),
            value('frame_period', int),
            value('filter_type', list, [str]),
            value('filter_by', str),
            value('num_workers', int)
        ]

    def __init__(self, **kwargs):
        assert 'root_dir' in kwargs
        assert 'frame_period' in kwargs
        assert 'filter_type' in kwargs
        assert 'filter_by' in kwargs
        assert 'num_workers' in kwargs

        self.sample_bound = 45
        self.root_dir = kwargs['root_dir']
        self.frame_period = kwargs['frame_period']
        self.filter_type = kwargs['filter_type']
        self.filter_by = kwargs['filter_by']
        self.num_workers = kwargs['num_workers']
        self.process_method = getattr(self, f'process_{kwargs["process_method"]}', self.process_raw)

        self.channel_min = np.load(str(Path(self.root_dir) / 'channel_min.npy'))
        self.channel_max = np.load(str(Path(self.root_dir) / 'channel_max.npy'))
        self.channel_mean = np.load(str(Path(self.root_dir) / 'channel_mean.npy'))
        self.channel_std = np.load(str(Path(self.root_dir) / 'channel_std.npy'))
        self.channel_25 = np.load(str(Path(self.root_dir) / 'channel_25.npy'))
        self.channel_75 = np.load(str(Path(self.root_dir) / 'channel_75.npy'))

        pool_map = Pool(self.num_workers).map if self.num_workers > 0 else map

        raw_data = OrderedDict(pool_map(
            self._lambda_load_proband,
            filter(
                (lambda it: it in self.filter_by) if self.filter_type == 'if' else (lambda it: it not in self.filter_by),
                filter(
                    lambda it: not it.endswith('.npy'),
                    listdir(self.root_dir)
                )
            )
        ))

        self.raw_features = np.concatenate(tuple(map(
            lambda it: it[1][0],
            sorted(raw_data.items(), key=lambda it: it[0])
        )), axis=0)
        self.raw_labels = np.concatenate(tuple(map(
            lambda it: it[1][1],
            sorted(raw_data.items(), key=lambda it: it[0])
        )), axis=0)

    def _lambda_load_proband(self, proband_name):
        return proband_name, self._load_proband(proband_name)

    def _load_proband(self, proband_name):
        labels = (
            'standing', 'sitting', 'lying', 'walking',
            'running', 'jumping', 'climbingup', 'climbingdown'
        )
        sensors = ('acc', 'gyr', 'mag')
        positions = (
            'chest', 'forearm', 'head', 'shin',
            'thigh', 'upperarm', 'waist'
        )

        def _sample_frame(frame):
            permutation = np.sort(np.random.permutation(frame[1].shape[0]))[:self.sample_bound * self.frame_period]
            return frame[0], frame[1].iloc[permutation, :]

        def _load_zipfile_csv(zip_file, filename):
            with zip_file.open(filename) as fp:
                frame = pd.read_csv(fp)
                frame['time_index'] = frame['attr_time'] // (1000 * self.frame_period)
                return OrderedDict(map(
                    lambda it: (it[0], it[1].loc[:, ('attr_x', 'attr_y', 'attr_y')]),
                    map(
                        _sample_frame,
                        filter(
                            lambda it: it[1].shape[0] > self.sample_bound * self.frame_period,
                            frame.groupby('time_index')
                        )
                    )
                ))

        def _load_zipfile(filename):
            try:
                with ZipFile(path.join(self.root_dir, proband_name, 'data', filename)) as zip_file:
                    raw_dict = OrderedDict(map(
                        lambda it: (it[0], _load_zipfile_csv(zip_file, it[1])),
                        map(
                            lambda position: (
                                position,
                                tuple(filter(lambda it: position in it.filename, zip_file.filelist))[0]
                            ),
                            positions
                        )
                    ))
                    common_indexes = tuple(sorted(reduce(
                        lambda s1, s2: s1 & s2,
                        map(lambda it: set(it.keys()), raw_dict.values())
                    )))
                    return OrderedDict(map(
                        lambda position: (position, OrderedDict(map(
                            lambda idx: (
                                idx,
                                raw_dict[position][idx].to_numpy().astype(np.float32).transpose()[np.newaxis, :, :]),
                            common_indexes
                        ))),
                        sorted(raw_dict)
                    ))
            except IndexError:
                return OrderedDict()

        per_sensor_label = OrderedDict(map(
            lambda it: (it[0], _load_zipfile(it[1])),
            map(
                lambda it: (it, f'{it[0]}_{it[1]}_csv.zip'),
                product(sensors, labels)
            )
        ))

        per_sensor_indexes = tuple(zip(map(
            lambda sensor: tuple(map(
                lambda it: reduce(
                    lambda s1, s2: s1 & s2,
                    map(
                        lambda it1: set(it1.keys()),
                        it[1].values()
                    )
                ) if it[1] else set(),
                filter(
                    lambda it: it[0][0] == sensor,
                    per_sensor_label.items()
                )
            )),
            sensors
        )))

        combined_indexes = reduce(
            lambda s1, s2: s1 & s2,
            map(
                lambda it: reduce(
                    lambda s1, s2: s1 | s2,
                    it[0]
                ),
                per_sensor_indexes
            )
        )

        features = np.concatenate(tuple(map(
            lambda sensor: np.concatenate(tuple(map(  # Single sensor group.
                lambda it_z: np.concatenate(tuple(map(  # Single label file.
                    lambda it_p: np.concatenate(tuple(map(  # Data frames for each position.
                        lambda it_t: it_t[1],  # Frames per time index.
                        sorted(filter(
                            lambda it_t: it_t[0] in combined_indexes,
                            it_p.items()
                        ), key=lambda it_t: it_t[0])
                    )), axis=0),
                    it_z.values()
                )), axis=1),
                filter(
                    lambda it: len(it) != 0,
                    map(lambda key: per_sensor_label[key], filter(lambda it: it[0] == sensor, per_sensor_label))
                )
            )), axis=0),
            sensors
        )), axis=1)

        extracted_labels = np.concatenate(tuple(map(
            lambda it: np.array([labels.index(it[0][1])] * len(
                tuple(filter(lambda key: key in combined_indexes, it[1][positions[0]]))
            ), dtype=np.int64),
            filter(lambda it: it[0][0] == sensors[0] and len(it[1]) != 0, per_sensor_label.items())
        )))

        self.logger.info(f'Proband {proband_name} loaded')

        return features, extracted_labels

    def process_minmax(self, array: np.ndarray) -> np.ndarray:
        return (np.clip(array, self.channel_min[:, np.newaxis], self.channel_max[:, np.newaxis]) - self.channel_min[:, np.newaxis]) / (self.channel_max - self.channel_min)[:, np.newaxis]

    def process_raw(self, array: np.ndarray) -> np.ndarray:
        return array

    def process_zscore(self, array: np.ndarray) -> np.ndarray:
        return (array - self.channel_mean[:, np.newaxis]) / self.channel_std[:, np.newaxis]

    def process_robust(self, array: np.ndarray) -> np.ndarray:
        upper = self.channel_75 + 1.5 * (self.channel_75 - self.channel_25)
        lower = self.channel_25 - 1.5 * (self.channel_75 - self.channel_25)
        return (np.clip(array, lower[:, np.newaxis], upper[:, np.newaxis]) - lower[:, np.newaxis]) / (upper - lower)[:, np.newaxis]

    def __getitem__(self, item):
        return self.process_method(self.raw_features[item]), self.raw_labels[item]

    def __len__(self):
        return self.raw_labels.shape[0]

