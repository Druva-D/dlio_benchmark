"""
   Copyright (c) 2025, UChicago Argonne, LLC
   All Rights Reserved

   Licensed under the Apache License, Version 2.0 (the "License");
   you may not use this file except in compliance with the License.
   You may obtain a copy of the License at

       http://www.apache.org/licenses/LICENSE-2.0

   Unless required by applicable law or agreed to in writing, software
   distributed under the License is distributed on an "AS IS" BASIS,
   WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
   See the License for the specific language governing permissions and
   limitations under the License.
"""
import logging
import math
import numpy as np

from dlio_benchmark.common.constants import MODULE_DATA_LOADER
from dlio_benchmark.common.enumerations import DataLoaderType
from dlio_benchmark.data_loader.base_data_loader import BaseDataLoader
from dlio_benchmark.utils.utility import Profile

dlp = Profile(MODULE_DATA_LOADER)

class SyntheticDataLoader(BaseDataLoader):
    @dlp.log_init
    def __init__(self, format_type, dataset_type, epoch):
        super().__init__(format_type, dataset_type, epoch, DataLoaderType.SYNTHETIC)
        shape = self._args.resized_image.shape
        
        # Calculate local samples for this rank
        total_samples = self.num_samples
        samples_per_proc = int(math.ceil(total_samples / self._args.comm_size))
        start_sample = self._args.my_rank * samples_per_proc
        end_sample = (self._args.my_rank + 1) * samples_per_proc - 1
        if end_sample > total_samples - 1:
            end_sample = total_samples - 1
        local_num_samples = end_sample - start_sample + 1
        
        # Calculate number of batches
        self.num_batches = int(math.ceil(local_num_samples / self.batch_size))
        
        # Pre-create a batch of zeros
        self.zero_batch = np.zeros((self.batch_size, shape[0], shape[1]), dtype=self._args.resized_image.dtype)

    @dlp.log
    def read(self, init=False):
        return

    @dlp.log
    def next(self):
        super().next()
        for step in dlp.iter(range(self.num_batches)):
            dlp.update(step=step)
            yield self.zero_batch

    @dlp.log
    def finalize(self):
        return