import glob
import numpy as np

from torch.utils.data.dataset import Dataset
from ..core.base_dataset import BaseDataset
from ..utils.utils import load_file

class SegRegDatasetTest(BaseDataset):

    def __init__(self, data_source, data_target, source_key, target_key, data_suffix, preprocesses=None, augmentation=None, is_assessment=False, num_assessment=None):

        assert isinstance(data_source, (list, np.ndarray)), 'Only accpet filename list.'
        assert isinstance(data_target, (list, np.ndarray)), 'Only accpet filename list.'

        if preprocesses:
            new_preprocesses = {}
            [new_preprocesses.update({source_key+k: v}) for k, v in preprocesses.items()]
            [new_preprocesses.update({target_key+k: v}) for k, v in preprocesses.items()]
        super().__init__(data_target, data_suffix, new_preprocesses, augmentation)
        
        self._data_source = data_source
        self._data_target = data_target
        self._source_key = source_key
        self._target_key = target_key
        self._is_assessment = is_assessment
        self._num_assessment = num_assessment

    def __getitem__(self, idx):
        data_dict = {}

        # target
        t_name = self._data_target[idx] # target image
        data_dict.update({self._target_key + self._org_suffix: load_file(t_name)})
        # for o_suffix in self._other_suffix:
        #     o_name = t_name.replace(self._org_suffix, o_suffix) # target label
        #     data_dict.update({self._target_key + o_suffix: load_file(o_name)})
            

        # source
        if not self._is_assessment:
            if len(self._data_source) > 0:
                s_name = self._data_source[idx % len(self._data_source)] # source image
                data_dict.update({self._source_key + self._org_suffix: load_file(s_name)})
                for o_suffix in self._other_suffix:
                    so_name = s_name.replace(self._org_suffix, o_suffix) # source label
                    data_dict.update({self._source_key + o_suffix: load_file(so_name)})

            data_dict = self.augmentation(data_dict)
            data_dict = self.pre_process(data_dict)
            return data_dict


        # quality assessment
        data_dict = self.augmentation(data_dict)
        data_dict = self.pre_process(data_dict)
        assessment_dicts = []
        tmp_assessment_source = self._data_source if self._num_assessment is None else np.random.choice(self._data_source, self._num_assessment)
        for si, s_name in enumerate(tmp_assessment_source):
            tmp_assessment_dict = {self._source_key + self._org_suffix: load_file(s_name)}
            for o_suffix in self._other_suffix:
                so_name = s_name.replace(self._org_suffix, o_suffix) # source labels
                tmp_assessment_dict.update({self._source_key + o_suffix: load_file(so_name)})
            tmp_assessment_dict = self.augmentation(tmp_assessment_dict)
            tmp_assessment_dict = self.pre_process(tmp_assessment_dict)
            assessment_dicts.append(tmp_assessment_dict)

        return {'data_dict': data_dict, 'assessment_dicts': assessment_dicts}

    def pre_process(self, data_dict):
        if self._preprocesses is None:
            return data_dict

        for key in self._preprocesses:
            for method in self._preprocesses[key]:
                if key in data_dict:
                    data_dict.update({key: method(data_dict[key])})
        return data_dict

        





