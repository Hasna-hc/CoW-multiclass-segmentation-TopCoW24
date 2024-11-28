import torch
from nnunetv2.training.loss.deep_supervision import DeepSupervisionWrapper
from nnunetv2.training.nnUNetTrainer.nnUNetTrainer import nnUNetTrainer
from nnunetv2.training.loss.compound_losses import DC_CE_and_BinDC_loss
from nnunetv2.training.loss.dice import MemoryEfficientSoftDiceLoss

import numpy as np


class nnUNetTrainerBinDice(nnUNetTrainer):
    def __init__(self, plans: dict, configuration: str, fold: int, dataset_json: dict, unpack_dataset: bool = True,
                 device: torch.device = torch.device('cuda')):
        super().__init__(plans, configuration, fold, dataset_json, unpack_dataset, device)
        if self.label_manager.has_regions:
            raise NotImplementedError("trainer not implemented for regions")
        
    def _build_loss(self):
        assert not self.label_manager.has_regions, "regions not supported by this trainer"
        loss = DC_CE_and_BinDC_loss(soft_dice_kwargs={'batch_dice': self.configuration_manager.batch_dice, 
                                                      'smooth': 1e-5, 'do_bg': False, 'ddp': self.is_ddp},
                                    ce_kwargs={}, weight_ce=0.8, weight_dice=0.8, weight_bdice=1, #FIXME: modified weights from 1 to 0.5
                                    ignore_label=self.label_manager.ignore_label, dice_class=MemoryEfficientSoftDiceLoss)
        

        # we give each output a weight which decreases exponentially (division by 2) as the resolution decreases
        # this gives higher resolution outputs more weight in the loss
        if self.enable_deep_supervision:
            deep_supervision_scales = self._get_deep_supervision_scales()
            weights = np.array([1 / (2**i) for i in range(len(deep_supervision_scales))])
            weights[-1] = 0

            # we don't use the lowest 2 outputs. Normalize weights so that they sum to 1
            weights = weights / weights.sum()
            # now wrap the loss
            loss = DeepSupervisionWrapper(loss, weights)
        return loss

