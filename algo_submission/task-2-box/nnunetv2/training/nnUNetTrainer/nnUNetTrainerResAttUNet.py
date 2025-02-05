from nnunetv2.training.nnUNetTrainer.nnUNetTrainer import nnUNetTrainer
from typing import Tuple, Union, List
import torch
from torch import nn
from nnunetv2.utilities.get_network_from_plans import get_network_from_plans
from nnunetv2.utilities.plans_handling.plans_handler import ConfigurationManager, PlansManager
from nnunetv2.training.loss.dice import get_tp_fp_fn_tn



class nnUNetTrainerResAttUNet(nnUNetTrainer):
    def __init__(self, plans: dict, configuration: str, fold: int, dataset_json: dict, unpack_dataset: bool = True,
                 device: torch.device = torch.device('cuda')):
        super().__init__(plans, configuration, fold, dataset_json, unpack_dataset, device)

    @staticmethod
    def build_network_architecture(architecture_class_name: str,
                                   arch_init_kwargs: dict,
                                   arch_init_kwargs_req_import: Union[List[str], Tuple[str, ...]],
                                   num_input_channels: int,
                                   num_output_channels: int,
                                   enable_deep_supervision: bool = True) -> nn.Module:
    
        return get_network_from_plans(
        architecture_class_name,
        arch_init_kwargs,
        arch_init_kwargs_req_import,
        num_input_channels,
        num_output_channels,
        allow_init=True,
        deep_supervision=enable_deep_supervision)

    def train_step(self, batch: dict) -> dict:
        data = batch['data']
        target = batch['target']

        data = data.to(self.device, non_blocking=True)
        if isinstance(target, list):
            target = [i.to(self.device, non_blocking=True) for i in target]
        else:
            target = target.to(self.device, non_blocking=True)

        self.optimizer.zero_grad(set_to_none=True)
        
        output = self.network(data)
        l = self.loss(output, target)
        l.backward()
        torch.nn.utils.clip_grad_norm_(self.network.parameters(), 12)
        self.optimizer.step()
        
        return {'loss': l.detach().cpu().numpy()}
    

    def validation_step(self, batch: dict) -> dict:
        data = batch['data']
        target = batch['target']

        data = data.to(self.device, non_blocking=True)
        if isinstance(target, list):
            target = [i.to(self.device, non_blocking=True) for i in target]
        else:
            target = target.to(self.device, non_blocking=True)

        self.optimizer.zero_grad(set_to_none=True)

        output = self.network(data)
        del data
        l = self.loss(output, target)

        # the following is needed for online evaluation. Fake dice (green line)
        axes = [0] + list(range(2, output.ndim))

        if self.label_manager.has_regions:
            predicted_segmentation_onehot = (torch.sigmoid(output) > 0.5).long()
        else:
            # no need for softmax
            output_seg = output.argmax(1)[:, None]
            predicted_segmentation_onehot = torch.zeros(output.shape, device=output.device, dtype=torch.float32)
            predicted_segmentation_onehot.scatter_(1, output_seg, 1)
            del output_seg

        if self.label_manager.has_ignore_label:
            if not self.label_manager.has_regions:
                mask = (target != self.label_manager.ignore_label).float()
                # CAREFUL that you don't rely on target after this line!
                target[target == self.label_manager.ignore_label] = 0
            else:
                mask = 1 - target[:, -1:]
                # CAREFUL that you don't rely on target after this line!
                target = target[:, :-1]
        else:
            mask = None

        tp, fp, fn, _ = get_tp_fp_fn_tn(predicted_segmentation_onehot, target, axes=axes, mask=mask)

        tp_hard = tp.detach().cpu().numpy()
        fp_hard = fp.detach().cpu().numpy()
        fn_hard = fn.detach().cpu().numpy()
        if not self.label_manager.has_regions:
            # if we train with regions all segmentation heads predict some kind of foreground. In conventional
            # (softmax training) there needs tobe one output for the background. We are not interested in the
            # background Dice
            # [1:] in order to remove background
            tp_hard = tp_hard[1:]
            fp_hard = fp_hard[1:]
            fn_hard = fn_hard[1:]

        return {'loss': l.detach().cpu().numpy(), 'tp_hard': tp_hard, 'fp_hard': fp_hard, 'fn_hard': fn_hard}
    
    # def configure_optimizers(self):

    #     optimizer = AdamW(self.network.parameters(), lr=self.initial_lr, weight_decay=self.weight_decay, eps=1e-5)
    #     scheduler = PolyLRScheduler(optimizer, self.initial_lr, self.num_epochs, exponent=1.0)

    #     self.print_to_log_file(f"Using optimizer {optimizer}")
    #     self.print_to_log_file(f"Using scheduler {scheduler}")

    #     return optimizer, scheduler
    
    # def set_deep_supervision_enabled(self, enabled: bool):
    #     pass




# class nnUNetTrainerResAttUNet(nnUNetTrainer):
#     """
#     Residual Encoder + Attention Gates + Residual Decoder + Skip Connections
#     """

#     @staticmethod
#     def build_network_architecture(architecture_class_name: str,
#                                    arch_init_kwargs: dict,
#                                    arch_init_kwargs_req_import: Union[List[str], Tuple[str, ...]],
#                                    num_input_channels: int,
#                                    num_output_channels: int,
#                                    enable_deep_supervision: bool = True) -> nn.Module:
#         """
#         This is where you build the architecture according to the plans. There is no obligation to use
#         get_network_from_plans, this is just a utility we use for the nnU-Net default architectures. You can do what
#         you want. Even ignore the plans and just return something static (as long as it can process the requested
#         patch size)
#         but don't bug us with your bugs arising from fiddling with this :-P
#         This is the function that is called in inference as well! This is needed so that all network architecture
#         variants can be loaded at inference time (inference will use the same nnUNetTrainer that was used for
#         training, so if you change the network architecture during training by deriving a new trainer class then
#         inference will know about it).

#         If you need to know how many segmentation outputs your custom architecture needs to have, use the following snippet:
#         > label_manager = plans_manager.get_label_manager(dataset_json)
#         > label_manager.num_segmentation_heads
#         (why so complicated? -> We can have either classical training (classes) or regions. If we have regions,
#         the number of outputs is != the number of classes. Also there is the ignore label for which no output
#         should be generated. label_manager takes care of all that for you.)

#         """
#         return get_network_from_plans(
#             architecture_class_name,
#             arch_init_kwargs,
#             arch_init_kwargs_req_import,
#             num_input_channels,
#             num_output_channels,
#             allow_init=True,
#             deep_supervision=enable_deep_supervision)