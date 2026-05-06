import math
import torch
import torch.nn as nn

from nnunetv2.training.nnUNetTrainer.variants.network_architecture.nnUNetTrainerNoDeepSupervision import \
    nnUNetTrainerNoDeepSupervision
from nnunetv2.utilities.plans_handling.plans_handler import ConfigurationManager, PlansManager
from nnunetv2.training.loss.dice import get_tp_fp_fn_tn

from nnunetv2.nets.SegMamba import SegMamba


# Adapted codes from GPT-5.4, referencing to nnUNetTrainerSwinUNETR.py
class nnUNetTrainerSegMamba(nnUNetTrainerNoDeepSupervision):
    """
    nnUNetv2 trainer for SegMamba.
    Assumes 3D input.
    """

    def __init__(
        self,
        plans: dict,
        configuration: str,
        fold: int,
        dataset_json: dict,
        unpack_dataset: bool = True,
        device: torch.device = torch.device("cuda"),
    ):
        super().__init__(plans, configuration, fold, dataset_json, unpack_dataset, device)

        # SegMamba downsamples 4 times by factor 2 => patch dims should be divisible by 16
        original_patch_size = self.configuration_manager.patch_size
        new_patch_size = [int(math.ceil(i / 16) * 16) for i in original_patch_size]

        self.configuration_manager.configuration["patch_size"] = new_patch_size
        self.plans_manager.plans["configurations"][self.configuration_name]["patch_size"] = new_patch_size

        self.print_to_log_file(f"Patch size changed from {original_patch_size} to {new_patch_size}")

        self.grad_scaler = None
        self.initial_lr = 1e-2
        self.weight_decay = 1e-5

    @staticmethod
    def build_network_architecture(
        plans_manager: PlansManager,
        dataset_json,
        configuration_manager: ConfigurationManager,
        num_input_channels,
        enable_deep_supervision: bool = False,
    ) -> nn.Module:
        label_manager = plans_manager.get_label_manager(dataset_json)
        patch_size = configuration_manager.patch_size
        spatial_dims = len(patch_size)

        if spatial_dims != 3:
            raise NotImplementedError("This SegMamba implementation currently supports only 3D nnUNet configurations.")

        model = SegMamba(
            in_chans=num_input_channels,
            out_chans=label_manager.num_segmentation_heads,
            depths=(2, 2, 2, 2),
            feat_size=(48, 96, 192, 384),
            hidden_size=768,
            norm_name="instance",
            res_block=True,
            spatial_dims=spatial_dims,
        )
        return model

    def train_step(self, batch: dict) -> dict:
        data = batch["data"].to(self.device, non_blocking=True)
        target = batch["target"]

        if isinstance(target, list):
            target = [t.to(self.device, non_blocking=True) for t in target]
        else:
            target = target.to(self.device, non_blocking=True)

        self.optimizer.zero_grad(set_to_none=True)

        output = self.network(data)
        loss = self.loss(output, target)
        loss.backward()

        torch.nn.utils.clip_grad_norm_(self.network.parameters(), 12.0)
        self.optimizer.step()

        return {"loss": loss.detach().cpu().numpy()}

    def validation_step(self, batch: dict) -> dict:
        data = batch["data"].to(self.device, non_blocking=True)
        target = batch["target"]

        if isinstance(target, list):
            target = [t.to(self.device, non_blocking=True) for t in target]
        else:
            target = target.to(self.device, non_blocking=True)

        output = self.network(data)
        loss = self.loss(output, target)

        axes = [0] + list(range(2, output.ndim))

        if self.label_manager.has_regions:
            predicted_segmentation_onehot = (torch.sigmoid(output) > 0.5).long()
        else:
            output_seg = output.argmax(1, keepdim=True)
            predicted_segmentation_onehot = torch.zeros(output.shape, device=output.device, dtype=torch.float32)
            predicted_segmentation_onehot.scatter_(1, output_seg, 1)

        if self.label_manager.has_ignore_label:
            if not self.label_manager.has_regions:
                mask = (target != self.label_manager.ignore_label).float()
                target = target.clone()
                target[target == self.label_manager.ignore_label] = 0
            else:
                mask = 1 - target[:, -1:]
                target = target[:, :-1]
        else:
            mask = None

        tp, fp, fn, _ = get_tp_fp_fn_tn(
            predicted_segmentation_onehot,
            target,
            axes=axes,
            mask=mask,
        )

        tp_hard = tp.detach().cpu().numpy()
        fp_hard = fp.detach().cpu().numpy()
        fn_hard = fn.detach().cpu().numpy()

        if not self.label_manager.has_regions:
            tp_hard = tp_hard[1:]
            fp_hard = fp_hard[1:]
            fn_hard = fn_hard[1:]

        return {
            "loss": loss.detach().cpu().numpy(),
            "tp_hard": tp_hard,
            "fp_hard": fp_hard,
            "fn_hard": fn_hard,
        }

    def set_deep_supervision_enabled(self, enabled: bool):
        pass


class nnUNetTrainerSegMambaLr5e3(nnUNetTrainerSegMamba):
    def __init__(self, plans: dict, configuration: str, fold: int, dataset_json: dict, unpack_dataset: bool = True,
                 device: torch.device = torch.device('cuda')):
        super().__init__(plans, configuration, fold, dataset_json, unpack_dataset, device)
        self.initial_lr = 5e-3
        
    
class nnUNetTrainerSegMambaLr1e3(nnUNetTrainerSegMamba):
    def __init__(self, plans: dict, configuration: str, fold: int, dataset_json: dict, unpack_dataset: bool = True,
                 device: torch.device = torch.device('cuda')):
        super().__init__(plans, configuration, fold, dataset_json, unpack_dataset, device)
        self.initial_lr = 1e-3
        

class nnUNetTrainerSegMambaLr5e4(nnUNetTrainerSegMamba):
    def __init__(self, plans: dict, configuration: str, fold: int, dataset_json: dict, unpack_dataset: bool = True,
                 device: torch.device = torch.device('cuda')):
        super().__init__(plans, configuration, fold, dataset_json, unpack_dataset, device)
        self.initial_lr = 5e-4
        