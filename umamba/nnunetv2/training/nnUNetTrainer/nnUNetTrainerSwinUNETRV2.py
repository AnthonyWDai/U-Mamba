import math
import inspect
import torch
from torch import nn
from torch.optim import AdamW
from torch.optim.lr_scheduler import CosineAnnealingLR
from monai.networks.nets import SwinUNETR

from nnunetv2.training.nnUNetTrainer.variants.network_architecture.nnUNetTrainerNoDeepSupervision import \
    nnUNetTrainerNoDeepSupervision
from nnunetv2.utilities.plans_handling.plans_handler import ConfigurationManager, PlansManager
from nnunetv2.training.loss.dice import get_tp_fp_fn_tn


class nnUNetTrainerSwinUNETRV2(nnUNetTrainerNoDeepSupervision):
    """
    nnUNet trainer for MONAI SwinUNETR v2
    Coded by GPT-5.4, referencing to nnUNetTrainerSwinUNETR.py
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

        original_patch_size = self.configuration_manager.patch_size
        new_patch_size = [int(math.ceil(i / 32) * 32) for i in original_patch_size]

        self.configuration_manager.configuration["patch_size"] = new_patch_size
        self.plans_manager.plans["configurations"][self.configuration_name]["patch_size"] = new_patch_size

        self.print_to_log_file(f"Patch size changed from {original_patch_size} to {new_patch_size}")

        self.grad_scaler = None
        self.initial_lr = 8e-4
        self.weight_decay = 0.01

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

        kwargs = dict(
            in_channels=num_input_channels,
            out_channels=label_manager.num_segmentation_heads,
            feature_size=48,
            depths=(2, 2, 2, 2),
            num_heads=(3, 6, 12, 24),
            norm_name="instance",
            drop_rate=0.0,
            attn_drop_rate=0.0,
            dropout_path_rate=0.0,
            normalize=True,
            use_checkpoint=False,
            spatial_dims=spatial_dims,
            downsample="mergingv2",
            use_v2=True,
        )

        sig = inspect.signature(SwinUNETR.__init__)
        if "img_size" in sig.parameters:
            kwargs["img_size"] = patch_size

        model = SwinUNETR(**kwargs)
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

        tp, fp, fn, _ = get_tp_fp_fn_tn(predicted_segmentation_onehot, target, axes=axes, mask=mask)

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

    def configure_optimizers(self):
        optimizer = AdamW(
            self.network.parameters(),
            lr=self.initial_lr,
            weight_decay=self.weight_decay,
            eps=1e-5,
        )
        scheduler = CosineAnnealingLR(
            optimizer,
            T_max=self.num_epochs,
            eta_min=1e-6,
        )
        self.print_to_log_file(f"Using optimizer {optimizer}")
        self.print_to_log_file(f"Using scheduler {scheduler}")
        return optimizer, scheduler

    def set_deep_supervision_enabled(self, enabled: bool):
        pass
    

