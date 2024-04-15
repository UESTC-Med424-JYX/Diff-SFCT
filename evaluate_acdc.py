import os
import numpy as np
import nibabel as nib

import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms

from module.unet import Diff_SFCT

from monai.inferers import SlidingWindowInferer
from monai.losses.dice import DiceLoss
from monai.utils import set_determinism

from dataloader_acdc import get_loader_acdc, ValidGenerator

from guided_diffusion.gaussian_diffusion import get_named_beta_schedule, ModelMeanType, ModelVarType, LossType
from guided_diffusion.resample import UniformSampler
from guided_diffusion.respace import SpacedDiffusion, space_timesteps

from light_training.evaluation.metric import dice, hausdorff_distance_95
from light_training.trainer import Trainer
from light_training.utils.files_helper import save_new_model_and_delete_last
from light_training.utils.lr_scheduler import LinearWarmupCosineAnnealingLR

set_determinism(123)

train_dir = ""
test_dir = ""

image_size = 224

logdir = "./logs_acdc/"
model_save_path = os.path.join(logdir, "model")

num_modality = 1
num_classes = 3

max_epoch = 8000
batch_size = 32
val_every = 200

env = "pytorch"
num_gpus = 1

device = "cuda:3"

checkpoint_path = "logs_acdc/model/best_model_0.9382.pt"


class DiffSFCT(nn.Module):
    def __init__(self) -> None:
        super().__init__()

        self.model = Diff_SFCT(2, num_modality, num_classes, [64, 128, 256, 512, 1024, 128],
                               act=("LeakyReLU", {"negative_slope": 0.1, "inplace": False}))

        betas = get_named_beta_schedule("linear", 1000)
        self.diffusion = SpacedDiffusion(use_timesteps=space_timesteps(1000, [1000]),
                                         betas=betas,
                                         model_mean_type=ModelMeanType.START_X,
                                         model_var_type=ModelVarType.FIXED_LARGE,
                                         loss_type=LossType.MSE,
                                         )

        self.sample_diffusion = SpacedDiffusion(use_timesteps=space_timesteps(1000, [10]),
                                                betas=betas,
                                                model_mean_type=ModelMeanType.START_X,
                                                model_var_type=ModelVarType.FIXED_LARGE,
                                                loss_type=LossType.MSE,
                                                )
        self.sampler = UniformSampler(1000)

    def forward(self, image=None, x=None, pred_type=None, step=None):
        if pred_type == "q_sample":
            noise = torch.randn_like(x).to(x.device)
            t, weight = self.sampler.sample(x.shape[0], x.device)
            return self.diffusion.q_sample(x, t, noise=noise), t, noise

        elif pred_type == "denoise":
            return self.model(x, t=step, image=image)

        elif pred_type == "ddim_sample":
            sample_out = self.sample_diffusion.ddim_sample_loop(self.model, (1, num_classes, image_size, image_size),
                                                                model_kwargs={"image": image})
            sample_out = sample_out["pred_xstart"]
            return sample_out


class ACDCTrainer(Trainer):
    def __init__(self, env_type, max_epochs, batch_size, device="cpu", val_every=1, num_gpus=1, logdir="./logs/",
                 master_ip='localhost', master_port=17750, training_script="train.py"):
        super().__init__(env_type, max_epochs, batch_size, device, val_every, num_gpus, logdir, master_ip, master_port,
                         training_script)
        self.window_infer = SlidingWindowInferer(roi_size=[image_size, image_size],
                                                 sw_batch_size=1,
                                                 overlap=0.6)
        self.model = DiffSFCT()

        if checkpoint_path is not None:
            print("-" * 60)
            print("加载预训练模型   ===>   ", checkpoint_path)
            self.model.load_state_dict(torch.load(checkpoint_path), strict=False)
            print("-" * 60)

    def get_input(self, batch):
        image = batch["image"]
        label = batch["label"]

        label = label.float()
        return image, label

    def validation_ds(self, ):
        transform_valid = transforms.Compose([
            ValidGenerator([image_size, image_size], 4),
        ])

        RV_dices = 0
        Myo_dices = 0
        LV_dices = 0
        cnt = 0

        RV_hd95s = 0
        Myo_hd95s = 0
        LV_hd95s = 0

        self.model.eval()
        self.model.to(device)

        for patient_dir in os.listdir(test_dir):
            valid_ds = ACDCTestDataset(os.path.join(test_dir, patient_dir), transform=transform_valid)
            val_loader = DataLoader(valid_ds, batch_size=1, shuffle=False)

            output_list, target_list = [], []
            for idx, batch in enumerate(val_loader):
                file = batch["file"]
                batch = {
                    x: batch[x].to(device)
                    for x in batch if isinstance(batch[x], torch.Tensor)
                }
                batch["file"] = file

                output, target = self.validation_step(batch)

                output_list.append(output)
                target_list.append(target)

            self.outputs = np.array(output_list).squeeze(1)
            self.targets = np.array(target_list).squeeze(1)

            o = self.outputs[:, 0]
            t = self.targets[:, 0]
            RV_dice = dice(o, t)
            RV_hd95 = hausdorff_distance_95(o, t)

            o = self.outputs[:, 1]
            t = self.targets[:, 1]
            Myo_dice = dice(o, t)
            Myo_hd95 = hausdorff_distance_95(o, t)

            o = self.outputs[:, 2]
            t = self.targets[:, 2]
            LV_dice = dice(o, t)
            LV_hd95 = hausdorff_distance_95(o, t)

            print(patient_dir)

            print(
                f"dice   ===>   RV is {RV_dice},   Myo is {Myo_dice},   LV is {LV_dice},   avg_dice is {(RV_dice + Myo_dice + LV_dice) / 3}"
            )

            print(
                f"hd95   ===>   RV is {RV_hd95},   Myo is {Myo_hd95},   LV is {LV_hd95},   avg_hd95 is {(RV_hd95 + Myo_hd95 + LV_hd95) / 3}"
            )

            RV_dices += RV_dice
            Myo_dices += Myo_dice
            LV_dices += LV_dice

            RV_hd95s += RV_hd95
            Myo_hd95s += Myo_hd95
            LV_hd95s += LV_hd95

            cnt += 1

        RV_dice_avg = RV_dices / cnt
        Myo_dice_avg = Myo_dices / cnt
        LV_dice_avg = LV_dices / cnt

        RV_hd95_avg = RV_hd95s / cnt
        Myo_hd95_avg = Myo_hd95s / cnt
        LV_hd95_avg = LV_hd95s / cnt

        print("Average")
        print(
            f"dice   ===>   RV is {RV_dice_avg},   Myo is {Myo_dice_avg},   LV is {LV_dice_avg},   avg_dice is {(RV_dice_avg + Myo_dice_avg + LV_dice_avg) / 3}"
        )

        print(
            f"hd95   ===>   RV is {RV_hd95_avg},   Myo is {Myo_hd95_avg},   LV is {LV_hd95_avg},   avg_hd95 is {(RV_hd95_avg + Myo_hd95_avg + LV_hd95_avg) / 3}"
        )

    def validation_step(self, batch):
        image, label = self.get_input(batch)

        output = self.window_infer(image, self.model, pred_type="ddim_sample")
        output = torch.sigmoid(output)

        output = (output > 0.5).float().cpu().numpy()
        target = label.cpu().numpy()

        return output, target

    def validation_end(self, mean_val_outputs):
        dices = mean_val_outputs
        print(dices)
        mean_dice = sum(dices) / len(dices)

        self.log("mean_dice", mean_dice, step=self.epoch)

        if mean_dice > self.best_mean_dice:
            self.best_mean_dice = mean_dice
            save_new_model_and_delete_last(self.model,
                                           os.path.join(model_save_path,
                                                        f"best_model_{mean_dice:.4f}.pt"),
                                           delete_symbol="best_model")

        save_new_model_and_delete_last(self.model,
                                       os.path.join(model_save_path,
                                                    f"final_model_{mean_dice:.4f}.pt"),
                                       delete_symbol="final_model")

        print(
            f"RV is {mean_val_outputs[0]}, Myo is {mean_val_outputs[1]},LV is {mean_val_outputs[2]}, mean_dice is {mean_dice}"
        )


class ACDCTestDataset(Dataset):
    def __init__(self, data_dir, transform=None):
        self.annotation_lines = []

        for file in os.listdir(os.path.join(data_dir)):
            if file.endswith("_gt_.nii.gz"):
                filename = file.split("_gt")[0]
                self.annotation_lines.append(filename)

        self.data_dir = data_dir
        self.transform = transform

    def __len__(self):
        return len(self.annotation_lines)

    def __getitem__(self, index):
        """Get the images"""
        name = self.annotation_lines[index] + '.nii.gz'

        img_path = os.path.join(self.data_dir, name)

        mask_name = self.annotation_lines[index] + '_gt_.nii.gz'
        mask_path = os.path.join(self.data_dir, mask_name)

        image = nib.load(img_path).get_fdata()
        label = nib.load(mask_path).get_fdata()

        sample = {
            "image": image,
            "label": label
        }

        if self.transform:
            state = torch.get_rng_state()
            torch.set_rng_state(state)
            sample = self.transform(sample)

        sample["file"] = self.annotation_lines[index]

        return sample


if __name__ == "__main__":
    trainer = ACDCTrainer(env_type=env,
                          max_epochs=max_epoch,
                          batch_size=batch_size,
                          device=device,
                          logdir=logdir,
                          val_every=val_every,
                          num_gpus=num_gpus,
                          master_port=17751,
                          training_script=__file__)

    trainer.validation_ds()
