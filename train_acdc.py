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

from light_training.evaluation.metric import dice
from light_training.trainer import Trainer
from light_training.utils.files_helper import save_new_model_and_delete_last
from light_training.utils.lr_scheduler import LinearWarmupCosineAnnealingLR

set_determinism(123)

train_dir = ""
eval_dir = ""

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

device = "cuda:0"


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
                                                 overlap=0.5)
        self.model = DiffSFCT()

        self.best_mean_dice = 0.0
        self.optimizer = torch.optim.AdamW(self.model.parameters(), lr=2e-4, weight_decay=1e-3)
        self.ce = nn.CrossEntropyLoss()
        self.mse = nn.MSELoss()
        self.scheduler = LinearWarmupCosineAnnealingLR(self.optimizer,
                                                       warmup_epochs=100,
                                                       max_epochs=max_epochs)
        self.bce = nn.BCEWithLogitsLoss()
        self.dice_loss = DiceLoss(sigmoid=True)

    def training_step(self, batch):
        image, label = self.get_input(batch)
        x_start = label

        x_start = (x_start) * 2 - 1
        x_t, t, noise = self.model(x=x_start, pred_type="q_sample")
        pred_xstart = self.model(x=x_t, step=t, image=image, pred_type="denoise")

        loss_dice = self.dice_loss(pred_xstart, label)
        loss_bce = self.bce(pred_xstart, label)

        pred_xstart = torch.sigmoid(pred_xstart)
        loss_mse = self.mse(pred_xstart, label)

        loss = loss_dice + loss_bce + loss_mse

        self.log("train_loss", loss, step=self.global_step)

        return loss

    def get_input(self, batch):
        image = batch["image"]
        label = batch["label"]

        label = label.float()
        return image, label

    def validation_ds(self, ):
        transform_valid = transforms.Compose([
            ValidGenerator([image_size, image_size], 4),
        ])

        RV_dice = 0
        Myo_dice = 0
        LV_dice = 0
        cnt = 0

        for patient_dir in os.listdir(eval_dir):
            valid_ds = ACDCTestDataset(os.path.join(eval_dir, patient_dir), transform=transform_valid)
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
            RV = dice(o, t)

            o = self.outputs[:, 1]
            t = self.targets[:, 1]
            Myo = dice(o, t)

            o = self.outputs[:, 2]
            t = self.targets[:, 2]
            LV = dice(o, t)

            print(
                f"dice   ===>   RV is {RV},   Myo is {Myo},   LV is {LV},   avg_dice is {(RV + Myo + LV) / 3}"
            )

            RV_dice += RV
            Myo_dice += Myo
            LV_dice += LV
            cnt += 1

        RV_avg = RV_dice / cnt
        Myo_avg = Myo_dice / cnt
        LV_avg = LV_dice / cnt

        print("Average")
        print(
            f"dice   ===>   RV is {RV_avg},   Myo is {Myo_avg},   LV is {LV_avg},   avg_dice is {(RV_avg + Myo_avg + LV_avg) / 3}"
        )

        return [RV_avg, Myo_avg, LV_avg]

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

    train_ds, test_ds = get_loader_acdc(train_dir=train_dir, test_dir=eval_dir, image_size=image_size,
                                        num_classes=num_classes + 1)

    trainer.train(train_dataset=train_ds, val_dataset=test_ds)
