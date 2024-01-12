"""Training script for vxm with a fixed scale and blur"""

import copy
import os
import random
import sys
from typing import Sequence

import cv2
import deep_trainer
import deep_trainer.pytorch.metric
import numpy as np
import torch
import torch.utils.data
import torchvision.transforms.v2 as transforms  # type: ignore
import tqdm

import byotrack

sys.path.append(f"{os.environ.get('EXPYRUN_CWD', '.')}/voxelmorph/")
os.environ["NEURITE_BACKEND"] = "pytorch"
os.environ["VXM_BACKEND"] = "pytorch"

import voxelmorph as vxm


SCALE = 4
BLUR = 1


class Loss:
    def __init__(self, weight=0.01):
        self.weight = weight
        self.image_loss = vxm.losses.MSE()
        self.penal_loss = vxm.losses.Grad()
        self.last_image_loss = torch.tensor(float("nan"))
        self.last_penal_loss = torch.tensor(float("nan"))

    def __call__(self, predictions, targets):
        self.last_image_loss = self.image_loss.loss(targets, predictions[0])
        self.last_penal_loss = self.penal_loss.loss(targets, predictions[1])
        return self.last_image_loss + self.weight * self.last_penal_loss


class Trainer(deep_trainer.PytorchTrainer):
    """Trainer for vxm"""

    def _default_process_batch(self, batch):
        sources, targets = super()._default_process_batch(batch)
        return (sources, targets), targets

    def train_step(self, batch, criterion):
        inputs, targets = self.process_train_batch(batch)

        with torch.cuda.amp.autocast(self.use_amp):
            predictions = self.model(*inputs)
            loss: torch.Tensor = criterion(predictions, targets)
            for prediction in predictions:
                prediction.detach()
            self.metrics_handler.update((inputs, targets), predictions)

        self.backward(loss)

        metrics = self.metrics_handler.last_values
        metrics["Loss"] = loss.item()

        if isinstance(criterion, Loss):
            metrics["ImageLoss"] = criterion.last_image_loss.item()
            metrics["PenalLoss"] = criterion.last_penal_loss.item()

        return metrics

    def eval_step(self, batch):
        inputs, targets = self.process_eval_batch(batch)

        with torch.cuda.amp.autocast(self.use_amp):
            predictions = self.model(*inputs)
            self.metrics_handler.update((inputs, targets), predictions)

        metrics = self.metrics_handler.last_values
        validation_metric = self.metrics_handler.get_validation_metric()
        if isinstance(validation_metric, deep_trainer.pytorch.metric.PytorchMetric):
            if isinstance(validation_metric.loss_function, Loss):
                metrics["ImageLoss"] = validation_metric.loss_function.last_image_loss.item()  # type: ignore
                metrics["PenalLoss"] = validation_metric.loss_function.last_penal_loss.item()  # type: ignore

        return metrics


class MultiVideoDataset(torch.utils.data.Dataset):
    """Load in ram videos with some preprocessing. All videos should share the same size

    In train, it returns pair of frames from -10 to 10 offset. In eval, it returns consecutive frames.
    """

    MAX_TRAIN_OFFSET = 10

    def __init__(self, videos: Sequence[Sequence[np.ndarray]], train=False) -> None:
        self.train = train
        width, height, channels = videos[0][0].shape[:3]  # Should be the same accross videos

        self.sequences = []
        self.lengths = []
        for video in tqdm.tqdm(videos):
            self.lengths.append(len(video))
            processed_frames = torch.zeros((len(video), channels, width // SCALE, height // SCALE))
            for i, frame in enumerate(video):
                frame = cv2.GaussianBlur(frame, (0, 0), BLUR, BLUR)
                frame = cv2.resize(frame, (0, 0), fx=1 / SCALE, fy=1 / SCALE, interpolation=cv2.INTER_LINEAR)
                if len(frame.shape) == 2:
                    frame = frame[..., None]

                processed_frames[i] = torch.tensor(frame).permute(2, 0, 1)

            self.sequences.append(processed_frames)

        self.train_transforms = transforms.Compose([transforms.RandomHorizontalFlip(), transforms.RandomVerticalFlip()])

    def __len__(self) -> int:
        return sum(self.lengths) - (not self.train) * len(self.lengths)

    def __getitem__(self, index: int) -> tuple[torch.Tensor, torch.Tensor]:
        i = 0
        length = self.lengths[0]

        for i, length in enumerate(self.lengths):
            length -= not self.train
            if index >= length:
                index -= length
            else:
                break

        source = self.sequences[i][index]

        if self.train:
            mini = max(0, index - self.MAX_TRAIN_OFFSET)
            maxi = min(length - 1, index + self.MAX_TRAIN_OFFSET)
            index = random.randint(mini, maxi)
            target = self.sequences[i][index]
            if self.train_transforms is not None:
                images = torch.cat([source, target], dim=1)  # concat on channel axis
                images = self.train_transforms(images)
                source = images[:, : source.shape[1]]
                target = images[:, source.shape[1] :]

            return source, target

        # Eval
        index += 1
        return source, self.sequences[i][index]


def train():
    """Training we used

    This function isn't directly reproducible. You should change the path we used.
    """

    # True video
    true_video = byotrack.Video("../data/dupre/20160412_stk_0001.tif")
    transform_config = byotrack.VideoTransformConfig(
        aggregate=True, normalize=True, q_min=0.02, q_max=0.999, smooth_clip=0.1
    )
    true_video.set_transform(transform_config)

    training_videos = [true_video]

    # Load 4 more synthetic videos for training

    paths = [
        "/home/rreme/data/pasteur/simulation/springs/1000/1.8/111/exp.0/video.mp4",
        "/home/rreme/data/pasteur/simulation/springs/1000/1.8/222/exp.0/video.mp4",
        "/home/rreme/data/pasteur/simulation/flow_20140829_1/1000/1.8/111/exp.0/video.mp4",
        "/home/rreme/data/pasteur/simulation/flow_20140829_1/1000/1.8/222/exp.0/video.mp4",
    ]

    for path in tqdm.tqdm(paths):
        video = byotrack.Video(path)
        video.set_transform(byotrack.VideoTransformConfig(aggregate=True, normalize=True, q_min=0.00, q_max=1.0))
        training_videos.append(video)

    trainset = MultiVideoDataset(training_videos, train=True)
    valset = copy.copy(trainset)  # Shallow copy, to keep the same data, but using it only in a frame-to-frame context
    valset.train = False

    train_loader = torch.utils.data.DataLoader(
        trainset,
        32,
        shuffle=True,
        num_workers=16,
        drop_last=True,
    )

    val_loader = torch.utils.data.DataLoader(
        valset,
        64,
        shuffle=False,
        num_workers=16,
        drop_last=False,
    )

    channels, *in_shape = trainset[0][0].shape
    model = vxm.networks.VxmDense(in_shape, src_feats=channels, trg_feats=channels, int_steps=0)
    model.flow.bias = None  # Disable bias

    optimizer = torch.optim.Adam(
        model.parameters(), lr=1e-3
    )  # Could add a scheduler, but we did not notice any real improvement

    trainer = Trainer(model, optimizer)  # Training can be followed with tensorboard
    trainer.train(150, train_loader, Loss(0.003), val_loader, epoch_size=100)

    torch.save(model.state_dict(), "vxm.pt")


if __name__ == "__main__":
    train()
