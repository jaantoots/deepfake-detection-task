"""Classification model training"""
import argparse
import inspect
import json
import logging
import logging.config
import os
from dataclasses import dataclass
from typing import Any

import click
import torch
import torch.nn as nn
import torch.optim as optim
from sklearn.metrics import roc_auc_score
from torch.nn.functional import binary_cross_entropy_with_logits, sigmoid
from torch.utils.data import DataLoader
from torchvision import transforms

import config
from dataset import Faces, MiniFaces, Split
from model import resnext50_32x4d
from utils import progress

LOGGER = logging.getLogger(__name__)

print("PyTorch version:", torch.__version__)
print("CUDA version:", torch.version.cuda)
print("cuDNN version:", torch.backends.cudnn.version())


@progress(fg="green")
def train(model, loader, optimizer, cuda=True):
    """Train classification model on data from loader"""
    model.train()

    def train_step(inputs, labels):
        labels = labels.float()
        if cuda:
            inputs = inputs.cuda(non_blocking=True)
            labels = labels.cuda(non_blocking=True)
        optimizer.zero_grad()
        outputs = torch.squeeze(model(inputs), 1)
        loss = binary_cross_entropy_with_logits(outputs, labels)
        loss.backward()
        optimizer.step()
        # Do not accumulate history
        LOGGER.debug("%8.5f", float(loss))
        return float(loss)

    losses = [train_step(inputs, labels) for inputs, labels, _ in loader]
    avg_loss = sum(losses) / len(losses)
    return avg_loss


@torch.no_grad()
@progress(fg="yellow")
def validate(model, loader, cuda=True):
    """Evaluate classification model on data from loader"""
    model.eval()

    def validate_step(inputs, labels):
        labels = labels.float()
        if cuda:
            inputs = inputs.cuda(non_blocking=True)
            labels = labels.cuda(non_blocking=True)
        outputs = torch.squeeze(model(inputs), 1)
        losses = binary_cross_entropy_with_logits(outputs, labels, reduction="none")
        scores = sigmoid(outputs)
        # Do not accumulate history
        return losses.tolist(), scores.tolist()

    files, labels, scores, losses = zip(
        *(
            (file, label, score, loss)
            for inputs, labels, files in loader
            for loss, score, file, label in zip(
                *validate_step(inputs, labels), files, labels.tolist()
            )
        )
    )

    results = {
        "results": [
            {"file": file, "label": label, "score": score}
            for file, label, score in zip(files, labels, scores)
        ],
    }
    avg_loss = sum(losses) / len(losses)
    return (
        avg_loss,
        roc_auc_score(labels, scores),
        results,
    )


@dataclass
class State:
    """Model state"""

    cnn: nn.Module
    optimizer: optim.Optimizer
    scheduler: Any = None
    count: int = 0

    def load(self, file: str):
        """Load model from file"""
        state = torch.load(file)
        self.cnn.load_state_dict(state["cnn"])
        self.optimizer.load_state_dict(state["optimizer"])
        if self.scheduler is not None:
            self.scheduler.load_state_dict(state["scheduler"])
        self.count = state["epoch"]
        LOGGER.warning("Load state from '%s' (epoch %d)", file, self.count)

    def save(self, directory: str):
        """Save model to directory"""
        state = {
            "epoch": self.count,
            "cnn": self.cnn.state_dict(),
            "optimizer": self.optimizer.state_dict(),
            "scheduler": self.scheduler.state_dict()
            if self.scheduler is not None
            else None,
        }
        torch.save(state, os.path.join(directory, f"model-{self.count:03}.pth"))


@dataclass
class Loop:
    """Training loop with logging and saving"""

    output: str
    num_epochs: int
    load: str = None
    cuda: bool = True

    def __call__(
        self, state: State, train_loader, val_loader,
    ):
        if self.load is not None:
            state.load(self.load)
        while self.num_epochs > 0:
            if state.count >= self.num_epochs:
                break
            state.count += 1
            train_loss = train(
                state.cnn, train_loader, state.optimizer, cuda=self.cuda,
            )
            val_loss, score, results = validate(state.cnn, val_loader, cuda=self.cuda,)
            if state.scheduler is not None:
                if "metrics" in inspect.signature(state.scheduler.step).parameters:
                    state.scheduler.step(val_loss)
                else:
                    state.scheduler.step()
            LOGGER.info(
                "%3d %8.5f %8.5f %8.5f", state.count, train_loss, val_loss, score,
            )
            with open(
                os.path.join(self.output, f"validate-{state.count:03}.json"), "w",
            ) as results_file:
                json.dump(results, results_file)
            state.save(self.output)
        else:
            val_loss, score, results = validate(state.cnn, val_loader, cuda=self.cuda,)
            LOGGER.warning("%8.5f %8.5f", val_loss, score)
            with open(os.path.join(self.output, f"validate.json"), "w") as results_file:
                json.dump(results, results_file)

    @staticmethod
    def parser() -> argparse.ArgumentParser:
        """Training loop arguments parser"""
        parser = argparse.ArgumentParser(add_help=False)
        parser.add_argument("-l", "--load", help="Load training state from file")
        parser.add_argument("-o", "--output", default=".", help="Output directory")
        parser.add_argument(
            "-n",
            "--num-epochs",
            type=int,
            default=config.NUM_EPOCHS,
            help="Number of epochs (use 0 for eval)",
        )
        return parser

    @classmethod
    def from_args(cls, args: argparse.Namespace, cuda=True):
        """Use training loop arguments from parser"""
        os.makedirs(args.output, exist_ok=True)
        log_config = config.LOGGING
        log_config["handlers"]["file"]["filename"] = os.path.join(
            args.output, "debug.log"
        )
        logging.config.dictConfig(log_config)
        LOGGER.setLevel(logging.DEBUG)
        return cls(args.output, args.num_epochs, load=args.load, cuda=cuda)


def main():
    """Train feature extraction model"""
    parser = argparse.ArgumentParser(description=__doc__, parents=[Loop.parser()],)
    parser.add_argument("--disable-cuda", action="store_true", help="Disable CUDA")
    parser.add_argument(
        "-p", "--parallel", action="store_true", help="Parallelize over all GPUs",
    )
    parser.add_argument(
        "-b", "--batch-size", type=int, default=config.BATCH_SIZE, help="Batch size",
    )
    parser.add_argument(
        "-r", "--lr", type=float, default=config.LR, help="Base learning rate"
    )
    parser.add_argument(
        "-m", "--momentum", type=float, default=config.MOMENTUM, help="Momentum",
    )
    parser.add_argument(
        "-d", "--decay-scale", type=int, help="Learning rate decay scale"
    )
    parser.add_argument(
        "--data-root", default=config.DATA_ROOT, help="Dataset root directory"
    )
    parser.add_argument(
        "-s", "--small", action="store_true", help="Use smaller dataset for testing",
    )
    args = parser.parse_args()
    cuda = not args.disable_cuda
    dataset = Faces if not args.small else MiniFaces
    loop = Loop.from_args(args, cuda=cuda)

    train_dataset = dataset(
        args.data_root,
        tf=transforms.Compose(
            [
                transforms.Resize((224, 224)),
                transforms.ToTensor(),
                # Use ImageNet normalization
                transforms.Normalize(
                    mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]
                ),
            ]
        ),
        split=Split.TRAIN,
    )
    train_loader = DataLoader(
        train_dataset,
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=os.cpu_count(),
        pin_memory=True,
        drop_last=True,
    )
    click.secho("Training dataset:", fg="green", bold=True)
    click.echo(train_loader.dataset)
    val_dataset = dataset(
        args.data_root,
        tf=transforms.Compose(
            [
                transforms.Resize((224, 224)),
                transforms.ToTensor(),
                # Use ImageNet normalization
                transforms.Normalize(
                    mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]
                ),
            ]
        ),
        split=Split.TVAL,
    )
    val_loader = DataLoader(
        val_dataset,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=os.cpu_count(),
        pin_memory=True,
        drop_last=False,
    )
    click.secho("Validation dataset:", fg="yellow", bold=True)
    click.echo(val_loader.dataset)
    click.echo()

    cnn = resnext50_32x4d(pretrained=True, num_classes=1)
    if cuda:
        cnn = cnn.cuda()
    if args.parallel:
        cnn = nn.DataParallel(cnn)
    optimizer = optim.SGD(
        cnn.parameters(),
        lr=args.lr,
        momentum=args.momentum,
        weight_decay=config.WEIGHT_DECAY,
    )
    scheduler = (
        optim.lr_scheduler.ExponentialLR(optimizer, 1 - 1 / args.decay_scale)
        if args.decay_scale is not None
        else optim.lr_scheduler.ReduceLROnPlateau(
            optimizer,
            mode="min",
            factor=0.1,
            patience=5,
            verbose=True,
            threshold=0.001,
            cooldown=3,
            eps=1e-08,
        )
    )

    state = State(cnn, optimizer, scheduler)
    with torch.autograd.detect_anomaly():
        loop(state, train_loader, val_loader)


if __name__ == "__main__":
    main()
