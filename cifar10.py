from functools import partial
from typing import Tuple
import chika
import homura
import torch
import torch.nn.functional as F
from homura import lr_scheduler, reporters, trainers
from homura.vision import DATASET_REGISTRY, MODEL_REGISTRY
from sam import SAMSGD as _SAMSGD
from pgn_sam import PGN_SAMSGD as _PGN_SAMSGD
from torch.utils.data import Subset, DataLoader
import random
import numpy as np

def SAM(lr=1e-1, momentum=0.0, dampening=0.0,
        weight_decay=0.0, nesterov=False, rho=0.05):
    return partial(_SAMSGD, **locals())

def PGN_SAM(lr=1e-1, momentum=0.0, dampening=0.0,
            weight_decay=0.0, nesterov=False, rho=0.05, alpha=0.1):
    return partial(_PGN_SAMSGD, **locals())

@chika.config
class Optim:
    epochs: int = 200
    name: str = chika.choices("pgn_sam", "sam", "sgd")
    lr: float = 0.1
    weight_decay: float = 5e-4
    rho: float = 5e-2
    alpha: float = 0.1

@chika.config
class Config:
    optim: Optim
    model: str = chika.choices("resnet20", "resnet56", "se_resnet56", "wrn28_2", "resnext29_32x4d")
    batch_size: int = 128
    use_amp: bool = False
    jit_model: bool = False
    seed: int = 1
    gpu: int = chika.bounded(0, 0, torch.cuda.device_count())
    subset_fraction: float = chika.bounded(0.2, 0.2, 1.0)  # New parameter for subset fraction

def get_subset_loader(loader, subset_fraction):
    dataset = loader.dataset
    num_samples = len(dataset)
    indices = list(range(num_samples))
    random.shuffle(indices)
    split = int(np.floor(subset_fraction * num_samples))
    subset_indices = indices[:split]
    subset = Subset(dataset, subset_indices)
    return DataLoader(subset, batch_size=loader.batch_size, shuffle=True, num_workers=loader.num_workers)


class Trainer(trainers.SupervisedTrainer):
    def iteration(self,
                  data: Tuple[torch.Tensor, torch.Tensor]
                  ) -> None:
        if not self.is_train:
            return super().iteration(data)
        input, target = data
        def closure():
            self.optimizer.zero_grad()
            output = self.model(input)
            loss = self.loss_f(output, target)
            loss.backward()
            return loss
        loss = self.optimizer.step(closure)
        self.reporter.add("loss", loss)


def _main(cfg):
    model = MODEL_REGISTRY(cfg.model)(num_classes=10)
    if cfg.jit_model:
        model = torch.jit.script(model)

    # Get the dataloaders
    train_loader, test_loader = DATASET_REGISTRY("cifar10")(cfg.batch_size, num_workers=4, download=True)

    # Create a subset of the training data if needed
    if cfg.subset_fraction < 1.0:
        train_loader = get_subset_loader(train_loader, cfg.subset_fraction)

    # Defining the optimizer
    if cfg.optim.name == "sam":
        optimizer = SAM(lr=cfg.optim.lr, momentum=0.9, weight_decay=cfg.optim.weight_decay, rho=cfg.optim.rho)
    elif cfg.optim.name == "pgn_sam":
        optimizer = PGN_SAM(lr=cfg.optim.lr, momentum=0.9, weight_decay=cfg.optim.weight_decay, rho=cfg.optim.rho, alpha=cfg.optim.alpha)
    else:
        optimizer = homura.optim.SGD(lr=cfg.optim.lr, momentum=0.9, weight_decay=cfg.optim.weight_decay)

    # Defining the scheduler
    scheduler = lr_scheduler.CosineAnnealingWithWarmup(cfg.optim.epochs, 4, 5)

    with Trainer(model,
                 optimizer,
                 F.cross_entropy,
                 reporters=[reporters.TensorboardReporter('.')],
                 scheduler=scheduler,
                 use_amp=cfg.use_amp,
                 ) as trainer:
        for _ in trainer.epoch_range(cfg.optim.epochs):
            trainer.train(train_loader)
            trainer.test(test_loader)
            trainer.scheduler.step()

        print(f"Max Test Accuracy={max(trainer.reporter.history('accuracy/test')):.3f}")

@chika.main(cfg_cls=Config, strict=True)
def main(cfg: Config):
    torch.cuda.set_device(cfg.gpu)
    with homura.set_seed(cfg.seed):
        _main(cfg)

if __name__ == '__main__':
    main()