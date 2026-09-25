"""Continued or from-scratch wide pre-training of nanoTabPFN (TabPFN-Wide style).

Trains a classifier on the on-the-fly TabICL prior with the three orthogonal TabPFN-Wide
levers enabled: continuous feature widening (Algorithm 1), categorical feature widening
(Algorithm 2) and MCAR missingness injection. Two modes:

  * default (continued pre-training): loads the released checkpoint (auto-downloaded and
    migrated to the value+indicator encoder) and keeps training it. The paper uses a low
    learning rate here (1e-5).
  * --from-scratch: builds a fresh model and trains it from random init (lr 1e-4).

Widening and missingness are opt-in; with the defaults they are ON, so the run actually
exercises them. Checkpoints are written by train() to workdir/<run-name>/latest_checkpoint.pth.

Quick smoke test (tiny model + prior, a few seconds per epoch):
    python examples/pretrain_wide.py --from-scratch --epochs 3 --steps 5 \
        --layers 2 --heads 2 --embeddingsize 32 --hiddensize 64 \
        --maxfeatures 20 --maxdatapoints 200 --addfeaturesmax 30

Real run, continued from the released checkpoint (paper-ish settings):
    python examples/pretrain_wide.py --epochs 2000 --steps 100 --lr 1e-5

Note: end-to-end reproducibility is not wired yet -- the widening/missingness sampling in
the training loop still uses the global RNG, so runs are not bit-exact even with a seed.
"""

import argparse

import torch
from torch import nn

from tfmplayground.callbacks import ConsoleLoggerCallback
from tfmplayground.external_priors import TabICLPriorDataLoader
from tfmplayground.interface import NanoTabPFNClassifier
from tfmplayground.models.nanotabpfn import NanoTabPFNModel
from tfmplayground.train import WideningConfig, train
from tfmplayground.utils import get_default_device, set_randomness_seed


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Wide pre-training of nanoTabPFN (continued or from scratch).")
    # mode
    p.add_argument("--from-scratch", action="store_true",
                   help="train a fresh model instead of continuing from the released checkpoint")
    # model architecture (only used with --from-scratch; otherwise read from the checkpoint)
    p.add_argument("--heads", type=int, default=6)
    p.add_argument("--embeddingsize", type=int, default=192)
    p.add_argument("--hiddensize", type=int, default=768)
    p.add_argument("--layers", type=int, default=6)
    # optimisation
    p.add_argument("--lr", type=float, default=None,
                   help="learning rate; default 1e-4 from scratch, 1e-5 for continued pre-training")
    p.add_argument("--epochs", type=int, default=2000)
    p.add_argument("--steps", type=int, default=100, help="batches per epoch")
    p.add_argument("--batchsize", type=int, default=1)
    p.add_argument("--accumulate", type=int, default=1, help="gradients to accumulate before an update")
    p.add_argument("--runname", type=str, default="pretrain_wide")
    # prior (TabICL, sampled on the fly)
    p.add_argument("--minfeatures", type=int, default=5)
    p.add_argument("--maxfeatures", type=int, default=50)
    p.add_argument("--mindatapoints", type=int, default=100)
    p.add_argument("--maxdatapoints", type=int, default=1000)
    p.add_argument("--maxclasses", type=int, default=10,
                   help="max classes for the prior (from scratch only; continued uses the checkpoint capacity)")
    # widening (Algorithm 1 + 2); addfeaturesmax=0 disables widening
    p.add_argument("--addfeaturesmin", type=int, default=0)
    p.add_argument("--addfeaturesmax", type=int, default=500,
                   help="max new features to add per dataset; 0 disables widening")
    p.add_argument("--sparsitymax", type=float, default=0.05)
    p.add_argument("--noisemax", type=float, default=1.0)
    p.add_argument("--maxcats", type=int, default=20, help="max categorical cardinality Kmax (Algorithm 2)")
    p.add_argument("--includeoriginalprob", type=float, default=0.5)
    # missingness; missingratemax=0 disables injection
    p.add_argument("--missingratemax", type=float, default=0.5,
                   help="max MCAR missing rate sampled per batch; 0 disables injection")
    return p.parse_args()


def main() -> None:
    args = parse_args()
    set_randomness_seed(2402)
    device = torch.device(get_default_device())

    # ---- model + prior ------------------------------------------------------
    if args.from_scratch:
        prior = TabICLPriorDataLoader(
            num_steps=args.steps, batch_size=args.batchsize,
            num_datapoints_min=args.mindatapoints, num_datapoints_max=args.maxdatapoints,
            min_features=args.minfeatures, max_features=args.maxfeatures,
            max_num_classes=args.maxclasses, device=device,
        )
        model = NanoTabPFNModel(
            num_attention_heads=args.heads, embedding_size=args.embeddingsize,
            mlp_hidden_size=args.hiddensize, num_layers=args.layers,
            num_outputs=prior.max_num_classes,
        )
        default_lr = 1e-4
        print(f"From scratch: {args.layers} layers, embedding {args.embeddingsize}, "
              f"{prior.max_num_classes} classes.")
    else:
        # NanoTabPFNClassifier() downloads the released checkpoint (if needed) and migrates its
        # feature encoder to the value+indicator layout; we take its model and keep training it.
        model = NanoTabPFNClassifier(device=device).model
        prior = TabICLPriorDataLoader(
            num_steps=args.steps, batch_size=args.batchsize,
            num_datapoints_min=args.mindatapoints, num_datapoints_max=args.maxdatapoints,
            min_features=args.minfeatures, max_features=args.maxfeatures,
            max_num_classes=model.num_outputs, device=device,  # match the checkpoint's class capacity
        )
        default_lr = 1e-5
        print(f"Continued pre-training from the released checkpoint "
              f"({model.num_outputs}-class capacity).")

    lr = args.lr if args.lr is not None else default_lr

    # ---- widening + missingness --------------------------------------------
    widening = WideningConfig(
        add_features_min=args.addfeaturesmin, add_features_max=args.addfeaturesmax,
        sparsity_max=args.sparsitymax, noise_max=args.noisemax,
        include_original_prob=args.includeoriginalprob, max_cats=args.maxcats,
    )
    print(f"Widening: up to +{args.addfeaturesmax} features "
          f"(sparsity<={args.sparsitymax}, noise<={args.noisemax}, max_cats={args.maxcats}) | "
          f"missing_rate_max={args.missingratemax} | lr={lr}")

    # ---- train --------------------------------------------------------------
    criterion = nn.CrossEntropyLoss()
    _trained_model, loss = train(
        model=model,
        prior=prior,
        criterion=criterion,
        epochs=args.epochs,
        accumulate_gradients=args.accumulate,
        lr=lr,
        device=device,
        callbacks=[ConsoleLoggerCallback()],
        run_name=args.runname,
        missing_rate_max=args.missingratemax,
        widening=widening,
    )
    print(f"Done. Final mean loss {loss:.4f}. Checkpoint: workdir/{args.runname}/latest_checkpoint.pth")


if __name__ == "__main__":
    main()