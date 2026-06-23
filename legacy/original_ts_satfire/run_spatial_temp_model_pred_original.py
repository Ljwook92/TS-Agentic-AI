import argparse
import csv
import json
import os
import random
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from monai.data import decollate_batch
from monai.losses.dice import DiceCELoss, DiceLoss
from monai.metrics import DiceMetric, MeanIoU
from monai.transforms import Activations, AsDiscrete, Compose
from sklearn.metrics import f1_score, jaccard_score
from torch import nn, optim
from torch.cuda.amp import GradScaler
from torch.utils.data import DataLoader
from tqdm import tqdm

from original_ts_satfire.data_generator_pred_torch_original import FireDataset
from original_ts_satfire.data_generator_pred_goes_spatial_original import FireDatasetWithGOESSpatial
from spatial_models.attentionunet import AttentionUnet
from spatial_models.swinunetr.swinunetr import SwinUNETR
from spatial_models.unet import UNet
from spatial_models.unetr.unetr import UNETR


SEED = 42
np.random.seed(SEED)
torch.manual_seed(SEED)


PAPER_MODEL_CONFIGS = {
    "unet3d": {"batch_size": 64, "learning_rate": 1e-3, "loss": "dice_ce"},
    "attunet": {"batch_size": 8, "learning_rate": 1.2e-3, "loss": "dice"},
    "unetr3d": {"batch_size": 4, "learning_rate": 1e-3, "loss": "dice_ce"},
    "swinunetr3d": {"batch_size": 8, "learning_rate": 1e-3, "loss": "dice_ce"},
}


def configure_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def resolve_training_config(args):
    if args.training_profile == "paper":
        defaults = PAPER_MODEL_CONFIGS[args.m]
        batch_size = args.b if args.b is not None else defaults["batch_size"]
        learning_rate = args.lr if args.lr is not None else defaults["learning_rate"]
        loss_name = defaults["loss"] if args.loss == "auto" else args.loss
        hidden_size = args.ed if args.ed is not None else 36
        max_epochs = args.epochs if args.epochs is not None else 200
    else:
        batch_size = args.b if args.b is not None else 1
        learning_rate = args.lr if args.lr is not None else 1e-4
        loss_name = "dice" if args.loss == "auto" else args.loss
        hidden_size = args.ed if args.ed is not None else 24
        max_epochs = args.epochs if args.epochs is not None else 100
    return batch_size, learning_rate, loss_name, hidden_size, max_epochs


def build_criterion(loss_name, activation, foreground_weight, device):
    activation_args = {
        "sigmoid": activation == "sigmoid",
        "softmax": activation == "softmax",
    }
    if loss_name == "dice":
        return DiceLoss(include_background=True, reduction="mean", **activation_args)
    if loss_name == "dice_ce":
        class_weight = torch.tensor([1.0, foreground_weight], dtype=torch.float32, device=device)
        return DiceCELoss(
            include_background=True,
            reduction="mean",
            weight=class_weight,
            lambda_dice=1.0,
            lambda_ce=1.0,
            **activation_args,
        )
    raise ValueError(f"Unsupported loss: {loss_name}")


def criterion_target(labels, loss_name):
    # MONAI DiceCELoss expects floating one-hot targets when target channels match logits.
    return labels.float() if loss_name == "dice_ce" else labels


def safe_tag(value):
    return "".join(char if char.isalnum() or char in "-_" else "_" for char in value)


def run_slug(args, batch_size, learning_rate, loss_name, hidden_size):
    return (
        f"{safe_tag(args.experiment_tag)}_{args.goes_variant}_{args.training_profile}_{loss_name}_seed{args.seed}_"
        f"model{args.m}_ts{args.ts}_b{batch_size}_lr{learning_rate:g}_ed{hidden_size}_nc{args.nc}"
    )


def write_csv(path, rows):
    if not rows:
        return
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def goes_train_path(root_path, split, ts_length, interval, goes_variant):
    candidates = [
        os.path.join(
            root_path,
            f"dataset_{split}",
            f"pred_{split}_{goes_variant}_seqtoseq_alll_{ts_length}i_{interval}.npy",
        ),
        os.path.join(
            root_path,
            f"GOES_{split}",
            f"pred_{split}_{goes_variant}_seqtoseq_alll_{ts_length}i_{interval}.npy",
        ),
    ]
    for path in candidates:
        if os.path.exists(path):
            return path
    raise FileNotFoundError(f"No GOES spatial {split} file found. Tried: {candidates}")


def goes_test_path(root_path, fire_id, ts_length, interval, goes_variant):
    filename = f"pred_{fire_id}_{goes_variant}_seqtoseql_{ts_length}i_{interval}.npy"
    candidates = [
        os.path.join(root_path, "dataset_test", filename),
        os.path.join(root_path, "GOES_test", filename),
    ]
    for path in candidates:
        if os.path.exists(path):
            return path
    raise FileNotFoundError(f"No GOES spatial test file found for {fire_id}. Tried: {candidates}")


def forward_model(model, batch, device, use_goes_concat):
    data = batch['data'].to(device)
    if use_goes_concat:
        goes = batch['goes_spatial'].to(device)
        data = torch.cat([data, goes], dim=1)
    return model(data).mean(2), data.size(0)


def foreground_probability(logits, activation):
    if activation == "softmax":
        return torch.softmax(logits, dim=1)[:, 1]
    return torch.sigmoid(logits)[:, 1]


class _DummyRun:
    name = ""
    id = "disabled"
    dir = "."


class _DummyWandb:
    def __init__(self):
        self.run = _DummyRun()
        self.config = {}

    def init(self, *args, **kwargs):
        return self.run

    def login(self, *args, **kwargs):
        return None

    def log(self, *args, **kwargs):
        return None


wandb = _DummyWandb()


def wandb_config(model_name, num_heads, hidden_size, batch_size, wandb_user_name):
    wandb.init(project="AFBAPred", entity=wandb_user_name)
    wandb.run.name = 'num_heads_' + str(num_heads) +'hidden_size_'+str(hidden_size)+'batchsize_'+str(batch_size)
    wandb.config = {
        "learning_rate": learning_rate,
        "weight_decay": weight_decay,
        "epochs": MAX_EPOCHS,
        "batch_size": batch_size,
    }


def build_model(model_name, n_channel, num_classes, image_size, window_size, num_heads, hidden_size):
    kernel_size_up_down = (1, 2, 2)
    if model_name == 'unet3d':
        return UNet(spatial_dims=3, in_channels=n_channel, out_channels=num_classes, channels=(64, 128, 256, 512, 1024), strides=(1, 2, 2))
    if model_name == 'attunet':
        return AttentionUnet(spatial_dims=3, in_channels=n_channel, out_channels=num_classes, channels=(64, 128, 256, 512, 1024), strides=(1, 2, 2))
    if model_name == 'unetr3d':
        patch_size = (1, 16, 16)
        return UNETR(in_channels=n_channel, out_channels=num_classes, img_size=image_size, spatial_dims=3, norm_name='batch',
                     feature_size=16, patch_size=patch_size, kernel_size_up_down=kernel_size_up_down,
                     hidden_size=384, mlp_dim=1536)
    if model_name == 'swinunetr3d':
        patch_size = (1, 2, 2)
        return SwinUNETR(
            image_size=image_size,
            patch_size=patch_size,
            window_size=window_size,
            in_channels=n_channel,
            out_channels=2,
            depths=(2, 2, 2, 2),
            num_heads=(num_heads, num_heads, num_heads, num_heads),
            feature_size=hidden_size,
            norm_name='batch',
            drop_rate=0.0,
            attn_drop_rate=0.0,
            drop_path_rate=0.0,
            attn_version='v1',
            normalize=True,
            use_checkpoint=False,
            spatial_dims=3
        )
    raise NotImplementedError(model_name)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(
        description='TS-SatFire prediction benchmark with legacy and paper-reproduction training profiles.'
    )
    parser.add_argument('-m', choices=sorted(PAPER_MODEL_CONFIGS), required=True, help='Spatial-temporal model')
    parser.add_argument('-mode', type=str, default='pred', help='Dataset filename prefix')
    parser.add_argument('-b', type=int, default=None, help='Batch size override')
    parser.add_argument('-r', type=int, default=0, help='Run identifier retained for compatibility')
    parser.add_argument('-lr', type=float, default=None, help='Learning rate override')
    parser.add_argument('-nh', type=int, default=2, help='SwinUNETR number of heads per stage')
    parser.add_argument('-ed', type=int, default=None, help='SwinUNETR feature size override')
    parser.add_argument('-nc', type=int, default=43, help='Number of preprocessed input channels')
    parser.add_argument('-ts', type=int, required=True, help='Time-series length in days')
    parser.add_argument('-it', type=int, default=1, help='Sampling interval')
    parser.add_argument('-test', dest='binary_flag', action='store_true', help='test latest configured checkpoint')
    parser.add_argument('-seed', type=int, default=42)
    parser.add_argument('-epochs', type=int, default=None, help='Maximum epochs; paper profile defaults to 200')
    parser.add_argument('--training-profile', choices=['legacy', 'paper'], default='legacy')
    parser.add_argument('--loss', choices=['auto', 'dice', 'dice_ce'], default='auto')
    parser.add_argument('--foreground-weight', type=float, default=446.7836)
    parser.add_argument('--activation', choices=['sigmoid', 'softmax'], default='sigmoid')
    parser.add_argument('--checkpoint-metric', choices=['val_loss', 'foreground_iou'], default='val_loss')
    parser.add_argument('--experiment-tag', default='viirs43')
    parser.add_argument(
        '--goes-variant',
        choices=['none', 'goes_spatial', 'goes_spatial_frontbuf'],
        default='none',
        help='Optionally concatenate six aligned GOES spatial channels to every paper model',
    )
    parser.add_argument(
        '--save-probability-maps',
        action='store_true',
        help='Save best-checkpoint validation/test foreground probability maps for candidate post-prior fusion',
    )
    parser.set_defaults(binary_flag=False)
    args = parser.parse_args()

    configure_seed(args.seed)

    model_name = args.m
    batch_size, lr, loss_name, hidden_size, MAX_EPOCHS = resolve_training_config(args)
    num_heads = args.nh
    ts_length = args.ts
    learning_rate = lr
    weight_decay = lr / 10
    num_classes = 2
    n_channel = args.nc
    interval = args.it
    mode = args.mode
    train = args.binary_flag
    test_after_train = True
    target_is_single_day = True
    use_goes_concat = args.goes_variant != 'none'
    model_input_channels = n_channel + (6 if use_goes_concat else 0)
    slug = run_slug(args, batch_size, lr, loss_name, hidden_size)

    root_path = os.environ.get("TS_SATFIRE_ORIG_DATASET_ROOT", "/home/jlc3q/data/SatFire/dataset/pred")
    roi_dir = os.environ.get("TS_SATFIRE_ORIG_ROI_DIR", "/home/jlc3q/New_project/TS-Agentic-AI/legacy/roi")
    checkpoint_dir = os.environ.get("TS_SATFIRE_ORIG_CHECKPOINT_ROOT", "/home/jlc3q/data/SatFire/checkpoints/original_ts_satfire")
    os.makedirs(checkpoint_dir, exist_ok=True)

    run_config = {
        "experiment_tag": args.experiment_tag,
        "training_profile": args.training_profile,
        "model": model_name,
        "mode": mode,
        "batch_size": batch_size,
        "learning_rate": lr,
        "loss": loss_name,
        "foreground_weight": args.foreground_weight if loss_name == "dice_ce" else None,
        "activation": args.activation,
        "checkpoint_metric": args.checkpoint_metric,
        "num_heads": num_heads,
        "feature_size": hidden_size,
        "input_channels": n_channel,
        "model_input_channels": model_input_channels,
        "goes_variant": args.goes_variant,
        "goes_fusion": "early_concat" if use_goes_concat else "none",
        "save_probability_maps": args.save_probability_maps,
        "time_series_days": ts_length,
        "interval": interval,
        "max_epochs": MAX_EPOCHS,
        "seed": args.seed,
        "dataset_root": root_path,
        "roi_dir": roi_dir,
        "checkpoint_dir": checkpoint_dir,
    }
    config_path = os.path.join(checkpoint_dir, f"config_{slug}.json")
    with open(config_path, "w") as handle:
        json.dump(run_config, handle, indent=2, sort_keys=True)
    print("Resolved experiment config:")
    print(json.dumps(run_config, indent=2, sort_keys=True))

    if not train:
        wandb_config(model_name, num_heads, hidden_size, batch_size, wandb_user_name="zhaoyutim")
        image_path = os.path.join(root_path, 'dataset_train/'+mode+'_train_img_seqtoseq_alll_'+str(ts_length)+'i_'+str(interval)+'.npy')
        label_path = os.path.join(root_path, 'dataset_train/'+mode+'_train_label_seqtoseq_alll_'+str(ts_length)+'i_'+str(interval)+'.npy')
        val_image_path = os.path.join(root_path, 'dataset_val/'+mode+'_val_img_seqtoseq_alll_'+str(ts_length)+'i_'+str(interval)+'.npy')
        val_label_path = os.path.join(root_path, 'dataset_val/'+mode+'_val_label_seqtoseq_alll_'+str(ts_length)+'i_'+str(interval)+'.npy')
        dataset_class = FireDatasetWithGOESSpatial if use_goes_concat else FireDataset
        train_kwargs = {}
        val_kwargs = {}
        if use_goes_concat:
            train_kwargs['goes_spatial_path'] = goes_train_path(
                root_path, 'train', ts_length, interval, args.goes_variant
            )
            val_kwargs['goes_spatial_path'] = goes_train_path(
                root_path, 'val', ts_length, interval, args.goes_variant
            )
        train_dataset = dataset_class(
            image_path=image_path,
            label_path=label_path,
            ts_length=ts_length,
            n_channel=n_channel,
            target_is_single_day=target_is_single_day,
            use_augmentations=True,
            **train_kwargs,
        )
        train_dataloader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
        val_dataset = dataset_class(
            image_path=val_image_path,
            label_path=val_label_path,
            ts_length=ts_length,
            n_channel=n_channel,
            target_is_single_day=target_is_single_day,
            use_augmentations=False,
            **val_kwargs,
        )
        val_dataloader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    image_size = (ts_length, 256, 256)
    window_size = (ts_length, 4, 4)

    model = build_model(
        model_name, model_input_channels, num_classes, image_size, window_size, num_heads, hidden_size
    )
    model = nn.DataParallel(model)
    model.to(device)

    criterion = build_criterion(loss_name, args.activation, args.foreground_weight, device)
    mean_iou = MeanIoU(include_background=True, reduction="mean", ignore_empty=False)
    dice_metric = DiceMetric(include_background=True, reduction="mean", ignore_empty=False)
    foreground_iou_metric = MeanIoU(include_background=False, reduction="mean", ignore_empty=False)
    foreground_dice_metric = DiceMetric(include_background=False, reduction="mean", ignore_empty=False)
    post_trans = Compose([
        Activations(sigmoid=args.activation == "sigmoid", softmax=args.activation == "softmax"),
        AsDiscrete(threshold=0.5),
    ])
    optimizer = optim.Adam(model.parameters(), lr=lr)
    scaler = GradScaler()
    best_checkpoints = []
    best_score = None
    history_rows = []

    if not train:
        for epoch in range(MAX_EPOCHS):
            model.train()
            train_loss = 0.0
            train_bar = tqdm(train_dataloader, total=len(train_dataloader))
            for i, batch in enumerate(train_bar):
                labels_batch = batch['labels'].to(torch.long).to(device)
                optimizer.zero_grad()
                outputs, current_batch_size = forward_model(model, batch, device, use_goes_concat)
                loss = criterion(outputs, criterion_target(labels_batch, loss_name))
                scaler.scale(loss).backward()
                scaler.step(optimizer)
                scaler.update()
                train_loss += loss.detach().item() * current_batch_size
                train_bar.set_description(
                    f"Epoch {epoch}/{MAX_EPOCHS}, Loss: {train_loss / min((i + 1) * batch_size, len(train_dataset)):.4f}"
                )
                if np.isnan(train_loss):
                    print(f"Loss is NaN, ending training at step {i}.")
                    raise SystemExit(1)

            train_loss /= len(train_dataset)
            wandb.log({'train_loss': train_loss})
            print(f"Epoch {epoch + 1}, Train Loss: {train_loss:.4f}")
            wandb.log({'epoch': epoch})

            model.eval()
            val_loss = 0.0
            iou_values = []
            dice_values = []
            foreground_iou_values = []
            foreground_dice_values = []
            val_bar = tqdm(val_dataloader, total=len(val_dataloader))
            for j, batch in enumerate(val_bar):
                val_labels_batch = batch['labels'].to(torch.long).to(device)
                with torch.no_grad():
                    outputs, current_batch_size = forward_model(model, batch, device, use_goes_concat)
                loss = criterion(outputs, criterion_target(val_labels_batch, loss_name))
                outputs = [post_trans(i) for i in decollate_batch(outputs)]
                val_labels_batch = decollate_batch(val_labels_batch)
                val_loss += loss.detach().item() * current_batch_size
                iou_values.append(mean_iou(outputs, val_labels_batch).mean().item())
                dice_values.append(dice_metric(y_pred=outputs, y=val_labels_batch).mean().item())
                foreground_iou_values.append(foreground_iou_metric(outputs, val_labels_batch).mean().item())
                foreground_dice_values.append(
                    foreground_dice_metric(y_pred=outputs, y=val_labels_batch).mean().item()
                )
                val_bar.set_description(
                    f"Epoch {epoch}/{MAX_EPOCHS}, Loss: {val_loss / min((j + 1) * batch_size, len(val_dataset)):.4f}"
                )

            val_loss /= len(val_dataset)
            mean_iou_val = np.mean(iou_values)
            mean_dice_val = np.mean(dice_values)
            foreground_iou_val = np.mean(foreground_iou_values)
            foreground_dice_val = np.mean(foreground_dice_values)
            epoch_row = {
                "epoch": epoch + 1,
                "train_loss": train_loss,
                "val_loss": val_loss,
                "val_mean_iou_with_background": mean_iou_val,
                "val_mean_dice_with_background": mean_dice_val,
                "val_foreground_iou": foreground_iou_val,
                "val_foreground_dice": foreground_dice_val,
            }
            history_rows.append(epoch_row)
            write_csv(os.path.join(checkpoint_dir, f"history_{slug}.csv"), history_rows)
            wandb.log(epoch_row)
            print(
                f"Epoch {epoch + 1}, Validation Loss: {val_loss:.4f}, "
                f"Mean IoU: {mean_iou_val:.4f}, Mean Dice: {mean_dice_val:.4f}, "
                f"Foreground IoU: {foreground_iou_val:.4f}, Foreground Dice: {foreground_dice_val:.4f}"
            )

            save_path = os.path.join(
                checkpoint_dir,
                f"model_{slug}_checkpoint_epoch_{epoch + 1}.pth",
            )
            score = val_loss if args.checkpoint_metric == "val_loss" else foreground_iou_val
            is_better = best_score is None or (
                score < best_score if args.checkpoint_metric == "val_loss" else score > best_score
            )
            if is_better:
                for _, previous_path in best_checkpoints:
                    if os.path.exists(previous_path):
                        os.remove(previous_path)
                torch.save({
                    'epoch': epoch,
                    'model_state_dict': model.state_dict(),
                    'optimizer_state_dict': optimizer.state_dict(),
                    'loss': loss,
                    'score': score,
                    'config': run_config,
                }, save_path)
                best_score = score
                best_checkpoints = [(score, save_path)]

        print("Top N best checkpoints:")
        for _, checkpoint in best_checkpoints:
            print(checkpoint)

    if train or test_after_train:
        df = pd.read_csv(os.path.join(roi_dir, 'us_fire_2021_out_new.csv'))
        ids = df['Id']
        ids = ids[~ids.isin(["US_2021_NV3700011641620210517"])].values.astype(str)

        if best_checkpoints:
            load_path = best_checkpoints[0][1]
        else:
            candidates = [
                os.path.join(checkpoint_dir, name)
                for name in os.listdir(checkpoint_dir)
                if name.startswith(f"model_{slug}_checkpoint_epoch_") and name.endswith(".pth")
            ]
            if not candidates:
                raise FileNotFoundError(f"No best-val checkpoint found in {checkpoint_dir}")
            load_path = max(candidates, key=os.path.getmtime)
        print(f"Loading best-val checkpoint: {load_path}")
        checkpoint = torch.load(load_path, map_location=device)
        model.load_state_dict(checkpoint['model_state_dict'])
        if 'optimizer_state_dict' in checkpoint:
            optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        model.eval()

        probability_dir = os.path.join(checkpoint_dir, f"probability_maps_{slug}")
        if args.save_probability_maps:
            if train:
                raise ValueError('--save-probability-maps requires a training run, not -test only')
            os.makedirs(probability_dir, exist_ok=True)
            val_probabilities = []
            with torch.no_grad():
                for batch in tqdm(val_dataloader, desc='Saving validation probabilities'):
                    logits, _ = forward_model(model, batch, device, use_goes_concat)
                    val_probabilities.append(
                        foreground_probability(logits, args.activation).cpu().numpy().astype(np.float16)
                    )
            val_probability_path = os.path.join(probability_dir, 'val.npy')
            np.save(val_probability_path, np.concatenate(val_probabilities, axis=0))
            print(f"Wrote {val_probability_path}")

        save_eval_plots = os.environ.get("TS_SATFIRE_ORIG_SAVE_PLOTS", "0") == "1"
        if save_eval_plots:
            import matplotlib.pyplot as plt
            import pathlib

        f1_all = 0
        iou_all = 0
        evaluated_ids = 0
        skipped_missing_ids = 0
        fire_metric_rows = []

        for i, id in enumerate(ids):
            test_image_path = os.path.join(root_path, f'dataset_test/{mode}_{id}_img_seqtoseql_{ts_length}i_{interval}.npy')
            test_label_path = os.path.join(root_path, f'dataset_test/{mode}_{id}_label_seqtoseql_{ts_length}i_{interval}.npy')
            if not (os.path.exists(test_image_path) and os.path.exists(test_label_path)):
                skipped_missing_ids += 1
                print(f"Skipping missing original TS-SatFire pred test arrays: {id}")
                continue

            test_kwargs = {}
            test_dataset_class = FireDatasetWithGOESSpatial if use_goes_concat else FireDataset
            if use_goes_concat:
                try:
                    test_kwargs['goes_spatial_path'] = goes_test_path(
                        root_path, id, ts_length, interval, args.goes_variant
                    )
                except FileNotFoundError as exc:
                    skipped_missing_ids += 1
                    print(f"Skipping missing GOES spatial test array: {id}. {exc}")
                    continue
            test_dataset = test_dataset_class(
                image_path=test_image_path,
                label_path=test_label_path,
                ts_length=ts_length,
                n_channel=n_channel,
                label_sel=0,
                target_is_single_day=True,
                use_augmentations=False,
                **test_kwargs,
            )
            test_dataloader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)

            def normalization(array):
                denom = array.max() - array.min()
                return (array-array.min()) / denom if denom != 0 else array

            f1 = 0
            iou = 0
            length = 0
            fire_probabilities = []
            for j, batch in enumerate(test_dataloader):
                test_data_batch = batch['data']
                test_labels_batch = batch['labels']
                with torch.no_grad():
                    logits, _ = forward_model(model, batch, device, use_goes_concat)
                if args.save_probability_maps:
                    fire_probabilities.append(
                        foreground_probability(logits, args.activation).cpu().numpy().astype(np.float16)
                    )
                outputs = logits
                outputs = [post_trans(item) for item in decollate_batch(outputs)]
                outputs = np.stack(outputs, axis=0)

                length += test_data_batch.shape[0]
                for k in range(test_data_batch.shape[0]):
                    output_stack = outputs[k, 1, ...]
                    label = (test_labels_batch[k, 1, ...] > 0).numpy()
                    f1 += f1_score(label.flatten(), output_stack.flatten(), zero_division=1.0)
                    iou += jaccard_score(label.flatten(), output_stack.flatten(), zero_division=1.0)

                    if save_eval_plots:
                        plt.imshow(normalization(test_data_batch[k, 3, -1, :]), cmap='gray')
                        img_tp = np.where(np.logical_and(output_stack == 1, label == 1), 1.0, 0.)
                        img_fp = np.where(np.logical_and(output_stack == 1, label == 0), 1.0, 0.)
                        img_fn = np.where(np.logical_and(output_stack == 0, label == 1), 1.0, 0.)
                        img_tp[img_tp == 0.] = np.nan
                        img_fp[img_fp == 0.] = np.nan
                        img_fn[img_fn == 0.] = np.nan
                        plt.imshow(img_tp, cmap='autumn', interpolation='nearest')
                        plt.imshow(img_fp, cmap='summer', interpolation='nearest')
                        plt.imshow(img_fn, cmap='brg', interpolation='nearest')
                        plt.axis('off')
                        plot_dir = 'evaluation_plot_original'
                        pathlib.Path(plot_dir).mkdir(parents=True, exist_ok=True)
                        plot_path = 'id_{}_nhead_{}_hidden_{}_nbatch_{}_nts_{}_ts_{}_nc_{}.png'.format(id, num_heads, hidden_size, j, k, i, n_channel)
                        image_path = os.path.join(plot_dir, plot_path)
                        plt.savefig(image_path, bbox_inches='tight')
                        plt.close()

            if length == 0:
                continue
            evaluated_ids += 1
            fire_iou = iou / length
            fire_f1 = f1 / length
            iou_all += fire_iou
            f1_all += fire_f1
            fire_metric_rows.append({
                "fire_id": id,
                "n_windows": length,
                "f1": fire_f1,
                "iou": fire_iou,
            })
            if args.save_probability_maps:
                fire_probability_path = os.path.join(probability_dir, f"test_{safe_tag(id)}.npy")
                np.save(fire_probability_path, np.concatenate(fire_probabilities, axis=0))
            print('ID{} IoU Score of the whole TS:{}'.format(id, fire_iou))
            print('ID{} F1 Score of the whole TS:{}'.format(id, fire_f1))

        if evaluated_ids == 0:
            raise SystemExit("No original TS-SatFire pred test samples were evaluated.")
        model_f1 = f1_all / evaluated_ids
        model_iou = iou_all / evaluated_ids
        print('model F1 Score: {} and iou score: {}'.format(model_f1, model_iou))
        print('evaluated_ids: {} skipped_missing_ids: {}'.format(evaluated_ids, skipped_missing_ids))
        wandb.log({"test_f1": model_f1, "test_iou": model_iou})
        write_csv(os.path.join(checkpoint_dir, f"test_fire_metrics_{slug}.csv"), fire_metric_rows)
        result = {
            **run_config,
            "checkpoint": load_path,
            "checkpoint_epoch": int(checkpoint.get("epoch", -1)) + 1,
            "test_f1": model_f1,
            "test_iou": model_iou,
            "evaluated_fires": evaluated_ids,
            "skipped_missing_fires": skipped_missing_ids,
        }
        result_path = os.path.join(checkpoint_dir, f"result_{slug}.json")
        with open(result_path, "w") as handle:
            json.dump(result, handle, indent=2, sort_keys=True)
        write_csv(os.path.join(checkpoint_dir, f"result_{slug}.csv"), [result])
        print(f"Wrote {result_path}")
