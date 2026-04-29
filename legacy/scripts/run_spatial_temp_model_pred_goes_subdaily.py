import argparse
import glob
import heapq
import os
import pathlib

import numpy as np
import pandas as pd
import torch
from monai.data import decollate_batch
from monai.losses.dice import DiceLoss
from monai.metrics import DiceMetric, MeanIoU
from monai.transforms import Activations, AsDiscrete, Compose
from sklearn.metrics import f1_score, jaccard_score
from torch import nn, optim
from torch.cuda.amp import GradScaler
from torch.utils.data import DataLoader
from tqdm import tqdm

from satimg_dataset_processor.data_generator_pred_goes_subdaily_torch import FireDatasetWithGOESSubdaily
from spatial_models.swinunetr.swinunetr_goes_fusion import SwinUNETRGOESFusion
from support.path_config import get_checkpoints_root, get_code_root, get_task_dataset_root


def resolve_checkpoint_path(
    checkpoint_dir,
    model_name,
    mode,
    num_heads,
    hidden_size,
    batch_size,
    n_channel,
    ts_length,
    requested_epoch=0,
):
    if requested_epoch and requested_epoch > 0:
        return os.path.join(
            checkpoint_dir,
            f"model_{model_name}_mode_{mode}_num_heads_{num_heads}_hidden_size_{hidden_size}_batchsize_{batch_size}_checkpoint_epoch_{requested_epoch}_nc_{n_channel}_ts_{ts_length}.pth",
        )

    pattern = os.path.join(
        checkpoint_dir,
        f"model_{model_name}_mode_{mode}_num_heads_{num_heads}_hidden_size_{hidden_size}_batchsize_{batch_size}_checkpoint_epoch_*_nc_{n_channel}_ts_{ts_length}.pth",
    )
    candidates = glob.glob(pattern)
    if not candidates:
        raise FileNotFoundError(f"No checkpoints matched pattern: {pattern}")

    def extract_epoch(path: str) -> int:
        fragment = os.path.basename(path).split("_checkpoint_epoch_")[-1].split("_nc_")[0]
        return int(fragment)

    return max(candidates, key=extract_epoch)


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


def goes_train_path(root_path: str, mode: str, ts_length: int, interval: int) -> str:
    candidates = [
        os.path.join(root_path, f"dataset_{mode}", f"pred_{mode}_goes_subdaily_seqtoseq_alll_{ts_length}i_{interval}.npy"),
        os.path.join(root_path, f"GOES_{mode}", f"pred_{mode}_goes_subdaily_seqtoseq_alll_{ts_length}i_{interval}.npy"),
    ]
    for path in candidates:
        if os.path.exists(path):
            return path
    raise FileNotFoundError(f"No GOES sub-daily {mode} file found. Tried: {candidates}")


def goes_test_path(root_path: str, fire_id: str, ts_length: int, interval: int) -> str:
    filename = f"pred_{fire_id}_goes_subdaily_seqtoseql_{ts_length}i_{interval}.npy"
    candidates = [
        os.path.join(root_path, "dataset_test", filename),
        os.path.join(root_path, "GOES_test", filename),
    ]
    for path in candidates:
        if os.path.exists(path):
            return path
    raise FileNotFoundError(f"No GOES sub-daily test file found for {fire_id}. Tried: {candidates}")


def build_model(model_name, image_size, window_size, n_channel, num_classes, num_heads, hidden_size, single_gpu_mode):
    if model_name != "swinunetr3d_goes_subdaily":
        raise ValueError("Only -m swinunetr3d_goes_subdaily is implemented in this script.")

    patch_size = (1, 2, 2)
    feature_size = 24 if single_gpu_mode else hidden_size
    effective_heads = 2 if single_gpu_mode else num_heads
    return SwinUNETRGOESFusion(
        image_size=image_size,
        patch_size=patch_size,
        window_size=window_size,
        in_channels=n_channel,
        out_channels=num_classes,
        depths=(2, 2, 2, 2),
        num_heads=(effective_heads, effective_heads, effective_heads, effective_heads),
        feature_size=feature_size,
        norm_name="batch",
        drop_rate=0.0,
        attn_drop_rate=0.0,
        drop_path_rate=0.0,
        attn_version="v1",
        normalize=True,
        use_checkpoint=False,
        spatial_dims=3,
        goes_in_features=4,
        goes_hidden_size=128,
    )


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Train pred SwinUNETR-3D with GOES sub-daily temporal fusion.")
    parser.add_argument("-m", type=str, required=True, help="Model name. Use swinunetr3d_goes_subdaily.")
    parser.add_argument("-mode", type=str, required=True, help="Task mode. Use pred.")
    parser.add_argument("-b", type=int, required=True, help="Batch size")
    parser.add_argument("-r", type=int, default=0, help="Run index")
    parser.add_argument("-lr", type=float, required=True, help="Learning rate")
    parser.add_argument("-nh", type=int, required=True, help="SwinUNETR num heads")
    parser.add_argument("-ed", type=int, required=True, help="SwinUNETR feature size")
    parser.add_argument("-nc", type=int, required=True, help="Input channels after preprocessing")
    parser.add_argument("-ts", type=int, required=True, help="Time-series length")
    parser.add_argument("-it", type=int, required=True, help="Interval")
    parser.add_argument("-test", dest="binary_flag", action="store_true", help="Skip training and test latest checkpoint")
    parser.add_argument("-seed", type=int, default=42)
    parser.add_argument("-epochs", type=int, default=100)
    args = parser.parse_args()

    np.random.seed(args.seed)
    torch.manual_seed(args.seed)

    model_name = args.m
    mode = args.mode
    batch_size = args.b
    num_heads = args.nh
    hidden_size = args.ed
    ts_length = args.ts
    interval = args.it
    n_channel = args.nc
    learning_rate = args.lr
    weight_decay = args.lr / 10
    max_epochs = args.epochs
    train = args.binary_flag
    test_after_train = True
    target_is_single_day = True
    num_classes = 2
    top_n_checkpoints = 1

    root_path = str(get_task_dataset_root("pred"))
    checkpoint_dir = str(get_checkpoints_root())
    roi_dir = get_code_root() / "legacy" / "roi"
    os.makedirs(checkpoint_dir, exist_ok=True)

    if not train:
        train_image_path = os.path.join(root_path, "dataset_train", f"{mode}_train_img_seqtoseq_alll_{ts_length}i_{interval}.npy")
        train_label_path = os.path.join(root_path, "dataset_train", f"{mode}_train_label_seqtoseq_alll_{ts_length}i_{interval}.npy")
        train_goes_path = goes_train_path(root_path, "train", ts_length, interval)
        val_image_path = os.path.join(root_path, "dataset_val", f"{mode}_val_img_seqtoseq_alll_{ts_length}i_{interval}.npy")
        val_label_path = os.path.join(root_path, "dataset_val", f"{mode}_val_label_seqtoseq_alll_{ts_length}i_{interval}.npy")
        val_goes_path = goes_train_path(root_path, "val", ts_length, interval)

        train_dataset = FireDatasetWithGOESSubdaily(
            image_path=train_image_path,
            label_path=train_label_path,
            goes_subdaily_path=train_goes_path,
            ts_length=ts_length,
            n_channel=n_channel,
            target_is_single_day=target_is_single_day,
            use_augmentations=True,
        )
        val_dataset = FireDatasetWithGOESSubdaily(
            image_path=val_image_path,
            label_path=val_label_path,
            goes_subdaily_path=val_goes_path,
            ts_length=ts_length,
            n_channel=n_channel,
            target_is_single_day=target_is_single_day,
            use_augmentations=False,
        )
        train_dataloader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
        val_dataloader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    single_gpu_mode = torch.cuda.is_available() and torch.cuda.device_count() <= 1
    image_size = (ts_length, 256, 256)
    window_size = (ts_length, 4, 4)

    model = build_model(
        model_name=model_name,
        image_size=image_size,
        window_size=window_size,
        n_channel=n_channel,
        num_classes=num_classes,
        num_heads=num_heads,
        hidden_size=hidden_size,
        single_gpu_mode=single_gpu_mode,
    )
    if torch.cuda.device_count() > 1:
        model = nn.DataParallel(model)
    model.to(device)

    criterion = DiceLoss(include_background=True, reduction="mean", sigmoid=True)
    mean_iou = MeanIoU(include_background=True, reduction="mean", ignore_empty=False)
    dice_metric = DiceMetric(include_background=True, reduction="mean", ignore_empty=False)
    post_trans = Compose([Activations(sigmoid=True), AsDiscrete(threshold=0.5)])
    optimizer = optim.Adam(model.parameters(), lr=learning_rate, weight_decay=weight_decay)
    scaler = GradScaler()
    best_checkpoints = []

    if not train:
        for epoch in range(max_epochs):
            model.train()
            train_loss = 0.0
            train_bar = tqdm(train_dataloader, total=len(train_dataloader))
            for i, batch in enumerate(train_bar):
                data_batch = batch["data"].to(device)
                goes_batch = batch["goes_subdaily"].to(device)
                labels_batch = batch["labels"].to(torch.long).to(device)

                optimizer.zero_grad()
                outputs = model(data_batch, goes_batch).mean(2)
                loss = criterion(outputs, labels_batch)
                scaler.scale(loss).backward()
                scaler.step(optimizer)
                scaler.update()

                train_loss += loss.detach().item() * data_batch.size(0)
                train_bar.set_description(f"Epoch {epoch}/{max_epochs}, Loss: {train_loss / ((i + 1) * data_batch.size(0)):.4f}")
                if np.isnan(train_loss):
                    raise RuntimeError(f"Loss is NaN at step {i}.")

            train_loss /= len(train_dataset)
            wandb.log({"train_loss": train_loss, "epoch": epoch})
            print(f"Epoch {epoch + 1}, Train Loss: {train_loss:.4f}")

            model.eval()
            val_loss = 0.0
            iou_values = []
            dice_values = []
            val_bar = tqdm(val_dataloader, total=len(val_dataloader))
            for j, batch in enumerate(val_bar):
                val_data_batch = batch["data"].to(device)
                val_goes_batch = batch["goes_subdaily"].to(device)
                val_labels_batch = batch["labels"].to(torch.long).to(device)
                with torch.no_grad():
                    outputs = model(val_data_batch, val_goes_batch).mean(2)
                loss = criterion(outputs, val_labels_batch)
                outputs_post = [post_trans(item) for item in decollate_batch(outputs)]
                labels_post = decollate_batch(val_labels_batch)
                val_loss += loss.detach().item() * val_data_batch.size(0)
                iou_values.append(mean_iou(outputs_post, labels_post).mean().item())
                dice_values.append(dice_metric(y_pred=outputs_post, y=labels_post).mean().item())
                val_bar.set_description(f"Epoch {epoch}/{max_epochs}, Loss: {val_loss / ((j + 1) * val_data_batch.size(0)):.4f}")

            val_loss /= len(val_dataset)
            mean_iou_val = np.mean(iou_values)
            mean_dice_val = np.mean(dice_values)
            wandb.log({"val_loss": val_loss, "miou": mean_iou_val, "mdice": mean_dice_val})
            print(
                f"Epoch {epoch + 1}, Validation Loss: {val_loss:.4f}, "
                f"Validation IoU: {mean_iou_val:.4f}, Validation Dice: {mean_dice_val:.4f}"
            )

            save_path = os.path.join(
                checkpoint_dir,
                f"model_{model_name}_mode_{mode}_num_heads_{num_heads}_hidden_size_{hidden_size}_batchsize_{batch_size}_checkpoint_epoch_{epoch + 1}_nc_{n_channel}_ts_{ts_length}.pth",
            )
            if len(best_checkpoints) < top_n_checkpoints or val_loss < best_checkpoints[0][0]:
                if len(best_checkpoints) == top_n_checkpoints:
                    _, remove_checkpoint = heapq.heappop(best_checkpoints)
                    if os.path.exists(remove_checkpoint):
                        os.remove(remove_checkpoint)
                torch.save(
                    {
                        "epoch": epoch,
                        "model_state_dict": model.state_dict(),
                        "optimizer_state_dict": optimizer.state_dict(),
                        "loss": loss,
                    },
                    save_path,
                )
                heapq.heappush(best_checkpoints, (val_loss, save_path))
                best_checkpoints = heapq.nlargest(top_n_checkpoints, best_checkpoints)
            print("Top N best checkpoints:")
            for _, checkpoint in best_checkpoints:
                print(checkpoint)

    if train or test_after_train:
        df = pd.read_csv(str(roi_dir / "us_fire_2021_out_new.csv"))
        ids = df["Id"]
        ids = ids[~ids.isin(["US_2021_NV3700011641620210517"])].values.astype(str)

        load_path = resolve_checkpoint_path(
            checkpoint_dir,
            model_name,
            mode,
            num_heads,
            hidden_size,
            batch_size,
            n_channel,
            ts_length,
            requested_epoch=0,
        )
        checkpoint = torch.load(load_path)
        model.load_state_dict(checkpoint["model_state_dict"])
        optimizer.load_state_dict(checkpoint["optimizer_state_dict"])
        model.eval()

        save_eval_plots = os.environ.get("TS_SATFIRE_SAVE_PRED_EVAL_PLOTS", "0") == "1"
        if save_eval_plots:
            import matplotlib.pyplot as plt

        f1_all = 0
        iou_all = 0
        evaluated_ids = 0
        skipped_empty_label_ids = 0
        total_pred_positive_pixels = 0
        total_label_positive_pixels = 0
        total_frames_evaluated = 0

        for fire_id in ids:
            test_image_path = os.path.join(root_path, "dataset_test", f"{mode}_{fire_id}_img_seqtoseql_{ts_length}i_{interval}.npy")
            test_label_path = os.path.join(root_path, "dataset_test", f"{mode}_{fire_id}_label_seqtoseql_{ts_length}i_{interval}.npy")
            if not (os.path.exists(test_image_path) and os.path.exists(test_label_path)):
                print(f"Skipping prediction test sample because prepared test arrays are missing: {fire_id}")
                continue
            try:
                test_goes_path = goes_test_path(root_path, fire_id, ts_length, interval)
            except FileNotFoundError as exc:
                print(f"Skipping prediction test sample because GOES array is missing: {fire_id}. {exc}")
                continue

            test_dataset = FireDatasetWithGOESSubdaily(
                image_path=test_image_path,
                label_path=test_label_path,
                goes_subdaily_path=test_goes_path,
                ts_length=ts_length,
                n_channel=n_channel,
                label_sel=0,
                target_is_single_day=True,
                use_augmentations=False,
            )
            test_dataloader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)

            def normalization(array):
                denom = array.max() - array.min()
                return (array - array.min()) / denom if denom != 0 else array

            f1 = 0
            iou = 0
            length = 0
            pred_positive_pixels = 0
            label_positive_pixels = 0
            zero_prediction_frames = 0
            for j, batch in enumerate(test_dataloader):
                test_data_batch = batch["data"]
                test_goes_batch = batch["goes_subdaily"]
                test_labels_batch = batch["labels"]
                with torch.no_grad():
                    outputs = model(test_data_batch.to(device), test_goes_batch.to(device)).mean(2)
                outputs = [post_trans(item) for item in decollate_batch(outputs)]
                outputs = np.stack(outputs, axis=0)

                length += test_data_batch.shape[0]
                for k in range(test_data_batch.shape[0]):
                    output_stack = outputs[k, 1, ...]
                    label = (test_labels_batch[k, 1, ...] > 0).numpy()
                    pred_positive_pixels += int(output_stack.sum())
                    label_positive_pixels += int(label.sum())
                    if output_stack.sum() == 0:
                        zero_prediction_frames += 1
                    f1 += f1_score(label.flatten(), output_stack.flatten(), zero_division=0.0)
                    iou += jaccard_score(label.flatten(), output_stack.flatten(), zero_division=0.0)

                    if save_eval_plots:
                        plt.imshow(normalization(test_data_batch[k, 3, -1, :]), cmap="gray")
                        pathlib.Path("evaluation_plot").mkdir(parents=True, exist_ok=True)
                        plt.savefig(
                            os.path.join(
                                "evaluation_plot",
                                f"goes_id_{fire_id}_batch_{j}_sample_{k}_nc_{n_channel}_ts_{ts_length}.png",
                            ),
                            bbox_inches="tight",
                        )
                        plt.close()

            if length == 0:
                continue
            if label_positive_pixels == 0:
                skipped_empty_label_ids += 1
                print(
                    f"ID{fire_id} skipped from aggregate metrics because label_positive_pixels=0 "
                    "(likely no valid growth target for this fire)."
                )
                print(
                    f"ID{fire_id} Diagnostics: pred_positive_pixels={pred_positive_pixels}, "
                    f"label_positive_pixels={label_positive_pixels}, zero_prediction_frames={zero_prediction_frames}/{length}"
                )
                continue

            evaluated_ids += 1
            total_pred_positive_pixels += pred_positive_pixels
            total_label_positive_pixels += label_positive_pixels
            total_frames_evaluated += length
            iou_all += iou / length
            f1_all += f1 / length
            print(f"ID{fire_id} Test IoU Score of the whole TS:{iou / length}")
            print(f"ID{fire_id} Test F1 Score of the whole TS:{f1 / length}")
            print(
                f"ID{fire_id} Diagnostics: pred_positive_pixels={pred_positive_pixels}, "
                f"label_positive_pixels={label_positive_pixels}, zero_prediction_frames={zero_prediction_frames}/{length}"
            )

        if evaluated_ids == 0:
            raise SystemExit("No prediction test samples with non-empty labels were evaluated.")

        mean_test_f1 = f1_all / evaluated_ids
        mean_test_iou = iou_all / evaluated_ids
        print(f"Model Test F1 Score: {mean_test_f1} and Test IoU Score: {mean_test_iou}")
        print(
            "Prediction diagnostics summary: total_pred_positive_pixels={}, "
            "total_label_positive_pixels={}, total_frames_evaluated={}, "
            "skipped_empty_label_ids={}".format(
                total_pred_positive_pixels,
                total_label_positive_pixels,
                total_frames_evaluated,
                skipped_empty_label_ids,
            )
        )
        wandb.log({"test_f1": mean_test_f1, "test_iou": mean_test_iou})
