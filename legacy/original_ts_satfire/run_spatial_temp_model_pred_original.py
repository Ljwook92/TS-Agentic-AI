import argparse
import heapq
import os

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

from original_ts_satfire.data_generator_pred_torch_original import FireDataset
from spatial_models.attentionunet import AttentionUnet
from spatial_models.swinunetr.swinunetr import SwinUNETR
from spatial_models.unet import UNet
from spatial_models.unetr.unetr import UNETR


SEED = 42
np.random.seed(SEED)
torch.manual_seed(SEED)


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
    parser = argparse.ArgumentParser(description='Original TS-SatFire pred runner with HPC paths and missing-file skip.')
    parser.add_argument('-m', type=str, help='Model to be executed')
    parser.add_argument('-mode', type=str, help='BA or Pred')
    parser.add_argument('-b', type=int, help='batch size')
    parser.add_argument('-r', type=int, help='run')
    parser.add_argument('-lr', type=float, help='learning rate')
    parser.add_argument('-nh', type=int, help='number-of-head')
    parser.add_argument('-ed', type=int, help='embedding dimension')
    parser.add_argument('-nc', type=int, help='n_channel')
    parser.add_argument('-ts', type=int, help='ts_length')
    parser.add_argument('-it', type=int, help='interval')
    parser.add_argument('-test', dest='binary_flag', action='store_true', help='test latest configured checkpoint')
    parser.add_argument('-seed', type=int, default=42)
    parser.add_argument('-epochs', type=int, default=100)
    parser.set_defaults(binary_flag=False)
    args = parser.parse_args()

    np.random.seed(args.seed)
    torch.manual_seed(args.seed)

    model_name = args.m
    batch_size = args.b
    num_heads = args.nh
    hidden_size = args.ed
    ts_length = args.ts
    lr = args.lr
    MAX_EPOCHS = args.epochs
    learning_rate = lr
    weight_decay = lr / 10
    num_classes = 2
    n_channel = args.nc
    interval = args.it
    mode = args.mode
    top_n_checkpoints = 1
    train = args.binary_flag
    test_after_train = True
    target_is_single_day = True

    root_path = os.environ.get("TS_SATFIRE_ORIG_DATASET_ROOT", "/home/jlc3q/data/SatFire/dataset/pred")
    roi_dir = os.environ.get("TS_SATFIRE_ORIG_ROI_DIR", "/home/jlc3q/New_project/TS-Agentic-AI/legacy/roi")
    checkpoint_dir = os.environ.get("TS_SATFIRE_ORIG_CHECKPOINT_ROOT", "/home/jlc3q/data/SatFire/checkpoints/original_ts_satfire")
    os.makedirs(checkpoint_dir, exist_ok=True)

    if not train:
        wandb_config(model_name, num_heads, hidden_size, batch_size, wandb_user_name="zhaoyutim")
        image_path = os.path.join(root_path, 'dataset_train/'+mode+'_train_img_seqtoseq_alll_'+str(ts_length)+'i_'+str(interval)+'.npy')
        label_path = os.path.join(root_path, 'dataset_train/'+mode+'_train_label_seqtoseq_alll_'+str(ts_length)+'i_'+str(interval)+'.npy')
        val_image_path = os.path.join(root_path, 'dataset_val/'+mode+'_val_img_seqtoseq_alll_'+str(ts_length)+'i_'+str(interval)+'.npy')
        val_label_path = os.path.join(root_path, 'dataset_val/'+mode+'_val_label_seqtoseq_alll_'+str(ts_length)+'i_'+str(interval)+'.npy')
        train_dataset = FireDataset(image_path=image_path, label_path=label_path, ts_length=ts_length, n_channel=n_channel, target_is_single_day=target_is_single_day, use_augmentations=True)
        train_dataloader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
        val_dataset = FireDataset(image_path=val_image_path, label_path=val_label_path, ts_length=ts_length, n_channel=n_channel, target_is_single_day=target_is_single_day, use_augmentations=False)
        val_dataloader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    image_size = (ts_length, 256, 256)
    window_size = (ts_length, 4, 4)

    model = build_model(model_name, n_channel, num_classes, image_size, window_size, num_heads, hidden_size)
    model = nn.DataParallel(model)
    model.to(device)

    criterion = DiceLoss(include_background=True, reduction='mean', sigmoid=True)
    mean_iou = MeanIoU(include_background=True, reduction="mean", ignore_empty=False)
    dice_metric = DiceMetric(include_background=True, reduction="mean", ignore_empty=False)
    post_trans = Compose([Activations(sigmoid=True), AsDiscrete(threshold=0.5)])
    optimizer = optim.Adam(model.parameters(), lr=lr)
    scaler = GradScaler()
    best_checkpoints = []

    if not train:
        for epoch in range(MAX_EPOCHS):
            model.train()
            train_loss = 0.0
            train_bar = tqdm(train_dataloader, total=len(train_dataloader))
            for i, batch in enumerate(train_bar):
                data_batch = batch['data'].to(device)
                labels_batch = batch['labels'].to(torch.long).to(device)
                optimizer.zero_grad()
                outputs = model(data_batch).mean(2)
                loss = criterion(outputs, labels_batch)
                scaler.scale(loss).backward()
                scaler.step(optimizer)
                scaler.update()
                train_loss += loss.detach().item() * data_batch.size(0)
                train_bar.set_description(f"Epoch {epoch}/{MAX_EPOCHS}, Loss: {train_loss/((i+1)* data_batch.size(0)):.4f}")
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
            val_bar = tqdm(val_dataloader, total=len(val_dataloader))
            for j, batch in enumerate(val_bar):
                val_data_batch = batch['data'].to(device)
                val_labels_batch = batch['labels'].to(torch.long).to(device)
                with torch.no_grad():
                    outputs = model(val_data_batch).mean(2)
                loss = criterion(outputs, val_labels_batch)
                outputs = [post_trans(i) for i in decollate_batch(outputs)]
                val_labels_batch = decollate_batch(val_labels_batch)
                val_loss += loss.detach().item() * val_data_batch.size(0)
                iou_values.append(mean_iou(outputs, val_labels_batch).mean().item())
                dice_values.append(dice_metric(y_pred=outputs, y=val_labels_batch).mean().item())
                val_bar.set_description(
                    f"Epoch {epoch}/{MAX_EPOCHS}, Loss: {val_loss / ((j + 1) * val_data_batch.size(0)):.4f}")

            val_loss /= len(val_dataset)
            mean_iou_val = np.mean(iou_values)
            mean_dice_val = np.mean(dice_values)
            wandb.log({'val_loss': val_loss, 'miou': mean_iou_val, 'mdice': mean_dice_val})
            print(f"Epoch {epoch + 1}, Validation Loss: {val_loss:.4f}, Mean IoU: {mean_iou_val:.4f}, Mean Dice: {mean_dice_val:.4f}")

            save_path = os.path.join(
                checkpoint_dir,
                f"model_{model_name}_mode_{mode}_num_heads_{num_heads}_hidden_size_{hidden_size}_batchsize_{batch_size}_checkpoint_epoch_{epoch + 1}_nc_{n_channel}_ts_{ts_length}.pth",
            )
            if len(best_checkpoints) < top_n_checkpoints or val_loss < best_checkpoints[0][0]:
                if len(best_checkpoints) == top_n_checkpoints:
                    _, remove_checkpoint = heapq.heappop(best_checkpoints)
                    if os.path.exists(remove_checkpoint):
                        os.remove(remove_checkpoint)
                torch.save({
                    'epoch': epoch,
                    'model_state_dict': model.state_dict(),
                    'optimizer_state_dict': optimizer.state_dict(),
                    'loss': loss,
                }, save_path)
                heapq.heappush(best_checkpoints, (val_loss, save_path))
                best_checkpoints = heapq.nlargest(top_n_checkpoints, best_checkpoints)

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
                if name.startswith(
                    f"model_{model_name}_mode_{mode}_num_heads_{num_heads}_hidden_size_{hidden_size}_batchsize_{batch_size}_checkpoint_epoch_"
                )
                and name.endswith(f"_nc_{n_channel}_ts_{ts_length}.pth")
            ]
            if not candidates:
                raise FileNotFoundError(f"No best-val checkpoint found in {checkpoint_dir}")
            load_path = max(candidates, key=os.path.getmtime)
        print(f"Loading best-val checkpoint: {load_path}")
        checkpoint = torch.load(load_path)
        model.load_state_dict(checkpoint['model_state_dict'])
        optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        model.eval()

        save_eval_plots = os.environ.get("TS_SATFIRE_ORIG_SAVE_PLOTS", "0") == "1"
        if save_eval_plots:
            import matplotlib.pyplot as plt
            import pathlib

        f1_all = 0
        iou_all = 0
        evaluated_ids = 0
        skipped_missing_ids = 0

        for i, id in enumerate(ids):
            test_image_path = os.path.join(root_path, f'dataset_test/{mode}_{id}_img_seqtoseql_{ts_length}i_1.npy')
            test_label_path = os.path.join(root_path, f'dataset_test/{mode}_{id}_label_seqtoseql_{ts_length}i_1.npy')
            if not (os.path.exists(test_image_path) and os.path.exists(test_label_path)):
                skipped_missing_ids += 1
                print(f"Skipping missing original TS-SatFire pred test arrays: {id}")
                continue

            test_dataset = FireDataset(image_path=test_image_path, label_path=test_label_path, ts_length=ts_length, n_channel=n_channel, label_sel=0, target_is_single_day=True, use_augmentations=False)
            test_dataloader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)

            def normalization(array):
                denom = array.max() - array.min()
                return (array-array.min()) / denom if denom != 0 else array

            f1 = 0
            iou = 0
            length = 0
            for j, batch in enumerate(test_dataloader):
                test_data_batch = batch['data']
                test_labels_batch = batch['labels']
                with torch.no_grad():
                    outputs = model(test_data_batch.to(device)).mean(2)
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
            iou_all += iou / length
            f1_all += f1 / length
            print('ID{} IoU Score of the whole TS:{}'.format(id, iou / length))
            print('ID{} F1 Score of the whole TS:{}'.format(id, f1 / length))

        if evaluated_ids == 0:
            raise SystemExit("No original TS-SatFire pred test samples were evaluated.")
        print('model F1 Score: {} and iou score: {}'.format(f1_all / evaluated_ids, iou_all / evaluated_ids))
        print('evaluated_ids: {} skipped_missing_ids: {}'.format(evaluated_ids, skipped_missing_ids))
        wandb.log({"test_f1": f1_all / evaluated_ids, "test_iou": iou_all / evaluated_ids})
