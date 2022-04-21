import math
from typing import Callable
from typing import Dict
from typing import Optional
from typing import Tuple
from pathlib import Path

import faiss
import numpy as np
import pandas as pd
import pytorch_lightning as pl
import timm
import torch
import torch.nn as nn
import torch.nn.functional as F
from PIL import Image
from pytorch_lightning.callbacks.model_checkpoint import ModelCheckpoint
from timm.data.transforms_factory import create_transform
from timm.optim import create_optimizer_v2
from torch.utils.data import DataLoader
from torch.utils.data import Dataset
from tqdm.notebook import tqdm
from sklearn.model_selection import StratifiedKFold
from sklearn.preprocessing import normalize
from sklearn.preprocessing import LabelEncoder

from model.model import LitDataModule, LitModule

INPUT_DIRECTORY = Path("..")
OUTPUT_DIRECTORY = Path("..") / "working"

DATA_ROOT_DIRECTORY = INPUT_DIRECTORY / "happy-whale-and-dolphin"
TRAIN_DIRECTORY = DATA_ROOT_DIRECTORY / "train_images"
TEST_DIRECTORY = DATA_ROOT_DIRECTORY / "test_images"
TRAIN_CSV_PATH = DATA_ROOT_DIRECTORY / "train.csv"
SAMPLE_SUBMISSION_CSV_PATH = DATA_ROOT_DIRECTORY / "sample_submission.csv"
PUBLIC_SUBMISSION_CSV_PATH = INPUT_DIRECTORY / "submission.csv"
IDS_WITHOUT_BACKFIN_PATH = INPUT_DIRECTORY / "without-backfin-ids" / "without_backfin_ids.npy"

N_SPLITS = 5

ENCODER_CLASSES_PATH = OUTPUT_DIRECTORY / "encoder_classes.npy"
TEST_CSV_PATH = OUTPUT_DIRECTORY / "test.csv"
TRAIN_CSV_ENCODED_FOLDED_PATH = OUTPUT_DIRECTORY / "trained_encoded_folded.csv"
CHECKPOINTS_DIRECTORY = OUTPUT_DIRECTORY / "checkpoints_dir"
SUBMISSION_CSV_PATH = OUTPUT_DIRECTORY / "submission.csv"

DEBUG = False

def get_image_path(id: str, dir: Path) -> str:
    return f"{dir / id}"

def load_eval_module(checkpoint_path: str, device: torch.device) -> LitModule:
    module = LitModule.load_from_checkpoint(checkpoint_path)
    module.to(device)
    module.eval()

    return module

def load_dataloaders(
    train_csv_encoded_folded,
    test_csv,
    val_fold,
    image_size,
    batch_size,
    num_workers,
) -> Tuple[DataLoader, DataLoader, DataLoader]:

    datamodule = LitDataModule(
        train_csv_encoded_folded=train_csv_encoded_folded,
        test_csv=test_csv,
        val_fold=val_fold,
        image_size=image_size,
        batch_size=batch_size,
        num_workers=num_workers,
    )

    datamodule.setup()

    train_dl = datamodule.train_dataloader()
    val_dl = datamodule.val_dataloader()
    test_dl = datamodule.test_dataloader()

    return train_dl, val_dl, test_dl


def load_encoder() -> LabelEncoder:
    encoder = LabelEncoder()
    encoder.classes_ = np.load(ENCODER_CLASSES_PATH, allow_pickle=True)

    return encoder

@torch.inference_mode()
def get_embeddings(
    module: pl.LightningModule, dataloader: DataLoader, encoder: LabelEncoder, stage: str
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:

    all_image_names = []
    all_embeddings = []
    all_targets = []

    for batch in tqdm(dataloader, desc=f"Creating {stage} embeddings"):
        image_names = batch["image_name"]
        images = batch["image"].to(module.device)
        targets = batch["target"].to(module.device)

        embeddings = module(images)

        all_image_names.append(image_names)
        all_embeddings.append(embeddings.cpu().numpy())
        all_targets.append(targets.cpu().numpy())
        
        if DEBUG:
            break

    all_image_names = np.concatenate(all_image_names)
    all_embeddings = np.vstack(all_embeddings)
    all_targets = np.concatenate(all_targets)

    all_embeddings = normalize(all_embeddings, axis=1, norm="l2")
    all_targets = encoder.inverse_transform(all_targets)

    return all_image_names, all_embeddings, all_targets

def create_and_search_index(embedding_size: int, train_embeddings: np.ndarray, val_embeddings: np.ndarray, k: int):
    index = faiss.IndexFlatIP(embedding_size)
    index.add(train_embeddings)
    D, I = index.search(val_embeddings, k=k)  # noqa: E741

    return D, I

def create_val_targets_df(
    train_targets: np.ndarray, val_image_names: np.ndarray, val_targets: np.ndarray
) -> pd.DataFrame:

    allowed_targets = np.unique(train_targets)
    val_targets_df = pd.DataFrame(np.stack([val_image_names, val_targets], axis=1), columns=["image", "target"])
    val_targets_df.loc[~val_targets_df.target.isin(allowed_targets), "target"] = "new_individual"

    return val_targets_df

def create_distances_df(
    image_names: np.ndarray, targets: np.ndarray, D: np.ndarray, I: np.ndarray, stage: str  # noqa: E741
) -> pd.DataFrame:

    distances_df = []
    for i, image_name in tqdm(enumerate(image_names), desc=f"Creating {stage}_df"):
        target = targets[I[i]]
        distances = D[i]
        subset_preds = pd.DataFrame(np.stack([target, distances], axis=1), columns=["target", "distances"])
        subset_preds["image"] = image_name
        distances_df.append(subset_preds)

    distances_df = pd.concat(distances_df).reset_index(drop=True)
    distances_df = distances_df.groupby(["image", "target"]).distances.max().reset_index()
    distances_df = distances_df.sort_values("distances", ascending=False).reset_index(drop=True)

    return distances_df

def get_predictions(df: pd.DataFrame, threshold: float = 0.2):
    sample_list = ["938b7e931166", "5bf17305f073", "7593d2aee842", "7362d7a01d00", "956562ff2888"]

    predictions = {}
    for i, row in tqdm(df.iterrows(), total=len(df), desc=f"Creating predictions for threshold={threshold}"):
        if row.image in predictions:
            if len(predictions[row.image]) == 5:
                continue
            predictions[row.image].append(row.target)
        elif row.distances > threshold:
            predictions[row.image] = [row.target, "new_individual"]
        else:
            predictions[row.image] = ["new_individual", row.target]

    for x in tqdm(predictions):
        if len(predictions[x]) < 5:
            remaining = [y for y in sample_list if y not in predictions]
            predictions[x] = predictions[x] + remaining
            predictions[x] = predictions[x][:5]

    return predictions

def map_per_image(label, predictions):
    """Computes the precision score of one image.

    Parameters
    ----------
    label : string
            The true label of the image
    predictions : list
            A list of predicted elements (order does matter, 5 predictions allowed per image)

    Returns
    -------
    score : double
    """
    try:
        return 1 / (predictions[:5].index(label) + 1)
    except ValueError:
        return 0.0

def get_best_threshold(val_targets_df: pd.DataFrame, valid_df: pd.DataFrame) -> Tuple[float, float]:
    best_th = 0
    best_cv = 0
    for th in [0.1 * x for x in range(11)]:
        all_preds = get_predictions(valid_df, threshold=th)

        cv = 0
        for i, row in val_targets_df.iterrows():
            target = row.target
            preds = all_preds[row.image]
            val_targets_df.loc[i, th] = map_per_image(target, preds)

        cv = val_targets_df[th].mean()

        print(f"th={th} cv={cv}")

        if cv > best_cv:
            best_th = th
            best_cv = cv

    print(f"best_th={best_th}")
    print(f"best_cv={best_cv}")

    # Adjustment: 10% 'new_individual' in public lb
    val_targets_df["is_new_individual"] = val_targets_df.target == "new_individual"
    val_scores = val_targets_df.groupby("is_new_individual").mean().T
    val_scores["adjusted_cv"] = val_scores[True] * 0.1 + val_scores[False] * 0.9
    best_th = val_scores["adjusted_cv"].idxmax()
    print(f"best_th_adjusted={best_th}")

    return best_th, best_cv

def create_predictions_df(test_df: pd.DataFrame, best_th: float) -> pd.DataFrame:
    predictions = get_predictions(test_df, best_th)

    predictions = pd.Series(predictions).reset_index()
    predictions.columns = ["image", "predictions"]
    predictions["predictions"] = predictions["predictions"].apply(lambda x: " ".join(x))

    return predictions

def calculate_top_x_accuracy(predictions, target, x=5):
    total = len(predictions)
    count = 0
    for index, row in enumerate(target):
        if target[index] in predictions[index][:x]:
            count += 1
    return count/total * 100.0

if '__name__' == '__main__':
    model_name = "convnext_base_384_in22ft1k"
    image_size = 384
    batch_size = 32

    checkpoint_path=CHECKPOINTS_DIRECTORY / f"{model_name}_{image_size}.ckpt"
    module = load_eval_module(checkpoint_path, torch.device("cuda"))

    train_dl, val_dl, test_dl = load_dataloaders(
        train_csv_encoded_folded=str(TRAIN_CSV_ENCODED_FOLDED_PATH),
        test_csv=str(TEST_CSV_PATH),
        val_fold=0.0,
        image_size=image_size,
        batch_size=32,
        num_workers=4,
    )

    encoder = load_encoder()

    train_image_names, train_embeddings, train_targets = get_embeddings(module, train_dl, encoder, stage="train")
    val_image_names, val_embeddings, val_targets = get_embeddings(module, val_dl, encoder, stage="val")
    test_image_names, test_embeddings, test_targets = get_embeddings(module, test_dl, encoder, stage="test")

    D, I = create_and_search_index(module.hparams.embedding_size, train_embeddings, val_embeddings, k=50)  # noqa: E741
    print("Created index with train_embeddings")

    val_df = create_distances_df(val_image_names, train_targets, D, I, "test")
    print(f"test_df=\n{val_df.head()}")

    best_th = 0.4
    predictions = create_predictions_df(val_df, best_th)
    print(f"predictions.head()={predictions.head()}")

    val_dataframe = pd.DataFrame({'image':val_image_names, 'target':val_targets})
    result = pd.merge(predictions, val_dataframe, on="image")
    #result.to_csv("../Results/ConvNext_ArcFace/Val_Results.csv", index=False)

    acc = calculate_top_x_accuracy(result.predictions.str.split(" "), result.target, x=5)
