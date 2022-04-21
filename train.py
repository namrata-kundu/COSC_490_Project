import math
from typing import Callable
from typing import Dict
from typing import Optional
from typing import Tuple
from pathlib import Path

import numpy as np
import pandas as pd
from tqdm.notebook import tqdm
from sklearn.model_selection import StratifiedKFold
from sklearn.preprocessing import normalize
from sklearn.preprocessing import LabelEncoder

from model.model import *

INPUT_DIRECTORY = Path("..")
OUTPUT_DIRECTORY = Path("..") / "working"

DATA_ROOT_DIRECTORY = INPUT_DIRECTORY / "happy-whale-and-dolphin"
TRAIN_DIRECTORY = DATA_ROOT_DIRECTORY / "training_images"
TEST_DIRECTORY = DATA_ROOT_DIRECTORY / "testing_images"
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

def train(
    train_csv_encoded_folded = str(TRAIN_CSV_ENCODED_FOLDED_PATH),
    test_csv = str(TEST_CSV_PATH),
    val_fold = 0.0,
    image_size = 256,
    batch_size = 64,
    num_workers = 4,
    model_name = "convnext_base_384_in22ft1k",
    pretrained = True,
    drop_rate = 0.0,
    embedding_size = 512,
    num_classes = 15587,
    arc_s = 30.0,
    arc_m = 0.5,
    arc_easy_margin = False,
    arc_ls_eps = 0.0,
    optimizer = "adam",
    learning_rate = 3e-4,
    weight_decay = 1e-6,
    CHECKPOINTS_DIRECTORY = str(CHECKPOINTS_DIRECTORY),
    accumulate_grad_batches = 1,
    auto_lr_find = False,
    auto_scale_batch_size = False,
    fast_dev_run = False,
    gpus = 1,
    max_epochs = 10,
    precision = 16,
    stochastic_weight_avg = True,
):
    pl.seed_everything(42)

    datamodule = LitDataModule(
        train_csv_encoded_folded=train_csv_encoded_folded,
        test_csv=test_csv,
        val_fold=val_fold,
        image_size=image_size,
        batch_size=batch_size,
        num_workers=num_workers,
    )
    
    datamodule.setup()
    len_train_dl = len(datamodule.train_dataloader())

    module = LitModule(
        model_name=model_name,
        pretrained=pretrained,
        drop_rate=drop_rate,
        embedding_size=embedding_size,
        num_classes=num_classes,
        arc_s=arc_s,
        arc_m=arc_m,
        arc_easy_margin=arc_easy_margin,
        arc_ls_eps=arc_ls_eps,
        optimizer=optimizer,
        learning_rate=learning_rate,
        weight_decay=weight_decay,
        len_train_dl=len_train_dl,
        epochs=max_epochs
    )
    
    model_checkpoint = ModelCheckpoint(
        CHECKPOINTS_DIRECTORY,
        filename=f"{model_name}_{image_size}",
        monitor="val_loss",
    )
        
    trainer = pl.Trainer(
        accumulate_grad_batches=accumulate_grad_batches,
        auto_lr_find=auto_lr_find,
        auto_scale_batch_size=auto_scale_batch_size,
        benchmark=True,
        callbacks=[model_checkpoint],
        deterministic=True,
        fast_dev_run=fast_dev_run,
        gpus=gpus,
        max_epochs=2 if DEBUG else max_epochs,
        precision=precision,
        stochastic_weight_avg=stochastic_weight_avg,
        limit_train_batches=0.1 if DEBUG else 1.0,
        limit_val_batches=0.1 if DEBUG else 1.0,
    )

    trainer.tune(module, datamodule=datamodule)

    trainer.fit(module, datamodule=datamodule)

if '__name__' == '__main__':

    train_df = pd.read_csv(TRAIN_CSV_PATH)

    train_df["image_path"] = train_df["image"].apply(get_image_path, dir=TRAIN_DIRECTORY)

    encoder = LabelEncoder()
    train_df["individual_id"] = encoder.fit_transform(train_df["individual_id"])
    np.save(ENCODER_CLASSES_PATH, encoder.classes_)

    skf = StratifiedKFold(n_splits=N_SPLITS)
    for fold, (_, val_) in enumerate(skf.split(X=train_df, y=train_df.individual_id)):
        train_df.loc[val_, "kfold"] = fold
        
    train_df.to_csv(TRAIN_CSV_ENCODED_FOLDED_PATH, index=False)

    # Use sample submission as template
    test_df = pd.read_csv(SAMPLE_SUBMISSION_CSV_PATH)
    test_df["image_path"] = test_df["image"].apply(get_image_path, dir=TEST_DIRECTORY)
    test_df.drop(columns=["predictions"], inplace=True)
    test_df["individual_id"] = 0
    test_df.to_csv(TEST_CSV_PATH, index=False)

    model_name = "convnext_base_384_in22ft1k"
    image_size = 384
    batch_size = 8

    train(model_name=model_name,
      image_size=image_size,
      batch_size=batch_size)


