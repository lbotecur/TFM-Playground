import os
import warnings

import numpy as np
import pandas as pd
import requests
import torch
import torch.nn.functional as F
from pfns.bar_distribution import FullSupportBarDistribution
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import FunctionTransformer, LabelEncoder, OrdinalEncoder

from tfmplayground.models.nanotabpfn import LEGACY_FEATURE_CLIP, NanoTabPFNModel
from tfmplayground.normalization import (
    compute_target_stats_numpy,
    denormalize_predictions,
    normalize_targets,
)
from tfmplayground.utils import autocast, get_default_device


def _migrate_feature_encoder_weights(model_state: dict) -> dict:
    """Expands a pre-indicator feature encoder weight of shape [E, 1] to [E, 2] so a
    checkpoint trained before the missing-indicator change loads into the current model.

    The value channel keeps the old weight and the new indicator channel starts at zero,
    which is function-preserving: with no missing entries the indicator is zero, so the
    embedding is identical to the old model's. The bias is unchanged. A checkpoint that
    is already [E, 2] is returned untouched.
    """
    key = "feature_encoder.linear_layer.weight"
    weight = model_state.get(key)
    if weight is not None and weight.shape[1] == 1:
        zeros = torch.zeros_like(weight)  # [E, 1], the indicator channel
        model_state = {**model_state, key: torch.cat([weight, zeros], dim=1)}  # -> [E, 2]
    return model_state


def _migrate_state_dict(model_state: dict) -> dict:
    """Makes a checkpoint from an older architecture load into the current model, preserving
    its function exactly:
    - feature encoder [E, 1] -> [E, 2], see _migrate_feature_encoder_weights;
    - target encoder [E, 1] -> [E, 2]: the new test-row indicator channel starts at zero, so
      the embedding equals the old one;
    - column embedding: absent in old checkpoints, added with zero weight and bias, so it adds
      nothing to the feature embeddings.
    A checkpoint already in the current format is returned unchanged.
    """
    model_state = _migrate_feature_encoder_weights(model_state)
    key = "target_encoder.linear_layer.weight"
    weight = model_state.get(key)
    if weight is not None and weight.shape[1] == 1:
        model_state = {**model_state, key: torch.cat([weight, torch.zeros_like(weight)], dim=1)}
    if "column_embedding.weight" not in model_state:
        embedding_size = model_state["feature_encoder.linear_layer.weight"].shape[0]
        model_state = {
            **model_state,
            "column_embedding.weight": torch.zeros(embedding_size, embedding_size // 4),
            "column_embedding.bias": torch.zeros(embedding_size),
        }
    return model_state


def init_model_from_state_dict_file(file_path):
    """
    reads model architecture from state dict, instantiates the architecture and loads the weights
    """
    state_dict = torch.load(file_path, map_location=torch.device("cpu"))
    model = NanoTabPFNModel(
        num_attention_heads=state_dict["architecture"]["num_attention_heads"],
        embedding_size=state_dict["architecture"]["embedding_size"],
        mlp_hidden_size=state_dict["architecture"]["mlp_hidden_size"],
        num_layers=state_dict["architecture"]["num_layers"],
        num_outputs=state_dict["architecture"]["num_outputs"],
        # checkpoints saved before the clip was configurable were all trained with 100
        feature_clip=state_dict["architecture"].get("feature_clip", LEGACY_FEATURE_CLIP),
    )
    model.load_state_dict(_migrate_state_dict(state_dict["model"]))
    # how the model is meant to be used at inference: shuffle the columns once (see _column_order); checkpoints
    # without the setting (the v2 and earlier) keep the preprocessor's order
    model.permute_columns = bool(state_dict["architecture"].get("permute_columns", False))
    return model


def export_checkpoint(source: str, destination: str, feature_clip: float | None = None,
                      permute_columns: bool | None = None) -> dict:
    """Writes a release copy of a training checkpoint: architecture (with the feature clip and the column
    permutation the model is meant to be used with) and weights, without the optimizer state. None keeps the
    source's own (LEGACY_FEATURE_CLIP and no permutation if it has none). Loading the copy with
    NanoTabPFNClassifier needs no other setting."""
    state = torch.load(source, map_location="cpu", weights_only=False)
    architecture = dict(state["architecture"])
    architecture["feature_clip"] = float(
        architecture.get("feature_clip", LEGACY_FEATURE_CLIP) if feature_clip is None else feature_clip
    )
    if permute_columns is not None:
        architecture["permute_columns"] = bool(permute_columns)
    release = {"epoch": state.get("epoch"), "architecture": architecture, "model": state["model"],
               "source": str(source)}
    torch.save(release, destination)
    return architecture


# low_memory="auto" switches to low-memory inference when the (rows, columns, embedding) table of one
# forward pass has at least this many elements. Above ~2**32 elements the normal path returned garbage
# (AUROC 0.5 or NaN on gisette: 7000 rows x 5000 features x 192) while low-memory was correct; 2**31
# keeps a margin. It depends only on the table, not on the GPU, so results do not change across GPUs.
LOW_MEMORY_AUTO_ELEMENTS = 2**31


def _configure(estimator, model, feature_clip, low_memory, permute_columns=None, column_seed=0):
    """Shared by the classifier and the regressor: feature clip, low-memory mode and column permutation."""
    if feature_clip is not None:
        model.feature_encoder.clip = float(feature_clip)
    estimator.permute_columns = (getattr(model, "permute_columns", False) if permute_columns is None
                                 else bool(permute_columns))
    estimator.column_seed = column_seed
    if low_memory not in (True, False, "auto"):
        raise ValueError(f"low_memory must be True, False or 'auto', got {low_memory!r}")
    estimator.low_memory = low_memory
    _set_low_memory(model, low_memory is True)


def _column_order(num_columns: int, permute: bool, seed: int) -> np.ndarray:
    """Order in which the preprocessed columns enter the model. The preprocessor puts the numeric columns first
    and the categorical ones last, a fixed pattern that training tables never have (their columns are in random
    order); with permute, one permutation fixed by the seed (deterministic, so attention scores are
    reproducible). Without, the preprocessor's order."""
    if not permute:
        return np.arange(num_columns)
    return np.random.default_rng(seed).permutation(num_columns)


def _fit_columns(estimator, X_train):
    """Fits the preprocessor on X_train and returns the preprocessed training rows, in the model's order."""
    estimator.feature_preprocessor = get_feature_preprocessor(
        X_train, categorical_features=estimator.categorical_features, infer_categorical=estimator.infer_categorical,
        max_unique_for_categorical=estimator.max_unique_for_categorical,
        min_samples_for_categorical_inference=estimator.min_samples_for_categorical_inference,
    )
    X = estimator.feature_preprocessor.fit_transform(X_train)
    estimator.column_order_ = _column_order(X.shape[1], estimator.permute_columns, estimator.column_seed)
    return X[:, estimator.column_order_]


def _transform_columns(estimator, X_test):
    """Preprocesses test rows as the training rows (same columns, same order)."""
    return estimator.feature_preprocessor.transform(X_test)[:, estimator.column_order_]


def _set_low_memory(model, on: bool):
    for block in getattr(model, "transformer_blocks", []):
        block.low_memory = on


def _run_model(estimator, x: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
    """One forward pass (under no_grad, in the estimator's amp_dtype) for x = [X_train; X_test].
    With low_memory="auto", large tables (see LOW_MEMORY_AUTO_ELEMENTS) run in low-memory mode, and a
    CUDA out-of-memory error in normal mode is retried once in low-memory mode, so a table either gets a
    correct answer or a real error. True/False force one mode."""
    model, mode = estimator.model, estimator.low_memory
    if mode == "auto":
        elements = x.shape[1] * (x.shape[2] + 1) * model.embedding_size  # +1: the target column
        use_low_memory = elements >= LOW_MEMORY_AUTO_ELEMENTS
    else:
        use_low_memory = bool(mode)

    def forward(low_memory: bool) -> torch.Tensor:
        _set_low_memory(model, low_memory)
        with autocast(estimator.device, estimator.amp_dtype):
            return model((x, y), train_test_split_index=len(estimator.X_train), num_mem_chunks=estimator.num_mem_chunks)

    if mode != "auto" or use_low_memory:
        return forward(use_low_memory)
    try:
        return forward(False)
    except torch.cuda.OutOfMemoryError:
        warnings.warn("Out of GPU memory in normal mode; retrying with low-memory inference.", stacklevel=3)
        torch.cuda.empty_cache()
        return forward(True)


# doing these as lambdas would cause NanoTabPFNClassifier to not be pickle-able,
# which would cause issues if we want to run it inside the tabarena codebase
def to_pandas(x):
    return pd.DataFrame(x) if not isinstance(x, pd.DataFrame) else x


def to_numeric(x):
    return x.apply(pd.to_numeric, errors="coerce").to_numpy()


def get_feature_preprocessor(
    X: np.ndarray | pd.DataFrame,
    categorical_features: list[int] | None = None,
    infer_categorical: bool = True,
    max_unique_for_categorical: int = 10,
    min_samples_for_categorical_inference: int = 30,
) -> ColumnTransformer:
    """
    fits a preprocessor that imputes NaNs, encodes categorical features and removes constant features.
    Columns whose positional index is in categorical_features are forced categorical (at any
    cardinality), overriding the automatic numeric/categorical detection; constant columns are still
    dropped. categorical_features=None keeps the automatic detection only.
    With infer_categorical=False, undeclared columns are treated as numeric and an undeclared
    non-numeric column raises ValueError (every categorical must be declared explicitly).
    When infer_categorical=True, an undeclared numeric column with at most
    max_unique_for_categorical distinct values is treated as categorical (e.g. integer-coded
    categories), provided it has at least min_samples_for_categorical_inference non-NaN rows;
    the sample floor keeps small datasets — where low cardinality is trivial — numeric. These
    thresholds are tuned for nanoTabPFN's small-data regime rather than copied from TabPFN.
    """
    X = pd.DataFrame(X)
    num_features = X.shape[1]
    declared_categorical = set(categorical_features or [])
    for index in declared_categorical:
        if index < 0 or index >= num_features:
            raise ValueError(
                f"categorical_features index {index} is out of range for {num_features} features."
            )
    num_mask = []
    cat_mask = []
    for position, col in enumerate(X):
        unique_non_nan_entries = X[col].dropna().unique()
        if len(unique_non_nan_entries) <= 1:
            num_mask.append(False)
            cat_mask.append(False)
            continue
        if position in declared_categorical:
            num_mask.append(False)
            cat_mask.append(True)
            continue
        non_nan_entries = X[col].notna().sum()
        numeric_entries = (
            pd.to_numeric(X[col], errors="coerce").notna().sum()
        )  # in case numeric columns are stored as strings
        is_numeric = non_nan_entries == numeric_entries
        if not infer_categorical and not is_numeric:
            raise ValueError(
                f"Column at index {position} is not numeric and was not declared categorical; "
                f"with infer_categorical=False every categorical column must be listed in "
                f"categorical_features."
            )
        # A low-cardinality numeric column (e.g. integer-coded categories) is treated as
        # categorical, but only with inference on and enough rows to trust the signal; the
        # sample floor keeps tiny datasets, where low cardinality is trivial, numeric.
        is_low_cardinality_numeric = (
            infer_categorical
            and is_numeric
            and non_nan_entries >= min_samples_for_categorical_inference
            and len(unique_non_nan_entries) <= max_unique_for_categorical
        )
        treat_as_numeric = is_numeric and not is_low_cardinality_numeric
        num_mask.append(treat_as_numeric)
        cat_mask.append(not treat_as_numeric)
        # num_mask.append(is_numeric_dtype(X[col]))  # Assumes pandas dtype is correct

    num_mask = np.array(num_mask)
    cat_mask = np.array(cat_mask)

    num_transformer = Pipeline(
        [
            ("to_pandas", FunctionTransformer(to_pandas)),  # to apply pd.to_numeric of pandas
            ("to_numeric", FunctionTransformer(to_numeric)),  # in case numeric columns are stored as strings
        ]
    )  # no imputation here: NaNs pass through and the model imputes (mean) and flags them
    cat_transformer = Pipeline(
        [
            ("encoder", OrdinalEncoder(handle_unknown="use_encoded_value", unknown_value=np.nan)),
        ]
    )  # no imputation here either: a missing category stays NaN for the model to handle
    preprocessor = ColumnTransformer(
        transformers=[("num", num_transformer, num_mask), ("cat", cat_transformer, cat_mask)]
    )
    return preprocessor


class NanoTabPFNClassifier:
    """scikit-learn like interface"""

    def __init__(
        self,
        model: NanoTabPFNModel | str | None = None,
        device: None | str | torch.device = None,
        num_mem_chunks: int = 8,
        categorical_features: list[int] | None = None,
        infer_categorical: bool = True,
        max_unique_for_categorical: int = 10,
        min_samples_for_categorical_inference: int = 30,
        amp_dtype: torch.dtype | None = None,
        low_memory: bool | str = True,
        feature_clip: float | None = None,
        permute_columns: bool | None = None,
        column_seed: int = 0,
    ):
        device = torch.device(get_default_device() if device is None else device)
        if model is None:
            model = "checkpoints/nanotabpfn.pth"
            if not os.path.isfile(model):
                os.makedirs("checkpoints", exist_ok=True)
                print("No cached model found, downloading model checkpoint.")
                response = requests.get(
                    "https://ml.informatik.uni-freiburg.de/research-artifacts/pfefferle/TFM-Playground/nanotabpfn_classifier.pth"
                )
                with open(model, "wb") as f:
                    f.write(response.content)
        if isinstance(model, str):
            model = init_model_from_state_dict_file(model)
        self.model = model.to(device)
        self.device = device
        self.num_mem_chunks = num_mem_chunks
        self.categorical_features = categorical_features
        self.infer_categorical = infer_categorical
        self.max_unique_for_categorical = max_unique_for_categorical
        self.min_samples_for_categorical_inference = min_samples_for_categorical_inference
        # Mixed precision for inference, as amp_dtype in train(): e.g. torch.bfloat16 runs matmuls and
        # attention in bf16 (much faster on wide tables). None keeps fp32.
        self.amp_dtype = amp_dtype
        # Low-memory inference (see TransformerEncoderLayer.low_memory), the default: as accurate as the normal
        # path (identical in fp32, rounding differences in bf16), faster in our benchmarks, and correct on large
        # tables where the normal path is not. "auto" uses the normal path for small tables (see _run_model).
        # feature_clip: bound of the normalized features (see normalize_features); None keeps the checkpoint's own.
        # permute_columns: shuffle the preprocessed columns once, with column_seed (see _column_order); None keeps
        # the checkpoint's own setting (off for the v2 and earlier).
        _configure(self, self.model, feature_clip, low_memory, permute_columns, column_seed)

    def fit(self, 
            X_train: np.ndarray | pd.DataFrame, 
            y_train: np.ndarray | pd.Series,
    ):
        """Stores X_train, label-encodes the targets to contiguous indices 0..num_classes-1
        (so arbitrary labels, e.g. non-contiguous integers or strings, are supported), and
        keeps the original labels in classes_ for decoding predictions"""
        self.X_train = _fit_columns(self, X_train)
        self.label_encoder = LabelEncoder()
        self.y_train = self.label_encoder.fit_transform(y_train)
        self.classes_ = self.label_encoder.classes_
        self.num_classes = len(self.classes_)
        if self.num_classes > self.model.num_outputs:
            raise ValueError(
                f"This model supports at most {self.model.num_outputs} classes, "
                f"but the training data has {self.num_classes}."
            )
        return self

    def predict(self, X_test: np.ndarray) -> np.ndarray:
        """calls predict_proba, picks the highest-probability class for each datapoint,
        and maps it back to the original label"""
        predicted_probabilities = self.predict_proba(X_test)
        encoded_predictions = predicted_probabilities.argmax(axis=1)
        return self.label_encoder.inverse_transform(encoded_predictions)

    def predict_proba(self, X_test: np.ndarray) -> np.ndarray:
        """
        creates (x,y), runs it through our PyTorch Model, cuts off the classes that didn't appear in the training data
        and applies softmax to get the probabilities
        """
        x = np.concatenate((self.X_train, _transform_columns(self, X_test)))
        y = self.y_train
        with torch.no_grad():
            x = torch.from_numpy(x).unsqueeze(0).to(torch.float).to(self.device)  # introduce batch size 1
            y = torch.from_numpy(y).unsqueeze(0).to(torch.float).to(self.device)
            out = _run_model(self, x, y)
            out = out.float().squeeze(0)  # softmax in fp32; remove batch size 1
            # our pretrained classifier supports up to num_outputs classes, if the dataset has less we cut off the rest
            out = out[:, : self.num_classes]
            # apply softmax to get a probability distribution
            probabilities = F.softmax(out, dim=1)
            return probabilities.to("cpu").numpy()

    def _feature_masks(self):
        """Recovers the numeric/categorical boolean masks (over original columns) from the
        fitted ColumnTransformer, so attention columns can be mapped back to original features.
        """
        num_mask = cat_mask = None
        for name, _, cols in self.feature_preprocessor.transformers_:
            if name == "num":
                num_mask = np.asarray(cols, dtype=bool)
            elif name == "cat":
                cat_mask = np.asarray(cols, dtype=bool)
        return num_mask, cat_mask

    def feature_attention_scores(self, X_test: np.ndarray) -> np.ndarray:
        """Feature importance via attention, in the style of TabPFN-Wide. Runs inference with
        attention capture enabled, averages the target column's attention to each feature over
        samples, heads and layers, and returns one score per ORIGINAL feature. Columns that were
        dropped as constant get NaN (the model never saw them). Requires fit() first and a model
        that exposes transformer_blocks.
        """
        x = np.concatenate((self.X_train, _transform_columns(self, X_test)))
        y = self.y_train
        blocks = self.model.transformer_blocks
        for block in blocks:
            block.save_feature_attention = True
            block.feature_attention = None
        try:
            with torch.no_grad():
                xt = torch.from_numpy(x).unsqueeze(0).to(torch.float).to(self.device)
                yt = torch.from_numpy(y).unsqueeze(0).to(torch.float).to(self.device)
                _run_model(self, xt, yt)
            # average over layers: stack to (num_layers, C) -> (C,); last entry is target->target
            per_layer = torch.stack([block.feature_attention for block in blocks], dim=0)
            attention_to_columns = per_layer.mean(dim=0)[:-1].to("cpu").numpy()  # (C-1,) transformed cols
        finally:
            for block in blocks:
                block.save_feature_attention = False
                block.feature_attention = None
        # map transformed columns ([num..., cat...]) back to original feature indices
        num_mask, cat_mask = self._feature_masks()
        original_of_output = np.asarray(np.where(num_mask)[0].tolist() + np.where(cat_mask)[0].tolist(), dtype=int)
        scores = np.full(num_mask.shape[0], np.nan)
        # model position k holds preprocessed column column_order_[k]
        scores[original_of_output[self.column_order_]] = attention_to_columns
        return scores

    def get_embeddings(self, X_test: np.ndarray) -> np.ndarray:
        """Extracts the per-row target-token embedding for X_test, using the fitted training
        set as in-context support (TabPFN-v2-style feature extraction). Returns an array of
        shape (n_test, embedding_size). For comparable TRAINING embeddings use the
        leave-one-fold-out extractor rather than embedding the training rows directly.
        """
        x = np.concatenate((self.X_train, _transform_columns(self, X_test)))
        y = self.y_train
        self.model.save_embeddings = True
        self.model.embeddings = None
        try:
            with torch.no_grad():
                xt = torch.from_numpy(x).unsqueeze(0).to(torch.float).to(self.device)
                yt = torch.from_numpy(y).unsqueeze(0).to(torch.float).to(self.device)
                _run_model(self, xt, yt)
            embeddings = self.model.embeddings[0, len(self.X_train):, :].float().to("cpu").numpy()
        finally:
            self.model.save_embeddings = False
            self.model.embeddings = None
        return embeddings


class NanoTabPFNRegressor:
    """scikit-learn like interface"""

    def __init__(
        self,
        model: NanoTabPFNModel | str | None = None,
        dist: FullSupportBarDistribution | str | None = None,
        device: str | torch.device | None = None,
        num_mem_chunks: int = 8,
        categorical_features: list[int] | None = None,
        infer_categorical: bool = True,
        max_unique_for_categorical: int = 10,
        min_samples_for_categorical_inference: int = 30,
        amp_dtype: torch.dtype | None = None,
        low_memory: bool | str = True,
        feature_clip: float | None = None,
        permute_columns: bool | None = None,
        column_seed: int = 0,
    ):
        device = torch.device(get_default_device() if device is None else device)
        if model is None:
            os.makedirs("checkpoints", exist_ok=True)
            model = "checkpoints/nanotabpfn_regressor.pth"
            dist = "checkpoints/nanotabpfn_regressor_buckets.pth"
            if not os.path.isfile(model):
                print("No cached model found, downloading model checkpoint.")
                response = requests.get(
                    "https://ml.informatik.uni-freiburg.de/research-artifacts/pfefferle/TFM-Playground/nanotabpfn_regressor.pth"
                )
                with open(model, "wb") as f:
                    f.write(response.content)
            if not os.path.isfile(dist):
                print("No cached bucket edges found, downloading bucket edges.")
                response = requests.get(
                    "https://ml.informatik.uni-freiburg.de/research-artifacts/pfefferle/TFM-Playground/nanotabpfn_regressor_buckets.pth"
                )
                with open(dist, "wb") as f:
                    f.write(response.content)
        if isinstance(model, str):
            model = init_model_from_state_dict_file(model)

        if isinstance(dist, str):
            bucket_edges = torch.load(dist, map_location=device)
            dist = FullSupportBarDistribution(bucket_edges).float()

        self.model = model.to(device)
        self.device = device
        self.dist = dist
        self.num_mem_chunks = num_mem_chunks
        self.categorical_features = categorical_features
        self.infer_categorical = infer_categorical
        self.max_unique_for_categorical = max_unique_for_categorical
        self.min_samples_for_categorical_inference = min_samples_for_categorical_inference
        # Mixed precision for inference, as amp_dtype in train(): e.g. torch.bfloat16 runs matmuls and
        # attention in bf16 (much faster on wide tables). None keeps fp32.
        self.amp_dtype = amp_dtype
        # Low-memory inference (see TransformerEncoderLayer.low_memory), the default: as accurate as the normal
        # path (identical in fp32, rounding differences in bf16), faster in our benchmarks, and correct on large
        # tables where the normal path is not. "auto" uses the normal path for small tables (see _run_model).
        # feature_clip: bound of the normalized features (see normalize_features); None keeps the checkpoint's own.
        # permute_columns: shuffle the preprocessed columns once, with column_seed (see _column_order); None keeps
        # the checkpoint's own setting (off for the v2 and earlier).
        _configure(self, self.model, feature_clip, low_memory, permute_columns, column_seed)

    def fit(self, 
            X_train:np.ndarray | pd.DataFrame, 
            y_train: np.ndarray | pd.Series,
    ):
        """
        Stores X_train and y_train for later use.
        Computes target normalization.
        """
        self.X_train = _fit_columns(self, X_train)
        self.y_train = y_train

        self.y_train_mean, self.y_train_std = compute_target_stats_numpy(self.y_train)
        self.y_train_n = normalize_targets(self.y_train, self.y_train_mean, self.y_train_std)
        return self

    def predict(self, X_test: np.ndarray) -> np.ndarray:
        """
        Performs in-context learning using X_train and y_train.
        Predicts the means of the output distributions for X_test.
        Renormalizes the predictions back to the original target scale.
        """
        X = np.concatenate((self.X_train, _transform_columns(self, X_test)))
        y = self.y_train_n

        with torch.no_grad():
            X_tensor = torch.tensor(X, dtype=torch.float32, device=self.device).unsqueeze(0)
            y_tensor = torch.tensor(y, dtype=torch.float32, device=self.device).unsqueeze(0)

            logits = _run_model(self, X_tensor, y_tensor)
            logits = logits.float().squeeze(0)  # distribution in fp32
            preds_n = self.dist.mean(logits)
            preds = denormalize_predictions(preds_n, self.y_train_mean, self.y_train_std)

        return preds.cpu().numpy()

    def get_embeddings(self, X_test: np.ndarray) -> np.ndarray:
        """Extracts the per-row target-token embedding for X_test, using the fitted training
        set as in-context support (TabPFN-v2-style feature extraction). Returns an array of
        shape (n_test, embedding_size). For comparable TRAINING embeddings use the
        leave-one-fold-out extractor rather than embedding the training rows directly.
        """
        x = np.concatenate((self.X_train, _transform_columns(self, X_test)))
        y = self.y_train_n
        self.model.save_embeddings = True
        self.model.embeddings = None
        try:
            with torch.no_grad():
                xt = torch.from_numpy(x).unsqueeze(0).to(torch.float).to(self.device)
                yt = torch.from_numpy(y).unsqueeze(0).to(torch.float).to(self.device)
                _run_model(self, xt, yt)
            embeddings = self.model.embeddings[0, len(self.X_train):, :].float().to("cpu").numpy()
        finally:
            self.model.save_embeddings = False
            self.model.embeddings = None
        return embeddings