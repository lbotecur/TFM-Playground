import numpy as np
import pandas as pd

from tfmplayground.benchmarks.analysis import (against_reference, friedman_nemenyi, holm, mean_ranks,
                                               merge_variants, short_name)


def _row(model, dataset, fold, auc):
    return {"benchmark": "b", "dataset": dataset, "n_features": 1, "model": model, "fold": fold,
            "accuracy": auc, "roc_auc": auc, "seconds": 1.0}


def test_merge_variants_keeps_plain_row_where_both_exist():
    r = pd.DataFrame([_row("m.pth", "A", 0, 0.9), _row("m.pth [low-memory]", "A", 0, 0.8),
                      _row("m.pth [low-memory]", "B", 0, 0.7)])
    merged = merge_variants(r)
    assert sorted(merged.model) == ["m.pth", "m.pth"]
    assert merged.set_index("dataset").roc_auc.to_dict() == {"A": 0.9, "B": 0.7}


def test_short_name():
    assert short_name("workdir/graph_scm_base_w5000/epoch_100.pth [bfloat16]") == "graph_scm_base_w5000/epoch_100"
    assert short_name("tabpfn-3.5:auto") == "tabpfn-3.5:auto"
    assert short_name("workdir/a/epoch_100.pth@ctx300 [bfloat16] [low-memory]") == "a/epoch_100@ctx300"


def test_holm_matches_hand_computation():
    np.testing.assert_allclose(holm([0.01, 0.04, np.nan, 0.03]), [0.03, 0.06, np.nan, 0.06])


def test_ranks_friedman_and_reference():
    table = pd.DataFrame({"a": [0.9, 0.8, 0.95, 0.7], "b": [0.8, 0.7, 0.9, 0.6], "c": [0.85, 0.75, 0.9, np.nan]})
    assert mean_ranks(table).index[0] == "a"
    res = friedman_nemenyi(table)
    assert res["datasets"] == 3 and res["models"] == 3 and res["cd"] > 0
    comp = against_reference(table, "a").set_index("model")
    assert comp.loc["b", "losses"] == 4 and comp.loc["c", "datasets"] == 3


def test_friedman_with_two_models_is_skipped():
    res = friedman_nemenyi(pd.DataFrame({"a": [0.9, 0.8, 0.7], "b": [0.8, 0.7, 0.6]}))
    assert res["models"] == 2 and np.isnan(res["p"])
