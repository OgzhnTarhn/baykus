from pathlib import Path

import pytest

from baykus.train.config import ConfigError, config_hash, load_config, parse_config

BASE = {
    "experiment": "e1_rgb",
    "mode": "rgb",
    "model": "yolo26s.pt",
    "data": "data/dv_rgbt/data_rgb.yaml",
}
EF = {**BASE, "mode": "early_fusion", "data": "data/dv_rgbt/data_rgbt4.yaml"}


def test_minimal_defaults():
    c = parse_config(dict(BASE))
    assert c.label_set == "ir" and c.modality == "rgb"
    assert c.train.epochs == 100 and c.train.seed == 42 and c.train.batch == 16
    assert c.first_conv_init is None and c.modality_dropout is None
    assert c.eval.n_boot == 1000 and c.eval.label_sets == ("ir", "rgb")
    assert c.mlflow_experiment == "baykus"


def test_unknown_top_level_key_names_key():
    with pytest.raises(ConfigError, match="epoch"):
        parse_config({**BASE, "epoch": 5})


def test_unknown_nested_key_names_section_and_key():
    with pytest.raises(ConfigError, match="train: bilinmeyen.*epoch"):
        parse_config({**BASE, "train": {"epoch": 5}})


def test_missing_required_key():
    with pytest.raises(ConfigError, match="model"):
        parse_config({k: v for k, v in BASE.items() if k != "model"})


def test_invalid_mode():
    with pytest.raises(ConfigError, match="mode"):
        parse_config({**BASE, "mode": "fusion"})


def test_data_file_must_match_mode():
    with pytest.raises(ConfigError, match="data_rgbt4.yaml"):
        parse_config({**BASE, "mode": "early_fusion"})


def test_fusion_options_rejected_for_single_modality():
    with pytest.raises(ConfigError, match="early_fusion"):
        parse_config({**BASE, "first_conv_init": "rgb_mean"})
    with pytest.raises(ConfigError, match="early_fusion"):
        parse_config({**BASE, "modality_dropout": {"p_rgb": 0.1, "p_ir": 0.1}})


def test_early_fusion_defaults_to_ul_default():
    c = parse_config(dict(EF))
    assert c.first_conv_init == "ul_default" and c.modality == "rgbt4"


def test_random_init_not_supported_yet():
    with pytest.raises(ConfigError, match="random"):
        parse_config({**EF, "first_conv_init": "random"})


def test_dropout_probabilities_validated():
    with pytest.raises(ConfigError, match="en fazla 1"):
        parse_config({**EF, "modality_dropout": {"p_rgb": 0.6, "p_ir": 0.5}})
    with pytest.raises(ConfigError, match="p_ir"):
        parse_config({**EF, "modality_dropout": {"p_rgb": 0.1, "p_ir": -0.1}})
    c = parse_config({**EF, "modality_dropout": {"p_rgb": 0.15, "p_ir": 0.15}})
    assert c.modality_dropout.p_rgb == 0.15 and c.modality_dropout.p_ir == 0.15


def test_eval_values_validated():
    with pytest.raises(ConfigError, match="label_sets"):
        parse_config({**BASE, "eval": {"label_sets": ["ir", "thermal"]}})
    with pytest.raises(ConfigError, match="slices"):
        parse_config({**BASE, "eval": {"slices": ["weather"]}})


def test_hash_stable_and_sensitive():
    assert config_hash(BASE) == config_hash(dict(reversed(list(BASE.items()))))
    assert config_hash(BASE) != config_hash({**BASE, "model": "yolo26n.pt"})
    assert config_hash(BASE).startswith("sha256:")
    assert parse_config(dict(BASE)).config_hash == config_hash(BASE)


def test_load_rejects_non_mapping(tmp_path: Path):
    p = tmp_path / "bad.yaml"
    p.write_text("- 1\n- 2\n", encoding="utf-8")
    with pytest.raises(ConfigError, match="sözlük"):
        load_config(p)


def test_repo_configs_load():
    root = Path(__file__).resolve().parents[2] / "configs"
    names = {f.stem for f in root.glob("*.yaml")}
    assert names >= {"e1_rgb", "e2_ir", "e3_early_fusion", "e3b_ul_default", "e3c_modality_dropout"}
    for f in root.glob("*.yaml"):
        load_config(f)
    e3c = load_config(root / "e3c_modality_dropout.yaml")
    assert e3c.first_conv_init == "rgb_mean" and e3c.modality_dropout.p_rgb == 0.15
