"""配置层测试：不需要网络，不需要 .env（见 config.py 的"调用时才校验"取舍）。"""
from __future__ import annotations

import pytest

from src import config


def test_paths_are_under_project_root():
    assert config.DATA_DIR == config.PROJECT_ROOT / "data"
    assert config.MILVUS_URI.startswith(str(config.DATA_DIR))


def test_require_env_raises_with_variable_name():
    with pytest.raises(config.ConfigError) as exc:
        config.require_env("MEDEVIDENCE_NOT_EXIST_XYZ")
    # 错误信息里必须带变量名，否则排查时只能靠猜
    assert "MEDEVIDENCE_NOT_EXIST_XYZ" in str(exc.value)


def test_require_env_falls_back_to_default(monkeypatch):
    monkeypatch.delenv("MEDEVIDENCE_NOT_EXIST_XYZ", raising=False)
    assert config.require_env("MEDEVIDENCE_NOT_EXIST_XYZ", "fallback") == "fallback"


def test_optional_env_treats_empty_string_as_unset(monkeypatch):
    """`.env.example` 里 `EMBEDDING_BATCH_SIZE=` 是空值，不能让它变成 0 或直接炸。"""
    monkeypatch.setenv("MEDEVIDENCE_MAYBE_XYZ", "")
    assert config.optional_env("MEDEVIDENCE_MAYBE_XYZ", "fallback") == "fallback"


def test_empty_batch_size_uses_default(monkeypatch):
    monkeypatch.setenv("EMBEDDING_BATCH_SIZE", "")
    monkeypatch.setenv("EMBEDDING_API_KEY", "k")
    monkeypatch.setenv("EMBEDDING_BASE_URL", "http://example.invalid/v1")
    monkeypatch.setenv("EMBEDDING_MODEL_NAME", "m")
    cfg = config.get_embedding_config()
    assert cfg.batch_size == config.DEFAULT_EMBEDDING_BATCH_SIZE


def test_mask_hides_secret_but_keeps_tail():
    shown = config._mask("sk-abcdef123456")
    assert shown.endswith("3456")
    assert "abcdef" not in shown


def test_ensure_dirs_is_idempotent():
    config.ensure_dirs()
    config.ensure_dirs()
    assert config.RAW_DIR.is_dir()
    assert config.PROCESSED_DIR.is_dir()
