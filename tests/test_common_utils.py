"""Tests for common_utils.py."""

import os
import tempfile
import pytest
import yaml
from pathlib import Path

from src.DynamicPricingEngine.utils.common_utils import create_dir, read_yaml, save_yaml


class TestCreateDir:
    def test_creates_single_dir(self):
        with tempfile.TemporaryDirectory() as tmp:
            new_dir = os.path.join(tmp, "test_dir")
            create_dir([new_dir])
            assert os.path.isdir(new_dir)

    def test_creates_nested_dirs(self):
        with tempfile.TemporaryDirectory() as tmp:
            nested = os.path.join(tmp, "a", "b", "c")
            create_dir([nested])
            assert os.path.isdir(nested)

    def test_no_error_on_existing(self):
        with tempfile.TemporaryDirectory() as tmp:
            create_dir([tmp])
            assert os.path.isdir(tmp)

    def test_multiple_dirs(self):
        with tempfile.TemporaryDirectory() as tmp:
            d1 = os.path.join(tmp, "d1")
            d2 = os.path.join(tmp, "d2")
            create_dir([d1, d2])
            assert os.path.isdir(d1)
            assert os.path.isdir(d2)


class TestReadYaml:
    def test_reads_valid_yaml(self):
        with tempfile.NamedTemporaryFile(mode='w', suffix='.yaml', delete=False) as f:
            yaml.dump({"key": "value", "num": 42}, f)
            fpath = f.name
        try:
            result = read_yaml(Path(fpath))
            assert result.key == "value"
            assert result.num == 42
        finally:
            os.unlink(fpath)

    def test_dotted_access(self):
        with tempfile.NamedTemporaryFile(mode='w', suffix='.yaml', delete=False) as f:
            yaml.dump({"nested": {"inner": "deep"}}, f)
            fpath = f.name
        try:
            result = read_yaml(Path(fpath))
            assert result.nested.inner == "deep"
        finally:
            os.unlink(fpath)

    def test_raises_on_empty_file(self):
        with tempfile.NamedTemporaryFile(suffix='.yaml', delete=False) as f:
            fpath = f.name
        try:
            with pytest.raises(ValueError):
                read_yaml(Path(fpath))
        finally:
            os.unlink(fpath)


class TestSaveYaml:
    def test_saves_and_reads_back(self):
        data = {"a": 1, "b": {"c": 2}}
        with tempfile.TemporaryDirectory() as tmp:
            path = os.path.join(tmp, "test.yaml")
            save_yaml(path, data)
            assert os.path.exists(path)
            result = read_yaml(Path(path))
            assert result.a == 1
            assert result.b.c == 2
