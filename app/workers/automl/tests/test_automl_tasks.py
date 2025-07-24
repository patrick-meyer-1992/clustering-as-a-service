
import pytest
import os
import json
from unittest.mock import patch, MagicMock
from workers.automl.automl_tasks import run_autocluster_logic
from workers.automl.automl_tasks import build_subprocess_args
from workers.automl.automl_tasks import build_environment
from workers.automl.automl_tasks import run_subprocess


def test_build_subprocess_args_structure():
    args = build_subprocess_args(
        job_id="abc123",
        dataset_name="test_dataset",
        columns=[{"name": "x", "type": "numeric"}],
        created_timestamp="2025-01-01T12:00:00+00:00",
        clustering_algorithms=["KMeans"],
        dim_reduction_algorithms=["PCA"],
        n_evaluations=10,
        cutoff_time=30,
        evaluator_ls=["silhouetteScore"],
        clustering_num=(2, 5),
        min_proportion=0.01,
        min_relative_proportion=0.05
    )

    assert args[0] == "python"
    assert "automl_worker.py" in args[1]
    assert args[2] == "abc123"
    assert args[3] == "test_dataset"

    columns_arg = json.loads(args[4])
    assert isinstance(columns_arg, list)
    assert columns_arg[0]["name"] == "x"

    config_arg = json.loads(args[6])
    assert config_arg["clustering_algorithms"] == ["KMeans"]
    assert config_arg["dim_reduction_algorithms"] == ["PCA"]
    assert config_arg["n_evaluations"] == 10


def test_build_subprocess_args_with_none():
    args = build_subprocess_args(
        job_id="abc123",
        dataset_name="test_dataset",
        columns=[],
        created_timestamp="2025-01-01T12:00:00+00:00",
        clustering_algorithms=None,
        dim_reduction_algorithms=None,
        n_evaluations=None,
        cutoff_time=None,
        evaluator_ls=None,
        clustering_num=None,
        min_proportion=None,
        min_relative_proportion=None
    )

    config_arg = json.loads(args[6])
    assert config_arg["clustering_algorithms"] is None
    assert config_arg["cutoff_time"] is None


def test_build_subprocess_args_json_is_valid():
    args = build_subprocess_args(
        job_id="id",
        dataset_name="ds",
        columns=[{"name": "a", "type": "numeric"}],
        created_timestamp="ts",
        clustering_algorithms=["DBSCAN"],
        dim_reduction_algorithms=[],
        n_evaluations=5,
        cutoff_time=60,
        evaluator_ls=["dbScore"],
        clustering_num=(3, 7),
        min_proportion=0.1,
        min_relative_proportion=0.2
    )

    # Test ob JSON wirklich valide ist
    try:
        json.loads(args[4])
        json.loads(args[6])
    except json.JSONDecodeError:
        pytest.fail("JSON-Argumente sind nicht korrekt serialisiert")


def test_build_environment_sets_pythonpath():
    env = build_environment()
    assert isinstance(env, dict)
    assert env["PYTHONPATH"] == "/app"


def test_build_environment_preserves_existing_env(monkeypatch):
    monkeypatch.setenv("HOME", "/home/testuser")
    env = build_environment()
    assert env["HOME"] == "/home/testuser"


def test_build_environment_on_empty_env(monkeypatch):
    monkeypatch.setattr(os, "environ", {})  # Leeres env
    env = build_environment()
    assert env["PYTHONPATH"] == "/app"


@patch("workers.automl.automl_tasks.run_subprocess")
@patch("workers.automl.automl_tasks.build_environment")
@patch("workers.automl.automl_tasks.build_subprocess_args")
def test_run_autocluster_logic_success(mock_args, mock_env, mock_subprocess):
    mock_args.return_value = ["python", "script.py", "id", "..."]
    mock_env.return_value = {"PYTHONPATH": "/app"}
    mock_subprocess.return_value = {"status": "SUCCESS", "job_id": "test123"}

    result = run_autocluster_logic(
        job_id="test123",
        dataset_name="iris",
        columns=[{"name": "x", "type": "numeric"}],
        created_timestamp="2025-01-01T12:00:00+00:00",
        clustering_algorithms=["KMeans"]
    )

    assert result == {"status": "SUCCESS", "job_id": "test123"}
    mock_args.assert_called_once()
    mock_env.assert_called_once()
    mock_subprocess.assert_called_once()


@patch("workers.automl.automl_tasks.build_subprocess_args", side_effect=ValueError("bad config"))
def test_run_autocluster_logic_error(mock_args):
    result = run_autocluster_logic(
        job_id="err-job",
        dataset_name="faulty",
        columns=[],
        created_timestamp="2025-01-01T12:00:00+00:00"
    )

    assert result["status"] == "error"
    assert result["job_id"] == "err-job"
    assert "bad config" in result["error"]


@patch("workers.automl.automl_tasks.subprocess.Popen")
def test_run_subprocess_success(mock_popen):
    mock_proc = MagicMock()
    mock_stdout = MagicMock()
    mock_stdout.__enter__.return_value = iter(["Everything ok\n"])
    mock_proc.stdout = mock_stdout
    mock_proc.wait.return_value = 0
    mock_popen.return_value = mock_proc

    result = run_subprocess("test-id", ["python", "dummy.py"], {"PYTHONPATH": "/app"})

    assert result == {"status": "SUCCESS", "job_id": "test-id"}


@patch("workers.automl.automl_tasks.subprocess.Popen")
def test_run_subprocess_failure_without_error_line(mock_popen):
    mock_proc = MagicMock()
    mock_stdout = MagicMock()
    mock_stdout.__enter__.return_value = iter(["Step 1", "Step 2"])
    mock_proc.stdout = mock_stdout
    mock_proc.wait.return_value = 1
    mock_popen.return_value = mock_proc

    result = run_subprocess("fail-id", ["python", "dummy.py"], {"PYTHONPATH": "/app"})

    assert result["status"] == "error"
    assert "return code" in result["error"]