from types import SimpleNamespace

from click.testing import CliRunner

from whisper_daemon import __main__ as cli
from whisper_daemon import env_file
from whisper_daemon.env_file import read_env_file


def test_env_file_parses_exports_quotes_and_comments(tmp_path):
    path = tmp_path / ".env"
    path.write_text(
        "# Hugging Face\n"
        "HF_TOKEN=hf_plain\n"
        "export QUOTED=\"two words\"\n"
        "SINGLE='x=y'\n"
        "\n"
        "not a line\n"
        "1BAD=skipped\n"
        "EMPTY=\n",
        encoding="utf-8",
    )
    assert read_env_file(path) == {"HF_TOKEN": "hf_plain", "QUOTED": "two words", "SINGLE": "x=y", "EMPTY": ""}


def test_missing_env_file_yields_nothing(tmp_path):
    assert read_env_file(tmp_path / "absent.env") == {}


def _prepare(monkeypatch, tmp_path, env_text):
    path = tmp_path / ".env"
    path.write_text(env_text, encoding="utf-8")
    monkeypatch.setattr(env_file, "PROJECT_ENV_FILE", path)
    calls = []
    monkeypatch.setattr(cli.subprocess, "run", lambda args, env: calls.append(env) or SimpleNamespace(returncode=0))
    result = CliRunner().invoke(cli.cli, ["models", "prepare", "--diarization"])
    assert result.exit_code == 0, result.output
    return calls[0]


def test_model_prepare_takes_the_token_from_the_project_env(monkeypatch, tmp_path):
    monkeypatch.delenv("HF_TOKEN", raising=False)
    env = _prepare(monkeypatch, tmp_path, "HF_TOKEN=hf_from_file\nHF_HUB_OFFLINE=1\n")
    assert env["HF_TOKEN"] == "hf_from_file"
    # Provisioning must stay online whatever the file says.
    assert env["HF_HUB_OFFLINE"] == "0"
    assert env["WHISPER_PREPARE_MODELS"] == "1"


def test_model_prepare_prefers_the_process_environment(monkeypatch, tmp_path):
    monkeypatch.setenv("HF_TOKEN", "hf_from_shell")
    env = _prepare(monkeypatch, tmp_path, "HF_TOKEN=hf_from_file\n")
    assert env["HF_TOKEN"] == "hf_from_shell"
