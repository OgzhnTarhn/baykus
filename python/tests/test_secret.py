import sys

import pytest

from baykus.data.secret import run_secret

PY = f'"{sys.executable}"'


def test_run_secret_success_is_silent(capsys):
    assert run_secret(f"{PY} -c \"print('SECRET123')\"", label="indirme") == 0
    out = capsys.readouterr()
    assert "SECRET123" not in out.out + out.err


def test_run_secret_never_leaks_on_failure(capsys):
    cmd = f"{PY} -c \"import sys; print('SECRET123'); sys.stderr.write('SECRET123'); sys.exit(3)\""
    assert run_secret(cmd, label="indirme") == 3
    out = capsys.readouterr()
    assert "SECRET123" not in out.out + out.err
    assert "indirme: başarısız (çıkış 3)" in out.out


def test_run_secret_bad_command_does_not_echo_it(capsys):
    rc = run_secret("SECRET123_no_such_command --cookie 'a=b'", label="indirme")
    out = capsys.readouterr()
    assert rc != 0 and "SECRET123" not in out.out + out.err


def test_run_secret_rejects_empty():
    with pytest.raises(ValueError):
        run_secret("   ", label="indirme")
