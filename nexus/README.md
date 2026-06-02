# Nexus Integration

Routes this project's Python dependency management through a
[Sonatype Nexus](https://www.sonatype.com/products/nexus-repository) repository —
a PyPI **group** repo that proxies `pypi.org` and serves any privately published
packages from a single index URL.

## Files

| File | Purpose |
|------|---------|
| `pip.conf` | Points `pip install` at the Nexus PyPI proxy. Placeholders only — no secrets. |
| `.pypirc.template` | Template for publishing wheels/sdists to a Nexus **hosted** repo via `twine`. Copy → fill in → keep out of git. |

## 1. Install dependencies through Nexus

Edit `nexus/pip.conf` and replace `nexus.example.com` with your server, then:

```bash
# project-local, current shell only
export PIP_CONFIG_FILE="$(pwd)/nexus/pip.conf"
pip install -r requirements.txt
```

To make it permanent for your user, copy `pip.conf` to:
- Linux/macOS: `~/.config/pip/pip.conf`
- Windows: `%APPDATA%\pip\pip.ini`

> Using `uv`? Point it at the same index:
> ```bash
> export UV_INDEX_URL="https://nexus.example.com/repository/pypi-group/simple"
> uv pip install -r requirements.txt
> ```

## 2. Publish a package to Nexus

```bash
cp nexus/.pypirc.template ~/.pypirc        # then edit in real credentials
python -m build                             # produces dist/*.whl + dist/*.tar.gz
twine upload -r nexus dist/*
```

Or keep credentials in env vars and skip the file entirely:

```bash
TWINE_USERNAME="$NEXUS_USER" \
TWINE_PASSWORD="$NEXUS_TOKEN" \
twine upload \
  --repository-url https://nexus.example.com/repository/pypi-hosted/ \
  dist/*
```

## Security

- The real `~/.pypirc` and any `.pypirc` / `.netrc` in the repo are **gitignored**.
- Only `*.template` files (placeholders, no secrets) are tracked.
- Prefer a Nexus **user token** over a raw password where supported.
