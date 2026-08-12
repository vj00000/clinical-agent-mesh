# Setting up on macOS (Apple Silicon)

## 1. Move the code across

**Don't copy-paste files.** Use a git bundle — a single file containing the entire
repository *including its commit history*, which is worth keeping: the commit
messages explain why each decision was made, and that is most of the project's
value when you come back to it in three months.

On the Linux machine:

```bash
cd ~/claudeCode/clinical-agent-mesh
git bundle create ~/clinical-agent-mesh.bundle --all
```

Copy that one file to the Mac (USB, AirDrop, cloud), then:

```bash
git clone ~/Downloads/clinical-agent-mesh.bundle clinical-agent-mesh
cd clinical-agent-mesh
git log --oneline          # full history is here
```

If you'd rather not use a bundle, push to a private GitHub repo and clone it. Plain
file copying works too but loses the history — avoid it if you can.

Either way, **`.env` will not travel** (it is gitignored, and secrets should never be
committed). You will create it fresh on the Mac.

## 2. Run the setup script

```bash
chmod +x scripts/setup-mac.sh
./scripts/setup-mac.sh
```

It is safe to re-run — every step checks before acting. It will:

1. Install `uv` (no sudo, no Homebrew needed)
2. Install Python 3.12 via uv — you do not need a system Python
3. Check Docker Desktop is running, and tell you how to install it if not
4. Install dependencies, then the optional `rerank` extra (skip with `SKIP_RERANK=1`)
5. Create `.env` from `.env.example`
6. Start Chroma and Postgres and wait for Chroma's heartbeat
7. Run the tests, ruff, and mypy

Then add your key:

```bash
$EDITOR .env       # OPENAI_API_KEY=sk-...
make ingest
```

## 3. What differs on macOS

| Thing | Linux | macOS arm64 |
|---|---|---|
| torch | pinned to the `pytorch-cpu` index (avoids 2.7GB of CUDA libraries) | standard PyPI wheel — already CPU/MPS, no CUDA payload |
| Chroma image | `linux/amd64` | native `linux/arm64` variant exists, no emulation |
| Docker | daemon via the system | needs **Docker Desktop** running |
| venv size | ~1.4GB with rerank | similar |

The torch difference is handled automatically: `pyproject.toml` carries
`marker = "sys_platform == 'linux'"` on the CPU index, because that index publishes
no macOS arm64 wheels. The lockfile holds both resolutions, so the same `uv.lock`
works on both machines.

## 4. Troubleshooting

**`uv: command not found` in a new terminal**
Add to `~/.zshrc`: `export PATH="$HOME/.local/bin:$PATH"`

**`Cannot connect to the Docker daemon`**
Docker Desktop is installed but not started. Open it and wait for the whale icon.

**Integration tests skip themselves**
Expected when Chroma is not running. `make up`, then re-run.

**`make ingest` exits with "OPENAI_API_KEY is not set"**
Working as designed. Put the key in `.env`.

**Rerank tests fail to import `sentence_transformers`**
The extra was skipped. `uv sync --extra rerank`.

**Port 8001 or 5432 already in use**
Something else is bound. Change the host side of the mapping in `compose.yaml`
(e.g. `"8002:8000"`) and set `CHROMA_PORT=8002` in `.env`.

## 5. Sanity check

```bash
make check          # lint + strict types + 149 fast tests, no network, no API key
```

If that passes, the environment is correct. Everything else depends only on your key
and Docker.
