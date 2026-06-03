# Deploying Dynamic DB Studio (Docker, single VPS)

The stack runs as two containers on one host: the Streamlit **app** and a
co-located **Ollama** server. Everything is wired by `docker-compose.yml`.

## Requirements

- A Linux VPS with Docker + Docker Compose v2
- **RAM:** ~8 GB recommended. `llama3.2:3b` needs ~4 GB resident; the embedding
  model and app add ~1–2 GB. 4 GB will swap and be slow.
- **Disk:** ~6 GB (model weights ~2.3 GB + image + data)
- Ports: `8501` (app). Ollama stays on the internal Docker network.

## First deploy

```bash
git clone <your-repo> && cd imswb
docker compose up -d --build
```

On first start, the `ollama-pull` service downloads `llama3.2:3b` and
`nomic-embed-text` (~2.3 GB) into the `ollama_models` volume, then exits. The
app is reachable once that's done:

```bash
docker compose logs -f ollama-pull      # watch the model download
curl -s -o /dev/null -w '%{http_code}\n' http://localhost:8501/_stcore/health
```

Open `http://<server>:8501`, create an account, and you're in.

## TLS / public access

Put a reverse proxy in front for HTTPS (the app speaks plain HTTP on 8501).
Minimal Caddy example (`Caddyfile`):

```
your.domain.com {
    reverse_proxy localhost:8501
}
```

Streamlit's WebSocket works through Caddy/nginx/Traefik out of the box. If you
use nginx, proxy `/_stcore/stream` with `Upgrade`/`Connection` headers.

## Data & backups

Two named volumes hold all state:

| Volume | Contents |
|--------|----------|
| `app_data` | per-user SQLite DBs + ChromaDB vector store (`/app/data`) |
| `ollama_models` | pulled model weights |

Back up `app_data` regularly:

```bash
docker run --rm -v imswb_app_data:/data -v "$PWD":/backup alpine \
    tar czf /backup/app_data-$(date +%F).tar.gz -C /data .
```

Restore by extracting the tarball back into the volume.

## Operations

```bash
docker compose ps                  # status
docker compose logs -f app         # app logs
docker compose up -d --build       # deploy a new version (data volumes persist)
docker compose restart app         # restart just the app
docker compose down                # stop (volumes are kept)
```

### Change the chat model

Edit `DEFAULT_MODEL` in `ai/query_parser.py`, add a `docker compose exec ollama
ollama pull <model>`, and restart. Note: the tool-calling prompts are tuned for
`llama3.2:3b`; other models may behave differently.

## Hardening notes (already applied)

- `.streamlit/config.toml`: headless, XSRF protection on, error details hidden
  from end users, 50 MB upload cap.
- Receipt HTML is escaped (no stored XSS); per-user table isolation enforced at
  the DB layer; passwords are PBKDF2-SHA256 (100k iterations).

### Recommended additions for a public deployment

- Terminate TLS at the proxy and force HTTPS.
- Add rate limiting / basic auth at the proxy if the instance is internet-facing.
- Set up off-host backups of the `app_data` tarball.
