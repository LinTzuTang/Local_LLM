#!/bin/sh
# Share the platform with colleagues: install Caddy (HTTPS reverse proxy with login), turn on
# accounts, and create the first admin. Run ON the GPU machine:
#   ./scripts/remote/setup_caddy.sh lab.example.com
# Before that, IT has to give the machine a static external IP, point the domain's DNS at it,
# and open TCP 80 and 443 (Caddy's certificate check and HTTPS) in the firewall.
set -eu
PROJECT_DIR=$(CDPATH= cd -- "$(dirname -- "$0")/../.." && pwd)
cd "$PROJECT_DIR"
DOMAIN="${1:?usage: setup_caddy.sh <domain, e.g. lab.example.com>}"
export PATH="$HOME/.local/bin:$PATH"

if ! command -v caddy >/dev/null; then
  # Caddy's official Debian/Ubuntu repository (https://caddyserver.com/docs/install).
  sudo apt-get install -y -q debian-keyring debian-archive-keyring apt-transport-https curl gpg
  curl -1sLf 'https://dl.cloudsmith.io/public/caddy/stable/gpg.key' \
    | sudo gpg --dearmor --yes -o /usr/share/keyrings/caddy-stable-archive-keyring.gpg
  curl -1sLf 'https://dl.cloudsmith.io/public/caddy/stable/debian.deb.txt' \
    | sudo tee /etc/apt/sources.list.d/caddy-stable.list >/dev/null
  sudo apt-get update -q
  sudo apt-get install -y -q caddy
fi

sed "s/^LAB_DOMAIN {/$DOMAIN {/" deploy/Caddyfile | sudo tee /etc/caddy/Caddyfile >/dev/null
sudo caddy validate --config /etc/caddy/Caddyfile --adapter caddyfile
sudo systemctl enable caddy >/dev/null 2>&1 || true
sudo systemctl reload-or-restart caddy

touch .env
set_env() {  # set_env KEY VALUE: replace or append one line of .env
  if grep -q "^$1=" .env; then
    sed -i "s|^$1=.*|$1=$2|" .env
  else
    echo "$1=$2" >> .env
  fi
}
set_env LOCAL_LLM_AUTH 1
set_env LOCAL_LLM_PUBLIC_URL "https://$DOMAIN"

if ! uv run local-agent users list | grep -q " admin "; then
  echo "Create the first admin account:"
  printf "Name: "
  read -r name
  uv run local-agent users add "$name" --admin
fi
echo
echo "Restart the platform so it picks up LOCAL_LLM_AUTH: ./scripts/tmux_down.sh && ./scripts/tmux_up.sh"
echo "Then open https://$DOMAIN . Add colleagues with: uv run local-agent users add <name>"
