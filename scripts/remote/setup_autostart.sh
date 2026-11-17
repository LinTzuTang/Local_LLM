#!/bin/sh
# Start the platform whenever the GPU machine boots (scheduled start, a manual start, or a restart
# after a Spot preemption), so the shared address works without anyone logging in. Run ON the
# GPU machine, as the user that owns the project:
#   ./scripts/remote/setup_autostart.sh            install and enable (takes effect at next boot)
#   ./scripts/remote/setup_autostart.sh --remove   turn it off again
# It installs a systemd service "local-llm" that runs scripts/tmux_up.sh as this user, so the
# tmux session, logs and `tmux attach -t local_llm` are the same as when started by hand. A
# session that is already running is left alone; `sudo systemctl start local-llm` starts it now.
set -eu
PROJECT_DIR=$(CDPATH= cd -- "$(dirname -- "$0")/../.." && pwd)
UNIT=/etc/systemd/system/local-llm.service

if [ "${1:-}" = --remove ]; then
  sudo systemctl disable local-llm >/dev/null 2>&1 || true
  sudo rm -f "$UNIT"
  sudo systemctl daemon-reload
  echo "Removed: the platform no longer starts at boot."
  exit 0
fi

# After the guest agent: OS Login users are looked up through the metadata server, so the
# service user only exists once the network is up. Retried in case that is still too early.
sudo tee "$UNIT" >/dev/null <<EOF
[Unit]
Description=Local Agent Lab (platform, model gateway, login service) in tmux
Wants=network-online.target
After=network-online.target google-guest-agent.service

[Service]
Type=oneshot
RemainAfterExit=yes
User=$(id -un)
WorkingDirectory=$PROJECT_DIR
Environment=PATH=$HOME/.local/bin:/usr/local/sbin:/usr/local/bin:/usr/sbin:/usr/bin:/sbin:/bin
ExecStart=$PROJECT_DIR/scripts/tmux_up.sh
ExecStop=$PROJECT_DIR/scripts/tmux_down.sh
TimeoutStartSec=300
Restart=on-failure
RestartSec=30

[Install]
WantedBy=multi-user.target
EOF
sudo systemd-analyze verify "$UNIT"
sudo systemctl daemon-reload
sudo systemctl enable local-llm
echo "Enabled: the platform starts at every boot (service local-llm)."
echo "Status: systemctl status local-llm   Logs: journalctl -u local-llm"
