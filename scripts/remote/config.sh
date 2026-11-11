# Shared settings for the scripts in scripts/remote/ (sourced, not run).
# The machine's details live in scripts/remote/config.local.sh (git-ignored, this repo is public):
#   GCP_INSTANCE=my-gpu-vm
#   GCP_PROJECT=my-project
#   GCP_ZONE=us-west1-c
#   REMOTE_USER=me_example_com      # OS Login user (gcloud compute os-login describe-profile)
# Any of them can also be set in the environment, e.g. GCP_INSTANCE=other-vm ./scripts/remote/up.sh
_here="${HERE:?set HERE to scripts/remote before sourcing config.sh}"  # every caller sets it
if [ -f "$_here/config.local.sh" ]; then
  # Environment values win over the file.
  _env_instance="${GCP_INSTANCE:-}" _env_project="${GCP_PROJECT:-}" _env_zone="${GCP_ZONE:-}" _env_user="${REMOTE_USER:-}"
  . "$_here/config.local.sh"
  GCP_INSTANCE="${_env_instance:-${GCP_INSTANCE:-}}"
  GCP_PROJECT="${_env_project:-${GCP_PROJECT:-}}"
  GCP_ZONE="${_env_zone:-${GCP_ZONE:-}}"
  REMOTE_USER="${_env_user:-${REMOTE_USER:-}}"
fi
for _v in GCP_INSTANCE GCP_PROJECT GCP_ZONE REMOTE_USER; do
  eval "[ -n \"\${$_v:-}\" ]" || {
    echo "$_v is not set: put the machine's details in scripts/remote/config.local.sh (see config.sh)." >&2
    exit 1
  }
done
# The key registered with `gcloud compute os-login ssh-keys add`.
SSH_KEY="${SSH_KEY:-$HOME/.ssh/google_compute_engine}"
REMOTE_DIR="${REMOTE_DIR:-Local_LLM}"  # relative to the remote home

gc() { gcloud compute instances "$@" "$GCP_INSTANCE" --project="$GCP_PROJECT" --zone="$GCP_ZONE"; }

vm_status() { gc describe --format="value(status)"; }

vm_ip() { gc describe --format="value(networkInterfaces[0].accessConfigs[0].natIP)"; }

# The external IP changes on every start (and an old one may now belong to another host), so
# the host key is checked under the instance name instead of the IP.
ssh_opts() {
  printf '%s\n' -i "$SSH_KEY" -o "HostKeyAlias=$GCP_INSTANCE" -o StrictHostKeyChecking=accept-new \
    -o ServerAliveInterval=30
}

remote() {  # remote <command...>: run a command on the VM
  ip=$(vm_ip)
  # shellcheck disable=SC2046
  ssh $(ssh_opts) "$REMOTE_USER@$ip" "$@"
}
