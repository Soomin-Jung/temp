# Buzz Linux X-session/NFS operator template (source from Bash).
# Replace the path, hostname and optional model defaults with approved values.
# No user identity, OAuth token, Nostr nsec or LLM API key belongs in this file.
export BUZZ_NFS_ROOT="/nfs/apps/buzz"
export BUZZ_RELAY_URL="wss://buzz.hpc.example.internal"
export BUZZ_RELAY_HTTP="https://buzz.hpc.example.internal"

# X11 only; remove or adjust for Wayland sessions.
export GDK_BACKEND="x11"

# Optional defaults ONLY when using buzz-agent (not universally applicable to
# Goose, Codex or Claude Code). Define per-user credentials outside this file.
# export BUZZ_AGENT_PROVIDER="openai"
# export OPENAI_COMPAT_BASE_URL="https://llm.hpc.example.internal/v1"
# export OPENAI_COMPAT_MODEL="<deployed-model-name>"
# export OPENAI_COMPAT_API="chat"

# Shell function instead of an executable named 'buzz' on PATH:
# agent subprocesses must still be able to invoke the real Buzz JSON CLI.
buzz() {
  command bash "${BUZZ_NFS_ROOT}/bin/buzz-desktop" "$@"
}
