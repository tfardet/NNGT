#!/bin/sh
ensure_ssh() {
  # ensure our SSH allows us to talk to GitHub.
  # Even if we ended up talking to a malicious host, all we'd do is push our
  # open source code to them, so no real concern there.
  KNOWN_HOSTS_FILE="~/.ssh/known_hosts"
  touch "$KNOWN_HOSTS_FILE"
  chmod 0600 "$KNOWN_HOSTS_FILE"
  ssh-keyscan github.com > "$KNOWN_HOSTS_FILE"
}

commit_on_branch() {
  # Returns whether the commit $1 is on the tip of the branch $2
  # This can be used with HEAD and master to check that you're on the
  # latest commit of the main branch
  git branch --format '%(refname:lstrip=2)' --contains "$1" | grep "$2"
}
