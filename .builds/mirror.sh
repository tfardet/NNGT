#!/bin/sh
set -eu

if [ "$#" -lt 1 ]; then
  echo "usage: mirror.sh <github_ssh_clone_url>" 1>&2
fi

basedir=$(dirname "$(realpath "$0")")

GITHUB_MIRROR_URL="$1"

# determine current commit
GIT_COMMIT="$(git rev-parse HEAD)"

# shellcheck source=.builds/lib.sh
. "$basedir/lib.sh"

# if our current commit is on the main branch, push to GitHub
mirror_branch="main"

if commit_on_branch "$GIT_COMMIT" "$mirror_branch"; then
  ensure_ssh
  remote_name="mirror"
  git remote add "$remote_name" "$GITHUB_MIRROR_URL" && \
  git checkout "$mirror_branch" && \
  git push -f -u "$remote_name" "$mirror_branch"
fi
