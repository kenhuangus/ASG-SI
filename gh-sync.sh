#!/usr/bin/env bash
set -euo pipefail

# 1. Ensure we are in a git repo
if ! git rev-parse --is-inside-work-tree >/dev/null 2>&1; then
  echo "Not inside a git repository."
  exit 1
fi

# 2. Check GitHub auth; if not logged in, run web-based login
if ! gh auth status >/dev/null 2>&1; then  # exits non‑zero when not authenticated [web:5]
  echo "You are not logged into GitHub. Opening web-based login..."
  gh auth login --web                        # default web flow for github.com [web:9]
  echo "Login completed."
fi

# 3. Ensure a remote repo is configured (default: origin)
remote_name="origin"

if ! git remote show "$remote_name" >/dev/null 2>&1; then  # non‑zero if remote missing [web:3]
  echo "No remote named '$remote_name' is configured."
  read -rp "Enter remote URL for '$remote_name' (e.g. git@github.com:USER/REPO.git): " remote_url
  if [ -z "$remote_url" ]; then
    echo "No URL provided, aborting."
    exit 1
  fi
  git remote add "$remote_name" "$remote_url"              # add remote [web:7]
  echo "Remote '$remote_name' set to '$remote_url'."
fi

# 4. Add and commit local changes if any
if ! git diff --quiet || ! git diff --cached --quiet; then
  echo "There are local changes."
  read -rp "Enter commit message (leave empty to skip commit and only push existing commits): " msg
  if [ -n "$msg" ]; then
    git add -A
    git commit -m "$msg"
  else
    echo "Skipping commit, will only push existing commits."
  fi
else
  echo "No local changes to commit."
fi

# 5. Push all branches and tags to remote
echo "Pushing all branches to $remote_name..."
git push "$remote_name" --all                           # push all branches [web:4]
echo "Pushing all tags to $remote_name..."
git push "$remote_name" --tags                          # push all tags [web:4]

echo "Done."
