#!/bin/bash
 
usage() {
    cat <<EOF
Usage: $0 <closed_path> <closed_branch> <open_path>
 
Copies files changed on <closed_branch> (vs main) from the closed-source
repo to the open-source repo, preserving directory structure.
 
Arguments:
  closed_path     Path to the closed-source repo
  closed_branch   Branch in the closed-source repo to sync from
  open_path       Path to the open-source repo
 
Example:
  $0 ~/code/closed-source feature-x ~/code/open-source
EOF
}
 
if [[ "$1" == "-h" || "$1" == "--help" ]]; then
    usage
    exit 0
fi
 
if [[ $# -ne 3 ]]; then
    usage
    exit 1
fi
 
CLOSED_PATH="$1"
CLOSED_BRANCH="$2"
OPEN_PATH="$3"
 
cd "$CLOSED_PATH"
git checkout "$CLOSED_BRANCH"
cp --parents $(git diff main --name-only) "$OPEN_PATH"
 
