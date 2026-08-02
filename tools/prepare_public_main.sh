#!/usr/bin/env bash
#
# Prepare a reproducible public `main` candidate from the canonical `test`
# branch. This script NEVER commits and NEVER pushes.
#
# Expected layout:
# .../ZeSolver/ -> worktree on branch test
# .../ZeSolver-main/ -> worktree on branch main
#
# Usage:
# tools/prepare_public_main.sh
# tools/prepare_public_main.sh --yes
# tools/prepare_public_main.sh --dry-run
# tools/prepare_public_main.sh --main-worktree /custom/path/ZeSolver-main
#
set -Eeuo pipefail

SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
TEST_ROOT="$(cd -- "${SCRIPT_DIR}/.." && pwd)"
DEFAULT_MAIN_WORKTREE="$(cd -- "${TEST_ROOT}/.." && pwd)/ZeSolver-main"

MAIN_WORKTREE="${ZESOLVER_MAIN_WORKTREE:-$DEFAULT_MAIN_WORKTREE}"
ASSUME_YES=0
DRY_RUN=0

usage() {
 cat <<'EOF'
Prepare the generated public ZeSolver `main` candidate from branch `test`.

Options:
 --yes Synchronize without asking for confirmation.
 --dry-run Build and validate the public tree, but do not
 synchronize the main worktree.
 --main-worktree PATH Override the default ../ZeSolver-main path.
 -h, --help Show this help.

Safety:
 - requires a clean `test` worktree;
 - requires local test HEAD == origin/test;
 - requires a clean `main` worktree;
 - requires local main == origin/main;
 - never commits;
 - never pushes;
 - never performs git merge or force push.
EOF
}

die() {
 printf 'ERROR: %s\n' "$*" >&2
 exit 1
}

note() {
 printf '\n==> %s\n' "$*"
}

assert_no_python_bytecode() {
 local root="$1"
 local found

 found="$(
 find "$root" \
  \( -type d -name __pycache__ -o -type f \( -name '*.pyc' -o -name '*.pyo' \) \) \
  -print
 )"
 [[ -z "$found" ]] || {
  printf '%s\n' "$found" >&2
  die "Python bytecode artifacts found under: $root"
 }
}

while (($#)); do
 case "$1" in
 --yes)
 ASSUME_YES=1
 shift
 ;;
 --dry-run)
 DRY_RUN=1
 shift
 ;;
 --main-worktree)
 (($# >= 2)) || die "--main-worktree requires a path"
 MAIN_WORKTREE="$2"
 shift 2
 ;;
 -h|--help)
 usage
 exit 0
 ;;
 *)
 die "unknown option: $1"
 ;;
 esac
done

for command in git rsync awk find; do
 command -v "$command" >/dev/null 2>&1 || die "required command not found: $command"
done

if [[ -x "${TEST_ROOT}/.venv/bin/python" ]]; then
 PYTHON_BIN="${TEST_ROOT}/.venv/bin/python"
elif command -v python3 >/dev/null 2>&1; then
 PYTHON_BIN="$(command -v python3)"
elif command -v python >/dev/null 2>&1; then
 PYTHON_BIN="$(command -v python)"
else
 die "Python was not found"
fi

BUILDER="${TEST_ROOT}/tools/build_public_tree.py"
MANIFEST="${TEST_ROOT}/packaging/public_manifest.txt"

[[ -f "$BUILDER" ]] || die "public tree builder not found: $BUILDER"
[[ -f "$MANIFEST" ]] || die "public manifest not found: $MANIFEST"
[[ -e "${MAIN_WORKTREE}/.git" ]] || die "main worktree not found: $MAIN_WORKTREE"

note "Checking canonical test worktree"

TEST_BRANCH="$(git -C "$TEST_ROOT" branch --show-current)"
[[ "$TEST_BRANCH" == "test" ]] || die "expected branch test in $TEST_ROOT, got: $TEST_BRANCH"

TEST_STATUS="$(git -C "$TEST_ROOT" status --porcelain=v1)"
[[ -z "$TEST_STATUS" ]] || {
 printf '%s\n' "$TEST_STATUS" >&2
 die "test worktree is not clean"
}

git -C "$TEST_ROOT" fetch origin

TEST_HEAD="$(git -C "$TEST_ROOT" rev-parse HEAD)"
ORIGIN_TEST="$(git -C "$TEST_ROOT" rev-parse origin/test)"
[[ "$TEST_HEAD" == "$ORIGIN_TEST" ]] || die \
 "test is not synchronized with origin/test; commit and push test first"

printf 'test HEAD: %s\n' "$TEST_HEAD"

note "Checking public main worktree"

MAIN_BRANCH="$(git -C "$MAIN_WORKTREE" branch --show-current)"
[[ "$MAIN_BRANCH" == "main" ]] || die \
 "expected branch main in $MAIN_WORKTREE, got: $MAIN_BRANCH"

MAIN_STATUS="$(git -C "$MAIN_WORKTREE" status --porcelain=v1)"
[[ -z "$MAIN_STATUS" ]] || {
 printf '%s\n' "$MAIN_STATUS" >&2
 die "main worktree already contains uncommitted changes"
}

git -C "$MAIN_WORKTREE" fetch origin

read -r MAIN_LEFT MAIN_RIGHT < <(
 git -C "$MAIN_WORKTREE" rev-list --left-right --count main...origin/main
)

[[ "$MAIN_LEFT" == "0" && "$MAIN_RIGHT" == "0" ]] || die \
 "local main and origin/main have diverged or are not synchronized: ${MAIN_LEFT} ${MAIN_RIGHT}"

TMP_ROOT="$(mktemp -d "${TMPDIR:-/tmp}/zesolver-public-main.XXXXXX")"
PUBLIC_TREE="${TMP_ROOT}/public"
REPORT="${TMP_ROOT}/public-tree-report.json"
PYCACHE_ROOT="${TMP_ROOT}/pycache"

export PYTHONPYCACHEPREFIX="$PYCACHE_ROOT"
export PYTHONDONTWRITEBYTECODE=1

cleanup() {
 rm -rf -- "$TMP_ROOT"
}
trap cleanup EXIT

note "Building public tree from test"

"$PYTHON_BIN" "$BUILDER" \
 --source "$TEST_ROOT" \
 --destination "$PUBLIC_TREE" \
 --manifest "$MANIFEST" \
 --report "$REPORT"

[[ -f "${PUBLIC_TREE}/ZESOLVER_SOURCE_REVISION" ]] || die \
 "generated tree is missing ZESOLVER_SOURCE_REVISION"

REVISION_SHA="$(
 awk -F= '$1 == "sha" { print $2 }' "${PUBLIC_TREE}/ZESOLVER_SOURCE_REVISION"
)"
[[ "$REVISION_SHA" == "$TEST_HEAD" ]] || die \
 "generated source revision does not match test HEAD"

note "Validating generated public tree"

"$PYTHON_BIN" -m compileall -q \
 "${PUBLIC_TREE}/zesolver" \
 "${PUBLIC_TREE}/zeblindsolver" \
 "${PUBLIC_TREE}/zewcs290"

PYTHONPATH="$PUBLIC_TREE" "$PYTHON_BIN" - <<'PY'
import zeblindsolver
import zesolver
import zewcs290

print("PUBLIC_IMPORT_SMOKE_OK")
PY

(
 cd "$PUBLIC_TREE"
 PYTHONPATH="$PUBLIC_TREE" "$PYTHON_BIN" \
 -m zesolver.gpu_diagnostic --json >/dev/null

 PYTHONPATH="$PUBLIC_TREE" "$PYTHON_BIN" \
 zesolver.py --help >/dev/null
)

FORBIDDEN_PATHS=(
 "tests"
 "tools"
 "reports"
 "docs/stabilization"
 "docs/architecture"
 "packaging"
 ".github"
 "AGENT.md"
 "memory.md"
 "followup.md"
 "structure.txt"
)

for path in "${FORBIDDEN_PATHS[@]}"; do
 [[ ! -e "${PUBLIC_TREE}/${path}" ]] || die \
 "forbidden path found in generated public tree: $path"
done

assert_no_python_bytecode "$PUBLIC_TREE"

printf 'Generated report: %s\n' "$REPORT"
cat "${PUBLIC_TREE}/ZESOLVER_SOURCE_REVISION"

if ((DRY_RUN)); then
 note "Dry run completed successfully"
 printf 'No file was synchronized to main.\n'
 exit 0
fi

if ((!ASSUME_YES)); then
 printf '\nThe validated public tree will replace the contents of:\n %s\n' "$MAIN_WORKTREE"
 printf 'The .git worktree metadata will be preserved.\n'
 read -r -p "Prepare the main candidate now? [y/N] " answer
 case "$answer" in
 y|Y|yes|YES|oui|OUI) ;;
 *)
 printf 'Cancelled. No change was made to main.\n'
 exit 0
 ;;
 esac
fi

note "Synchronizing generated tree into main worktree"

rsync -a --delete \
 --exclude='.git' \
 "${PUBLIC_TREE}/" \
 "${MAIN_WORKTREE}/"

note "Checking prepared main candidate"

[[ "$(git -C "$MAIN_WORKTREE" branch --show-current)" == "main" ]] || die \
 "main worktree changed branch unexpectedly"

git -C "$MAIN_WORKTREE" diff --check

MAIN_REVISION_SHA="$(
 awk -F= '$1 == "sha" { print $2 }' \
 "${MAIN_WORKTREE}/ZESOLVER_SOURCE_REVISION"
)"
[[ "$MAIN_REVISION_SHA" == "$TEST_HEAD" ]] || die \
 "main candidate provenance does not match test HEAD"

for path in "${FORBIDDEN_PATHS[@]}"; do
 [[ ! -e "${MAIN_WORKTREE}/${path}" ]] || die \
 "forbidden path found in prepared main candidate: $path"
done

assert_no_python_bytecode "$MAIN_WORKTREE"

note "Prepared main candidate summary"

git -C "$MAIN_WORKTREE" status -sb
printf '\n'
git -C "$MAIN_WORKTREE" diff --stat
printf '\nSource commit: %s\n' "$TEST_HEAD"
printf 'Main worktree: %s\n' "$MAIN_WORKTREE"

SHORT_SHA="${TEST_HEAD:0:7}"

cat <<EOF

MAIN CANDIDATE READY — nothing has been committed or pushed.

Review the changes:

 cd "$MAIN_WORKTREE"
 git status -sb
 git diff --check
 git diff --stat
 git diff

After Tristan explicitly approves the candidate:

 cd "$MAIN_WORKTREE"
 git add -A
 git diff --cached --check
 git diff --cached --stat
 git commit -m "Publish minimal ZeSolver distribution from test $SHORT_SHA"
 git push origin main
 git status -sb

To abandon this uncommitted candidate:

 cd "$MAIN_WORKTREE"
 git reset --hard HEAD
 git clean -fd

EOF
