#!/usr/bin/env bash
# Coverage of the lines a branch adds: crosses cargo-llvm-cov's lcov report with
# `git diff -U0 <base>...HEAD` and fails below COVERAGE_NEW_MIN percent (default 99, the
# HappyDev new-code gate). Same idea as diff-cover, without its Python dependency.
# Shared verbatim with the owner's other Rust workspaces; keep the copies in sync.
#
#   scripts/coverage-new.sh                     run the workspace tests, compare with origin/main
#   scripts/coverage-new.sh --base <ref>        compare with another ref
#   scripts/coverage-new.sh --lcov F --diff F --root DIR   check existing files (tests)
#
# Only `*.rs` lines the report instruments count: comments, blank lines and declarations
# have no line record, so they neither help nor hurt. Files with added lines that are not
# in the report at all are listed for the reviewer: the report cannot tell a file of
# declarations (a `mod.rs`, serde structs) from one no test compiles. The diff only sees
# commits, so uncommitted `.rs` changes are refused. Exit 0 at or above the minimum, 1
# below it, 2 on a usage error.
set -euo pipefail

usage='usage: coverage-new.sh [--base <ref>] [--lcov <file> --diff <file> --root <dir>]'
min=${COVERAGE_NEW_MIN:-99}
base='' lcov='' diff='' root=''

fail() {
  printf 'coverage-new.sh: %s\n' "$1" >&2
  exit 2
}

while (($# > 0)); do
  case $1 in
    --base | --lcov | --diff | --root)
      (($# >= 2)) || { echo "$usage" >&2; exit 2; }
      declare "${1#--}=$2"
      shift 2
      ;;
    *) echo "$usage" >&2; exit 2 ;;
  esac
done

[[ $min =~ ^[0-9]+(\.[0-9]+)?$ ]] || fail "COVERAGE_NEW_MIN must be a number, got $min"

if [[ -n $lcov || -n $diff || -n $root ]]; then
  [[ -n $lcov && -n $diff && -n $root && -z $base ]] || { echo "$usage" >&2; exit 2; }
else
  # Everything that can be refused is refused before the tests run, which takes minutes.
  base=${base:-origin/main}
  if [[ $base == -* ]] || ! git rev-parse --verify --quiet "$base^{commit}" > /dev/null; then
    fail "--base must name a commit, got $base"
  fi
  root=$(git rev-parse --show-toplevel)
  dirty=$(git -C "$root" status --porcelain -- '*.rs')
  [[ -z $dirty ]] || fail "commit or stash the .rs changes first; the diff only sees commits:
$dirty"
  lcov=$root/target/llvm-cov/lcov.info
  diff=$(mktemp)
  trap 'rm -f "$diff"' EXIT
  mkdir -p "$(dirname "$lcov")"
  cargo llvm-cov --workspace --lcov --output-path "$lcov" >&2
  # Fixed prefixes, unquoted paths and no external diff, whatever the user's git config.
  git -C "$root" -c core.quotePath=false diff --no-ext-diff --src-prefix=a/ --dst-prefix=b/ \
    -U0 --no-color "$base...HEAD" -- '*.rs' > "$diff"
fi

awk -v root="${root%/}/" -v min="$min" -v diff_file="$diff" '
  # The diff comes first: record every added line as "path:line".
  FILENAME == diff_file {
    # Inside a hunk body, count its lines down so an added line that starts with "++ "
    # (printed "+++ ") is never read as a file header.
    if (left > 0) {
      c = substr($0, 1, 1)
      if (c == "+") {
        added[file ":" line] = 1
        if (!(file in has_added)) { has_added[file] = 1; added_files[++n_files] = file }
        line++
        left--
      } else if (c == "-") {
        left--
      }
      next
    }
    if ($0 ~ /^\+\+\+ /) {
      # git ends the header with a tab when the path has a space.
      file = substr($0, 5)
      sub(/\t$/, "", file)
      sub(/^b\//, "", file)
    } else if ($0 ~ /^@@ /) {
      # @@ -a[,b] +c[,d] @@: b removed and d added lines follow, from line c; counts
      # default to 1.
      split($2, minus, ",")
      split($3, plus, ",")
      line = substr(plus[1], 2) + 0
      left = ((2 in minus) ? minus[2] + 0 : 1) + ((2 in plus) ? plus[2] + 0 : 1)
    }
    next
  }
  /^SF:/ {
    file = substr($0, 4)
    if (index(file, root) == 1) file = substr(file, length(root) + 1)
    in_report[file] = 1
    next
  }
  /^DA:/ {
    # DA:line,hits[,checksum]; a file can have several records, so keep the highest hits.
    split(substr($0, 4), da, ",")
    key = file ":" (da[1] + 0)
    if (!(key in added)) next
    if (!(key in hits)) { order[++n_keys] = key; key_file[n_keys] = file; key_line[n_keys] = da[1] + 0; hits[key] = 0 }
    if (da[2] + 0 > hits[key]) hits[key] = da[2] + 0
  }
  END {
    for (i = 1; i <= n_keys; i++) {
      total++
      if (hits[order[i]] > 0) { covered++; continue }
      # Uncovered lines arrive in order per file; merge consecutive ones into ranges.
      if (key_file[i] == last_file && key_line[i] == last_line + 1) {
        ranges[n_ranges] = key_file[i] ":" first_line "-" key_line[i]
      } else {
        first_line = key_line[i]
        ranges[++n_ranges] = key_file[i] ":" first_line
      }
      last_file = key_file[i]
      last_line = key_line[i]
    }
    for (i = 1; i <= n_files; i++) {
      if (!(added_files[i] in in_report)) missing[++n_missing] = added_files[i]
    }
    if (total == 0) {
      print "new lines: 0"
    } else {
      # Truncated, not rounded, so 98.996 % never prints as 99.00 % next to a FAIL.
      printf "new lines: %d, covered: %d (%.2f %%)\n", total, covered, int(covered * 10000 / total) / 100
    }
    if (n_ranges > 0) {
      print "uncovered:"
      for (i = 1; i <= n_ranges; i++) print "  " ranges[i]
    }
    if (n_missing > 0) {
      print "not in the coverage report (declarations only, or no test compiles them):"
      for (i = 1; i <= n_missing; i++) print "  " missing[i]
    }
    if (total == 0) { print "OK (no new executable lines)"; exit 0 }
    if (covered * 100 >= min * total) { print "OK (>= " min " %)"; exit 0 }
    print "FAIL (< " min " %)"
    exit 1
  }
' "$diff" "$lcov"
