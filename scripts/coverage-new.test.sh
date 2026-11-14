#!/usr/bin/env bash
# Tests for coverage-new.sh: feeds it a fixed diff and lcov report, checks its output and
# exit code. Run from anywhere: scripts/coverage-new.test.sh
set -euo pipefail

here=$(cd "$(dirname "$0")" && pwd)
script="$here/coverage-new.sh"
work=$(mktemp -d)
trap 'rm -rf "$work"' EXIT
failures=0

# Added lines: lib.rs 11-13 and 23 (32-33 are only deleted); new.rs 1-4.
cat > "$work/diff" <<'DIFF'
diff --git a/crates/a/src/lib.rs b/crates/a/src/lib.rs
--- a/crates/a/src/lib.rs
+++ b/crates/a/src/lib.rs
@@ -10,0 +11,3 @@ fn x() {
+    one();
+    two();
+    three();
@@ -20 +23 @@ fn y() {
+    four();
@@ -30,2 +32,0 @@ fn z() {
-    gone();
-    gone();
diff --git a/crates/b/src/new.rs b/crates/b/src/new.rs
new file mode 100644
--- /dev/null
+++ b/crates/b/src/new.rs
@@ -0,0 +1,4 @@
+fn a() {}
+fn b() {}
+// a comment, not instrumented
+fn c() {}
DIFF

# lib.rs:5 and old.rs are not new; new.rs:3 is not instrumented.
cat > "$work/partial.lcov" <<'LCOV'
SF:/repo/crates/a/src/lib.rs
DA:5,1
DA:11,3
DA:12,0
DA:13,0
DA:23,1
end_of_record
SF:/repo/crates/b/src/new.rs
DA:1,1
DA:2,1
DA:4,0
end_of_record
SF:/repo/crates/c/src/old.rs
DA:1,0
end_of_record
LCOV

sed 's/,0$/,7/' "$work/partial.lcov" > "$work/full.lcov"
: > "$work/empty.diff"

# expect <name> <expected exit> <expected output> <command...>
expect() {
  local name=$1 want_code=$2 want_out=$3
  shift 3
  local out code=0
  out=$("$@" 2>&1) || code=$?
  if [[ $code != "$want_code" || $out != "$want_out" ]]; then
    printf 'FAIL %s\n  exit %s, want %s\n  got:\n%s\n  want:\n%s\n' \
      "$name" "$code" "$want_code" "$out" "$want_out"
    failures=$((failures + 1))
  else
    printf 'ok   %s\n' "$name"
  fi
}

args=(--root /repo --diff "$work/diff")

expect "counts only added, instrumented lines and lists the uncovered ones" 1 \
'new lines: 7, covered: 4 (57.14 %)
uncovered:
  crates/a/src/lib.rs:12-13
  crates/b/src/new.rs:4
FAIL (< 99 %)' "$script" "${args[@]}" --lcov "$work/partial.lcov"

expect "passes when every new line is covered" 0 \
'new lines: 7, covered: 7 (100.00 %)
OK (>= 99 %)' "$script" "${args[@]}" --lcov "$work/full.lcov"

expect "honors COVERAGE_NEW_MIN" 0 \
'new lines: 7, covered: 4 (57.14 %)
uncovered:
  crates/a/src/lib.rs:12-13
  crates/b/src/new.rs:4
OK (>= 50 %)' env COVERAGE_NEW_MIN=50 "$script" "${args[@]}" --lcov "$work/partial.lcov"

expect "passes with no new executable lines" 0 \
'new lines: 0
OK (no new executable lines)' "$script" --root /repo --diff "$work/empty.diff" --lcov "$work/partial.lcov"

expect "rejects an unknown option" 2 \
'usage: coverage-new.sh [--base <ref>] [--lcov <file> --diff <file> --root <dir>]' "$script" --bogus

# A path with a space (git ends its header with a tab), an added line whose text starts
# with "++ " (it prints as "+++ "), a no-newline marker, and a deleted file.
printf '%s\n' \
  'diff --git a/crates/t/src/with space.rs b/crates/t/src/with space.rs' \
  '--- a/crates/t/src/with space.rs' \
  $'+++ b/crates/t/src/with space.rs\t' \
  '@@ -3 +3 @@ const S: &str = "' \
  '-++ y";' \
  '+++ z";' \
  '@@ -5,0 +6,2 @@ fn g() {' \
  '+    h();' \
  '+    k();' \
  '\ No newline at end of file' \
  'diff --git a/crates/t/src/gone.rs b/crates/t/src/gone.rs' \
  'deleted file mode 100644' \
  '--- a/crates/t/src/gone.rs' \
  '+++ /dev/null' \
  '@@ -1,2 +0,0 @@' \
  '-fn gone() {}' \
  '-fn gone2() {}' > "$work/tricky.diff"

# A checksum on a DA line, and the same file in two records (line 6 is hit in the second).
cat > "$work/tricky.lcov" <<'LCOV'
SF:/repo/crates/t/src/with space.rs
DA:3,1,c2ba
DA:6,0
DA:7,1
end_of_record
SF:/repo/crates/t/src/with space.rs
DA:6,4
end_of_record
LCOV
sed 's/^DA:7,1$/DA:7,0/' "$work/tricky.lcov" > "$work/tricky-miss.lcov"

cat "$work/tricky.diff" - > "$work/orphan.diff" <<'DIFF'
diff --git a/crates/t/src/orphan.rs b/crates/t/src/orphan.rs
new file mode 100644
--- /dev/null
+++ b/crates/t/src/orphan.rs
@@ -0,0 +1 @@
+fn never_compiled() {}
DIFF

tricky=(--root /repo --diff "$work/tricky.diff")

expect "keeps the file of an added line that starts with ++" 0 \
'new lines: 3, covered: 3 (100.00 %)
OK (>= 99 %)' "$script" "${tricky[@]}" --lcov "$work/tricky.lcov"

expect "truncates the percentage and takes a fractional minimum" 1 \
'new lines: 3, covered: 2 (66.66 %)
uncovered:
  crates/t/src/with space.rs:7
FAIL (< 66.67 %)' env COVERAGE_NEW_MIN=66.67 "$script" "${tricky[@]}" --lcov "$work/tricky-miss.lcov"

expect "lists files with new lines that are not in the report" 0 \
'new lines: 3, covered: 3 (100.00 %)
not in the coverage report (declarations only, or no test compiles them):
  crates/t/src/orphan.rs
OK (>= 99 %)' \
  "$script" --root /repo --diff "$work/orphan.diff" --lcov "$work/tricky.lcov"

expect "rejects a minimum that is not a number" 2 \
'coverage-new.sh: COVERAGE_NEW_MIN must be a number, got abc' \
  env COVERAGE_NEW_MIN=abc "$script" "${args[@]}" --lcov "$work/partial.lcov"

expect "rejects --base with files given" 2 \
'usage: coverage-new.sh [--base <ref>] [--lcov <file> --diff <file> --root <dir>]' \
  "$script" --base main "${args[@]}" --lcov "$work/partial.lcov"

# The checks below run before any test does, in a throwaway repository.
repo="$work/repo"
git init -q "$repo"
git -C "$repo" -c user.name=t -c user.email=t@t commit -q --allow-empty -m init
in_repo() { (cd "$repo" && "$script" "$@"); }

expect "rejects a base that is not a commit before running anything" 2 \
'coverage-new.sh: --base must name a commit, got --output=x' \
  in_repo --base --output=x

expect "rejects an unknown base before running anything" 2 \
'coverage-new.sh: --base must name a commit, got no-such-ref' \
  in_repo --base no-such-ref

echo 'fn wip() {}' > "$repo/wip.rs"
expect "refuses uncommitted .rs changes, which the diff would not see" 2 \
'coverage-new.sh: commit or stash the .rs changes first; the diff only sees commits:
?? wip.rs' \
  in_repo --base HEAD

if ((failures > 0)); then
  echo "$failures test(s) failed"
  exit 1
fi
echo "all tests passed"
