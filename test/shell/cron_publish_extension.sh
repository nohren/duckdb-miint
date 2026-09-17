#!/bin/bash
# Scenario tests for scripts/cron-publish-extension.sh: which run each stream
# selects, and what it alerts, deploys and records when the run listing comes
# back stale once or several ticks in a row, when a pull_request run shares the
# branch's name, when a branch looks like or is named like a release tag, when
# state comes from another branch, DuckDB version or format, when a run or tag
# listing fails, and when a stream name is misspelled.
#
# gh, mail, ssh, rsync and hostname are stubs (cron_publish_extension_stubs/),
# so nothing leaves the machine. The gh stub serves canned run listings and
# applies --branch / --event / --limit the way the real listing does: --branch
# matches a run's head branch whatever the event, results come newest-first by
# run id.
#
# Needs jq, flock and GNU coreutils (numfmt, stat -c), as the cron host does;
# skips otherwise (e.g. on macOS). To run it there:
#   docker run --rm -v "$PWD":/w -w /w ubuntu:24.04 bash -c \
#     'apt-get update -qq && apt-get install -y -qq jq >/dev/null && bash test/shell/cron_publish_extension.sh'
# The scenarios' temporary directory is removed when all pass and kept otherwise.
#
# Usage: bash test/shell/cron_publish_extension.sh [script-under-test]

set -e

HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
SCRIPT="${1:-$HERE/../../scripts/cron-publish-extension.sh}"
FAILED=0
ALL_PASSED=0

missing=""
for tool in jq flock numfmt; do
    command -v "$tool" > /dev/null 2>&1 || missing="$missing $tool"
done
stat -c %s /dev/null > /dev/null 2>&1 || missing="$missing stat-c"
if [ -n "$missing" ]; then
    echo "cron-publish tests need jq, flock and GNU coreutils; missing:$missing. Skipping."
    exit 0
fi

export PATH="$HERE/cron_publish_extension_stubs:$PATH"
TMP_ROOT="$(mktemp -d "${TMPDIR:-/tmp}/cron_publish_extension.XXXXXX")"
trap 'if [ "$ALL_PASSED" = 1 ]; then rm -rf "$TMP_ROOT"; fi' EXIT
SCENARIO=0

# run_json ID EVENT HEAD_BRANCH CONCLUSION [STATUS] [HEAD_SHA]: a run with the
# fields the script requests from gh run list --json, plus the event the gh stub
# filters on. STATUS defaults to completed, HEAD_SHA to "sha<id>".
run_json() {
    jq -nc --argjson id "$1" --arg ev "$2" --arg hb "$3" --arg c "$4" --arg st "${5:-completed}" --arg sha "${6:-}" \
        '{databaseId: $id, event: $ev, headBranch: $hb,
          headSha: (if $sha == "" then "sha" + ($id | tostring) else $sha end),
          displayTitle: "t", createdAt: "2026-01-01T00:00:00Z", url: ("https://example/" + ($id | tostring)),
          status: $st, conclusion: $c}'
}
set_runs() { printf '%s\n' "$@" | jq -sc '.' > "$SD/runs.json"; }
set_artifacts() {
    local id="$1"
    shift
    printf '%s\n' "$@" | jq -Rsc '{artifacts: (split("\n") | map(select(length > 0)) | map({name: .}))}' > "$SD/artifacts_${id}.json"
}
# set_tags NAME=RUN_ID ...: a tag whose commit is "sha<RUN_ID>".
set_tags() {
    printf '%s\n' "$@" |
        jq -Rsc 'split("\n") | map(select(length > 0) | split("=") | {name: .[0], commit: {sha: ("sha" + .[1])}})' > "$SD/tags.json"
}
set_state() { printf '%s\n' "$2" > "$SD/state/last_run_id.$1"; }
state_of() { cat "$SD/state/last_run_id.$1" 2> /dev/null || true; }
set_stale() { printf '%s\n' "$2" > "$SD/state/stale_ticks.$1"; }
stale_of() { cat "$SD/state/stale_ticks.$1" 2> /dev/null || echo 0; }

new_scenario() {
    SCENARIO=$((SCENARIO + 1))
    SD="$TMP_ROOT/$SCENARIO"
    export SCENARIO_DIR="$SD"
    mkdir -p "$SD/state"
    printf '[]' > "$SD/runs.json"
    printf '[]' > "$SD/tags.json"
    cat > "$SD/env" << EOF
REPO=the-miint/duckdb-miint
BRANCH=${1:-v1.5-variegata}
WORKFLOW=MainDistributionPipeline.yml
DUCKDB_VERSION=v1.5.4
DEST_USER=deploy
DEST_HOST=repo.example
DEST_BASE=/srv/miint
NOTIFY_EMAIL=nobody@example
STATE_DIR=$SD/state
EOF
}

invoke() {
    printf 'STREAMS="%s"\n' "$1" >> "$SD/env"
    EXIT=0
    bash "$SCRIPT" "$SD/env" || EXIT=$?
    SUBJECT="$(cat "$SD/mail.subject" 2> /dev/null || true)"
    BODY="$(cat "$SD/mail.body" 2> /dev/null || true)"
    EMAILS="$(grep -c '' "$SD/mail.subjects" 2> /dev/null || echo 0)"
}

report() {
    local name="$1" why="$2" ok="$3"
    echo "Running: $name"
    if [ "$ok" = 1 ]; then
        echo "  PASS"
    else
        echo "  FAIL"
        echo "  Why it matters: $why"
        echo "  Exit: $EXIT  Emails: $EMAILS  Subject: $SUBJECT"
        echo "  State: $(cat "$SD"/state/last_run_id.* 2> /dev/null | tr '\t\n' '  ' || true)"
        echo "  Stale ticks: $(cat "$SD"/state/stale_ticks.* 2> /dev/null | tr '\n' ' ' || true)"
        echo "  Report: $(printf '%s\n' "$BODY" | grep '^\[' | tr '\n' '|' || true)"
        echo "  Log: $SD/state/last_run.log"
        FAILED=1
    fi
}

D=$'deployed\t'
R=$'rejected\t'
V=$'\tv1.5.4'
SB=$'\tv1.5-variegata'
ST=$'\ttags'

# --- branch stream: runs older than the last handled run ---

# The newest visible run is older than the one already deployed: the
# 2026-09-10 false FAIL-CI on a green, already-deployed branch. The state uses
# the three-field format written before the selection was recorded.
new_scenario
set_state branch "${D}33135547066${V}"
set_runs "$(run_json 32880410496 push v1.5-variegata failure)" "$(run_json 32788471971 push v1.5-variegata success)"
invoke branch
ok=0
[[ $EXIT -eq 0 && "$SUBJECT" != "[FAIL-CI]"* && "$(state_of branch)" == "${D}33135547066${V}" ]] &&
    [[ "$BODY" == *"is older than last handled run 33135547066"* ]] && ok=1
report "branch: a listing older than the last deployed run is reported as skipped, not as a CI failure" \
    "an incomplete listing is not a CI failure; alerting on it sent FAIL-CI for a green, deployed branch" "$ok"

new_scenario
set_state branch "${D}33135547066${V}${SB}"
set_runs "$(run_json 32788471971 push v1.5-variegata success)"
set_artifacts 32788471971 miint-v1.5.4-extension-linux_amd64
invoke branch
ok=0
[[ $EXIT -eq 0 && ! -e "$SD/ssh.calls" && "$(state_of branch)" == "${D}33135547066${V}${SB}" ]] && ok=1
report "branch: an older successful run is not deployed over the newer build" \
    "acting on a stale listing would silently roll the live extension back to an older build" "$ok"

new_scenario
set_state branch "${D}33135547066${V}${SB}"
set_runs "$(run_json 32000000000 push v1.5-variegata "" in_progress)"
invoke branch
ok=0
[[ $EXIT -eq 0 && "$BODY" == *"is older than last handled run 33135547066"* && "$BODY" != *"is in_progress"* ]] && ok=1
report "branch: an older run still in progress is skipped as older, not reported as the latest run" \
    "the floor is checked before the in-progress gate, so an old run that is still running is never presented as the latest" "$ok"

new_scenario
set_state branch "${R}33135547066${V}${SB}"
set_runs "$(run_json 32788471971 push v1.5-variegata success)"
set_artifacts 32788471971 miint-v1.5.2-extension-linux_amd64
invoke branch
ok=0
[[ $EXIT -eq 0 && "$SUBJECT" != "[FAIL]"* && "$(state_of branch)" == "${R}33135547066${V}${SB}" ]] && ok=1
report "branch: a rejected run also holds back older runs" \
    "otherwise a stale listing after a rejection could reject an older run and overwrite the state" "$ok"

# After a DuckDB version bump on the same branch, an older run comes from a
# stale listing and carries the old version's artifacts.
new_scenario
set_state branch $'deployed\t33135547066\tv1.5.3\tv1.5-variegata'
set_runs "$(run_json 32788471971 push v1.5-variegata success)"
set_artifacts 32788471971 miint-v1.5.3-extension-linux_amd64
invoke branch
ok=0
[[ $EXIT -eq 0 && "$SUBJECT" != "[FAIL]"* && "$(state_of branch)" == $'deployed\t33135547066\tv1.5.3\tv1.5-variegata' ]] && ok=1
report "branch: a DuckDB version bump does not lift the floor on the same branch" \
    "acting on an older run after a bump would reject its old-version artifacts and overwrite the state" "$ok"

new_scenario
set_state branch "33135547066"
set_runs "$(run_json 32880410496 push v1.5-variegata failure)"
invoke branch
ok=0
[[ $EXIT -eq 0 && "$SUBJECT" != "[FAIL-CI]"* ]] && ok=1
report "branch: a legacy single-field state file still sets the floor" \
    "read_state accepts pre-streams state, and its run id is still the last handled run" "$ok"

# --- branch stream: what the floor must not change ---

new_scenario
set_state branch "${D}33135547066${V}"
set_runs "$(run_json 34000000001 push v1.5-variegata failure)" "$(run_json 33135547066 push v1.5-variegata success)"
invoke branch
ok=0
[[ $EXIT -eq 0 && "$SUBJECT" == "[FAIL-CI]"* && "$BODY" == *"FAIL-CI: latest run 34000000001"* ]] && ok=1
report "branch: a newer failed run is still reported as FAIL-CI" \
    "the floor must never mask a real post-merge CI failure" "$ok"

new_scenario
set_state branch "${D}33135547066${V}"
set_runs "$(run_json 35000000000 push main success)" "$(run_json 34000000002 push v1.5-variegata success)" \
    "$(run_json 33135547066 push v1.5-variegata success)"
set_artifacts 35000000000 miint-v1.5.4-extension-linux_amd64
set_artifacts 34000000002 miint-v1.5.4-extension-linux_amd64 miint-v1.5.4-extension-osx_arm64
invoke branch
ok=0
[[ $EXIT -eq 0 && "$SUBJECT" == "[SUCCEED]"* && "$(state_of branch)" == "${D}34000000002${V}${SB}" ]] &&
    grep -q "releases/v1.5.4-run34000000002" "$SD/ssh.calls" && ! grep -q "run35000000000" "$SD/ssh.calls" && ok=1
report "branch: the newest successful run on BRANCH is deployed and recorded with its selection" \
    "the floor must not stop new builds from shipping, and a newer run on another branch is not this stream's to publish" "$ok"

new_scenario
set_runs "$(run_json 34000000004 push v1.5-variegata success)"
set_artifacts 34000000004 miint-v1.5.2-extension-linux_amd64
invoke branch
ok=0
[[ $EXIT -eq 0 && "$SUBJECT" == "[FAIL]"* && "$BODY" == *"artifact naming mismatch on run 34000000004"* ]] &&
    [[ "$(state_of branch)" == "${R}34000000004${V}${SB}" && ! -e "$SD/ssh.calls" ]] && ok=1
report "branch: a run whose artifacts don't match DUCKDB_VERSION is rejected and recorded with its selection" \
    "a rejection sets the floor too, so it has to record the selection it came from" "$ok"

new_scenario
set_state branch "${R}33135547066${V}${SB}"
set_runs "$(run_json 33135547066 push v1.5-variegata success)"
invoke branch
ok=0
[[ $EXIT -eq 0 && "$SUBJECT" == "[FAIL]"* && "$BODY" == *"was previously REJECTED"* ]] && ok=1
report "branch: the same run that was rejected last time is still reported as FAIL" \
    "only a strictly older run is held back; the same run keeps its no-op or rejection outcome" "$ok"

# Moving BRANCH at the same DuckDB version.
new_scenario main
set_state branch "${D}33135547066${V}${SB}"
set_runs "$(run_json 32000000000 push main success)"
set_artifacts 32000000000 miint-v1.5.4-extension-linux_amd64
invoke branch
ok=0
[[ $EXIT -eq 0 && "$SUBJECT" == "[SUCCEED]"* && "$(state_of branch)" == "${D}32000000000${V}"$'\tmain' ]] && ok=1
report "branch: state recorded for another BRANCH does not hold back this branch's runs" \
    "the floor only means something for the same selection; otherwise moving BRANCH would stall the stream" "$ok"

new_scenario
# shellcheck disable=SC2016 # the $(...) must reach the state file unexpanded
set_state branch "$(printf 'deployed\ta[$(touch %s/pwned)]\tv1.5.4\tv1.5-variegata' "$SD")"
set_runs "$(run_json 32880410496 push v1.5-variegata failure)"
invoke branch
ok=0
[[ $EXIT -eq 0 && "$SUBJECT" == "[FAIL-CI]"* && ! -e "$SD/pwned" ]] && ok=1
report "branch: a non-numeric run id in state is never evaluated as arithmetic" \
    "bash (( )) runs command substitutions hidden in a variable's value; a tampered state file must not execute code" "$ok"

new_scenario
set_state branch $'deployed\t020\tv1.5.4\tv1.5-variegata'
set_runs "$(run_json 15 push v1.5-variegata failure)"
invoke branch
ok=0
[[ $EXIT -eq 0 && "$SUBJECT" == "[FAIL-CI]"* ]] && ok=1
report "branch: a run id with a leading zero in state is ignored, not read as octal" \
    "bash reads 020 as 16, so a hand-edited id would set a floor nobody recorded" "$ok"

new_scenario
set_state branch $'deployed\t99999999999999999999\tv1.5.4\tv1.5-variegata'
set_runs "$(run_json 15 push v1.5-variegata failure)"
invoke branch
ok=0
[[ $EXIT -eq 0 && "$SUBJECT" == "[FAIL-CI]"* ]] && ok=1
report "branch: a run id in state too large for bash arithmetic is ignored" \
    "bash wraps values above 2^63-1, so an oversized hand-edited id would set a floor nobody recorded" "$ok"

new_scenario
set_state branch $'deployed\t99999\tv1.5.4\tv1.5-variegata'
set_runs "$(run_json 100000 push v1.5-variegata failure)"
invoke branch
ok=0
[[ $EXIT -eq 0 && "$SUBJECT" == "[FAIL-CI]"* ]] && ok=1
report "branch: run ids are compared as numbers, not as text" \
    "run 100000 is newer than 99999 but sorts before it as text; a text comparison would skip a real failure" "$ok"

new_scenario
set_runs "$(run_json 32880410496 push v1.5-variegata failure)"
invoke branch
ok=0
[[ $EXIT -eq 0 && "$SUBJECT" == "[FAIL-CI]"* ]] && ok=1
report "branch: with no prior state, a failed latest run is reported as FAIL-CI" \
    "with nothing to compare against, a first run must behave exactly as before" "$ok"

# --- branch stream: pull_request runs ---

new_scenario
set_state branch "${D}33135547066${V}"
set_runs "$(run_json 34000000003 pull_request v1.5-variegata success)" "$(run_json 33135547066 push v1.5-variegata success)"
set_artifacts 34000000003 miint-v1.5.4-extension-linux_amd64
invoke branch
ok=0
[[ $EXIT -eq 0 && ! -e "$SD/ssh.calls" && "$BODY" == *"OK: no new build since last deploy (run 33135547066"* ]] && ok=1
report "branch: a pull_request run named like BRANCH is never published" \
    "only push builds of the branch are publishable; a PR build must never reach the public extension repository" "$ok"

# --- tagged stream ---

# Around the real v1.0.0-rc.3 tag run: a newer non-push run carrying the tag's
# name and commit, newer pushes to a branch that looks like a tag, to a branch
# that has the tag's name at another commit, and to a branch at the tag's own
# commit, plus an older run of another real tag.
new_scenario
set_tags v1.0.0-rc.3=29940753947 v1.0.0-rc.2=25000000000
set_runs "$(run_json 30000000003 workflow_dispatch v1.0.0-rc.3 success completed sha29940753947)" \
    "$(run_json 30000000002 push v1.0.0-rc.3 success)" "$(run_json 30000000001 push v1.6-experimental success)" \
    "$(run_json 29940753990 push v1.5-variegata success completed sha29940753947)" \
    "$(run_json 29940753947 push v1.0.0-rc.3 success)" "$(run_json 25000000000 push v1.0.0-rc.2 success)"
for id in 30000000003 30000000002 30000000001 29940753990 25000000000; do
    set_artifacts "$id" miint-v1.5.4-extension-linux_amd64
done
set_artifacts 29940753947 miint-v1.5.4-extension-linux_amd64 miint-v1.5.4-extension-wasm_eh
invoke tagged
ok=0
[[ $EXIT -eq 0 && "$(state_of tagged)" == "${D}29940753947${V}${ST}" ]] &&
    grep -q "tagged-v1.5.4-run29940753947" "$SD/ssh.calls" &&
    ! grep -q -E "run(30000000003|30000000002|30000000001|29940753990|25000000000)" "$SD/ssh.calls" && ok=1
report "tagged: the newest run of a real tag's name and commit is published, and no look-alike" \
    "the tagged path serves the in-browser wasm build; branch v1.5-per-sample-table-functions was once picked as a release" "$ok"

# The listing omits the deployed v1.0.0-rc.3 run. Of what it shows, the old
# pattern would pick the April branch run (the 2026-09-09 rejection that
# overwrote the state) and tag matching picks the older v1.0.0-rc.2 run. Neither
# may alert or overwrite the state.
new_scenario
set_tags v1.0.0-rc.3=29940753947 v1.0.0-rc.2=23000000000
set_state tagged "${D}29940753947${V}${ST}"
set_runs "$(run_json 24874001218 push v1.5-per-sample-table-functions success)" "$(run_json 23000000000 push v1.0.0-rc.2 success)"
set_artifacts 24874001218 miint-v1.5.2-extension-linux_amd64
set_artifacts 23000000000 miint-v1.5.2-extension-linux_amd64
invoke tagged
ok=0
[[ $EXIT -eq 0 && "$SUBJECT" != "[FAIL]"* && "$(state_of tagged)" == "${D}29940753947${V}${ST}" ]] && ok=1
report "tagged: a stale listing neither alerts nor overwrites state" \
    "an incomplete listing once made the tagged stream reject an old branch run and overwrite its state, forcing a re-deploy" "$ok"

# The listing has no run of any tag this tick, but the stream has a deployed
# release.
new_scenario
set_tags v1.0.0-rc.3=29940753947
set_state tagged "${D}29940753947${V}${ST}"
set_runs "$(run_json 34000000002 push v1.5-variegata success)"
invoke tagged
ok=0
[[ $EXIT -eq 0 && "$BODY" == *"last handled run 29940753947 (deployed) is unchanged"* && "$BODY" != *"nothing to publish yet"* ]] && ok=1
report "tagged: a listing with no matching run doesn't claim there is nothing to publish when a release is deployed" \
    "'nothing to publish yet' next to a live release reads as if the stream had never published" "$ok"

new_scenario
set_tags v1.0.0-rc.3=29940753947
set_runs "$(run_json 34000000002 push v1.5-variegata success)"
invoke tagged
ok=0
[[ $EXIT -eq 0 && "$SUBJECT" == "[OK]"* && "$BODY" == *"[tagged] no matching run found (nothing to publish yet)."* ]] && ok=1
report "tagged: with no matching run and no prior state, the stream reports nothing to publish yet" \
    "a stream that has never published keeps the plain first-run message" "$ok"

# --- stale ticks in a row ---

new_scenario
set_state branch "${D}33135547066${V}${SB}"
set_runs "$(run_json 32880410496 push v1.5-variegata failure)"
invoke branch
ok=0
[[ $EXIT -eq 0 && "$SUBJECT" == "[OK]"* && "$(stale_of branch)" == 1 ]] && ok=1
report "branch: a first tick below the floor is [OK] and starts the stale count" \
    "a single stale listing clears on its own; alerting on it would bring back the false alarms this change removes" "$ok"

new_scenario
set_state branch "${D}33135547066${V}${SB}"
set_stale branch 1
set_runs "$(run_json 32880410496 push v1.5-variegata failure)"
invoke branch
ok=0
[[ $EXIT -eq 0 && "$SUBJECT" == "[STALE]"* && "$BODY" == *"[branch] STALE: newest selectable run 32880410496"*"2 ticks in a row"* ]] &&
    [[ "$(stale_of branch)" == 2 && "$(state_of branch)" == "${D}33135547066${V}${SB}" && ! -e "$SD/ssh.calls" ]] && ok=1
report "branch: the second tick in a row below the floor alerts as [STALE]" \
    "a stream that stays below its last handled run is stuck, and nothing is published until someone looks" "$ok"

new_scenario
set_tags v1.0.0-rc.3=29940753947
set_state tagged "${D}29940753947${V}${ST}"
set_stale tagged 1
set_runs "$(run_json 34000000002 push v1.5-variegata success)"
invoke tagged
ok=0
[[ $EXIT -eq 0 && "$SUBJECT" == "[STALE]"* && "$BODY" == *"[tagged] STALE: no matching run in the listing, 2 ticks in a row"* && "$(stale_of tagged)" == 2 ]] && ok=1
report "tagged: the second tick in a row with no matching run alerts as [STALE]" \
    "a listing that keeps coming back without any tag run leaves the tagged stream unable to publish" "$ok"

new_scenario
set_state branch "${D}33135547066${V}${SB}"
set_stale branch 1
set_runs "$(run_json 33135547066 push v1.5-variegata success)"
invoke branch
ok=0
[[ $EXIT -eq 0 && "$SUBJECT" == "[OK]"* && "$(stale_of branch)" == 0 ]] && ok=1
report "branch: a tick whose listing reaches the last handled run resets the stale count" \
    "the count is for ticks in a row; one listing that reaches the last handled run means the stream is not stuck" "$ok"

new_scenario
set_state branch "${D}33135547066${V}${SB}"
set_stale branch 1
printf 'FAIL: HTTP 502: runs unavailable' > "$SD/runs.json"
invoke branch
ok=0
[[ $EXIT -eq 0 && "$SUBJECT" == "[FAIL]"* && "$(stale_of branch)" == 1 ]] && ok=1
report "branch: a failed listing leaves the stale count as it was" \
    "a failed listing says nothing about whether the stream is stuck, and it is already reported as FAIL" "$ok"

new_scenario
set_tags v1.0.0-rc.3=29940753947
set_state branch "${D}33135547066${V}${SB}"
set_state tagged "${D}29940753947${V}${ST}"
set_stale tagged 1
set_runs "$(run_json 34000000002 push v1.5-variegata success)" "$(run_json 33135547066 push v1.5-variegata success)"
set_artifacts 34000000002 miint-v1.5.4-extension-linux_amd64
invoke "branch tagged"
ok=0
[[ $EXIT -eq 0 && "$SUBJECT" == "[STALE]"* && "$BODY" == *"[branch] SUCCEED: deployed run 34000000002"* && "$BODY" == *"[tagged] STALE:"* ]] && ok=1
report "both streams: a stale stream sets the subject even when the other stream deployed" \
    "a [SUCCEED] subject would bury the one stream that has stopped publishing" "$ok"

new_scenario
set_state branch "${D}33135547066${V}${SB}"
set_stale branch 08
set_runs "$(run_json 32880410496 push v1.5-variegata failure)"
invoke branch
ok=0
[[ $EXIT -eq 0 && "$SUBJECT" == "[OK]"* && "$(stale_of branch)" == 1 ]] && ok=1
report "branch: a malformed stale count reads as zero" \
    "bash reads 08 as an invalid octal number, so a hand-edited count must be ignored rather than used in arithmetic" "$ok"

# --- listing failures and configuration errors ---

new_scenario
printf 'FAIL: HTTP 502: runs unavailable' > "$SD/runs.json"
invoke branch
ok=0
[[ $EXIT -eq 0 && "$SUBJECT" == "[FAIL]"* && "$BODY" == *"[branch] FAIL: could not list runs; nothing"* ]] &&
    [[ "$BODY" == *"HTTP 502: runs unavailable"* && "$BODY" != *"no matching run found"* ]] && ok=1
report "branch: a failed run listing is reported as FAIL with its error, not as nothing to publish" \
    "an [OK] 'no matching run found' would hide that the stream cannot see its runs, and the email is the only place the error outlives the next run" "$ok"

new_scenario
printf 'FAIL: HTTP 502: tags unavailable' > "$SD/tags.json"
set_runs "$(run_json 29940753947 push v1.0.0-rc.3 success)"
set_artifacts 29940753947 miint-v1.5.4-extension-linux_amd64
invoke tagged
ok=0
[[ $EXIT -eq 0 && "$SUBJECT" == "[FAIL]"* && "$BODY" == *"[tagged] FAIL: could not list runs or tags; nothing"* && ! -e "$SD/ssh.calls" ]] &&
    [[ "$BODY" == *"HTTP 502: tags unavailable"* && "$BODY" != *"no matching run found"* ]] && ok=1
report "tagged: a failed tag lookup is reported as FAIL with its error, not as nothing to publish" \
    "an [OK] 'no matching run found', or no email at all, would hide that the tagged stream cannot see its releases" "$ok"

new_scenario
set_state branch "${D}33135547066${V}${SB}"
set_runs "$(run_json 34000000002 push v1.5-variegata success)" "$(run_json 33135547066 push v1.5-variegata success)"
set_artifacts 34000000002 miint-v1.5.4-extension-linux_amd64
printf 'FAIL: HTTP 502: tags unavailable' > "$SD/tags.json"
invoke "branch tagged"
ok=0
[[ $EXIT -eq 0 && "$SUBJECT" == "[FAIL]"* && "$(state_of branch)" == "${D}34000000002${V}${SB}" ]] &&
    [[ "$BODY" == *"[branch] SUCCEED: deployed run 34000000002"* && "$BODY" == *"[tagged] FAIL: could not list runs or tags; nothing"* ]] &&
    [[ "$BODY" != *"no matching run found"* ]] && ok=1
report "both streams: a later stream failing to list still sends the email reporting the earlier stream's deploy" \
    "a listing failure in one stream must not swallow the report of what another stream deployed in the same run" "$ok"

new_scenario
set_state branch "${D}33135547066${V}${SB}"
set_runs "$(run_json 34000000002 push v1.5-variegata success)" "$(run_json 33135547066 push v1.5-variegata success)"
set_artifacts 34000000002 miint-v1.5.4-extension-linux_amd64
printf 'FAIL: HTTP 502: tags unavailable' > "$SD/tags.json"
invoke "tagged branch"
ok=0
[[ $EXIT -eq 0 && "$SUBJECT" == "[FAIL]"* && "$(state_of branch)" == "${D}34000000002${V}${SB}" ]] &&
    [[ "$BODY" == *"[tagged] FAIL: could not list runs or tags; nothing"* && "$BODY" == *"[branch] SUCCEED: deployed run 34000000002"* ]] && ok=1
report "both streams: a stream after a failed listing still runs and deploys" \
    "a listing failure in one stream must not stop the streams that come after it" "$ok"

new_scenario
invoke "branch taged"
ok=0
[[ $EXIT -ne 0 && "$EMAILS" == 1 && "$SUBJECT" == *"unknown stream 'taged'"* ]] && ok=1
report "a misspelled stream is reported once, as an unknown stream" \
    "a configuration error has to say what it is, not also show up as a listing failure in a second email" "$ok"

echo ""
if [ $FAILED -eq 0 ]; then
    ALL_PASSED=1
    echo "All tests passed!"
    exit 0
else
    echo "Some tests failed! Scenario files kept in $TMP_ROOT"
    exit 1
fi
