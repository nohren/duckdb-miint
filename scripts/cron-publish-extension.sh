#!/usr/bin/env bash
#
# cron-publish-extension.sh
#
# Poll GitHub Actions for the latest passing build(s) of duckdb-miint,
# download the extension artifacts, gzip them, and publish them to a custom
# DuckDB extension repository via rsync. Designed to run from cron on a host
# other than the builder.
#
# TWO PUBLISH STREAMS, both handled in a single invocation (see $STREAMS):
#
#   branch  — the latest push run on $BRANCH, published if it passed. These
#             are NATIVE-only builds (CI excludes wasm on branch pushes).
#             Published to the continuously-updated live path:
#                 $DEST_BASE/<duckdb_version>
#
#   tagged  — the latest push run of a release tag (CI builds tags matching
#             v[0-9]+*), published if it passed. Tag runs additionally carry
#             the wasm_eh build, so this stream is what serves the in-browser
#             DuckDB-Wasm console.
#             Published, keyed by DuckDB version (latest tag wins), to:
#                 $DEST_BASE/tagged/<duckdb_version>
#
# Keeping the streams in separate locations means the branch's native-only
# builds never overwrite the tagged release's wasm (and vice versa), while a
# stable client URL — .../tagged/<duckdb_version>/wasm_eh/... — always points
# at the newest tagged release for that DuckDB version.
#
# Deployment is atomic via a symlink swap: each release lives under
# $DEST_BASE/releases/<name>/ and the live symlink flips to it in a single
# rename(2) call. Release dir names are stream-scoped so the two streams
# never collide:
#   branch:  releases/<duckdb_version>-run<run_id>/
#   tagged:  releases/tagged-<duckdb_version>-run<run_id>/
# Old releases are left in place; prune them manually if disk fills up.
#
# Idempotent per stream: a stream with no new passing build is a no-op. The
# script always emails one combined status covering every stream. Concurrent
# runs are prevented via flock.
#
# Email subject prefixes (worst stream disposition wins):
#   [SUCCEED]  at least one stream deployed a new build, none failed
#   [OK]       ran, no action needed (no new builds, a stream's latest run is
#              still in progress upstream, or a stream's listing came back
#              older than, or without, the run it last handled on fewer than
#              $STALE_TICKS ticks in a row)
#   [STALE]    a stream's listing has come back older than, or without, the run
#              it last handled on $STALE_TICKS or more ticks in a row, so nothing
#              is being published from it
#   [FAIL]     the script itself errored, a stream could not list its runs or
#              tags, or a stream's latest passing build's artifacts don't match
#              $DUCKDB_VERSION (rejected)
#   [FAIL-CI]  a stream's latest run did not succeed (failure, cancelled, ...)
#
# Prereqs on the cron host (none scripted — all assumed ready):
#   - `gh` authenticated (check with: gh auth status)
#   - SSH public key trusted by $DEST_USER@$DEST_HOST (no-password rsync/ssh)
#   - `mail` configured to deliver to $NOTIFY_EMAIL
#   - gzip, rsync, ssh, flock, jq, find, mktemp (standard)
#
# First-deploy note: if a live path ($DEST_BASE/<duckdb_version> or
# $DEST_BASE/tagged/<duckdb_version>) already exists as a plain directory, the
# symlink swap will fail (rename(2) refuses to replace a non-empty directory
# with a non-symlink). Move that directory aside before the first cron run.
#
# Configure via env vars or by passing an env file as $1. All vars marked
# REQUIRED must be set; others have sensible defaults. See bottom of file
# for an example env file.
#
# Example crontab (daily at 03:17 local time):
#   17 3 * * * /path/to/repo/scripts/cron-publish-extension.sh /etc/miint-publish.env

set -euo pipefail

# ----- config -----

if [[ -n "${1:-}" ]]; then
    # shellcheck disable=SC1090
    source "$1"
fi

: "${REPO:=the-miint/duckdb-miint}"
: "${BRANCH:=v1.5-variegata}"
: "${WORKFLOW:=MainDistributionPipeline.yml}"

# Which streams to publish this invocation, space-separated. Default is both.
# Valid tokens: "branch" "tagged".
: "${STREAMS:=branch tagged}"

# How far back to scan push runs when hunting for the latest release-tag run.
# Tags are infrequent while branch pushes are not, so the newest tag run may
# sit well below the newest branch run. If a tag build isn't found within this
# window the tagged stream logs a skip (see "no tag run found" below).
: "${TAG_SCAN_LIMIT:=100}"

# How many ticks in a row a stream may go without a selectable run at or above
# the run it last handled (see the floor in process_stream) before the email
# says [STALE] rather than [OK].
: "${STALE_TICKS:=2}"

# DUCKDB_VERSION must match both the duckdb submodule pin and the
# duckdb_version input in .github/workflows/MainDistributionPipeline.yml.
# When the extension targets a new DuckDB version, bump all three together.
: "${DUCKDB_VERSION:=v1.5.4}"

: "${DEST_USER:?DEST_USER is required}"
: "${DEST_HOST:?DEST_HOST is required}"
: "${DEST_BASE:?DEST_BASE is required (absolute path on \$DEST_HOST)}"
: "${NOTIFY_EMAIL:?NOTIFY_EMAIL is required}"

: "${STATE_DIR:=${HOME}/.local/state/miint-publish}"
LOCK_FILE="${STATE_DIR}/lock"
WORK_DIR="${STATE_DIR}/work"
LOG_FILE="${STATE_DIR}/last_run.log"

ARTIFACT_PREFIX="miint-${DUCKDB_VERSION}-extension-"
HOSTNAME_SHORT="$(hostname -s)"

mkdir -p "$STATE_DIR" "$WORK_DIR" "$WORK_DIR/.empty"

# Log everything to $LOG_FILE (truncated each run). Cron swallows stderr
# unless captured; this keeps a record for the FAIL email tail.
: > "$LOG_FILE"
exec >>"$LOG_FILE" 2>&1

log() { printf '[%s] %s\n' "$(date -u +%FT%TZ)" "$*"; }

send_mail() {
    local subject="$1" body="$2"
    printf '%s\n' "$body" | mail -s "$subject" "$NOTIFY_EMAIL"
}

# Per-stream state file. Format: TAB-separated
# "<status>\t<run_id>\t<duckdb_version>\t<selection>", where <selection> is what
# the stream selected from: $BRANCH for "branch", "tags" for "tagged".
# status ∈ {deployed, rejected}. Old single-field format (from before streams
# existed) is treated as deployed with an unknown version, which forces a
# re-examination on next run. The three-field format from before <selection>
# was recorded reads back with an empty selection. <selection> does not include
# $REPO or $WORKFLOW: before pointing an existing $STATE_DIR at another
# repository or workflow, clear that stream's state files (last_run_id.<stream>
# and stale_ticks.<stream>).
state_file() { printf '%s/last_run_id.%s' "$STATE_DIR" "$1"; }

read_state() {
    local sf; sf=$(state_file "$1")
    [[ -f "$sf" ]] || return 0
    local line fields
    line=$(cat "$sf")
    IFS=$'\t' read -ra fields <<<"$line"
    case ${#fields[@]} in
        0) ;;
        1) printf 'deployed\t%s\t\t\n' "${fields[0]}" ;;
        2) printf '%s\t%s\t\t\n' "${fields[0]}" "${fields[1]}" ;;
        3) printf '%s\t%s\t%s\t\n' "${fields[0]}" "${fields[1]}" "${fields[2]}" ;;
        *) printf '%s\t%s\t%s\t%s\n' "${fields[0]}" "${fields[1]}" "${fields[2]}" "${fields[3]}" ;;
    esac
}

write_state() {
    local sf; sf=$(state_file "$1")
    printf '%s\t%s\t%s\t%s\n' "$2" "$3" "$4" "$5" > "$sf"
}

# Per-stream count of ticks in a row whose listing came back older than, or
# without, the run the stream last handled. A plain number in
# stale_ticks.<stream>; anything else reads as 0.
stale_file() { printf '%s/stale_ticks.%s' "$STATE_DIR" "$1"; }

read_stale() {
    local sf n=""
    sf=$(stale_file "$1")
    [[ -f "$sf" ]] && n=$(cat "$sf")
    [[ "$n" =~ ^(0|[1-9][0-9]{0,8})$ ]] || n=0
    printf '%s\n' "$n"
}

write_stale() {
    local sf; sf=$(stale_file "$1")
    printf '%s\n' "$2" > "$sf"
}

# count_stale <state_key>: one more such tick; sets STALE_N to the new count.
count_stale() {
    STALE_N=$(( $(read_stale "$1") + 1 ))
    write_stale "$1" "$STALE_N"
}

# ----- combined-report accumulators -----
#
# process_stream() never exits; it appends a section to REPORT and bumps the
# worst-disposition tracker. main() composes one email from these at the end.
# WORST_RANK: 0 none/noop, 1 in-progress, 2 stale, 3 ci-fail, 4 rejected or listing error.
REPORT=""
WORST_RANK=0
WORST_KIND="ok"
ANY_DEPLOYED=0
STALE_N=0

bump() {
    # bump <rank> <kind>
    if (( $1 > WORST_RANK )); then WORST_RANK=$1; WORST_KIND="$2"; fi
}

add_report() { REPORT+="$1"$'\n'; }

fail() {
    trap - ERR
    local msg="$1"
    log "FAIL: $msg"
    local subject="[FAIL] miint publish on ${HOSTNAME_SHORT} — ${msg}"
    local body
    body=$(cat <<EOF
Host:          $HOSTNAME_SHORT
Repo:          $REPO
Branch:        $BRANCH
Workflow:      $WORKFLOW
DuckDB ver:    $DUCKDB_VERSION
Destination:   ${DEST_USER}@${DEST_HOST}:${DEST_BASE}

--- last log lines ---
$(tail -n 200 "$LOG_FILE")
EOF
)
    send_mail "$subject" "$body" || true
    exit 1
}

trap 'fail "unexpected error at line $LINENO"' ERR

# ----- lock -----

exec 9>"$LOCK_FILE"
if ! flock -n 9; then
    log "another instance is holding the lock; exiting quietly"
    exit 0
fi

# ----- work-dir reset helper (rsync --delete trick; no rm) -----

reset_dir() {
    local d="$1"
    mkdir -p "$d"
    rsync -a --delete "$WORK_DIR/.empty/" "$d/"
}

# ----- select the run for a stream -----
#
# Echoes a single-run JSON object (or empty), and returns non-zero if the runs or
# tags can't be listed. For "branch" it's the newest push run on $BRANCH; for
# "tagged" it's the newest push run whose head branch and commit match one of
# the repository's tags (e.g. v1.5.4 / v1.0.0-rc.2).
select_run() {
    local stream="$1"
    case "$stream" in
        branch)
            # --branch matches a run's head branch whatever triggered it, so
            # without --event push a pull_request run whose head branch is also
            # named $BRANCH would be selected and published. It also leaves out
            # workflow_dispatch runs: a dispatched build is never published from
            # this stream.
            gh run list --repo "$REPO" --workflow "$WORKFLOW" --branch "$BRANCH" \
                --event push --limit 1 \
                --json databaseId,headBranch,headSha,displayTitle,createdAt,url,status,conclusion \
                | jq -c '.[0] // empty'
            ;;
        tagged)
            # A push run's headBranch is the tag name for a tag push and the
            # branch name for a branch push, and nothing else on the run says
            # which it was. So no pattern can tell them apart (a branch like
            # v1.5-per-sample-table-functions matches the CI tag glob `v[0-9]+*`
            # just as v1.5.4 does), and neither can the name alone, since a
            # branch may share a tag's name. A run counts as a tag run when its
            # headBranch and headSha are an existing tag's name and commit; a
            # branch that shares a tag's name and sits at that tag's commit
            # would still match.
            local tag_pairs
            if ! tag_pairs=$(gh api "repos/$REPO/tags" --paginate --jq '.[] | [.name, .commit.sha]'); then
                # Fail rather than match against an empty list; process_stream
                # reports it as a failure, not as "no matching run found".
                return 1
            fi
            gh run list --repo "$REPO" --workflow "$WORKFLOW" --event push \
                --limit "$TAG_SCAN_LIMIT" \
                --json databaseId,headBranch,headSha,displayTitle,createdAt,url,status,conclusion \
                | jq -c --argjson tags "$(jq -sc '.' <<<"$tag_pairs")" \
                    '[.[] | select([.headBranch, .headSha] as $r | any($tags[]; . == $r))] | .[0] // empty'
            ;;
        *)
            fail "unknown stream '$stream'"
            ;;
    esac
}

# ----- process one stream (no exit; reports via accumulators) -----

process_stream() {
    local stream="$1"

    # Stream-scoped destination + naming.
    local link_dir link_target release_name state_key ref_label ensure_dir selection listing
    case "$stream" in
        branch)
            link_dir="$DEST_BASE"
            release_name="${DUCKDB_VERSION}-run__RUNID__"
            link_target="releases/$release_name"
            state_key="branch"
            selection="$BRANCH"
            listing="runs"
            ref_label="$REPO @ $BRANCH (native)"
            ensure_dir=""
            ;;
        tagged)
            link_dir="$DEST_BASE/tagged"
            release_name="tagged-${DUCKDB_VERSION}-run__RUNID__"
            link_target="../releases/$release_name"
            state_key="tagged"
            selection="tags"
            listing="runs or tags"
            ref_label="$REPO @ tags (native + wasm)"
            ensure_dir="$DEST_BASE/tagged"
            ;;
        *)
            # Checked here, before select_run: select_run's own unknown-stream
            # fail runs inside $(...), where the listing-failure handling below
            # would report it a second time as "could not list runs or tags".
            fail "unknown stream '$stream'"
            ;;
    esac

    # Prior disposition for this stream.
    local LAST_STATUS="" LAST_RUN_ID="" LAST_DUCKDB_VERSION="" LAST_SELECTION="" last_state
    last_state=$(read_state "$state_key")
    if [[ -n "$last_state" ]]; then
        IFS=$'\t' read -r LAST_STATUS LAST_RUN_ID LAST_DUCKDB_VERSION LAST_SELECTION <<<"$last_state"
    fi

    log "== stream '$stream': selecting run =="
    local run_json select_err="$WORK_DIR/select.$stream.err"
    if ! run_json=$(select_run "$stream" 2>"$select_err"); then
        # A failed listing is not "nothing to publish". Report it with the error
        # output, since the log is cleared on the next run, and still run the
        # other streams and send the combined email.
        cat "$select_err"
        log "stream '$stream': could not list $listing"
        bump 4 error
        local err_lines
        err_lines=$(tail -n 10 "$select_err" | sed 's/^/      /')
        add_report "[$stream] FAIL: could not list $listing; nothing was selected this tick. Error output:
${err_lines:-      (none)}"
        return 0
    fi
    cat "$select_err"

    if [[ -z "$run_json" ]]; then
        log "stream '$stream': no matching run found"
        if [[ -n "$LAST_RUN_ID" ]]; then
            count_stale "$state_key"
            if (( STALE_N >= STALE_TICKS )); then
                bump 2 stale
                add_report "[$stream] STALE: no matching run in the listing, $STALE_N ticks in a row; last handled run $LAST_RUN_ID ($LAST_STATUS) is unchanged. Check $(state_file "$state_key")."
            else
                add_report "[$stream] OK: no matching run in this tick's listing; last handled run $LAST_RUN_ID ($LAST_STATUS) is unchanged."
            fi
        else
            add_report "[$stream] no matching run found (nothing to publish yet)."
        fi
        return 0
    fi

    local RUN_ID RUN_SHA RUN_TITLE RUN_TIME RUN_URL RUN_STATUS RUN_CONCLUSION RUN_REF
    RUN_ID=$(jq -r '.databaseId // empty' <<<"$run_json")
    RUN_SHA=$(jq -r '.headSha // empty' <<<"$run_json")
    RUN_TITLE=$(jq -r '.displayTitle // empty' <<<"$run_json")
    RUN_TIME=$(jq -r '.createdAt // empty' <<<"$run_json")
    RUN_URL=$(jq -r '.url // empty' <<<"$run_json")
    RUN_STATUS=$(jq -r '.status // empty' <<<"$run_json")
    RUN_CONCLUSION=$(jq -r '.conclusion // empty' <<<"$run_json")
    RUN_REF=$(jq -r '.headBranch // empty' <<<"$run_json")

    # Resolve the run-id into the release name / link target placeholders.
    release_name="${release_name/__RUNID__/$RUN_ID}"
    link_target="${link_target/__RUNID__/$RUN_ID}"

    log "stream '$stream': run $RUN_ID ref=$RUN_REF status=$RUN_STATUS conclusion=$RUN_CONCLUSION (last=${LAST_RUN_ID:-none}/${LAST_STATUS:-none})"

    # Gate: never go back past the run this stream last handled. Run ids grow
    # with creation time (true of every run listed when this gate was written),
    # so a selected run older than that one means the listing came back
    # incomplete -- gh run list has been seen omitting the newest runs -- or the
    # handled run no longer qualifies (e.g. its tag was deleted). Acting on it
    # could report FAIL-CI for a run that isn't the latest, reject an old run and
    # overwrite this stream's state, or deploy an older build over a newer one.
    # This can't catch a listing that hides a newer green run but still shows a
    # newer failed one; that is indistinguishable from a real failure. Each such
    # tick counts toward $STALE_TICKS in a row, after which the email says
    # [STALE]; a tick whose listing reaches the last handled run resets it.
    #
    # The floor only holds for the same selection: after $BRANCH changes, the
    # new branch's runs are examined normally. State from before the selection
    # was recorded still sets it. The id must be 1-18 plain decimal digits
    # without a leading zero before it reaches (( )): bash reads a leading zero
    # as octal, wraps values above 2^63-1, and would otherwise evaluate a
    # hand-edited value as an expression, command substitutions included.
    if [[ ( -z "$LAST_SELECTION" || "$LAST_SELECTION" == "$selection" ) && "$LAST_RUN_ID" =~ ^[1-9][0-9]{0,17}$ ]] &&
        (( RUN_ID < LAST_RUN_ID )); then
        count_stale "$state_key"
        log "stream '$stream': run $RUN_ID is older than last handled run $LAST_RUN_ID; skipping ($STALE_N in a row)"
        if (( STALE_N >= STALE_TICKS )); then
            bump 2 stale
            add_report "[$stream] STALE: newest selectable run $RUN_ID is older than last handled run $LAST_RUN_ID, $STALE_N ticks in a row (stale listing, or $LAST_RUN_ID no longer qualifies); nothing is published until that changes. Check $(state_file "$state_key")."
        else
            add_report "[$stream] OK: newest selectable run $RUN_ID is older than last handled run $LAST_RUN_ID (stale listing, or $LAST_RUN_ID no longer qualifies); skipping this tick. If this repeats, check $(state_file "$state_key")."
        fi
        return 0
    fi
    # This tick's listing reached the last handled run, so the stream isn't stuck.
    [[ "$(read_stale "$state_key")" == 0 ]] || write_stale "$state_key" 0

    # Gate: not completed yet.
    if [[ "$RUN_STATUS" != "completed" ]]; then
        bump 1 inprogress
        add_report "[$stream] latest run $RUN_ID ($RUN_REF) is $RUN_STATUS; skipping this tick. $RUN_URL"
        return 0
    fi

    # Gate: completed but not success.
    if [[ "$RUN_CONCLUSION" != "success" ]]; then
        bump 3 ci
        add_report "[$stream] FAIL-CI: latest run $RUN_ID ($RUN_REF) ended conclusion=$RUN_CONCLUSION; no deploy. $RUN_URL"
        return 0
    fi

    # No-op: already handled this exact run at this version.
    if [[ "$RUN_ID" == "$LAST_RUN_ID" && "$DUCKDB_VERSION" == "$LAST_DUCKDB_VERSION" ]]; then
        if [[ "$LAST_STATUS" == "rejected" ]]; then
            bump 4 reject
            add_report "[$stream] FAIL: run $RUN_ID ($RUN_REF) was previously REJECTED (artifact naming mismatch); awaiting a newer passing run."
        else
            add_report "[$stream] OK: no new build since last deploy (run $RUN_ID, $RUN_REF)."
        fi
        return 0
    fi

    # Artifact precheck.
    log "stream '$stream': listing artifacts on run $RUN_ID"
    local art_names matched=()
    art_names=$(gh api "repos/$REPO/actions/runs/$RUN_ID/artifacts" --paginate --jq '.artifacts[].name')
    local n
    while IFS= read -r n; do
        [[ -z "$n" ]] && continue
        [[ "$n" == "$ARTIFACT_PREFIX"* ]] && matched+=("$n")
    done <<<"$art_names"

    if [[ ${#matched[@]} -eq 0 ]]; then
        write_state "$state_key" rejected "$RUN_ID" "$DUCKDB_VERSION" "$selection"
        bump 4 reject
        local avail_fmt
        avail_fmt=$(while IFS= read -r n; do [[ -n "$n" ]] && printf '      - %s\n' "$n"; done <<<"$art_names")
        add_report "[$stream] FAIL: artifact naming mismatch on run $RUN_ID ($RUN_REF). Expected prefix $ARTIFACT_PREFIX; present:
$avail_fmt    (recorded; will not re-alert until a newer run appears.)"
        return 0
    fi

    log "stream '$stream': matched ${#matched[@]} artifact(s) against prefix $ARTIFACT_PREFIX"

    # Fresh per-stream work dirs.
    local art_dir="$WORK_DIR/artifacts.$stream" stage_dir="$WORK_DIR/stage.$stream"
    reset_dir "$art_dir"
    reset_dir "$stage_dir"

    log "stream '$stream': downloading artifacts into $art_dir"
    gh run download "$RUN_ID" --repo "$REPO" --dir "$art_dir" --pattern "${ARTIFACT_PREFIX}*"

    # Stage: gzip natives, copy wasm/.gz as-is, laid out by platform.
    local release_dir="$stage_dir/$release_name"
    mkdir -p "$release_dir"

    local deployed=() art name platform ext_path pat out_dir out_file kind size
    while IFS= read -r -d '' art; do
        name=$(basename "$art")
        platform="${name#"$ARTIFACT_PREFIX"}"
        if [[ "$platform" == "$name" ]]; then
            log "  skip $name (no prefix match)"
            continue
        fi

        ext_path=""
        for pat in '*.duckdb_extension' '*.duckdb_extension.gz' '*.duckdb_extension.wasm'; do
            ext_path=$(find "$art" -type f -name "$pat" -print -quit)
            [[ -n "$ext_path" ]] && break
        done
        [[ -n "$ext_path" ]] || fail "artifact $name contains no .duckdb_extension{,.gz,.wasm} file"

        out_dir="$release_dir/$platform"
        mkdir -p "$out_dir"

        case "$ext_path" in
            *.duckdb_extension.wasm)
                out_file="$out_dir/miint.duckdb_extension.wasm"
                cp "$ext_path" "$out_file"
                kind=wasm
                ;;
            *.duckdb_extension.gz)
                out_file="$out_dir/miint.duckdb_extension.gz"
                cp "$ext_path" "$out_file"
                kind=gz
                ;;
            *)
                out_file="$out_dir/miint.duckdb_extension.gz"
                gzip -9 -c "$ext_path" > "$out_file"
                kind=gz
                ;;
        esac

        size=$(stat -c %s "$out_file")
        deployed+=("$platform ($(numfmt --to=iec --suffix=B "$size") $kind)")
        log "  staged $platform <- $(basename "$ext_path") ($size bytes $kind)"
    done < <(find "$art_dir" -mindepth 1 -maxdepth 1 -type d -print0)

    [[ ${#deployed[@]} -gt 0 ]] || fail "stream '$stream': no artifacts matched prefix $ARTIFACT_PREFIX after download"

    # Push to remote + atomic symlink swap.
    local remote_releases="$DEST_BASE/releases" remote_release="$DEST_BASE/releases/$release_name"

    log "stream '$stream': ensuring remote dirs exist on $DEST_HOST"
    # shellcheck disable=SC2029  # intentional local expansion
    ssh "${DEST_USER}@${DEST_HOST}" "mkdir -p '$remote_releases'${ensure_dir:+ '$ensure_dir'}"

    log "stream '$stream': syncing $release_dir/ -> ${DEST_USER}@${DEST_HOST}:${remote_release}/"
    rsync -az --chmod=D755,F644 "$release_dir/" "${DEST_USER}@${DEST_HOST}:${remote_release}/"

    log "stream '$stream': flipping $link_dir/$DUCKDB_VERSION -> $link_target"
    # shellcheck disable=SC2029  # intentional local expansion
    ssh "${DEST_USER}@${DEST_HOST}" "
        set -euo pipefail
        cd '$link_dir'
        ln -sfn '$link_target' '${DUCKDB_VERSION}.tmp'
        mv -Tf '${DUCKDB_VERSION}.tmp' '$DUCKDB_VERSION'
    "

    write_state "$state_key" deployed "$RUN_ID" "$DUCKDB_VERSION" "$selection"
    ANY_DEPLOYED=1

    local plat_list
    plat_list=$(printf '      - %s\n' "${deployed[@]}")
    add_report "[$stream] SUCCEED: deployed run $RUN_ID ($RUN_REF)
    ref:   $ref_label
    sha:   $RUN_SHA
    title: $RUN_TITLE
    time:  $RUN_TIME
    url:   $RUN_URL
    live:  ${DEST_USER}@${DEST_HOST}:${link_dir}/${DUCKDB_VERSION}  ->  $link_target
    platforms (${#deployed[@]}):
$plat_list"
    log "stream '$stream': done."
}

# ----- run every requested stream, then send one combined email -----

for s in $STREAMS; do
    process_stream "$s"
done

case "$WORST_KIND" in
    reject|error) prefix="[FAIL]" ;;
    ci)     prefix="[FAIL-CI]" ;;
    stale)  prefix="[STALE]" ;;
    *)      if (( ANY_DEPLOYED )); then prefix="[SUCCEED]"; else prefix="[OK]"; fi ;;
esac

subject="$prefix miint publish on ${HOSTNAME_SHORT} — streams: ${STREAMS}"
body=$(cat <<EOF
Host:          $HOSTNAME_SHORT
Repo:          $REPO
Branch:        $BRANCH
Workflow:      $WORKFLOW
DuckDB ver:    $DUCKDB_VERSION
Streams:       $STREAMS
Destination:   ${DEST_USER}@${DEST_HOST}:${DEST_BASE}

$REPORT
EOF
)
send_mail "$subject" "$body"
log "All streams processed ($prefix)."

# ──────────────────────────────────────────────────────────────────────
# Example env file (save as e.g. /etc/miint-publish.env, chmod 600):
#
#   REPO=the-miint/duckdb-miint
#   BRANCH=v1.5-variegata
#   WORKFLOW=MainDistributionPipeline.yml
#   DUCKDB_VERSION=v1.5.4
#   STREAMS="branch tagged"          # or just "branch" / just "tagged"
#   DEST_USER=deploy
#   DEST_HOST=repo.example.com
#   DEST_BASE=/var/www/miint-ext
#   NOTIFY_EMAIL=releases@example.com
#   # optional:
#   # STATE_DIR=/var/lib/miint-publish
#   # TAG_SCAN_LIMIT=100
#
# After install:
#   chmod +x scripts/cron-publish-extension.sh
#   scripts/cron-publish-extension.sh /etc/miint-publish.env   # run once to verify
#   crontab -e
#     17 3 * * * /path/to/repo/scripts/cron-publish-extension.sh /etc/miint-publish.env
# ──────────────────────────────────────────────────────────────────────
