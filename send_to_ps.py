#!/usr/bin/env python3
"""
JSONL File Processor and HTTP POST Client for AI label predictions for Project Sidewalk.

This script reads a JSONL (JSON Lines) file produced by main.py, converts each record's
normalized detections to the pixel-coordinate label format expected by Project Sidewalk,
and sends the records via POST requests to a Project Sidewalk ingest endpoint.

Usage:
    python send_to_ps.py bend.jsonl --endpoint http://localhost:9000/ai/submitLabelsOnPano

Submission progress is tracked in a sidecar file (<file>.submitted) so a re-run resumes
where it left off instead of re-POSTing every line. The sidecar is line numbers only, so a
git-tracked <file>.submission.json records, per endpoint, what those lines were (sha256 of
the file) and how many went where; before POSTing anything the two are checked against
each other (see check_resume_state), and a campaign that moves from a test instance to
production starts the sidecar afresh rather than skipping the lines test already took.
"""

import argparse
import hashlib
import ipaddress
import json
import os
import socket
import time
from contextlib import nullcontext
from datetime import datetime, timezone
from typing import Dict, Any, Optional, Set
from pathlib import Path
from urllib.parse import urlparse

import requests
from dotenv import load_dotenv

from detectors import OPERATIONAL_CONFIDENCE

# Local secrets (e.g. PS_INTERNAL_API_KEY) from ./.env; real env vars win.
load_dotenv()

DEFAULT_ENDPOINT_URL = "http://localhost:9000/ai/submitLabelsOnPano"
MAX_ATTEMPTS = 3
RETRY_BACKOFF_SECONDS = [2, 8]

# Project Sidewalk's pano_source enum (PanoSource in SidewalkWebpage). Legacy Stage-1
# records store streetlevel's raw source string ("launch", "scout", ...) instead; any
# value outside this set is GSV imagery.
PS_PANO_SOURCES = {"gsv", "mapillary", "infra3d"}


def is_loopback(host: Optional[str]) -> bool:
    """True if `host` means this machine (loopback, or a local server's bind address), so a
    request to it never reaches the wire."""
    if not host:
        return False
    host = host.strip('[]').lower()          # strip the brackets of an IPv6 literal
    if host == 'localhost' or host.endswith('.localhost'):
        return True
    try:
        address = ipaddress.ip_address(host)
    except ValueError:
        return False                          # a real hostname; resolving it is not our job
    # `0.0.0.0` / `::` are what a dev server binds to, and people paste the bind address
    # into --endpoint. As a destination they mean "this machine" too (or fail outright),
    # so like loopback they never put the key on the wire.
    return address.is_loopback or address.is_unspecified


def check_endpoint_security(endpoint_url: str, api_key: Optional[str]) -> None:
    """
    Refuse to send the internal API key in cleartext.

    The key is a *shared* Project Sidewalk secret (the server's INTERNAL_API_KEY), so a
    mistyped `--endpoint` that drops the 's' would leak it to every hop in between — and
    a leak is silent, since the request otherwise succeeds. Fail fast instead.

    Loopback is exempt (the request never leaves the machine), which keeps the default
    `http://localhost:9000` dev endpoint working. Sending no key over plain HTTP is also
    fine: there is nothing to protect, and deployments that don't require auth exist.

    Raises:
        ValueError: if a key would travel unencrypted to a remote host.
    """
    if not api_key:
        return
    parsed = urlparse(endpoint_url)
    if parsed.scheme == 'https' or is_loopback(parsed.hostname):
        return
    raise ValueError(
        f"Refusing to send the API key over {parsed.scheme or 'an unknown scheme'}:// to "
        f"remote host '{parsed.hostname}' — it would travel in cleartext. Use https://, "
        f"or unset the key's environment variable to submit without auth."
    )


def transform_pano(pano: Dict[str, Any]) -> Dict[str, Any]:
    """
    Map a Stage-1 pano block onto the field names and enum values the Project Sidewalk
    reader (PanoSubmission in SidewalkWebpage) expects. Tolerates legacy records
    produced before the field names were aligned with the server:

    - 'panorama_id' -> 'pano_id'
    - 'source' outside the pano_source enum (raw streetlevel strings) -> 'gsv'
    - links[].'target_gsv_panorama_id' -> 'target_pano_id'
    - 'links'/'history' are required (possibly empty) arrays server-side

    Every other key is forwarded as-is, deliberately: we submit all the provenance we
    have, so it's already in the payload the day PS learns to store it. Extra keys are
    safe — PanoSubmission's reader (ExploreFormats.scala) is path-based and ignores
    what it doesn't name — but they are also discarded server-side today: pano_data has
    no column for source_metadata, camera_make/model/type, sequence_id or quality_score.
    Landing them needs a SidewalkWebpage change, not a change here.
    """
    pano = dict(pano)
    if 'panorama_id' in pano:
        pano['pano_id'] = pano.pop('panorama_id')
    if pano.get('source') not in PS_PANO_SOURCES:
        pano['source'] = 'gsv'
    pano['links'] = [
        {
            "target_pano_id": link.get('target_pano_id', link.get('target_gsv_panorama_id')),
            "yaw_deg": link['yaw_deg'],
            "description": link.get('description'),
        } for link in pano.get('links') or []
    ]
    pano['history'] = pano.get('history') or []
    return pano


def transform_record(data: Dict[str, Any], min_confidence: float = OPERATIONAL_CONFIDENCE) -> Dict[str, Any]:
    """
    Convert a main.py JSONL record into the payload expected by Project Sidewalk.

    Detections at or above min_confidence are converted from normalized coordinates to
    pixel coordinates using the pano dimensions stored in the record, renamed
    'detections' -> 'labels'; the pano block is mapped onto the server's field names
    (see transform_pano). results.jsonl stores candidate peaks down to a storage floor
    below the operational threshold (see detectors/__init__.py); a record whose
    detections all fall below min_confidence still submits with empty labels — it is a
    processed "checked, nothing found" pano, not a droppable one.
    """
    modified_data = data.copy()
    modified_data['pano'] = transform_pano(data['pano'])
    modified_data['labels'] = [
        {
            "pano_x": round(detection['x_normalized'] * modified_data['pano']['width']),
            "pano_y": round(detection['y_normalized'] * modified_data['pano']['height']),
            "confidence": detection['confidence']
        } for detection in data['detections'] if detection['confidence'] >= min_confidence
    ]
    modified_data.pop('detections', None)
    return modified_data


def send_to_project_sidewalk(
    payload: Dict[str, Any], endpoint_url: str, api_key: Optional[str] = None
) -> Optional[requests.Response]:
    """
    Send a POST request with JSON data to the specified PS endpoint, retrying on
    transient failures.

    Args:
        payload: The transformed JSON data to send in the POST request.
        endpoint_url: The target endpoint URL.
        api_key: Optional Project Sidewalk internal API key. When provided, it is sent as an
            ``Authorization: Bearer`` header so the request can authenticate to the ingest endpoint.

    Returns:
        The response object if successful, None if an error occurred. Retries up to
        MAX_ATTEMPTS times with backoff on connection errors and 5xx responses; 4xx
        responses are treated as permanent and not retried.
    """
    headers = {
        'Content-Type': 'application/json',
        'Accept': 'application/json'
    }
    if api_key:
        headers['Authorization'] = f'Bearer {api_key}'

    for attempt in range(1, MAX_ATTEMPTS + 1):
        try:
            response = requests.post(
                endpoint_url,
                json=payload,
                headers=headers,
                timeout=30
            )

            if response.status_code == 200:
                return response

            # Print the error response body for diagnosis.
            try:
                print(json.dumps(response.json(), indent=2))
            except json.JSONDecodeError:
                print(response.text)

            # 4xx means the payload or auth is wrong; retrying won't help.
            if 400 <= response.status_code < 500:
                print(f"Permanent error (HTTP {response.status_code}); not retrying.")
                return None

        except requests.exceptions.RequestException as e:
            print(f"Error sending POST request (attempt {attempt}/{MAX_ATTEMPTS}): {e}")

        if attempt < MAX_ATTEMPTS:
            backoff = RETRY_BACKOFF_SECONDS[attempt - 1]
            print(f"Retrying in {backoff}s...")
            time.sleep(backoff)

    return None


def load_submitted_lines(sidecar_path: Path) -> Set[int]:
    """Loads the set of already-submitted line numbers from the sidecar file."""
    if not sidecar_path.exists():
        return set()
    with open(sidecar_path, 'r', encoding='utf-8') as f:
        return {int(line) for line in f if line.strip()}


SUBMISSION_RECORD_SUFFIX = ".submission.json"


def submission_record_path(file_path: str) -> Path:
    """Path of a campaign's submission record, beside the JSONL and its `.submitted` sidecar."""
    return Path(f"{file_path}{SUBMISSION_RECORD_SUFFIX}")


def hash_and_count(input_file: Path) -> tuple:
    """(sha256 of the bytes, number of non-blank lines) of the JSONL.

    The hash is what ties the line-numbered sidecar to the file it was written against; the
    count is what a complete campaign's sidecar reaches, so the record can say how far there
    is still to go. Lines are counted the way the submit loop reads them (text mode, blank
    lines skipped), so the two can't disagree over a trailing newline or a stray blank.
    """
    digest = hashlib.sha256()
    with open(input_file, 'rb') as f:
        for chunk in iter(lambda: f.read(1 << 20), b''):
            digest.update(chunk)
    with open(input_file, 'r', encoding='utf-8') as f:
        lines = sum(1 for line in f if line.strip())
    return digest.hexdigest(), lines


def canonical_endpoint(endpoint_url: str) -> str:
    """The form an endpoint is recorded under: scheme and host lower-cased, no trailing slash.

    The record is keyed by endpoint, so `https://HOST/ai/submitLabelsOnPano/` and the same URL
    without the slash must not read as two different servers.
    """
    parts = urlparse(endpoint_url)
    return parts._replace(scheme=parts.scheme.lower(), netloc=parts.netloc.lower(),
                          path=parts.path.rstrip('/')).geturl()


def load_submission_record(record_path: Path) -> Dict[str, Any]:
    """The campaign's record, or {} when there has never been one (a first submission).

    Raises:
        ValueError: if a record EXISTS but cannot be read as one. That file is the memory of
            what is already live on a server, so "unreadable" must not decay into "nothing
            was sent": a merge conflict (the record is git-tracked, and two machines are
            exactly the case it guards) or a truncated write are the usual causes.
    """
    if not record_path.exists():
        return {}
    try:
        with open(record_path, 'r', encoding='utf-8') as f:
            record = json.load(f)
        endpoints = record['endpoints']
        if not isinstance(endpoints, dict):
            raise ValueError("'endpoints' is not a per-endpoint mapping")
        for state in endpoints.values():
            int(state['submitted_lines'])
    except (json.JSONDecodeError, OSError, KeyError, TypeError, ValueError) as e:
        raise ValueError(
            f"{record_path.name} exists but is not a readable submission record ({e}). It says "
            f"what is already live on a server, so it can't be treated as 'nothing was sent'. "
            f"Repair it (a merge conflict, a truncated write, or a record from before the "
            f"per-endpoint format - `git show HEAD:{record_path.as_posix()}` recovers the "
            f"committed copy) or, for a case checked by hand, --ignore-submission-guard, "
            f"which starts a fresh record."
        )
    return record


def check_resume_state(record: Dict[str, Any], digest: str, submitted_lines: Set[int],
                       endpoint_url: str, min_confidence: float, input_file: Path,
                       record_path: Path, sidecar_path: Path) -> None:
    """
    Refuse to submit when the resume sidecar no longer describes this file and this endpoint.

    The sidecar is a set of LINE NUMBERS and nothing else - it doesn't know which file they
    index or which server they went to - so three things silently turn it into a lie:
    editing or reordering the JSONL under it, losing it (deleted, or a second machine that
    never had it), and pointing the same file at a second server. The first two re-POST
    records that are already live and duplicate their labels - a 9,000-record city doubles
    itself without a single error being raised. The third does the opposite: the lines a
    staging run claimed are skipped on the production run and never reach the live city.
    The per-endpoint record written beside the sidecar is what makes all three visible.

    Raises:
        ValueError: if the input file changed; if the sidecar accounts for fewer lines than
            the record says already went to this endpoint; if the sidecar holds lines that
            went to a different endpoint than the one being submitted to; or if this run's
            --min-confidence differs from the one this endpoint was submitted at.
    """
    if not record:
        return

    recorded_digest = record.get('sha256')
    if recorded_digest and recorded_digest != digest:
        raise ValueError(
            f"{input_file.name} has changed since the submission recorded in {record_path.name} "
            f"(sha256 {digest[:12]}... != {recorded_digest[:12]}...). {sidecar_path.name} holds "
            f"line numbers against the old file, so resuming would skip and re-send the wrong "
            f"records. Restore the file that was submitted, or start a new campaign under a new "
            f"name. --ignore-submission-guard overrides."
        )

    endpoint = canonical_endpoint(endpoint_url)
    states = record['endpoints']
    already = states.get(endpoint, {}).get('submitted_lines', 0)
    if already > len(submitted_lines):
        complete = already >= record.get('total_lines', already + 1)
        raise ValueError(
            f"{record_path.name} says {already} line(s) of {input_file.name} were already "
            f"submitted to {endpoint}, but {sidecar_path.name} accounts for only "
            f"{len(submitted_lines)}. "
            + ("That campaign ran to completion, so there is nothing left to send there. "
               if complete else
               "The resume sidecar is missing, truncated, or from a different machine - "
               "submitting now would re-POST records that are already live and duplicate "
               "their labels. Restore the sidecar first (a backup copy). ")
            + "--ignore-submission-guard overrides."
        )

    # More sidecar lines than this endpoint is recorded to have, while another endpoint has
    # a count: those extra lines went there, not here. (`already` is 0 for an endpoint the
    # record has never seen.) The one innocent way to reach this state is a crash between
    # the sidecar flush and the record write, which the message names for the override.
    elsewhere = sorted(url for url, state in states.items()
                       if url != endpoint and state.get('submitted_lines'))
    if len(submitted_lines) > already and elsewhere:
        raise ValueError(
            f"{sidecar_path.name} holds {len(submitted_lines)} line(s) but {record_path.name} "
            f"says only {already} went to {endpoint}; the rest went to {', '.join(elsewhere)}. "
            f"The sidecar is endpoint-agnostic, so submitting now would skip exactly those "
            f"records here and they would never reach this server. Move the sidecar aside "
            f"(e.g. rename it {sidecar_path.name}.staging) so this endpoint starts from line 1; "
            f"the record keeps the other endpoint's count. --ignore-submission-guard sends only "
            f"the remainder - right only if the extra lines DID go to {endpoint} and a crash "
            f"kept the record from catching up."
        )

    recorded_confidence = states.get(endpoint, {}).get('min_confidence')
    if recorded_confidence is not None and recorded_confidence != min_confidence:
        raise ValueError(
            f"{record_path.name} says {endpoint} was submitted at --min-confidence "
            f"{recorded_confidence}, but this run uses {min_confidence}. One server should hold "
            f"one threshold's labels, and the record can only count at one. Use "
            f"--min-confidence {recorded_confidence}. --ignore-submission-guard overrides."
        )


def count_labels(input_file: Path, line_numbers: Set[int], min_confidence: float) -> int:
    """How many labels the given lines of the JSONL submit at `min_confidence`.

    Recounted from the file and the sidecar rather than accumulated in memory, so the record
    stays right across interrupted runs, resumed runs, and lines sent before the record existed.
    """
    if not line_numbers:
        return 0
    labels = 0
    with open(input_file, 'r', encoding='utf-8') as f:
        for line_number, line in enumerate(f, 1):
            if line_number in line_numbers and line.strip():
                labels += len(transform_record(json.loads(line), min_confidence)['labels'])
    return labels


def write_submission_record(record_path: Path, input_file: Path, digest: str, total_lines: int,
                            submitted_lines: Set[int], endpoint_url: str, min_confidence: float,
                            previous: Dict[str, Any]) -> None:
    """Record what went where, so a campaign survives the loss of its sidecar.

    One entry per endpoint - a campaign legitimately hits a test instance before prod, and
    each server's count is its own. Both counts come from the sidecar and the file, never
    from this run's tallies, so the record and the sidecar cannot disagree. Small and stable
    enough to commit, unlike the sidecar and the JSONL, which are gitignored run state - so
    it also answers "what did we submit to that server, and when?" long after the run
    directory is gone.
    """
    now = datetime.now(timezone.utc).strftime('%Y-%m-%dT%H:%M:%SZ')
    endpoint = canonical_endpoint(endpoint_url)
    states = dict(previous.get('endpoints') or {})
    state = dict(states.get(endpoint) or {})
    state.update({
        "submitted_lines": len(submitted_lines),
        "labels_submitted": count_labels(input_file, submitted_lines, min_confidence),
        "min_confidence": min_confidence,
        "first_submission_utc": state.get('first_submission_utc', now),
        "last_submission_utc": now,
        "last_run_host": socket.gethostname(),
    })
    states[endpoint] = state
    record = {
        "input_file": input_file.name,
        "sha256": digest,
        "total_lines": total_lines,
        "endpoints": states,
    }
    # Written whole-or-not-at-all: a crash mid-write would leave a truncated file, and an
    # unreadable record refuses the next run (load_submission_record). LF regardless of
    # platform, since the file is committed.
    tmp_path = record_path.with_name(record_path.name + ".tmp")
    with open(tmp_path, 'w', encoding='utf-8', newline='\n') as f:
        json.dump(record, f, indent=2)
        f.write("\n")
    os.replace(tmp_path, record_path)


def process_jsonl_file(
    file_path: str,
    endpoint_url: str = DEFAULT_ENDPOINT_URL,
    api_key: Optional[str] = None,
    dry_run: bool = False,
    min_confidence: float = OPERATIONAL_CONFIDENCE,
    limit: Optional[int] = None,
    ignore_guard: bool = False,
) -> None:
    """
    Process a JSONL file containing detections from main.py by reading each line and sending
    POST requests to PS.

    Args:
        file_path: Path to the JSONL file to process.
        endpoint_url: The endpoint URL to send POST requests to.
        api_key: Optional Project Sidewalk internal API key, forwarded to each request (see
            ``send_to_project_sidewalk``).
        dry_run: If True, print the transformed payloads instead of POSTing them, and do not
            record submission progress.
        min_confidence: Only detections at/above this confidence are submitted as labels
            (see ``transform_record``).
        limit: Stop after this many records are submitted *this run*; already-submitted
            lines don't count against it, so successive capped runs walk the file.
        ignore_guard: Submit even when the submission record disagrees with the resume
            sidecar (see ``check_resume_state``).
    """
    input_file = Path(file_path)

    # Validate file exists.
    if not input_file.exists():
        print(f"Error: File '{file_path}' does not exist.")
        return

    # Fail before opening the file: a cleartext key leaks on the very first request.
    check_endpoint_security(endpoint_url, None if dry_run else api_key)

    # Load resume state: line numbers that already got a 200 on a previous run.
    sidecar_path = Path(f"{file_path}.submitted")
    submitted_lines = load_submitted_lines(sidecar_path)

    # ...and check it still describes this file and this endpoint. A dry run POSTs nothing
    # and writes nothing, so it is exempt from all of it (it must stay usable on a stale
    # file or beside a broken record); the override downgrades every refusal to a warning
    # (and an unreadable record to a fresh one).
    record_path = submission_record_path(file_path)
    digest, total_lines = hash_and_count(input_file)
    previous_record: Dict[str, Any] = {}
    try:
        if not dry_run:
            previous_record = load_submission_record(record_path)
            check_resume_state(previous_record, digest, submitted_lines, endpoint_url,
                               min_confidence, input_file, record_path, sidecar_path)
    except ValueError as e:
        if not ignore_guard:
            raise
        print(f"WARNING (--ignore-submission-guard): {e}")

    success_count = 0
    error_count = 0
    skipped_count = 0
    filtered_detections = 0
    attempted = 0          # records touched this run; what --limit caps
    limit_reached = False
    interrupted = False

    print(f"Processing JSONL file: {file_path}")
    print(f"Target endpoint: {endpoint_url}")
    print(f"Auth: {'Bearer key supplied' if api_key else 'none (no key set)'}")
    if dry_run:
        print("DRY RUN: payloads will be printed, not sent.")
    if limit is not None:
        print(f"LIMIT: stopping after {limit} record(s) this run.")
    if submitted_lines:
        print(f"Resuming: {len(submitted_lines)} lines already submitted (per {sidecar_path.name}).")
    print("-" * 50)

    try:
        # A dry run records nothing, so don't create (or touch) the sidecar file.
        with open(input_file, 'r', encoding='utf-8') as file, \
             (nullcontext() if dry_run else open(sidecar_path, 'a', encoding='utf-8')) as f_sidecar:
            for line_number, line in enumerate(file, 1):
                line = line.strip()

                # Skip empty lines and lines submitted on a previous run.
                if not line:
                    continue
                if line_number in submitted_lines:
                    skipped_count += 1
                    continue

                # Stop once this run has touched --limit records. Checked here, after the
                # resume skip, so a capped run always advances into unsubmitted lines.
                if limit is not None and attempted >= limit:
                    limit_reached = True
                    break
                attempted += 1

                try:
                    # Parse a line of JSON and convert to the PS payload format.
                    json_data = json.loads(line)
                    payload = transform_record(json_data, min_confidence)
                    filtered_detections += len(json_data.get('detections', [])) - len(payload['labels'])

                    if dry_run:
                        print(json.dumps(payload, indent=2))
                        success_count += 1
                        continue

                    # Send POST request.
                    response = send_to_project_sidewalk(payload, endpoint_url, api_key)

                    if response:
                        success_count += 1
                        f_sidecar.write(f"{line_number}\n")
                        f_sidecar.flush()
                    else:
                        error_count += 1
                        print(f"Line {line_number}: Failed to send POST request")

                except json.JSONDecodeError as e:
                    error_count += 1
                    print(f"Line {line_number}: Invalid JSON - {e}")

                except Exception as e:
                    error_count += 1
                    print(f"Line {line_number}: Unexpected error - {e}")

    except IOError as e:
        print(f"Error reading file: {e}")
        return
    except KeyboardInterrupt:
        # Every line that got a 200 is already in the sidecar (flushed per POST), so the
        # record can still be brought up to date below before the interrupt is reported.
        interrupted = True

    # The record is derived from the sidecar and the file, not from this run's tallies, so
    # it is written after any run that landed something - complete, capped, or interrupted -
    # and never after one that didn't: a run where every POST failed must not create a
    # record claiming an endpoint that took nothing.
    record_written = False
    if not dry_run and success_count:
        write_submission_record(record_path, input_file, digest, total_lines,
                                load_submitted_lines(sidecar_path), endpoint_url,
                                min_confidence, previous_record)
        record_written = True

    # Print summary.
    print("-" * 50)
    print("Interrupted." if interrupted else "Processing complete!")
    print(f"Successfully processed:        {success_count} records")
    print(f"Skipped (already submitted):   {skipped_count} records")
    print(f"Errors encountered:            {error_count} records")
    print(f"Detections below --min-confidence {min_confidence} (not submitted): {filtered_detections}")
    if record_written:
        print(f"Submission record:             {record_path.name}")
    if limit_reached:
        print(f"Stopped at --limit {limit}. "
              + ("Dry run — nothing was recorded, so a re-run starts from this same line."
                 if dry_run else
                 f"Re-run to continue from here ({sidecar_path.name} records what already landed)."))
    if interrupted:
        raise SystemExit(130)


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Submit a main.py JSONL file of AI label predictions to a Project Sidewalk endpoint."
    )
    parser.add_argument(
        "jsonl_file",
        help="Path to the JSONL file produced by main.py."
    )
    parser.add_argument(
        "--endpoint",
        default=DEFAULT_ENDPOINT_URL,
        help=f"Project Sidewalk ingest endpoint URL (default: {DEFAULT_ENDPOINT_URL})."
    )
    parser.add_argument(
        "--api-key-env",
        default="PS_INTERNAL_API_KEY",
        help="Name of the environment variable holding Project Sidewalk's internal API key "
             "(matches Project Sidewalk's INTERNAL_API_KEY), read from the environment or a "
             "gitignored ./.env. If the variable is unset, no auth header is sent (works "
             "against a deployment that doesn't require it yet). When a key IS set, a remote "
             "--endpoint must be https:// or the run is refused."
    )
    parser.add_argument(
        "--limit", type=int, metavar="N",
        help="Submit at most N records this run, then stop. Already-submitted lines don't "
             "count, so repeated capped runs walk the file — use it to send a handful and "
             "inspect them in Project Sidewalk before committing to the whole city."
    )
    parser.add_argument(
        "--ignore-submission-guard",
        action="store_true",
        help="Submit even when the submission record disagrees with the resume sidecar - a "
             "changed input file, a sidecar accounting for fewer lines than were already "
             "sent to this endpoint, a sidecar whose lines went to a different endpoint, or "
             "an unreadable record (which is then replaced). Only for a case you have checked "
             "by hand: the guard is what stops a lost sidecar from re-POSTing a whole city "
             "and duplicating its labels."
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Print transformed payloads instead of POSTing them; no progress is recorded."
    )
    parser.add_argument(
        "--min-confidence", type=float, default=OPERATIONAL_CONFIDENCE,
        help="Minimum detection confidence submitted as a label (default: %(default)s, the "
             "operational threshold). results.jsonl stores candidate peaks down to a lower "
             "storage floor for multi-view fusion; records whose detections all fall below "
             "this still submit as 'checked, nothing found'."
    )
    args = parser.parse_args()

    if args.limit is not None and args.limit < 1:
        parser.error("--limit must be at least 1.")

    api_key = os.environ.get(args.api_key_env)
    try:
        process_jsonl_file(args.jsonl_file, args.endpoint, api_key, args.dry_run,
                           args.min_confidence, args.limit, args.ignore_submission_guard)
    except ValueError as e:
        raise SystemExit(f"Error: {e}")


if __name__ == "__main__":
    main()
