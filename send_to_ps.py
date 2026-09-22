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
git-tracked <file>.submission.json records, per endpoint, what those lines were (sha256 and
byte length of the file) and how many went where; before POSTing anything the two are checked against
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

import position_check
from detectors import OPERATIONAL_CONFIDENCE, on_camera_rig

# Local secrets (e.g. PS_INTERNAL_API_KEY) from ./.env; real env vars win.
load_dotenv()

DEFAULT_ENDPOINT_URL = "http://localhost:9000/ai/submitLabelsOnPano"
MAX_ATTEMPTS = 3
RETRY_BACKOFF_SECONDS = [2, 8]

# Project Sidewalk's pano_source enum (PanoSource in SidewalkWebpage). Legacy Stage-1
# records store streetlevel's raw source string ("launch", "scout", ...) instead; any
# value outside this set is GSV imagery. `panoramax` is listed ahead of the server: its
# enum value lands with SidewalkWebpage's Panoramax support, and until it does the server
# rejects such records — better than this script silently relabeling them as GSV.
PS_PANO_SOURCES = {"gsv", "mapillary", "infra3d", "panoramax"}


class CampaignComplete(Exception):
    """The guard proved there is nothing left to send; stop without POSTing anything.

    Deliberately NOT a ValueError: every guard refusal is a ValueError that
    ``--ignore-submission-guard`` downgrades to a warning and carries on from. This is the
    opposite case — the campaign is *finished*, and carrying on would re-POST it. The only
    way to reach it is a lost or truncated sidecar on a band the record shows complete, and
    there the override would be actively harmful, so it must not apply.
    """


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


def transform_record(data: Dict[str, Any], min_confidence: float = OPERATIONAL_CONFIDENCE,
                     max_confidence: Optional[float] = None,
                     mask_rig: bool = True) -> Dict[str, Any]:
    """
    Convert a main.py JSONL record into the payload expected by Project Sidewalk.

    Detections at or above min_confidence are converted from normalized coordinates to
    pixel coordinates using the pano dimensions stored in the record, renamed
    'detections' -> 'labels'; the pano block is mapped onto the server's field names
    (see transform_pano). results.jsonl stores candidate peaks down to a storage floor
    below the operational threshold (see detectors/__init__.py); a record whose
    detections all fall below min_confidence still submits with empty labels — it is a
    processed "checked, nothing found" pano, not a droppable one.

    max_confidence (exclusive) selects a BAND instead: only detections with
    min_confidence <= c < max_confidence. That is how a city already live at one
    threshold receives the labels a lower one adds (issue #20): PS is insert-only, so
    the labels the server already holds must not be sent again.

    mask_rig drops detections that are too steeply below the horizon to be on the street
    at all — they are on the camera vehicle (see detectors.on_camera_rig). Pass False only
    to reconstruct what a campaign that predates the mask actually sent; for anything
    being submitted now it stays on.
    """
    modified_data = data.copy()
    modified_data['pano'] = transform_pano(data['pano'])
    modified_data['labels'] = [
        {
            "pano_x": round(detection['x_normalized'] * modified_data['pano']['width']),
            "pano_y": round(detection['y_normalized'] * modified_data['pano']['height']),
            "confidence": detection['confidence']
        } for detection in data['detections']
        if detection['confidence'] >= min_confidence
        and (max_confidence is None or detection['confidence'] < max_confidence)
        and not (mask_rig and on_camera_rig(detection['y_normalized']))
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


def band_key(min_confidence: float, max_confidence: float) -> str:
    """The name a band campaign goes by in the record and in its sidecar: '0.3-0.55'."""
    return f"{min_confidence:g}-{max_confidence:g}"


def sidecar_path_for(file_path: str, min_confidence: float,
                     max_confidence: Optional[float]) -> Path:
    """The resume sidecar: `<file>.submitted` for a whole-file campaign, or
    `<file>.band-<min>-<max>.submitted` for a band campaign, so a band never touches the
    line numbers the base campaign recorded."""
    if max_confidence is None:
        return Path(f"{file_path}.submitted")
    return Path(f"{file_path}.band-{band_key(min_confidence, max_confidence)}.submitted")


def hash_and_count(input_file: Path) -> tuple:
    """(sha256 of the bytes, number of non-blank lines, byte length) of the JSONL.

    The hash is what ties the line-numbered sidecar to the file it was written against; the
    count is what a complete campaign's sidecar reaches, so the record can say how far there
    is still to go. Lines are counted the way the submit loop reads them (text mode, blank
    lines skipped), so the two can't disagree over a trailing newline or a stray blank. The
    byte length is what later makes an APPEND provable rather than indistinguishable from an
    edit (see append_check); it is counted from the same read as the hash, so the two always
    describe the same bytes.
    """
    digest = hashlib.sha256()
    size = 0
    with open(input_file, 'rb') as f:
        for chunk in iter(lambda: f.read(1 << 20), b''):
            digest.update(chunk)
            size += len(chunk)
    with open(input_file, 'r', encoding='utf-8') as f:
        lines = sum(1 for line in f if line.strip())
    return digest.hexdigest(), lines, size


def hash_prefix(input_file: Path, length: int) -> tuple:
    """(sha256, non-blank lines, ends-with-newline) of the file's first `length` bytes - the
    recorded prefix an append check has to re-hash.

    Counting is bytewise (split on b'\\n', ASCII-blank lines skipped) where hash_and_count
    reads text (universal newlines, Unicode whitespace). For a UTF-8 JSONL the two agree;
    where they could not - a lone \\r as a line break, a line of exotic whitespace - the
    append check compares this count against the recorded one and REFUSES on any
    disagreement, so the difference can only ever fail closed. A file shorter than `length`
    (it shrank under us) hashes fewer bytes and so fails the digest comparison, which is the
    same refusal.
    """
    digest = hashlib.sha256()
    lines, read, carry = 0, 0, b''
    with open(input_file, 'rb') as f:
        while read < length:
            chunk = f.read(min(1 << 20, length - read))
            if not chunk:
                break
            read += len(chunk)
            digest.update(chunk)
            parts = (carry + chunk).split(b'\n')
            carry = parts.pop()
            lines += sum(1 for part in parts if part.strip())
    if carry.strip():
        lines += 1
    return digest.hexdigest(), lines, not carry


def hash_through_lines(input_file: Path, lines: int) -> tuple:
    """(sha256, byte length, blank-inclusive sha256, blank-inclusive byte length) of the
    file's first `lines` non-blank lines - i.e. of the prefix a submission record with
    `total_lines` but no `total_bytes` describes. All four are None if the file holds fewer
    non-blank lines than that.

    This is the migration tool for a record written before `total_bytes` existed
    (`--prefix-digest N`): if the digest it prints matches the record's `sha256`, the
    already-submitted region is intact and the length beside it is the `total_bytes` to add
    to the record by hand. Counted bytewise, exactly as hash_prefix counts.

    Two candidates, because `lines` alone cannot say where the recorded file ENDED. A record
    always describes a whole file (hash_and_count hashes all of it), so a file that ended
    with blank line(s) has them inside its recorded digest while a count of non-blank lines
    stops short of them. The second pair therefore extends to the last blank line that
    follows the Nth non-blank one; it is (None, None) when no blank line follows, which is
    every file main.py wrote. Match whichever digest the record holds.
    """
    digest = hashlib.sha256()
    count, size = 0, 0
    with open(input_file, 'rb') as f:
        for raw in f:
            digest.update(raw)
            size += len(raw)
            if raw.strip():
                count += 1
                if count == lines:
                    break
        else:
            return None, None, None, None
        through_lines, through_bytes = digest.hexdigest(), size
        for raw in f:                        # ...and any blank tail the record would cover
            if raw.strip():
                break
            digest.update(raw)
            size += len(raw)
    if size == through_bytes:
        return through_lines, through_bytes, None, None
    return through_lines, through_bytes, digest.hexdigest(), size


def pano_id_of(raw: bytes) -> Optional[str]:
    """The `panorama_id` of one raw JSONL line, or None if the line can't yield one."""
    try:
        pano = (json.loads(raw.decode('utf-8-sig')) or {}).get('pano') or {}
        return pano.get('panorama_id') or pano.get('pano_id')
    except (ValueError, AttributeError, UnicodeDecodeError):
        return None


def repeated_pano_ids(input_file: Path, prefix_bytes: int) -> tuple:
    """(first repeated panorama_id, appended lines repeating one, appended lines read,
    appended lines that yield no id at all).

    New BYTES are not new PANOS. A gap-fill only fetches ids the run never processed, so the
    appended region of a genuine append is disjoint from the submitted one - while a doubled
    file, or a re-run after the gitignored `already_processed.txt` was lost, appends panos
    that are already live on the server. The sidecar cannot see that: its line numbers still
    line up, so every repeat would be POSTed a second time and duplicate its labels.

    One streaming pass with a set of the prefix's ids. `prefix_bytes` is known to end on a
    line boundary by the time this runs, so no line straddles the two regions. An appended
    line that yields no id is counted separately and refused by the caller: an unparseable
    one is only an error line the submit loop never sidecars, but a `pano` block WITHOUT a
    `panorama_id` transforms and POSTs happily (as `pano_id: null`) while being invisible
    here - the one shape that could let "new bytes" outrun "new panos". In the prefix the
    same line is simply not a known id, which can only make the check stricter.
    """
    seen, first, repeats, appended, idless, offset = set(), None, 0, 0, 0, 0
    with open(input_file, 'rb') as f:
        for raw in f:
            in_prefix = offset < prefix_bytes
            offset += len(raw)
            if not raw.strip():
                continue
            pano_id = pano_id_of(raw)
            if in_prefix:
                if pano_id is not None:
                    seen.add(pano_id)
                continue
            appended += 1
            if pano_id is None:
                idless += 1
            elif pano_id in seen:
                repeats += 1
                if first is None:
                    first = pano_id
    return first, repeats, appended, idless


def append_check(record: Dict[str, Any], input_file: Path, total_lines: int,
                 total_bytes: int) -> tuple:
    """Is this file's changed hash explained by a PURE APPEND to the recorded content?

    Issue #59: `main.py --gap-fill-only` (#32) adds panos to the end of a results.jsonl that
    may already be submitted, and "submit, then gap-fill, then submit the rest" is the
    expected sequence for the GSV runs. An append leaves every recorded line at the line
    number the sidecar holds for it, so resuming is exactly right - but a whole-file hash
    cannot tell that from an edit. The recorded byte length can: re-hash exactly that many
    bytes and compare against the recorded digest.

    Fail-closed. Anything that leaves the already-submitted region in doubt is NOT an
    append: a record with no length (written before this check existed), a file that is
    shorter or the same size, a prefix that no longer hashes the same, a prefix whose last
    line has no newline (so the added bytes ran onto it instead of starting a new line), or
    a prefix whose line count is missing from the record or disagrees with it. Nor is new
    bytes enough: every appended line must name a pano this file has not submitted before
    (see repeated_pano_ids).

    Returns:
        (note, reason): exactly one is not None. `note` is the "N new line(s)" line to print
        when the append is proven; `reason` says why it could not be, for the refusal.
    """
    recorded_bytes = record.get('total_bytes')
    recorded_lines = record.get('total_lines')
    recorded_digest = record.get('sha256')
    if not isinstance(recorded_bytes, int) or isinstance(recorded_bytes, bool) or recorded_bytes <= 0:
        migrate = (
            f"Check the recorded prefix with `python send_to_ps.py {input_file.as_posix()} "
            f"--prefix-digest {recorded_lines}`; if the sha256 it prints is the record's, add "
            f'its byte length to the record as "total_bytes" and re-run under the normal guard. '
            f"Prefer that to --ignore-submission-guard, which also silences the lost-sidecar "
            f"and wrong-endpoint checks for this run."
            if isinstance(recorded_lines, int) else
            "The record has no total_lines either, so there is no prefix to check at all: "
            "repair the record by hand (`git log` on it) rather than submitting against it.")
        return None, (
            f"the record predates the append check, so it carries no byte length "
            f"(`total_bytes`) and a pure append - what a gap-fill does - cannot be told from "
            f"an edit. {migrate}")
    if total_bytes <= recorded_bytes:
        verb = "truncated" if total_bytes < recorded_bytes else "edited in place"
        return None, (f"the file is {total_bytes} bytes against the {recorded_bytes} recorded, "
                      f"so it was {verb}, not appended to.")
    prefix_digest, prefix_lines, ends_with_newline = hash_prefix(input_file, recorded_bytes)
    if prefix_digest != recorded_digest:
        return None, (f"its first {recorded_bytes} bytes no longer hash to the recorded digest "
                      f"({prefix_digest[:12]}... != {str(recorded_digest)[:12]}...), so the "
                      f"already-submitted records were edited, not merely appended to.")
    if not ends_with_newline:
        return None, (f"the recorded {recorded_bytes} bytes do not end with a newline, so what "
                      f"was added ran onto the last submitted record instead of starting a new "
                      f"line.")
    if not isinstance(recorded_lines, int) or isinstance(recorded_lines, bool):
        return None, (f"the record carries a byte length but no line count, so the {prefix_lines} "
                      f"line(s) in the recorded {recorded_bytes} bytes cannot be checked against "
                      f"it; repair the record by hand rather than submitting against it.")
    if prefix_lines != recorded_lines:
        return None, (f"the recorded {recorded_bytes} bytes hold {prefix_lines} line(s) but the "
                      f"record says {recorded_lines}, so the record does not describe them.")
    repeated, repeats, appended, idless = repeated_pano_ids(input_file, recorded_bytes)
    if repeats:
        return None, (f"the bytes are new but the panos are not - {repeats} of the {appended} "
                      f"appended line(s) carry a panorama_id the submitted region already "
                      f"holds (first: {repeated}). A gap-fill only fetches ids the run never "
                      f"processed, so a real append is disjoint; this is a doubled file, or a "
                      f"re-run after `already_processed.txt` was lost, and submitting it would "
                      f"POST those panos a second time and duplicate their labels.")
    if idless:
        return None, (f"{idless} of the {appended} appended line(s) name no pano - an "
                      f"unparseable line (a partial write, which the submit loop would only "
                      f"log), or a pano block with no panorama_id, which POSTs as pano_id "
                      f"null. Either way it cannot be checked against what is already live, "
                      f"so the append is not provably new. Repair or drop those lines.")
    return (f"{input_file.name} has GROWN since the recorded submission: its first "
            f"{prefix_lines} line(s) are byte-for-byte unchanged and {appended} new line(s) "
            f"were appended, none repeating a panorama_id already submitted (a gap-fill, "
            f"issue #32). The sidecar's line numbers still index the same records, so this is "
            f"an ordinary resume: the new lines are submitted and the record is rewritten with "
            f"the new hash and length."), None


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
                       record_path: Path, sidecar_path: Path, total_lines: int,
                       total_bytes: int, max_confidence: Optional[float] = None) -> Optional[str]:
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

    The one change that is not a lie is a pure APPEND (issue #59): a gap-fill adds panos to
    the end of the file, every recorded line keeps its number, and resuming is correct. That
    case is proven byte-for-byte by append_check and returned as a note to print; every
    other difference still refuses.

    A BAND campaign (max_confidence set, issue #20) is the one legitimate way to send a
    second threshold to a server: it ships exactly the labels between the new threshold and
    the one the server holds. So it is allowed only on top of a complete campaign at
    exactly max_confidence, on the unchanged file, and it resumes from its own sidecar
    (see check_band_state).

    Returns:
        A note to print when the file grew by a pure append, else None.

    Raises:
        ValueError: if the input file changed; if the sidecar accounts for fewer lines than
            the record says already went to this endpoint; if the sidecar holds lines that
            went to a different endpoint than the one being submitted to; or if this run's
            --min-confidence differs from the one this endpoint was submitted at.
    """
    if max_confidence is not None:
        check_band_state(record, digest, submitted_lines, endpoint_url, min_confidence,
                         max_confidence, input_file, record_path, sidecar_path, total_lines)
        return None
    if not record:
        return None

    append_note = None
    recorded_digest = record.get('sha256')
    if recorded_digest and recorded_digest != digest:
        append_note, reason = append_check(record, input_file, total_lines, total_bytes)
        if append_note is None:
            raise ValueError(
                f"{input_file.name} has changed since the submission recorded in {record_path.name} "
                f"(sha256 {digest[:12]}... != {recorded_digest[:12]}...). {sidecar_path.name} holds "
                f"line numbers against the old file, so resuming would skip and re-send the wrong "
                f"records. A pure append - a gap-fill (issue #32) on an already-submitted run - "
                f"would resume instead, but this is not one: {reason} Restore the file that was "
                f"submitted, or start a new campaign under a new name. --ignore-submission-guard "
                f"overrides."
            )

    endpoint = canonical_endpoint(endpoint_url)
    states = record['endpoints']
    already = states.get(endpoint, {}).get('submitted_lines', 0)
    if already > len(submitted_lines):
        # Judged against the file as it is NOW, not as the record found it: a run that was
        # complete and has since been gap-filled has lines left to send, and saying otherwise
        # would either lose them or push the user to an override that re-POSTs the lot.
        recorded_lines = record.get('total_lines')
        recorded_lines = recorded_lines if isinstance(recorded_lines, int) else None
        covered = recorded_lines is not None and already >= recorded_lines
        grown = total_lines - recorded_lines if recorded_lines is not None else 0
        lost = ("The resume sidecar is missing, truncated, or from a different machine - "
                "submitting now would re-POST records that are already live and duplicate "
                "their labels. Restore the sidecar first (a backup copy). ")
        if grown > 0 and covered:
            detail = (f"That campaign covered the whole file as recorded, but {input_file.name} "
                      f"has since grown by {grown} line(s), and without the sidecar there is no "
                      f"way to tell which lines those are - the {already} already-submitted ones "
                      f"cannot be skipped. Restore the sidecar (a backup copy); it is the only "
                      f"thing that separates the new lines from the live ones. ")
        elif grown > 0:
            detail = (lost + f"{input_file.name} has also grown by {grown} line(s) since that "
                      f"record, so the sidecar is the only thing that tells the new lines from "
                      f"the live ones. ")
        elif covered:
            detail = "That campaign ran to completion, so there is nothing left to send there. "
        else:
            detail = lost
        raise ValueError(
            f"{record_path.name} says {already} line(s) of {input_file.name} were already "
            f"submitted to {endpoint}, but {sidecar_path.name} accounts for only "
            f"{len(submitted_lines)}. " + detail + "--ignore-submission-guard overrides."
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
        # The band route only exists DOWNWARD: a band adds the labels between a lower
        # threshold and the one the server holds. When this run's threshold is ABOVE the
        # recorded one the server already has these labels and more, so naming a band would
        # print an impossible `--min-confidence 0.55 --max-confidence 0.3` (a band is
        # [min, max)). Say what is actually true instead.
        if min_confidence < recorded_confidence:
            route = (f"to ADD the labels a lower threshold finds, send them as a band: "
                     f"--min-confidence {min_confidence} --max-confidence {recorded_confidence}.")
        else:
            route = (f"This run's threshold is HIGHER, so {endpoint} already holds every label "
                     f"it would send - there is nothing to add, and no band goes upward.")
        raise ValueError(
            f"{record_path.name} says {endpoint} holds labels down to --min-confidence "
            f"{recorded_confidence}, but this run uses {min_confidence}. One server should hold "
            f"one threshold's labels, and the record can only count at one. Use "
            f"--min-confidence {recorded_confidence}; {route} --ignore-submission-guard overrides."
        )

    return append_note


def check_band_state(record: Dict[str, Any], digest: str, band_lines: Set[int],
                     endpoint_url: str, min_confidence: float, max_confidence: float,
                     input_file: Path, record_path: Path, sidecar_path: Path,
                     total_lines: int) -> None:
    """Refuse a band campaign that would not be exactly "the labels this server lacks".

    The band [min, max) is only new to the server if the server holds everything at or above
    max and nothing below it, and only indexes the right records if the file is the one the
    base campaign was recorded against. The band's own sidecar is endpoint-agnostic like the
    base one, so the same two lies apply to it (lost, or carried over from another server)
    and are caught the same way from the per-endpoint band entry in the record.
    """
    if not min_confidence < max_confidence:
        raise ValueError(f"--min-confidence {min_confidence} must be below --max-confidence "
                         f"{max_confidence}: a band is [min, max).")
    endpoint = canonical_endpoint(endpoint_url)
    state = (record.get('endpoints') or {}).get(endpoint) if record else None
    if not state:
        raise ValueError(
            f"{record_path.name} records no campaign to {endpoint}, so there is nothing for a "
            f"band [{min_confidence}, {max_confidence}) to sit on top of. Submit the file "
            f"normally (--min-confidence {min_confidence}, no --max-confidence) instead."
        )
    if record.get('sha256') != digest:
        raise ValueError(
            f"{input_file.name} is not the file {record_path.name} was recorded against "
            f"(sha256 {digest[:12]}... != {str(record.get('sha256'))[:12]}...). A band re-reads "
            f"the recorded lines, so it needs that exact file; if the file was gap-filled, "
            f"submit the appended lines normally first."
        )
    recorded_lines = record.get('total_lines')
    if state.get('submitted_lines', 0) < (recorded_lines if isinstance(recorded_lines, int)
                                          else total_lines):
        raise ValueError(
            f"{endpoint} holds {state.get('submitted_lines', 0)} of {recorded_lines} lines of "
            f"{input_file.name}; a band only makes sense on top of a COMPLETE campaign. Finish "
            f"the base campaign first."
        )
    key = band_key(min_confidence, max_confidence)
    bands = state.get('bands') or {}
    held = state.get('min_confidence')
    if held is None:
        # A record written before `min_confidence` was tracked. There is no way to tell what
        # tier the server holds, and guessing either way risks a re-send or a gap.
        raise ValueError(
            f"{record_path.name} records a campaign to {endpoint} but no --min-confidence, so "
            f"it predates threshold tracking and cannot say what tier that server holds. A band "
            f"is only safe on top of a known tier. Establish it by hand: confirm what was "
            f"submitted, add \"min_confidence\": <tier> to that endpoint's entry in "
            f"{record_path.name}, then re-run. (--ignore-submission-guard overrides, but it "
            f"also silences the lost- and foreign-sidecar checks.)"
        )
    already = (bands.get(key) or {}).get('submitted_lines', 0)
    # A band that already completed moved the endpoint's threshold down to its floor, so a
    # re-run of that same band is an idempotent resume (nothing left to send), not a gap.
    # "Completed" must be judged against the file as it is NOW, not just from the record:
    # a gap-fill (issue #32) appends panos AFTER a band completes, the base campaign ships
    # them at the band's floor via the append path, and an accidental re-run of the band
    # command would then find a sidecar that never saw those line numbers and POST their
    # [min, max) labels a second time. PS is insert-only, so that is permanent.
    band_complete = key in bands and already >= total_lines
    if held != max_confidence and not (band_complete and held == min_confidence):
        if key in bands and held == min_confidence:
            raise ValueError(
                f"the {key} band to {endpoint} completed over {already} line(s), but "
                f"{input_file.name} now holds {total_lines}. Those {total_lines - already} new "
                f"line(s) are a gap-fill, and they belong to the BASE campaign at the tier the "
                f"server now holds - re-running the band would re-send the band labels of every "
                f"line the current sidecar is missing. Submit them normally instead: "
                f"--min-confidence {min_confidence} (no --max-confidence)."
            )
        raise ValueError(
            f"{endpoint} holds labels down to --min-confidence {held}, so the band that adds "
            f"the next tier is --max-confidence {held}, not {max_confidence}: anything else "
            f"would re-send labels the server has (PS is insert-only) or leave a gap."
        )
    # A file with nothing stored in [min, max) would run to "completion" without a single
    # POST - every line marked done as an empty band - and the completion branch would then
    # drop the endpoint's recorded threshold to `min_confidence`, so the record would claim
    # the server holds a tier it has never been sent. Runs from before the storage floor
    # (#28) are exactly this shape: `results.jsonl` holds nothing below 0.55, so the obvious
    # first command on such a city is the one that corrupts its record.
    if count_band_labels_in_file(input_file, min_confidence, max_confidence) == 0:
        floor = _storage_floor_for(input_file)
        why = ("" if floor is None or floor < max_confidence else
               f" {input_file.name} comes from a run whose detection_storage_floor is {floor}, "
               f"at or above --max-confidence {max_confidence}, so it can never hold one.")
        raise ValueError(
            f"{input_file.name} holds no labels at all in [{min_confidence}, {max_confidence}), "
            f"so this band would mark every line done without sending anything and then record "
            f"{endpoint} as holding labels down to {min_confidence} - which it would not.{why} "
            f"Re-infer the run's panos at the current storage floor first "
            f"(scripts/reinfer.py) and ship the band from the file that produces."
        )
    if already > len(band_lines):
        if band_complete:
            # Nothing is left to send, so the missing sidecar cannot cause a re-send: the
            # record accounts for every line itself. Refusing here would push the user to
            # --ignore-submission-guard, which silences the checks that DO still matter -
            # and which, with an empty sidecar, would then re-POST the whole band. So this
            # is a hard stop, not a warning: CampaignComplete is NOT a ValueError and is
            # never downgraded by the override.
            raise CampaignComplete(
                f"the {key} band to {endpoint} is already complete over all {total_lines} "
                f"line(s) per {record_path.name}; {sidecar_path.name} accounts for only "
                f"{len(band_lines)}, so it was lost or truncated. Nothing is left to send. "
                f"Restore the sidecar if you want a resumable record of it."
            )
        raise ValueError(
            f"{record_path.name} says {already} line(s) of the {key} band already went to "
            f"{endpoint}, but {sidecar_path.name} accounts for only {len(band_lines)}. The band "
            f"sidecar is missing or truncated; submitting now would re-send band labels that "
            f"are already live. Restore it first. --ignore-submission-guard overrides."
        )
    elsewhere = sorted(url for url, st in (record.get('endpoints') or {}).items()
                       if url != endpoint and ((st.get('bands') or {}).get(key) or {})
                       .get('submitted_lines'))
    if len(band_lines) > already and elsewhere:
        raise ValueError(
            f"{sidecar_path.name} holds {len(band_lines)} line(s) but {record_path.name} says "
            f"only {already} of the {key} band went to {endpoint}; the rest went to "
            f"{', '.join(elsewhere)}. Move the band sidecar aside so this endpoint starts "
            f"from line 1. --ignore-submission-guard overrides."
        )


def first_pano_source(input_file: Path) -> Optional[str]:
    """`pano.source` of the first record - a results file is single-source, so that is the
    file's source. None for an empty or malformed head."""
    with open(input_file, 'r', encoding='utf-8') as f:
        for line in f:
            if line.strip():
                try:
                    return (json.loads(line).get('pano') or {}).get('source')
                except (ValueError, AttributeError):
                    return None
    return None


def check_position_state(input_file: Path, digest: str) -> Optional[Dict[str, Any]]:
    """Refuse (ValueError) to submit a Mapillary file whose pano-position check is missing,
    stale or flagged; return the check otherwise (None for a non-Mapillary file).

    SidewalkWebpage#5361: Mapillary SfM sequences can sit 8-10 m off the street as a block
    and every label inherits it 1:1. main.py ends every run with the check and writes
    position_check.json beside results.jsonl (or <stem>.position_check.json beside a
    scripts/reposition.py output); this is the fail-closed half - the only way to ship an
    unchecked or flagged file is --ignore-position-check. Only Mapillary carries two
    positions to choose between, so other sources are not gated. Staleness is the
    results file's sha256 against the one the check recorded, so a check cannot vouch for
    a file edited after it ran.
    """
    if first_pano_source(input_file) != 'mapillary':
        return None
    check, reason = position_check.load_check(input_file)
    run_dir = input_file.parent
    base = f"python scripts/position_check.py {run_dir.as_posix()}"
    rerun = base if input_file.name == 'results.jsonl' else f"{base} --results {input_file.as_posix()}"
    if check is None:
        raise ValueError(f"{reason}. Run: {rerun}  (--ignore-position-check overrides)")
    recorded = check.get('results_sha256')
    if recorded != digest:
        raise ValueError(
            f"{position_check.check_path_for(input_file).name} describes a different version of "
            f"{input_file.name} (sha256 {digest[:12]}... != {str(recorded)[:12]}...): the file changed "
            f"after the check ran, or the check predates results_sha256. Re-run: {rerun}  "
            f"(--ignore-position-check overrides)")
    flagged = check.get('flagged_sequences') or []
    if flagged:
        raise ValueError(
            f"{len(flagged)} sequence(s) in {input_file.name} sit off the street on the submitted "
            f"position and the other Mapillary field fixes it ({position_check.check_path_for(input_file).name}). "
            f"Run: python scripts/reposition.py {input_file.as_posix()} --from-check, check the output "
            f"with {base} --results <output>, and submit that file instead  "
            f"(--ignore-position-check overrides)")
    return check


def count_labels(input_file: Path, line_numbers: Set[int], min_confidence: float,
                 max_confidence: Optional[float] = None) -> int:
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
                labels += len(transform_record(json.loads(line), min_confidence,
                                               max_confidence)['labels'])
    return labels


def _storage_floor_for(input_file: Path) -> Optional[float]:
    """The run's recorded ``detection_storage_floor``, or None if it cannot be read.

    Only ever used to explain a refusal, never to make one, so every failure to read it
    (no manifest, a submission artifact sitting outside a run dir, unreadable JSON) is a
    silent None rather than an error.
    """
    try:
        manifest = json.loads((input_file.parent / 'manifest.json').read_text(encoding='utf-8'))
    except (OSError, ValueError):
        return None
    floor = manifest.get('detection_storage_floor')
    return floor if isinstance(floor, (int, float)) else None


def count_band_labels_in_file(input_file: Path, min_confidence: float,
                              max_confidence: Optional[float],
                              mask_rig: bool = True) -> int:
    """How many labels the WHOLE file holds in [min_confidence, max_confidence).

    Separate from ``count_labels`` because that one answers "what did these sidecar lines
    send?" and this one answers "is there anything here to send at all?" — the question a
    band has to settle before it marks a single line done (see ``check_band_state``).

    mask_rig=False reconstructs what a campaign sent BEFORE the nadir mask existed, which
    is what a derived record has to match; see ``transform_record``.
    """
    labels = 0
    with open(input_file, 'r', encoding='utf-8') as f:
        for line in f:
            if line.strip():
                labels += len(transform_record(json.loads(line), min_confidence,
                                               max_confidence, mask_rig)['labels'])
    return labels


def write_submission_record(record_path: Path, input_file: Path, digest: str, total_lines: int,
                            total_bytes: int, submitted_lines: Set[int], endpoint_url: str,
                            min_confidence: float, previous: Dict[str, Any],
                            check: Optional[Dict[str, Any]] = None,
                            max_confidence: Optional[float] = None) -> None:
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
    if max_confidence is not None:
        # A band campaign (issue #20) on top of the base one: its own line count and label
        # count under `bands`, recounted from ITS sidecar. Once it covers the whole file the
        # server holds everything down to the band's floor, so the endpoint's threshold and
        # label total move down with it - the base count is kept as recorded, since the base
        # sidecar may legitimately have been moved aside since.
        key = band_key(min_confidence, max_confidence)
        # A band normally sits on a recorded base campaign, so `state` already carries the
        # base's counts. It can be reached without one via --ignore-submission-guard on a
        # file that has no record (the reinfer -> band route before it wrote one), and the
        # branch below would then write an endpoint entry with `bands` but no
        # `submitted_lines` - a shape load_submission_record refuses as unreadable, which
        # strands the campaign entirely. Write the honest 0 instead, so the next run says
        # "holds 0 of N lines, finish the base campaign first" and can still be repaired.
        state.setdefault("submitted_lines", 0)
        state.setdefault("labels_submitted", 0)
        bands = dict(state.get('bands') or {})
        band = dict(bands.get(key) or {})
        band_labels = count_labels(input_file, submitted_lines, min_confidence, max_confidence)
        was_complete = band.get('submitted_lines', 0) >= total_lines
        band.update({
            "submitted_lines": len(submitted_lines),
            "labels_submitted": band_labels,
            "first_submission_utc": band.get('first_submission_utc', now),
            "last_submission_utc": now,
            "last_run_host": socket.gethostname(),
        })
        bands[key] = band
        state["bands"] = bands
        if len(submitted_lines) >= total_lines and not was_complete:
            state["labels_submitted"] = state.get('labels_submitted', 0) + band_labels
            state["min_confidence"] = min_confidence
        state["last_submission_utc"] = now
        state["last_run_host"] = socket.gethostname()
    else:
        state.update({
            "submitted_lines": len(submitted_lines),
            "labels_submitted": count_labels(input_file, submitted_lines, min_confidence),
            "min_confidence": min_confidence,
            "first_submission_utc": state.get('first_submission_utc', now),
            "last_submission_utc": now,
            "last_run_host": socket.gethostname(),
        })
    if check is not None:  # the position verdict this campaign shipped under
        if check.get('overridden'):
            state["position_check"] = {"overridden": True, "reason": check.get('reason')}
        else:
            state["position_check"] = {"checked_at": check.get('checked_at'),
                                       "results_sha256": check.get('results_sha256'),
                                       "flagged": len(check.get('flagged_sequences') or [])}
    states[endpoint] = state
    record = {
        "input_file": input_file.name,
        "sha256": digest,
        "total_lines": total_lines,
        # The length the digest covers, so that a later hash mismatch can be proven to be a
        # pure append (a gap-fill) rather than an edit - see append_check.
        "total_bytes": total_bytes,
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
    ignore_position_check: bool = False,
    max_confidence: Optional[float] = None,
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
        limit: Stop after this many records are TOUCHED *this run*; already-submitted lines
            don't count against it, so successive capped runs walk the file. "Touched", not
            "POSTed": in band mode a record with nothing in the band is marked done without
            a request (see ``max_confidence``) and still counts, so a capped band run
            advances through the file by `limit` lines whatever the label counts are.
        ignore_guard: Submit even when the submission record disagrees with the resume
            sidecar (see ``check_resume_state``).
        ignore_position_check: Submit a Mapillary file whose pano-position check is missing,
            stale or flagged (see ``check_position_state``).
        max_confidence: Send a BAND, ``min_confidence <= c < max_confidence``, on top of a
            complete campaign at exactly ``max_confidence`` (issue #20). Records with no band
            label are not POSTed (the server already holds them as checked) but are marked
            done in the band's own sidecar, ``<file>.band-<min>-<max>.submitted``.
    """
    input_file = Path(file_path)

    # Validate file exists.
    if not input_file.exists():
        print(f"Error: File '{file_path}' does not exist.")
        return

    # Fail before opening the file: a cleartext key leaks on the very first request.
    check_endpoint_security(endpoint_url, None if dry_run else api_key)

    # Load resume state: line numbers that already got a 200 on a previous run (or, for a
    # band campaign, lines already handled by that band - sent, or empty and skipped).
    sidecar_path = sidecar_path_for(file_path, min_confidence, max_confidence)
    submitted_lines = load_submitted_lines(sidecar_path)

    # ...and check it still describes this file and this endpoint. A dry run POSTs nothing
    # and writes nothing, so it is exempt from all of it (it must stay usable on a stale
    # file or beside a broken record); the override downgrades every refusal to a warning
    # (and an unreadable record to a fresh one).
    record_path = submission_record_path(file_path)
    digest, total_lines, total_bytes = hash_and_count(input_file)
    previous_record: Dict[str, Any] = {}
    append_note: Optional[str] = None
    try:
        if not dry_run:
            previous_record = load_submission_record(record_path)
            append_note = check_resume_state(previous_record, digest, submitted_lines,
                                             endpoint_url, min_confidence, input_file,
                                             record_path, sidecar_path, total_lines, total_bytes,
                                             max_confidence)
    except CampaignComplete as e:
        # Not a refusal: nothing is left to send, so stop here without POSTing and without
        # rewriting the record (which would only restate what it already says).
        print(f"Nothing to do: {e}")
        return
    except ValueError as e:
        if not ignore_guard:
            raise
        print(f"WARNING (--ignore-submission-guard): {e}")

    # ...and that its pano positions were checked and passed (SidewalkWebpage#5361). Same
    # shape: a dry run is exempt, the override downgrades the refusal to a warning.
    position_state: Optional[Dict[str, Any]] = None
    try:
        if not dry_run:
            position_state = check_position_state(input_file, digest)
    except ValueError as e:
        if not ignore_position_check:
            raise
        print(f"WARNING (--ignore-position-check): {e}")
        # The record must say the gate was bypassed, and why — otherwise an overridden
        # Mapillary campaign is indistinguishable from an ungated GSV one.
        position_state = {'overridden': True, 'reason': str(e)}

    success_count = 0

    empty_band_count = 0
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
    if append_note:
        print(f"APPEND: {append_note}")
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
                    payload = transform_record(json_data, min_confidence, max_confidence)
                    filtered_detections += len(json_data.get('detections', [])) - len(payload['labels'])

                    if max_confidence is not None and not payload['labels']:
                        # Nothing in the band for this pano: the server already has it as
                        # "checked", so no POST - but the band's sidecar still marks the
                        # line handled, or the band could never be recorded as complete.
                        empty_band_count += 1
                        if not dry_run:
                            f_sidecar.write(f"{line_number}\n")
                            f_sidecar.flush()
                        continue

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
    if not dry_run and (success_count or (max_confidence is not None and empty_band_count)):
        write_submission_record(record_path, input_file, digest, total_lines, total_bytes,
                                load_submitted_lines(sidecar_path), endpoint_url,
                                min_confidence, previous_record, position_state,
                                max_confidence)
        record_written = True

    # Print summary.
    print("-" * 50)
    print("Interrupted." if interrupted else "Processing complete!")
    print(f"Successfully processed:        {success_count} records")
    print(f"Skipped (already submitted):   {skipped_count} records")
    print(f"Errors encountered:            {error_count} records")
    if max_confidence is None:
        print(f"Detections below --min-confidence {min_confidence} (not submitted): {filtered_detections}")
    else:
        print(f"Band [{min_confidence}, {max_confidence}): detections outside it (not submitted): "
              f"{filtered_detections}; records with nothing in the band (not POSTed, marked "
              f"done): {empty_band_count}")
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
        help="Handle at most N records this run, then stop. Already-submitted lines don't "
             "count, so repeated capped runs walk the file — use it to send a handful and "
             "inspect them in Project Sidewalk before committing to the whole city. In band "
             "mode a record with nothing in the band is handled without a POST and still "
             "counts, so N caps lines advanced, not labels sent."
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
        "--ignore-position-check",
        action="store_true",
        help="Submit a Mapillary file whose pano-position check (position_check.json beside "
             "it) is missing, stale or flagged. Only for a case you have checked by hand: the "
             "check is what stops a run whose SfM positions drifted off the street "
             "(SidewalkWebpage#5361) from placing every label metres off."
    )
    parser.add_argument(
        "--prefix-digest", type=int, metavar="N",
        help="Print the sha256 and byte length of the file's first N non-blank lines, then "
             "exit without submitting anything. Use it to migrate a submission record written "
             "before the record carried a byte length: run it with the record's total_lines, "
             "and if the sha256 matches the record's, the already-submitted region is intact - "
             "add the printed length to the record as \"total_bytes\" and a later gap-fill "
             "append resumes under the normal guard. A record describes a whole file, so if "
             "that file ended with blank line(s) its digest covers them: a second pair is then "
             "printed that includes the blank tail. Use whichever sha256 the record holds."
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
    parser.add_argument(
        "--max-confidence", type=float, default=None,
        help="Submit only the BAND --min-confidence <= confidence < this value, on top of a "
             "campaign already complete at exactly this threshold (issue #20: the operating "
             "point moved from 0.55 to 0.30, and PS is insert-only, so a live city gets the "
             "new labels as `--min-confidence 0.30 --max-confidence 0.55`). Records with "
             "nothing in the band are not POSTed. Progress lives in "
             "<file>.band-<min>-<max>.submitted; the record gains a `bands` entry per endpoint."
    )
    args = parser.parse_args()

    if args.limit is not None and args.limit < 1:
        parser.error("--limit must be at least 1.")
    if args.max_confidence is not None and not args.min_confidence < args.max_confidence:
        parser.error(f"--min-confidence {args.min_confidence} must be below --max-confidence "
                     f"{args.max_confidence}.")

    if args.prefix_digest is not None:
        if args.prefix_digest < 1:
            parser.error("--prefix-digest must be at least 1.")
        if not Path(args.jsonl_file).exists():
            raise SystemExit(f"Error: File '{args.jsonl_file}' does not exist.")
        digest, size, blank_digest, blank_size = hash_through_lines(
            Path(args.jsonl_file), args.prefix_digest)
        if digest is None:
            raise SystemExit(f"Error: {args.jsonl_file} holds fewer than {args.prefix_digest} "
                             f"non-blank line(s).")
        print(f"sha256      {digest}")
        print(f"total_bytes {size}")
        if blank_digest is not None:
            print(f"...or, if the file as recorded ended with the blank line(s) that follow "
                  f"line {args.prefix_digest}:")
            print(f"sha256      {blank_digest}")
            print(f"total_bytes {blank_size}")
        return

    api_key = os.environ.get(args.api_key_env)
    try:
        process_jsonl_file(args.jsonl_file, args.endpoint, api_key, args.dry_run,
                           args.min_confidence, args.limit, args.ignore_submission_guard,
                           args.ignore_position_check, args.max_confidence)
    except ValueError as e:
        raise SystemExit(f"Error: {e}")


if __name__ == "__main__":
    main()
