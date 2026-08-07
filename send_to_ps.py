#!/usr/bin/env python3
"""
JSONL File Processor and HTTP POST Client for AI label predictions for Project Sidewalk.

This script reads a JSONL (JSON Lines) file produced by main.py, converts each record's
normalized detections to the pixel-coordinate label format expected by Project Sidewalk,
and sends the records via POST requests to a Project Sidewalk ingest endpoint.

Usage:
    python send_to_ps.py bend.jsonl --endpoint http://localhost:9000/ai/submitLabelsOnPano

Submission progress is tracked in a sidecar file (<file>.submitted) so a re-run resumes
where it left off instead of re-POSTing every line.
"""

import argparse
import ipaddress
import json
import os
import time
from contextlib import nullcontext
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
    """True if `host` names the local machine, so a request to it never reaches the wire."""
    if not host:
        return False
    host = host.strip('[]').lower()          # strip the brackets of an IPv6 literal
    if host == 'localhost' or host.endswith('.localhost'):
        return True
    try:
        return ipaddress.ip_address(host).is_loopback
    except ValueError:
        return False                          # a real hostname; resolving it is not our job


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


def process_jsonl_file(
    file_path: str,
    endpoint_url: str = DEFAULT_ENDPOINT_URL,
    api_key: Optional[str] = None,
    dry_run: bool = False,
    min_confidence: float = OPERATIONAL_CONFIDENCE,
    limit: Optional[int] = None,
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

    success_count = 0
    error_count = 0
    skipped_count = 0
    filtered_detections = 0
    attempted = 0          # records touched this run; what --limit caps
    limit_reached = False

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

    # Print summary.
    print("-" * 50)
    print(f"Processing complete!")
    print(f"Successfully processed:        {success_count} records")
    print(f"Skipped (already submitted):   {skipped_count} records")
    print(f"Errors encountered:            {error_count} records")
    print(f"Detections below --min-confidence {min_confidence} (not submitted): {filtered_detections}")
    if limit_reached:
        print(f"Stopped at --limit {limit}. Re-run to continue from here"
              f"{'' if dry_run else f' ({sidecar_path.name} records what already landed)'}.")


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
                           args.min_confidence, args.limit)
    except ValueError as e:
        raise SystemExit(f"Error: {e}")


if __name__ == "__main__":
    main()
