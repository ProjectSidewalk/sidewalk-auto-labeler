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
import json
import os
import time
from contextlib import nullcontext
from typing import Dict, Any, Optional, Set
from pathlib import Path

import requests

DEFAULT_ENDPOINT_URL = "http://localhost:9000/ai/submitLabelsOnPano"
MAX_ATTEMPTS = 3
RETRY_BACKOFF_SECONDS = [2, 8]


def transform_record(data: Dict[str, Any]) -> Dict[str, Any]:
    """
    Convert a main.py JSONL record into the payload expected by Project Sidewalk.

    Detections are converted from normalized coordinates to pixel coordinates using the
    pano dimensions stored in the record, renamed 'detections' -> 'labels'.
    """
    modified_data = data.copy()
    modified_data['labels'] = [
        {
            "pano_x": round(detection['x_normalized'] * modified_data['pano']['width']),
            "pano_y": round(detection['y_normalized'] * modified_data['pano']['height']),
            "confidence": detection['confidence']
        } for detection in data['detections']
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
    """
    input_file = Path(file_path)

    # Validate file exists.
    if not input_file.exists():
        print(f"Error: File '{file_path}' does not exist.")
        return

    # Load resume state: line numbers that already got a 200 on a previous run.
    sidecar_path = Path(f"{file_path}.submitted")
    submitted_lines = load_submitted_lines(sidecar_path)

    success_count = 0
    error_count = 0
    skipped_count = 0

    print(f"Processing JSONL file: {file_path}")
    print(f"Target endpoint: {endpoint_url}")
    if dry_run:
        print("DRY RUN: payloads will be printed, not sent.")
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

                try:
                    # Parse a line of JSON and convert to the PS payload format.
                    json_data = json.loads(line)
                    payload = transform_record(json_data)

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
             "(matches Project Sidewalk's INTERNAL_API_KEY). If the variable is unset, no auth "
             "header is sent (works against a deployment that doesn't require it yet)."
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Print transformed payloads instead of POSTing them; no progress is recorded."
    )
    args = parser.parse_args()

    api_key = os.environ.get(args.api_key_env)
    process_jsonl_file(args.jsonl_file, args.endpoint, api_key, args.dry_run)


if __name__ == "__main__":
    main()
