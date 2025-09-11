#!/usr/bin/env python3
"""
JSONL File Processor and HTTP POST Client for AI label predictions for Project Sidewalk.

This script reads a JSONL (JSON Lines) file containing AI label predictions, applies modifications to each JSON object,
and sends the modified data via POST requests to a localhost endpoint from Project Sidewalk.
"""

import json
import requests
from typing import Dict, Any, Optional
from pathlib import Path


def send_to_project_sidewalk(data: Dict[str, Any], endpoint_url: str) -> Optional[requests.Response]:
    """
    Send a POST request with JSON data to the specified PS endpoint.

    Args:
        data: The JSON data to send in the POST request.
        endpoint_url: The target endpoint URL.

    Returns:
        The response object if successful, None if an error occurred.
    """
    modified_data = data.copy()
    # modified_data['pano']['panorama_id'] = modified_data['pano']['gsv_panorama_id']
    # modified_data.pop('gsv_panorama_id', None)  # Remove detections to avoid redundancy
    # modified_data['model_training_date'] = "08-21-2025"
    # modified_data['pano']['camera_heading'] = math.degrees(modified_data['pano']['camera_heading'])
    # modified_data['pano']['camera_pitch'] = math.degrees(modified_data['pano']['camera_pitch'])

    # Transform detections to format expected by Project Sidewalk (pixel coordinates rather than normalized).
    modified_data['labels'] = [
        {
            "pano_x": round(detection['x_normalized'] * modified_data['pano']['width']),
            "pano_y": round(detection['y_normalized'] * modified_data['pano']['height']),
            "confidence": detection['confidence']
        } for detection in data['detections']
    ]
    modified_data.pop('detections', None)

    print(json.dumps(modified_data, indent=4))

    # Send the POST request.
    try:
        headers = {
            'Content-Type': 'application/json',
            'Accept': 'application/json'
        }
        response = requests.post(
            endpoint_url,
            json=modified_data,
            headers=headers,
            timeout=30
        )

        # Attempt to print response if it's not 200 (success).
        if response.status_code != 200:
            try:
                print(json.dumps(response.json(), indent=2)) # Attempt to parse as JSON if applicable
            except json.JSONDecodeError:
                print(response.text) # If not JSON, print as plain text

        # Raise an exception for bad status codes.
        response.raise_for_status()

        return response

    except requests.exceptions.RequestException as e:
        print(f"Error sending POST request: {e}")
        return None


def process_jsonl_file(file_path: str, endpoint_url: str = "http://localhost:9000/ai/submitLabel") -> None:
    """
    Process a JSONL file containing detections from main.py by reading each line and sending POST requests to PS.

    Args:
        file_path: Path to the JSONL file to process.
        endpoint_url: The endpoint URL to send POST requests to.
    """
    input_file = Path(file_path)

    # Validate file exists.
    if not input_file.exists():
        print(f"Error: File '{file_path}' does not exist.")
        return

    processed_count = 0
    error_count = 0

    print(f"Processing JSONL file: {file_path}")
    print(f"Target endpoint: {endpoint_url}")
    print("-" * 50)

    try:
        with open(input_file, 'r', encoding='utf-8') as file:
            for line_number, line in enumerate(file, 1):
                line = line.strip()

                # Skip empty lines.
                if not line:
                    continue

                try:
                    # Parse a line of JSON.
                    json_data = json.loads(line)

                    # Send POST request.
                    response = send_to_project_sidewalk(json_data, endpoint_url)

                    if response:
                        processed_count += 1
                        # print(f"Line {line_number}: Successfully processed")
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
    print(f"Successfully processed: {processed_count} records")
    print(f"Errors encountered: {error_count} records")


if __name__ == "__main__":
    JSONL_FILE_PATH = "vancouver.jsonl"
    ENDPOINT_URL = "http://localhost:9000/ai/submitLabel" # TODO send directly to PS server from main.py after pilot

    process_jsonl_file(JSONL_FILE_PATH, ENDPOINT_URL)
