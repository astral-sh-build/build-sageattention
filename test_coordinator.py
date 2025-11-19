#!/usr/bin/env python3
"""
Coordinator script for testing SageAttention wheels from GitHub Actions.

This script fetches artifact information from a GitHub Actions run and orchestrates
running run_test.py for each wheel via subprocess. The wheels are downloaded during
the Modal image build rather than locally.

Run with:
    GITHUB_TOKEN=... python test_coordinator.py <actions_job_url>

Example:
    GITHUB_TOKEN=... python test_coordinator.py https://github.com/astral-sh/build-sageattention/actions/runs/18633682272
"""

import os
import re
import subprocess
import sys
from pathlib import Path


def get_wheel_artifacts_from_github_actions(
    job_url: str, github_token: str
) -> tuple[str, str, str, list[dict[str, str]]]:
    """Get wheel artifact information from a GitHub Actions job.

    Returns a tuple of (owner, repo, run_id, list of artifact info dicts).
    Each artifact info dict contains 'name' and 'download_url'.
    """
    import requests

    # Parse the job URL to get owner, repo, run_id
    # Example: https://github.com/astral-sh/build-sageattention/actions/runs/18634809438/job/53124508228
    match = re.match(
        r"https://github\.com/([^/]+)/([^/]+)/actions/runs/(\d+)(?:/job/\d+)?", job_url
    )
    if not match:
        raise ValueError(f"Invalid GitHub Actions job URL: {job_url}")

    owner, repo, run_id = match.groups()

    print(f"Fetching artifacts from {owner}/{repo} run {run_id}...")

    # Get the artifacts for this run
    headers = {
        "Authorization": f"token {github_token}",
        "Accept": "application/vnd.github.v3+json",
    }

    artifacts_url = (
        f"https://api.github.com/repos/{owner}/{repo}/actions/runs/{run_id}/artifacts"
    )
    response = requests.get(artifacts_url, headers=headers)
    response.raise_for_status()

    artifacts = response.json()["artifacts"]
    print(f"Found {len(artifacts)} artifacts")

    # Collect artifact information (without downloading)
    wheel_artifacts = []
    for artifact in artifacts:
        artifact_name = artifact["name"]

        # Skip ARM artifacts
        if "aarch64" in artifact_name.lower() or "arm64" in artifact_name.lower():
            print(f"  Skipping ARM artifact: {artifact_name}")
            continue

        # Assume the artifact name is the wheel name (this is typical for wheel artifacts)
        wheel_artifacts.append({
            "name": artifact_name,
            "download_url": artifact["archive_download_url"],
        })
        print(f"  Found artifact: {artifact_name}")

    print(f"\nFound {len(wheel_artifacts)} wheel artifacts")
    return owner, repo, run_id, wheel_artifacts


def filter_x86_64_wheels(wheel_artifacts: list[dict[str, str]]) -> tuple[list[dict[str, str]], list[str]]:
    """Filter to x86_64 wheel artifacts only.

    Returns tuple of (x86_64_artifacts, skipped_arm_artifacts).
    """
    x86_64_artifacts = []
    skipped_arm_artifacts = []

    for artifact in wheel_artifacts:
        name = artifact["name"]
        # Skip ARM wheels explicitly
        if "aarch64" in name or "arm64" in name:
            skipped_arm_artifacts.append(name)
            continue
        # Include x86_64 wheels
        if "x86_64" in name or "manylinux" in name:
            x86_64_artifacts.append(artifact)

    return x86_64_artifacts, skipped_arm_artifacts


def run_test_for_wheel(
    wheel_artifact: dict[str, str],
    owner: str,
    repo: str,
    run_id: str,
    script_path: Path,
    github_token: str,
) -> bool:
    """Run the Modal test script for a single wheel.

    Returns True if all tests passed, False otherwise.
    """
    wheel_name = wheel_artifact["name"]
    print(f"\n{'=' * 80}")
    print(f"Running tests for: {wheel_name}")
    print(f"{'=' * 80}\n")

    env = os.environ.copy()
    env["WHEEL_NAME"] = wheel_name
    env["GITHUB_OWNER"] = owner
    env["GITHUB_REPO"] = repo
    env["GITHUB_RUN_ID"] = run_id
    env["GITHUB_TOKEN"] = github_token

    try:
        result = subprocess.run(
            ["uv", "run", "--with", "modal", "modal", "run", str(script_path)],
            env=env,
            check=True,
            capture_output=False,
        )
        return result.returncode == 0
    except subprocess.CalledProcessError:
        print(f"\n❌ Tests failed for {wheel_name}")
        return False


def main():
    if len(sys.argv) < 2:
        print("Usage: python test_coordinator.py <actions_job_url>")
        print("\nExample:")
        print(
            "  GITHUB_TOKEN=... python test_coordinator.py https://github.com/owner/repo/actions/runs/12345/job/67890"
        )
        sys.exit(1)

    job_url = sys.argv[1]

    # Get GitHub token
    github_token = os.environ.get("GITHUB_TOKEN")
    if not github_token:
        print("Error: GITHUB_TOKEN environment variable must be set")
        sys.exit(1)

    # Get wheel artifact information
    print("=" * 80)
    print("SageAttention Wheel Test Coordinator")
    print("=" * 80)
    print()

    owner, repo, run_id, wheel_artifacts = get_wheel_artifacts_from_github_actions(
        job_url, github_token
    )

    if not wheel_artifacts:
        print("Error: No wheel artifacts found in GitHub Actions run")
        sys.exit(1)

    # Filter to x86_64 wheels
    x86_64_artifacts, skipped_arm_artifacts = filter_x86_64_wheels(wheel_artifacts)

    if skipped_arm_artifacts:
        print(
            f"\nSkipped {len(skipped_arm_artifacts)} ARM wheels (Modal doesn't support ARM GPUs):"
        )
        for wheel_name in sorted(skipped_arm_artifacts):
            print(f"  - {wheel_name}")

    if not x86_64_artifacts:
        print("Error: No x86_64 wheel artifacts found")
        sys.exit(1)

    print(f"\nFound {len(x86_64_artifacts)} x86_64 wheels to test:")
    for artifact in sorted(x86_64_artifacts, key=lambda x: x["name"]):
        print(f"  - {artifact['name']}")

    # Get the path to run_test.py (should be in same directory as this script)
    script_dir = Path(__file__).parent
    test_script = script_dir / "run_test.py"

    if not test_script.exists():
        print(f"Error: Test script not found at {test_script}")
        sys.exit(1)

    # Run tests for each wheel
    print(f"\n{'=' * 80}")
    print(f"Starting tests for {len(x86_64_artifacts)} wheels")
    print(f"{'=' * 80}\n")

    results = {}
    for artifact in sorted(x86_64_artifacts, key=lambda x: x["name"]):
        success = run_test_for_wheel(artifact, owner, repo, run_id, test_script, github_token)
        results[artifact["name"]] = success

    # Print final summary
    print(f"\n{'=' * 80}")
    print("FINAL TEST SUMMARY")
    print(f"{'=' * 80}\n")

    passed = [name for name, success in results.items() if success]
    failed = [name for name, success in results.items() if not success]

    if passed:
        print(f"✅ Passed ({len(passed)}):")
        for wheel_name in passed:
            print(f"  - {wheel_name}")
        print()

    if failed:
        print(f"❌ Failed ({len(failed)}):")
        for wheel_name in failed:
            print(f"  - {wheel_name}")
        print()

    print(f"{'=' * 80}")
    print(f"Total: {len(results)} wheels tested")
    print(f"Passed: {len(passed)}")
    print(f"Failed: {len(failed)}")
    print(f"{'=' * 80}")

    if failed:
        sys.exit(1)
    else:
        print("\n✅ All wheels passed testing!")
        sys.exit(0)


if __name__ == "__main__":
    main()
