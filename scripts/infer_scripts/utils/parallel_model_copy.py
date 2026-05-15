#!/usr/bin/env python3

import argparse
import os
import shlex
import shutil
import subprocess
import sys
import threading
import time
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path


def run_command(cmd):
    subprocess.run(cmd, check=True)


def list_hdfs_entries(src_root):
    result = subprocess.run(
        ["hdfs", "dfs", "-ls", "-R", src_root],
        check=True,
        text=True,
        capture_output=True,
    )

    directories = set()
    files = []
    for raw_line in result.stdout.splitlines():
        line = raw_line.strip()
        if not line or line.startswith("Found "):
            continue

        parts = line.split(None, 7)
        if len(parts) < 8:
            continue

        entry_type = parts[0][0]
        entry_path = parts[-1]
        rel_path = os.path.relpath(entry_path, src_root)
        if rel_path == ".":
            continue

        if entry_type == "d":
            directories.add(rel_path)
        else:
            files.append((entry_path, rel_path))

    return sorted(directories), files


def list_local_entries(src_root):
    directories = set()
    files = []
    for root, dirnames, filenames in os.walk(src_root):
        root_path = Path(root)
        for dirname in dirnames:
            dir_path = root_path / dirname
            directories.add(os.path.relpath(dir_path, src_root))
        for filename in filenames:
            file_path = root_path / filename
            files.append((str(file_path), os.path.relpath(file_path, src_root)))

    return sorted(directories), files


def copy_local_file(src_path, dst_path):
    Path(dst_path).parent.mkdir(parents=True, exist_ok=True)
    shutil.copy2(src_path, dst_path)


def copy_hdfs_file(src_path, dst_path):
    dst_parent = Path(dst_path).parent
    dst_parent.mkdir(parents=True, exist_ok=True)
    run_command(["hdfs", "dfs", "-get", src_path, str(dst_parent)])


def copy_with_retry(copy_fn, src_path, dst_path, rel_path, max_retries, retry_delay_sec, log_lock):
    last_error = None
    attempts = max_retries + 1
    for attempt in range(1, attempts + 1):
        try:
            copy_fn(src_path, dst_path)
            return
        except Exception as exc:
            last_error = exc
            with log_lock:
                print(
                    f"WARNING: Copy failed for {rel_path} on attempt {attempt}/{attempts}: {exc}",
                    flush=True,
                )
            if attempt < attempts:
                time.sleep(retry_delay_sec * attempt)

    raise RuntimeError(
        f"Failed to copy {rel_path} after {attempts} attempts. Last error: {last_error}"
    ) from last_error


def main():
    parser = argparse.ArgumentParser(description="Copy model files in parallel.")
    parser.add_argument("--src", required=True, help="Source directory path.")
    parser.add_argument("--dst", required=True, help="Destination directory path.")
    parser.add_argument(
        "--workers",
        type=int,
        default=min(16, os.cpu_count() or 4),
        help="Number of concurrent copy workers.",
    )
    parser.add_argument(
        "--max-retries",
        type=int,
        default=3,
        help="Retry count for each file copy after the initial attempt fails.",
    )
    parser.add_argument(
        "--retry-delay-sec",
        type=float,
        default=1.0,
        help="Base delay in seconds between retries.",
    )
    parser.add_argument(
        "--source-type",
        choices=("hdfs", "local"),
        required=True,
        help="Source backend type.",
    )
    args = parser.parse_args()

    dst_root = Path(args.dst)
    dst_root.mkdir(parents=True, exist_ok=True)

    if args.source_type == "hdfs":
        directories, files = list_hdfs_entries(args.src)
        copy_fn = copy_hdfs_file
    else:
        directories, files = list_local_entries(args.src)
        copy_fn = copy_local_file

    for rel_dir in directories:
        (dst_root / rel_dir).mkdir(parents=True, exist_ok=True)

    if not directories and not files:
        print(f"INFO: No files found under {args.src}")
        return 0

    print(
        f"INFO: Copying {len(files)} files from {args.source_type} source "
        f"{shlex.quote(args.src)} to {shlex.quote(args.dst)} with {args.workers} workers.",
        flush=True,
    )

    completed = 0
    total = len(files)
    progress_lock = threading.Lock()
    log_lock = threading.Lock()

    def do_copy(entry):
        nonlocal completed
        src_path, rel_path = entry
        dst_path = dst_root / rel_path
        copy_with_retry(
            copy_fn=copy_fn,
            src_path=src_path,
            dst_path=str(dst_path),
            rel_path=rel_path,
            max_retries=max(0, args.max_retries),
            retry_delay_sec=max(0.0, args.retry_delay_sec),
            log_lock=log_lock,
        )
        with progress_lock:
            completed += 1
            if completed == total or completed % 10 == 0:
                print(f"INFO: Copied {completed}/{total} files.", flush=True)

    failures = []
    with ThreadPoolExecutor(max_workers=max(1, args.workers)) as executor:
        futures = {executor.submit(do_copy, entry): entry for entry in files}
        for future, entry in futures.items():
            try:
                future.result()
            except Exception as exc:
                failures.append((entry[1], str(exc)))

    if failures:
        print("ERROR: Parallel model copy failed. Failed files:", flush=True)
        for rel_path, error_msg in failures:
            print(f"ERROR: {rel_path}: {error_msg}", flush=True)
        return 1

    return 0


if __name__ == "__main__":
    sys.exit(main())
