#!/usr/bin/env python3

"""
Build the x86 base Docker image, then one or more per-issue images.

Usage
-----
    ./build_images.py                        # build the base image (tag: latest), then all issues/* with 4 workers
    ./build_images.py 42 43 44               # build the base image and the given issues (default tag: latest)
    ./build_images.py --tag v1.2.3            # build and tag images with :v1.2.3
    ./build_images.py 28565_1001 --tag beta   # build issue 28565_1001 only with tag :beta
    ./build_images.py -w 1                    # same as above but sequential (workers = 1)
    ./build_images.py -w 8                    # use up to 8 concurrent workers
    ./build_images.py --clean-up              # build all issues and delete per-issue images after push
"""

from __future__ import annotations

import argparse
import concurrent.futures
import logging
import os
import subprocess
import sys
from pathlib import Path

logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")


def get_root() -> Path:
    """Get the root directory of the repository."""

    return Path(__file__).resolve().parent.parent


def run(cmd: list[str], cwd: Path) -> None:
    """Wrapper around subprocess.run with `check=True`."""

    subprocess.run(cmd, cwd=cwd, check=True)


def build_base_image() -> None:
    """Build the common x86 base image (swelancer_x86)."""

    logging.info("Building Docker base image swelancer_x86")
    dockerfile = get_root() / "Dockerfile_x86_base"

    assert dockerfile.is_file(), f"Required Dockerfile not found: {dockerfile}"

    cmd = [
        "docker",
        "buildx",
        "build",
        "-f",
        "Dockerfile_x86_base",
        "--platform",
        "linux/amd64",
    ]

    ssh_sock = os.environ.get("SSH_AUTH_SOCK")

    if ssh_sock:
        cmd += ["--ssh", f"default={ssh_sock}"]

    cmd += ["-t", "swelancer_x86:latest", "."]

    run(cmd, cwd=get_root())


def build_issue_image(issue_id: str, tag: str) -> None:
    """Build the per-issue image (swelancer_x86_<ISSUE_ID>)."""

    logging.info("Building Docker image for issue %s", issue_id)
    dockerfile = get_root() / "Dockerfile_x86_per_task"

    assert dockerfile.is_file(), f"Required Dockerfile not found: {dockerfile}"

    cmd = [
        "docker",
        "buildx",
        "build",
        "--build-arg",
        f"ISSUE_ID={issue_id}",
        "-f",
        "Dockerfile_x86_per_task",
        "--platform",
        "linux/amd64",
        "-t",
        f"swelancer_x86_{issue_id}:{tag}",
        ".",
    ]

    run(cmd, cwd=get_root())


def build_monolith_image(tag: str) -> None:
    """Build the monolith image"""

    logging.info("Building Monolith Docker image")
    dockerfile = get_root() / "Dockerfile_x86_monolith"

    assert dockerfile.is_file(), f"Required Dockerfile not found: {dockerfile}"

    cmd = [
        "docker",
        "buildx",
        "build",
        "-f",
        "Dockerfile_x86_monolith",
        "--platform",
        "linux/amd64",
        "-t",
        f"swelancer_x86_monolith:{tag}",
        ".",
    ]

    run(cmd, cwd=get_root())


def push_image(issue_id: str, tag: str, registry: str) -> None:
    """Push the per-issue image to the container registry."""

    logging.info("Pushing image for issue %s to %s", issue_id, registry)

    local_tag = f"swelancer_x86_{issue_id}:{tag}"
    remote_tag = f"{registry}/swelancer_x86_{issue_id}:{tag}"

    run(["docker", "tag", local_tag, remote_tag], cwd=get_root())
    run(["docker", "push", remote_tag], cwd=get_root())


def push_monolith_image(tag: str, registry: str) -> None:
    """Push the monolith image to the container registry."""

    logging.info("Pushing monolith image to %s", registry)

    local_tag = f"swelancer_x86_monolith:{tag}"
    remote_tag = f"{registry}/swelancer_x86_monolith:{tag}"

    run(["docker", "tag", local_tag, remote_tag], cwd=get_root())
    run(["docker", "push", remote_tag], cwd=get_root())


def issue_worker(issue_id: str, push: bool, cleanup: bool, tag: str, registry: str) -> None:
    logging.info("Worker started for issue %s", issue_id)

    build_issue_image(issue_id, tag)
    if push:
        push_image(issue_id, tag, registry)

    if cleanup:
        logging.info("Removing image swelancer_x86_%s:%s", issue_id, tag)
        run(["docker", "rmi", f"swelancer_x86_{issue_id}:{tag}"], cwd=get_root())


def monolith_worker(push: bool, cleanup: bool, tag: str, registry: str) -> None:
    logging.info("Building monolithic image swelancer_x86_monolith")

    build_monolith_image(tag)
    if push:
        push_monolith_image(tag, registry)

    if cleanup:
        logging.info("Removing image swelancer_x86_monolith:%s", tag)
        run(["docker", "rmi", f"swelancer_x86_monolith:{tag}"], cwd=get_root())


def worker(issue_id: str, push: bool, cleanup: bool, tag: str, registry: str) -> None:
    """Build an issue image, push it, and optionally remove it afterwards."""

    if issue_id == "monolith":
        monolith_worker(push, cleanup, tag, registry)
    else:
        issue_worker(issue_id, push, cleanup, tag, registry)


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Build the x86 base image and per-issue images, tagging them with the specified tag (default: latest)."
    )
    parser.add_argument(
        "issue_ids",
        nargs="*",
        metavar="ISSUE_ID",
        help="Optional one or more ISSUE_IDs (otherwise builds all issues/*)",
    )
    parser.add_argument(
        "-w",
        "--workers",
        type=int,
        default=4,
        help="Number of concurrent workers (default: 4; use 1 for sequential execution).",
    )
    parser.add_argument(
        "--tag",
        default="latest",
        help="Tag to apply to built images (prefix ':' optional, default: latest)",
    )
    parser.add_argument(
        "-c",
        "--clean-up",
        dest="cleanup",
        action="store_true",
        help="Delete per-issue images after a successful build.",
    )
    parser.add_argument(
        "--skip-push",
        dest="skip_push",
        action="store_true",
        help="Skip pushing images to the container registry",
    )
    parser.add_argument(
        "--registry",
        help="Container registry (required unless --skip-push is passed)",
    )
    args = parser.parse_args()

    if not args.skip_push and not args.registry:
        parser.error("--registry is required unless --skip-push is passed")

    tag = args.tag.lstrip(":")
    registry = args.registry.rstrip("/") if args.registry else ""
    issues_dir = get_root() / "issues"

    if not issues_dir.is_dir():
        sys.exit("No issues/ directory found")

    build_base_image()

    issue_ids = args.issue_ids or [p.name for p in sorted(issues_dir.iterdir()) if p.is_dir()]

    push = not args.skip_push
    cleanup = args.cleanup

    if args.workers <= 1 or len(issue_ids) <= 1:
        for issue in issue_ids:
            worker(issue, push, cleanup, tag, registry)
    else:
        with concurrent.futures.ThreadPoolExecutor(max_workers=args.workers) as pool:
            pool.map(
                lambda iid: worker(iid, push, cleanup, tag, registry),
                issue_ids,
            )


if __name__ == "__main__":
    try:
        main()
    except (subprocess.CalledProcessError, Exception) as exc:
        sys.exit(str(exc))
