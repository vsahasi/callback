"""Resumable dataset downloads for the Callback detector.

Datasets land in CALLBACK_DATA_DIR (default ~/data/callback), never in the repo.

Checksums: the official SHA-256 digests for these archives are not all
published in a machine-readable place, so this script uses trust-on-first-use.
The first successful download of a file records its digest in
<data-dir>/checksums.json, and every later run verifies against that record. If
you have an official digest, paste it into the registry below as `sha256` and
it is enforced from the first byte instead. Entries that still need an official
digest are marked NEEDS-OFFICIAL-DIGEST.

Usage:

    # what would happen, no network
    uv run python -m training.download --dry-run

    # disk check only
    uv run python -m training.download --check-space

    # fetch everything (long; run it detached)
    tmux new-window -t callback -n downloads \\
        'cd ~/projects/callback && uv run python -m training.download'

    # one dataset
    uv run python -m training.download --only musan rirs_noises
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import shutil
import sys
import time
import urllib.error
import urllib.request
from dataclasses import dataclass, field
from pathlib import Path

USER_AGENT = "callback-detector/0.1 (dataset fetcher)"
CHUNK = 1 << 20  # 1 MiB
DEFAULT_DATA_DIR = Path(os.environ.get("CALLBACK_DATA_DIR", "~/data/callback"))

# Headroom on top of the sum of archive sizes, for unpacking later.
HEADROOM_GIB = 15.0


@dataclass(frozen=True)
class Dataset:
    key: str
    url: str
    filename: str
    approx_gib: float
    purpose: str
    sha256: str | None = None
    notes: str = ""


REGISTRY: tuple[Dataset, ...] = (
    Dataset(
        key="asvspoof2019_la",
        url="https://datashare.ed.ac.uk/bitstream/handle/10283/3336/LA.zip",
        filename="asvspoof2019_LA.zip",
        approx_gib=7.6,
        purpose="Primary spoof/bonafide training and dev data.",
        notes="NEEDS-OFFICIAL-DIGEST. Edinburgh DataShare DS_10283_3336.",
    ),
    Dataset(
        key="asvspoof2021_la_eval",
        url="https://zenodo.org/records/4837263/files/ASVspoof2021_LA_eval.tar.gz",
        filename="ASVspoof2021_LA_eval.tar.gz",
        approx_gib=8.4,
        purpose="Held-out eval set. Never train on this.",
        notes="NEEDS-OFFICIAL-DIGEST. Zenodo record 4837263.",
    ),
    Dataset(
        key="asvspoof2021_la_keys",
        url="https://www.asvspoof.org/asvspoof2021/LA-keys-full.tar.gz",
        filename="ASVspoof2021_LA_keys_full.tar.gz",
        approx_gib=0.01,
        purpose="Labels for the 2021 LA eval set.",
        notes="NEEDS-OFFICIAL-DIGEST.",
    ),
    Dataset(
        key="in_the_wild",
        url="https://owncloud.fraunhofer.de/index.php/s/JZgXh0JEAF0elxa/download",
        filename="release_in_the_wild.zip",
        approx_gib=7.4,
        purpose="Out-of-domain generalisation check (real-world deepfakes).",
        notes=(
            "NEEDS-OFFICIAL-DIGEST and NEEDS-URL-CHECK. The Fraunhofer share "
            "link changes; confirm against deepfake-total.com before trusting it."
        ),
    ),
    Dataset(
        key="musan",
        url="https://www.openslr.org/resources/17/musan.tar.gz",
        filename="musan.tar.gz",
        approx_gib=10.6,
        purpose="Additive noise for augmentation.",
        notes="NEEDS-OFFICIAL-DIGEST. OpenSLR 17.",
    ),
    Dataset(
        key="rirs_noises",
        url="https://www.openslr.org/resources/28/rirs_noises.zip",
        filename="rirs_noises.zip",
        approx_gib=0.9,
        purpose="Room impulse responses for reverberation augmentation.",
        notes="NEEDS-OFFICIAL-DIGEST. OpenSLR 28.",
    ),
)

REGISTRY_BY_KEY = {d.key: d for d in REGISTRY}


@dataclass
class Ledger:
    """Recorded digests, so a resumed or repeated run can verify."""

    path: Path
    entries: dict[str, dict[str, object]] = field(default_factory=dict)

    @classmethod
    def load(cls, data_dir: Path) -> Ledger:
        path = data_dir / "checksums.json"
        if path.exists():
            return cls(path=path, entries=json.loads(path.read_text("utf-8")))
        return cls(path=path)

    def save(self) -> None:
        self.path.write_text(
            json.dumps(self.entries, indent=2, sort_keys=True) + "\n", encoding="utf-8"
        )

    def recorded(self, key: str) -> str | None:
        entry = self.entries.get(key)
        if not entry:
            return None
        value = entry.get("sha256")
        return value if isinstance(value, str) else None

    def record(self, dataset: Dataset, digest: str, size: int) -> None:
        self.entries[dataset.key] = {
            "sha256": digest,
            "bytes": size,
            "filename": dataset.filename,
            "url": dataset.url,
            "recorded_at": time.strftime("%Y-%m-%dT%H:%M:%S%z"),
            "source": "official" if dataset.sha256 else "trust-on-first-use",
        }
        self.save()


def human(gib: float) -> str:
    return f"{gib:.1f} GiB"


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(CHUNK), b""):
            digest.update(block)
    return digest.hexdigest()


def check_space(data_dir: Path, datasets: tuple[Dataset, ...]) -> tuple[bool, str]:
    data_dir.mkdir(parents=True, exist_ok=True)
    usage = shutil.disk_usage(data_dir)
    free_gib = usage.free / 2**30
    needed = sum(d.approx_gib for d in datasets) + HEADROOM_GIB
    ok = free_gib >= needed
    message = (
        f"free {human(free_gib)} at {data_dir}, "
        f"need about {human(needed)} "
        f"({human(needed - HEADROOM_GIB)} of archives plus {human(HEADROOM_GIB)} headroom)"
    )
    return ok, message


def remote_size(url: str) -> int | None:
    request = urllib.request.Request(
        url, method="HEAD", headers={"User-Agent": USER_AGENT}
    )
    try:
        with urllib.request.urlopen(request, timeout=30) as response:
            length = response.headers.get("Content-Length")
            return int(length) if length else None
    except (urllib.error.URLError, ValueError, TimeoutError):
        return None


def download_resumable(dataset: Dataset, target: Path, *, retries: int = 5) -> Path:
    """Download with HTTP Range resume. Safe to interrupt and re-run."""

    target.parent.mkdir(parents=True, exist_ok=True)
    partial = target.with_suffix(target.suffix + ".part")

    expected_total = remote_size(dataset.url)

    for attempt in range(1, retries + 1):
        have = partial.stat().st_size if partial.exists() else 0

        if expected_total is not None and have == expected_total:
            break

        headers = {"User-Agent": USER_AGENT}
        if have:
            headers["Range"] = f"bytes={have}-"

        request = urllib.request.Request(dataset.url, headers=headers)
        try:
            with urllib.request.urlopen(request, timeout=60) as response:
                if have and response.status != 206:
                    # Server ignored the range; start over rather than corrupt.
                    print(
                        f"    server ignored Range (status {response.status}); "
                        "restarting from zero",
                        flush=True,
                    )
                    partial.unlink(missing_ok=True)
                    have = 0
                    mode = "wb"
                else:
                    mode = "ab" if have else "wb"

                with partial.open(mode) as sink:
                    copied = 0
                    last_report = time.monotonic()
                    while True:
                        block = response.read(CHUNK)
                        if not block:
                            break
                        sink.write(block)
                        copied += len(block)
                        now = time.monotonic()
                        if now - last_report >= 10:
                            total = have + copied
                            if expected_total:
                                pct = 100.0 * total / expected_total
                                print(
                                    f"    {total / 2**30:.2f} GiB "
                                    f"of {expected_total / 2**30:.2f} GiB ({pct:.1f}%)",
                                    flush=True,
                                )
                            else:
                                print(f"    {total / 2**30:.2f} GiB", flush=True)
                            last_report = now
            break
        except (urllib.error.URLError, TimeoutError, ConnectionError) as exc:
            if attempt == retries:
                raise
            backoff = min(60, 2**attempt)
            print(
                f"    attempt {attempt}/{retries} failed ({exc}); "
                f"retrying in {backoff}s",
                flush=True,
            )
            time.sleep(backoff)

    partial.replace(target)
    return target


def fetch(dataset: Dataset, data_dir: Path, ledger: Ledger) -> bool:
    target = data_dir / dataset.filename
    print(f"[{dataset.key}] {dataset.purpose}")
    if dataset.notes:
        print(f"    note: {dataset.notes}")

    expected = dataset.sha256 or ledger.recorded(dataset.key)

    if target.exists():
        if expected is None:
            digest = sha256_file(target)
            ledger.record(dataset, digest, target.stat().st_size)
            print(f"    present, digest recorded {digest[:16]}...")
            return True
        print("    present, verifying...")
        digest = sha256_file(target)
        if digest == expected:
            print("    digest OK")
            return True
        print(
            f"    DIGEST MISMATCH\n"
            f"      expected {expected}\n"
            f"      actual   {digest}\n"
            f"    refusing to use it. Delete {target} and re-run to refetch.",
            file=sys.stderr,
        )
        return False

    print(f"    downloading about {human(dataset.approx_gib)} from {dataset.url}")
    try:
        download_resumable(dataset, target)
    except Exception as exc:  # noqa: BLE001 - report and continue with the rest
        print(f"    FAILED: {type(exc).__name__}: {exc}", file=sys.stderr)
        return False

    digest = sha256_file(target)
    size = target.stat().st_size

    if expected is None:
        ledger.record(dataset, digest, size)
        print(f"    done, digest recorded {digest[:16]}...")
        return True

    if digest != expected:
        print(
            f"    DIGEST MISMATCH after download\n"
            f"      expected {expected}\n"
            f"      actual   {digest}",
            file=sys.stderr,
        )
        return False

    ledger.record(dataset, digest, size)
    print("    done, digest OK")
    return True


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--data-dir",
        type=Path,
        default=DEFAULT_DATA_DIR,
        help="Where archives land (default ~/data/callback).",
    )
    parser.add_argument(
        "--only",
        nargs="+",
        choices=sorted(REGISTRY_BY_KEY),
        help="Fetch only these datasets.",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Print the plan and the disk check, touch no network.",
    )
    parser.add_argument(
        "--check-space", action="store_true", help="Disk check only, then exit."
    )
    parser.add_argument(
        "--force-space",
        action="store_true",
        help="Continue even if the disk check fails.",
    )
    args = parser.parse_args(argv)

    data_dir = args.data_dir.expanduser().resolve()
    datasets = tuple(REGISTRY_BY_KEY[k] for k in args.only) if args.only else REGISTRY

    print(f"data dir: {data_dir}")
    ok, message = check_space(data_dir, datasets)
    print(f"disk    : {message}")

    if args.check_space:
        return 0 if ok else 1

    if not ok and not args.force_space:
        print(
            "not enough free space. Free some up or pass --force-space.",
            file=sys.stderr,
        )
        return 1

    print(f"datasets: {len(datasets)}")
    for dataset in datasets:
        marker = "official digest" if dataset.sha256 else "trust-on-first-use"
        print(f"  - {dataset.key:24s} {human(dataset.approx_gib):>10s}  [{marker}]")

    if args.dry_run:
        print("\ndry run, nothing downloaded.")
        return 0

    ledger = Ledger.load(data_dir)
    failures: list[str] = []
    for dataset in datasets:
        if not fetch(dataset, data_dir, ledger):
            failures.append(dataset.key)

    print()
    if failures:
        print(f"FAILED: {', '.join(failures)}", file=sys.stderr)
        return 1
    print(f"all {len(datasets)} dataset(s) present and verified in {data_dir}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
