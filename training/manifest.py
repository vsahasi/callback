"""One manifest format behind several incompatible protocol file layouts.

ASVspoof 2019 LA, the ASVspoof 2021 LA eval keys and In-the-Wild all label
their audio differently. Training code should not know that. Everything here
funnels into `Utterance` records and a `Manifest`, and the parsers are the only
place that knows a column index.

Pure standard library on purpose, so it is unit-testable with no dataset and no
torch, from tiny inline protocol strings.

EXPECTED ON-DISK LAYOUT
-----------------------
`training/download.py` drops archives in `CALLBACK_DATA_DIR` (default
`~/data/callback`). After unpacking, the roots this module expects are:

    <data>/LA/                                  (from asvspoof2019_LA.zip)
      ASVspoof2019_LA_train/flac/*.flac
      ASVspoof2019_LA_dev/flac/*.flac
      ASVspoof2019_LA_eval/flac/*.flac
      ASVspoof2019_LA_cm_protocols/
        ASVspoof2019.LA.cm.train.trn.txt
        ASVspoof2019.LA.cm.dev.trl.txt
        ASVspoof2019.LA.cm.eval.trl.txt
      ASVspoof2019_LA_asv_scores/
        ASVspoof2019.LA.asv.dev.gi.trl.scores.txt
        ASVspoof2019.LA.asv.eval.gi.trl.scores.txt

    <data>/ASVspoof2021_LA_eval/flac/*.flac      (from the 2021 tarball)
    <data>/keys/LA/CM/trial_metadata.txt         (from LA-keys-full.tar.gz)
    <data>/keys/LA/ASV/ASV_scores/*.txt

    <data>/release_in_the_wild/
      meta.csv
      *.wav

`discover_roots` locates these tolerantly (it searches a couple of levels down
for the marker directories) because archive nesting varies with the unpack
command used. Nothing here unpacks anything.

ASSUMPTIONS TO VERIFY ONCE THE REAL FILES LAND
----------------------------------------------
No protocol file was available while this was written, so column positions come
from the published evaluation plans and from the reference parsers in the
official ASVspoof baselines. Two of them are worth re-checking, and both are
handled defensively rather than by blind indexing:

1. ASVspoof 2019 LA CM protocol, 5 whitespace-separated columns. Column 0 is
   the speaker, column 1 the utterance id, column 4 the key
   (`bonafide`/`spoof`). The attack id (`A01`..`A19`) is in column 3 and column
   2 is an unused `-` for the LA condition, the reverse of the PA condition
   where column 2 carries the environment id. Rather than trust that, the
   parser scans columns 2 and 3 for a token matching `A\\d+` and takes that as
   the attack id. If neither matches and the key is `spoof`, it raises, because
   a spoof trial with no attack id means the layout differs from what is
   assumed here.

2. ASVspoof 2021 LA CM `trial_metadata.txt`, 8 whitespace-separated columns:
   speaker, utterance id, codec, transmission, attack id, key, trim flag,
   phase (`progress`/`eval`/`hidden`). The parser locates the key column by
   looking for the `bonafide`/`spoof` token rather than indexing column 5
   blindly, and takes the attack id from the column immediately before it. The
   phase is read from the last column when it is one of the known phase names.
   Pin the real indices here once `keys/LA/CM/trial_metadata.txt` is unpacked.

A third thing to confirm: the 2021 eval audio is `.flac` and In-the-Wild is
`.wav`. `audio_suffix` on each source encodes that guess; fix it if wrong.
"""

from __future__ import annotations

import csv
import os
import re
from collections.abc import Iterable, Iterator, Mapping, Sequence
from dataclasses import dataclass, field, replace
from enum import StrEnum
from pathlib import Path

__all__ = [
    "BONAFIDE",
    "SPOOF",
    "DataRoots",
    "Label",
    "Manifest",
    "Subset",
    "Utterance",
    "default_data_dir",
    "discover_roots",
    "load_asvspoof2019_la",
    "load_asvspoof2021_la_eval",
    "load_in_the_wild",
    "parse_asv_scores",
    "parse_asvspoof2019_cm_protocol",
    "parse_asvspoof2021_cm_keys",
]


BONAFIDE = "bonafide"
SPOOF = "spoof"

_ATTACK_RE = re.compile(r"^A\d+$")
_PHASES = frozenset({"progress", "eval", "hidden"})
_PLACEHOLDERS = frozenset({"-", "", "none", "null"})


class Label(StrEnum):
    """Binary countermeasure label. `str` mixin so it serialises plainly."""

    BONAFIDE = BONAFIDE
    SPOOF = SPOOF

    @classmethod
    def parse(cls, raw: str) -> Label:
        token = raw.strip().lower()
        # In-the-Wild writes "bona-fide"; ASVspoof writes "bonafide".
        if token in {"bonafide", "bona-fide", "bona fide", "genuine", "real"}:
            return cls.BONAFIDE
        if token in {"spoof", "spoofed", "fake", "deepfake"}:
            return cls.SPOOF
        raise ValueError(f"unrecognised label {raw!r}")

    @property
    def target(self) -> int:
        """Integer class index. 0 = spoof, 1 = bonafide.

        Chosen so that the positive class matches the score convention in
        `training.metrics`, where a higher score means more bonafide.
        """

        return 1 if self is Label.BONAFIDE else 0


class Subset(StrEnum):
    """Which split an utterance belongs to.

    `EVAL_2021` and `IN_THE_WILD` are held out by construction: nothing in this
    module will hand them to a training split, and `Manifest.for_training`
    refuses them outright.
    """

    TRAIN = "train"
    DEV = "dev"
    EVAL = "eval"
    EVAL_2021 = "eval_2021"
    IN_THE_WILD = "in_the_wild"

    @property
    def is_held_out(self) -> bool:
        return self in {Subset.EVAL_2021, Subset.IN_THE_WILD}


@dataclass(frozen=True, slots=True)
class Utterance:
    """One labelled audio file, whatever corpus it came from.

    `path` may point at a file that does not exist yet: manifests are parsed
    from protocol text, and existence is checked separately by
    `Manifest.missing_files` so that a parse works with no audio on disk.
    """

    utterance_id: str
    path: Path
    label: Label
    subset: Subset
    attack_id: str | None = None
    speaker_id: str | None = None
    # Free-form extras that only some corpora have: the 2021 codec and phase,
    # the In-the-Wild speaker name. Kept out of the core fields so training
    # code never has to branch on corpus.
    meta: Mapping[str, str] = field(default_factory=dict)

    @property
    def is_bonafide(self) -> bool:
        return self.label is Label.BONAFIDE

    @property
    def target(self) -> int:
        return self.label.target


@dataclass(frozen=True)
class Manifest:
    """An immutable, ordered collection of utterances with a few views."""

    utterances: tuple[Utterance, ...]
    source: str = ""

    def __len__(self) -> int:
        return len(self.utterances)

    def __iter__(self) -> Iterator[Utterance]:
        return iter(self.utterances)

    def __getitem__(self, index: int) -> Utterance:
        return self.utterances[index]

    @classmethod
    def from_iterable(
        cls, utterances: Iterable[Utterance], source: str = ""
    ) -> Manifest:
        return cls(tuple(utterances), source)

    def filter(
        self,
        *,
        subset: Subset | Sequence[Subset] | None = None,
        label: Label | None = None,
        attack_ids: Sequence[str] | None = None,
    ) -> Manifest:
        subsets = (
            None
            if subset is None
            else frozenset([subset] if isinstance(subset, Subset) else subset)
        )
        attacks = None if attack_ids is None else frozenset(attack_ids)
        kept = [
            utterance
            for utterance in self.utterances
            if (subsets is None or utterance.subset in subsets)
            and (label is None or utterance.label is label)
            and (attacks is None or utterance.attack_id in attacks)
        ]
        return Manifest(tuple(kept), self.source)

    def for_training(self) -> Manifest:
        """Trainable utterances only.

        Held-out subsets raise rather than being silently dropped, so a config
        that points training at the 2021 eval set fails loudly. Leaking the
        held-out set into training is the one mistake that cannot be detected
        from the resulting numbers, so it is made impossible here.
        """

        leaked = sorted(
            {u.subset.value for u in self.utterances if u.subset.is_held_out}
        )
        if leaked:
            raise ValueError(
                "refusing to train on held-out data; this manifest contains "
                f"subsets {leaked}. Filter them out explicitly if you really "
                "mean it."
            )
        return self.filter(subset=[Subset.TRAIN])

    @property
    def counts(self) -> dict[str, int]:
        counts: dict[str, int] = {}
        for utterance in self.utterances:
            key = f"{utterance.subset.value}/{utterance.label.value}"
            counts[key] = counts.get(key, 0) + 1
        return dict(sorted(counts.items()))

    @property
    def attack_ids(self) -> tuple[str, ...]:
        return tuple(sorted({u.attack_id for u in self.utterances if u.attack_id}))

    def label_map(self) -> dict[str, str]:
        """`utt_id -> label` mapping, for `metrics.split_scores_by_label`."""

        return {u.utterance_id: u.label.value for u in self.utterances}

    def class_weights(self) -> tuple[float, float]:
        """Inverse-frequency weights as `(spoof_weight, bonafide_weight)`.

        ASVspoof LA train is roughly 10% bonafide to 90% spoof, so an unweighted
        cross entropy quietly learns to answer "spoof". Index order matches
        `Label.target`, so the tuple drops straight into a weighted loss.
        """

        n_bonafide = sum(1 for u in self.utterances if u.is_bonafide)
        n_spoof = len(self.utterances) - n_bonafide
        if n_bonafide == 0 or n_spoof == 0:
            return (1.0, 1.0)
        total = n_bonafide + n_spoof
        return (total / (2 * n_spoof), total / (2 * n_bonafide))

    def sample_weights(self) -> list[float]:
        """Per-utterance weights for a `WeightedRandomSampler`."""

        spoof_weight, bonafide_weight = self.class_weights()
        return [
            bonafide_weight if u.is_bonafide else spoof_weight for u in self.utterances
        ]

    def missing_files(self, limit: int | None = None) -> list[Path]:
        """Paths that are not on disk. Cheap sanity check before a long run."""

        missing: list[Path] = []
        for utterance in self.utterances:
            if not utterance.path.exists():
                missing.append(utterance.path)
                if limit is not None and len(missing) >= limit:
                    break
        return missing

    def with_root(self, root: Path) -> Manifest:
        """Re-base every relative path on `root`. Absolute paths are kept."""

        return Manifest(
            tuple(
                utterance
                if utterance.path.is_absolute()
                else replace(utterance, path=root / utterance.path)
                for utterance in self.utterances
            ),
            self.source,
        )

    def __add__(self, other: Manifest) -> Manifest:
        if not isinstance(other, Manifest):  # pragma: no cover - type guard
            return NotImplemented
        sources = [s for s in (self.source, other.source) if s]
        return Manifest(self.utterances + other.utterances, source=" + ".join(sources))


# ---------------------------------------------------------------------------
# Locating the data
# ---------------------------------------------------------------------------


def default_data_dir() -> Path:
    """`CALLBACK_DATA_DIR` or `~/data/callback`, matching download.py."""

    return Path(os.environ.get("CALLBACK_DATA_DIR", "~/data/callback")).expanduser()


@dataclass(frozen=True)
class DataRoots:
    """Where each corpus was found, or None when it is not unpacked yet."""

    data_dir: Path
    asvspoof2019_la: Path | None = None
    asvspoof2021_la_eval: Path | None = None
    asvspoof2021_keys: Path | None = None
    in_the_wild: Path | None = None
    musan: Path | None = None
    rirs_noises: Path | None = None

    def describe(self) -> str:
        lines = [f"data dir: {self.data_dir}"]
        for name in (
            "asvspoof2019_la",
            "asvspoof2021_la_eval",
            "asvspoof2021_keys",
            "in_the_wild",
            "musan",
            "rirs_noises",
        ):
            value = getattr(self, name)
            lines.append(f"  {name:22s} {value if value else 'not found'}")
        return "\n".join(lines)


def _find_dir(root: Path, names: Sequence[str], max_depth: int = 3) -> Path | None:
    """First directory under `root` whose name is in `names`.

    Breadth-first to a shallow depth, because unpacked archives nest
    inconsistently (`LA/` versus `LA/LA/` depending on the unzip invocation)
    but never deeply. Returns None instead of raising, so a partial download is
    a normal state.
    """

    if not root.is_dir():
        return None
    wanted = {name.lower() for name in names}
    frontier = [(root, 0)]
    while frontier:
        current, depth = frontier.pop(0)
        try:
            children = sorted(child for child in current.iterdir() if child.is_dir())
        except (PermissionError, OSError):  # pragma: no cover - hostile fs
            continue
        for child in children:
            if child.name.lower() in wanted:
                return child
        if depth < max_depth:
            frontier.extend((child, depth + 1) for child in children)
    return None


def discover_roots(data_dir: Path | str | None = None) -> DataRoots:
    """Locate whichever corpora are unpacked. Never raises for absent ones."""

    root = Path(data_dir).expanduser() if data_dir else default_data_dir()

    la_2019 = _find_dir(root, ["LA"])
    # Guard against matching ASVspoof2021_LA_eval/.. or keys/LA when looking
    # for the 2019 tree: the 2019 root is the one with the CM protocols.
    if la_2019 is not None and not _find_dir(
        la_2019, ["ASVspoof2019_LA_cm_protocols"], max_depth=1
    ):
        la_2019 = None

    return DataRoots(
        data_dir=root,
        asvspoof2019_la=la_2019,
        asvspoof2021_la_eval=_find_dir(root, ["ASVspoof2021_LA_eval"]),
        asvspoof2021_keys=_find_dir(root, ["keys"]),
        in_the_wild=_find_dir(root, ["release_in_the_wild"]),
        musan=_find_dir(root, ["musan"]),
        rirs_noises=_find_dir(root, ["RIRS_NOISES", "rirs_noises"]),
    )


# ---------------------------------------------------------------------------
# ASVspoof 2019 LA
# ---------------------------------------------------------------------------

_PROTOCOL_2019 = {
    Subset.TRAIN: "ASVspoof2019.LA.cm.train.trn.txt",
    Subset.DEV: "ASVspoof2019.LA.cm.dev.trl.txt",
    Subset.EVAL: "ASVspoof2019.LA.cm.eval.trl.txt",
}

_FLAC_DIR_2019 = {
    Subset.TRAIN: "ASVspoof2019_LA_train",
    Subset.DEV: "ASVspoof2019_LA_dev",
    Subset.EVAL: "ASVspoof2019_LA_eval",
}


def _attack_from_columns(columns: Sequence[str]) -> str | None:
    for value in columns:
        if _ATTACK_RE.match(value):
            return value
    return None


def parse_asvspoof2019_cm_protocol(
    text: str,
    subset: Subset,
    audio_dir: Path | None = None,
    *,
    audio_suffix: str = ".flac",
) -> Manifest:
    """Parse one ASVspoof 2019 LA CM protocol file from its text.

    Expected line shape, 5 whitespace-separated columns::

        LA_0079 LA_T_1138215 - -   bonafide
        LA_0079 LA_T_1271820 - A01 spoof

    speaker, utterance id, unused for LA, attack id, key. The attack id is
    located by pattern among the middle columns rather than by fixed index; see
    the module docstring for why.

    Lines that are blank or start with `#` are skipped. A header line naming
    the columns is tolerated.
    """

    utterances: list[Utterance] = []
    for line_number, raw in enumerate(text.splitlines(), start=1):
        line = raw.strip()
        if not line or line.startswith("#"):
            continue
        columns = line.split()
        if len(columns) < 5:
            raise ValueError(
                f"line {line_number}: expected at least 5 columns in a 2019 LA "
                f"CM protocol line, got {len(columns)}: {raw!r}"
            )
        if columns[-1].lower() in {"key", "label"}:  # header row
            continue

        speaker_id, utterance_id = columns[0], columns[1]
        label = Label.parse(columns[-1])
        attack_id = _attack_from_columns(columns[2:-1])

        if attack_id is None and label is Label.SPOOF:
            raise ValueError(
                f"line {line_number}: spoof trial {utterance_id!r} carries no "
                f"attack id matching A<digits> in columns 3..{len(columns) - 1}. "
                "The protocol layout differs from what manifest.py assumes; "
                "check the real file and fix parse_asvspoof2019_cm_protocol."
            )
        if attack_id is not None and label is Label.BONAFIDE:
            raise ValueError(
                f"line {line_number}: bonafide trial {utterance_id!r} carries "
                f"attack id {attack_id!r}, which should not happen."
            )

        path = (
            audio_dir / f"{utterance_id}{audio_suffix}"
            if audio_dir is not None
            else Path(f"{utterance_id}{audio_suffix}")
        )
        utterances.append(
            Utterance(
                utterance_id=utterance_id,
                path=path,
                label=label,
                subset=subset,
                attack_id=attack_id,
                speaker_id=speaker_id,
            )
        )

    return Manifest.from_iterable(utterances, source=f"asvspoof2019_la/{subset.value}")


def load_asvspoof2019_la(
    la_root: Path | str,
    subsets: Sequence[Subset] = (Subset.TRAIN, Subset.DEV, Subset.EVAL),
    *,
    audio_suffix: str = ".flac",
) -> Manifest:
    """Load one or more ASVspoof 2019 LA splits from the unpacked `LA/` tree."""

    root = Path(la_root)
    protocol_dir = root / "ASVspoof2019_LA_cm_protocols"
    manifest = Manifest(())

    for subset in subsets:
        if subset not in _PROTOCOL_2019:
            raise ValueError(
                f"{subset.value!r} is not an ASVspoof 2019 LA subset; "
                f"expected one of {[s.value for s in _PROTOCOL_2019]}"
            )
        protocol_path = protocol_dir / _PROTOCOL_2019[subset]
        if not protocol_path.exists():
            raise FileNotFoundError(
                f"CM protocol not found: {protocol_path}. Unpack "
                "asvspoof2019_LA.zip into the data dir first."
            )
        audio_dir = root / _FLAC_DIR_2019[subset] / "flac"
        manifest = manifest + parse_asvspoof2019_cm_protocol(
            protocol_path.read_text(encoding="utf-8"),
            subset,
            audio_dir,
            audio_suffix=audio_suffix,
        )

    return manifest


# ---------------------------------------------------------------------------
# ASVspoof 2021 LA eval
# ---------------------------------------------------------------------------


def parse_asvspoof2021_cm_keys(
    text: str,
    audio_dir: Path | None = None,
    *,
    phases: Sequence[str] | None = None,
    audio_suffix: str = ".flac",
) -> Manifest:
    """Parse `keys/LA/CM/trial_metadata.txt` from the 2021 keys archive.

    Expected line shape, 8 whitespace-separated columns::

        LA_0009 LA_E_9332881 alaw ita_tx A07 spoof notrim progress
        LA_0012 LA_E_1000147 -    -      -   bonafide notrim eval

    speaker, utterance id, codec, transmission, attack id, key, trim flag,
    phase. The key column is located by looking for the `bonafide`/`spoof`
    token rather than by fixed index, and the attack id is taken from the
    column immediately before it. See the module docstring.

    `phases` filters on the last column, which is how the official package
    separates the progress, eval and hidden-track subsets. Pass
    `phases=("eval",)` to reproduce the headline 2021 LA number;
    pass None to keep every trial.

    Codec and phase are carried through in `Utterance.meta`, because codec is
    exactly the condition the 2019-trained model will degrade on, and you will
    want the per-codec breakdown.
    """

    wanted_phases = frozenset(phases) if phases else None
    utterances: list[Utterance] = []

    for line_number, raw in enumerate(text.splitlines(), start=1):
        line = raw.strip()
        if not line or line.startswith("#"):
            continue
        columns = line.split()
        if len(columns) < 3:
            raise ValueError(
                f"line {line_number}: expected at least 3 columns in a 2021 LA "
                f"CM key line, got {len(columns)}: {raw!r}"
            )

        label_index = _find_label_index(columns)
        if label_index is None:
            raise ValueError(
                f"line {line_number}: no 'bonafide' or 'spoof' token found in "
                f"{raw!r}. The 2021 key layout differs from what manifest.py "
                "assumes; check keys/LA/CM/trial_metadata.txt and fix "
                "parse_asvspoof2021_cm_keys."
            )

        speaker_id = columns[0] if columns[0] not in _PLACEHOLDERS else None
        utterance_id = columns[1]
        label = Label.parse(columns[label_index])

        attack_id = None
        if label_index >= 1:
            candidate = columns[label_index - 1]
            if _ATTACK_RE.match(candidate):
                attack_id = candidate
        if attack_id is None:
            attack_id = _attack_from_columns(columns[2:label_index])

        phase = None
        if columns[-1].lower() in _PHASES:
            phase = columns[-1].lower()
        if wanted_phases is not None and phase not in wanted_phases:
            continue

        meta: dict[str, str] = {}
        # Columns between the utterance id and the attack/key block describe
        # the transmission condition. Keep them verbatim under stable names.
        condition = [c for c in columns[2:label_index] if not _ATTACK_RE.match(c)]
        if condition:
            meta["codec"] = condition[0]
        if len(condition) > 1:
            meta["transmission"] = condition[1]
        if phase:
            meta["phase"] = phase

        path = (
            audio_dir / f"{utterance_id}{audio_suffix}"
            if audio_dir is not None
            else Path(f"{utterance_id}{audio_suffix}")
        )
        utterances.append(
            Utterance(
                utterance_id=utterance_id,
                path=path,
                label=label,
                subset=Subset.EVAL_2021,
                attack_id=attack_id,
                speaker_id=speaker_id,
                meta=meta,
            )
        )

    return Manifest.from_iterable(utterances, source="asvspoof2021_la/eval")


def _find_label_index(columns: Sequence[str]) -> int | None:
    """Index of the bonafide/spoof token, searched from the right."""

    for index in range(len(columns) - 1, 1, -1):
        if columns[index].lower() in {"bonafide", "spoof"}:
            return index
    return None


def load_asvspoof2021_la_eval(
    keys_root: Path | str,
    audio_root: Path | str | None = None,
    *,
    phases: Sequence[str] | None = ("eval",),
    audio_suffix: str = ".flac",
) -> Manifest:
    """Load the held-out 2021 LA eval manifest.

    `keys_root` is the unpacked `keys/` directory; `audio_root` the unpacked
    `ASVspoof2021_LA_eval/` directory (its `flac/` subdirectory is used when
    present).
    """

    keys_path = Path(keys_root)
    metadata = keys_path / "LA" / "CM" / "trial_metadata.txt"
    if not metadata.exists():
        # Tolerate being handed the CM directory or the file itself.
        for candidate in (
            keys_path / "CM" / "trial_metadata.txt",
            keys_path / "trial_metadata.txt",
            keys_path,
        ):
            if candidate.is_file():
                metadata = candidate
                break
        else:
            raise FileNotFoundError(
                f"2021 LA CM keys not found under {keys_path}; expected "
                "keys/LA/CM/trial_metadata.txt from LA-keys-full.tar.gz."
            )

    audio_dir: Path | None = None
    if audio_root is not None:
        audio_dir = Path(audio_root)
        if (audio_dir / "flac").is_dir():
            audio_dir = audio_dir / "flac"

    return parse_asvspoof2021_cm_keys(
        metadata.read_text(encoding="utf-8"),
        audio_dir,
        phases=phases,
        audio_suffix=audio_suffix,
    )


# ---------------------------------------------------------------------------
# In-the-Wild
# ---------------------------------------------------------------------------


def load_in_the_wild(
    release_root: Path | str, *, meta_name: str = "meta.csv"
) -> Manifest:
    """Load the In-the-Wild release from its `meta.csv`.

    ASSUMPTION TO VERIFY. The release is documented as a flat directory of
    `.wav` files plus a `meta.csv` with the columns `file`, `speaker`, `label`,
    where label is `bona-fide` or `spoof`. The reader is tolerant: it matches
    column names case-insensitively, accepts `filename`/`path`/`audio` as
    aliases for `file`, and falls back to positional columns if the header is
    missing. Check it against the real archive.

    Every utterance comes back with `subset=IN_THE_WILD`, which
    `Manifest.for_training` refuses, because this corpus exists only to measure
    out-of-domain generalisation.
    """

    root = Path(release_root)
    meta_path = root / meta_name
    if not meta_path.exists():
        raise FileNotFoundError(
            f"In-the-Wild metadata not found: {meta_path}. Unpack "
            "release_in_the_wild.zip into the data dir first."
        )

    utterances: list[Utterance] = []
    with meta_path.open("r", encoding="utf-8", newline="") as handle:
        rows = list(csv.reader(handle))

    if not rows:
        return Manifest((), source="in_the_wild")

    header = [cell.strip().lower() for cell in rows[0]]
    file_aliases = ("file", "filename", "path", "audio", "utterance")
    has_header = any(cell in file_aliases for cell in header) or "label" in header

    if has_header:
        file_col = next(
            (header.index(name) for name in file_aliases if name in header), 0
        )
        label_col = header.index("label") if "label" in header else len(header) - 1
        speaker_col = header.index("speaker") if "speaker" in header else None
        body = rows[1:]
    else:
        file_col, speaker_col, label_col = 0, 1, 2
        body = rows

    for line_number, row in enumerate(body, start=2 if has_header else 1):
        cells = [cell.strip() for cell in row]
        if not cells or not cells[0]:
            continue
        if max(file_col, label_col) >= len(cells):
            raise ValueError(
                f"{meta_path}:{line_number}: row has {len(cells)} columns, "
                f"need at least {max(file_col, label_col) + 1}: {row!r}"
            )
        filename = cells[file_col]
        label = Label.parse(cells[label_col])
        speaker = (
            cells[speaker_col]
            if speaker_col is not None and speaker_col < len(cells)
            else None
        )
        utterances.append(
            Utterance(
                utterance_id=Path(filename).stem,
                path=root / filename,
                label=label,
                subset=Subset.IN_THE_WILD,
                attack_id=None if label is Label.BONAFIDE else "itw",
                speaker_id=speaker or None,
                meta={"speaker_name": speaker} if speaker else {},
            )
        )

    return Manifest.from_iterable(utterances, source="in_the_wild")


# ---------------------------------------------------------------------------
# ASV scores, needed for t-DCF
# ---------------------------------------------------------------------------


def parse_asv_scores(text: str) -> tuple[list[float], list[float], list[float]]:
    """Split an ASVspoof ASV score file into target, nontarget and spoof lists.

    ASSUMPTION TO VERIFY. The 2019 ASV score files
    (`ASVspoof2019_LA_asv_scores/ASVspoof2019.LA.asv.{dev,eval}.gi.trl.scores.txt`)
    are documented as three whitespace-separated columns: speaker id, trial
    type (`target`/`nontarget`/`spoof`) and the ASV score. Some releases carry a
    fourth utterance-id column. This reader takes the score from the last
    column and the trial type from whichever column holds one of the three type
    tokens, so either shape parses. Confirm against the real file.

    The result feeds `metrics.obtain_asv_error_rates`, which is the only way to
    get a real `AsvErrorRates` for min t-DCF.
    """

    targets: list[float] = []
    nontargets: list[float] = []
    spoofs: list[float] = []
    bins = {"target": targets, "nontarget": nontargets, "spoof": spoofs}

    for line_number, raw in enumerate(text.splitlines(), start=1):
        line = raw.strip()
        if not line or line.startswith("#"):
            continue
        columns = line.split()
        if len(columns) < 2:
            raise ValueError(
                f"line {line_number}: expected at least 2 columns, got {raw!r}"
            )
        trial_type = next((c.lower() for c in columns if c.lower() in bins), None)
        if trial_type is None:
            raise ValueError(
                f"line {line_number}: no target/nontarget/spoof token in {raw!r}"
            )
        bins[trial_type].append(float(columns[-1]))

    return targets, nontargets, spoofs
