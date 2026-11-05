"""Protocol parsing tests, driven by inline protocol text.

No dataset and no torch: every parser is fed a few lines in the documented
column layout, so the assumptions in `manifest.py` are written down executably
as well as in prose.
"""

from __future__ import annotations

from pathlib import Path

import pytest

from training.manifest import (
    Label,
    Manifest,
    Subset,
    Utterance,
    default_data_dir,
    discover_roots,
    load_asvspoof2019_la,
    load_asvspoof2021_la_eval,
    load_in_the_wild,
    parse_asv_scores,
    parse_asvspoof2019_cm_protocol,
    parse_asvspoof2021_cm_keys,
)

PROTOCOL_2019 = """\
LA_0079 LA_T_1138215 - - bonafide
LA_0079 LA_T_1271820 - A01 spoof
LA_0080 LA_T_0000001 - A06 spoof
LA_0081 LA_T_9999999 - - bonafide
"""

KEYS_2021 = """\
LA_0009 LA_E_9332881 alaw ita_tx A07 spoof notrim progress
LA_0012 LA_E_1000147 - - - bonafide notrim eval
LA_0023 LA_E_5013670 opus loc_tx A10 spoof notrim eval
LA_0030 LA_E_7000001 g722 ita_tx - bonafide trim hidden
"""

ITW_META = """\
file,speaker,label
0.wav,Alec Guinness,spoof
1.wav,Alec Guinness,bona-fide
2.wav,Barack Obama,spoof
"""


# ---------------------------------------------------------------------------
# Labels and subsets
# ---------------------------------------------------------------------------


class TestLabel:
    @pytest.mark.parametrize(
        "raw", ["bonafide", "BONAFIDE", " bonafide ", "bona-fide", "genuine"]
    )
    def test_bonafide_spellings(self, raw):
        assert Label.parse(raw) is Label.BONAFIDE

    @pytest.mark.parametrize("raw", ["spoof", "SPOOF", "deepfake", "fake"])
    def test_spoof_spellings(self, raw):
        assert Label.parse(raw) is Label.SPOOF

    def test_unknown_label_is_rejected(self):
        with pytest.raises(ValueError, match="unrecognised label"):
            Label.parse("maybe")

    def test_target_indices_match_the_score_convention(self):
        # metrics treats higher scores as more bonafide, so bonafide is the
        # positive class.
        assert Label.BONAFIDE.target == 1
        assert Label.SPOOF.target == 0


class TestSubset:
    def test_only_2021_eval_and_in_the_wild_are_held_out(self):
        assert Subset.EVAL_2021.is_held_out
        assert Subset.IN_THE_WILD.is_held_out
        assert not Subset.TRAIN.is_held_out
        assert not Subset.DEV.is_held_out
        assert not Subset.EVAL.is_held_out


# ---------------------------------------------------------------------------
# ASVspoof 2019 LA
# ---------------------------------------------------------------------------


class TestParse2019:
    def test_parses_every_line(self):
        manifest = parse_asvspoof2019_cm_protocol(PROTOCOL_2019, Subset.TRAIN)
        assert len(manifest) == 4

    def test_extracts_each_field(self):
        manifest = parse_asvspoof2019_cm_protocol(PROTOCOL_2019, Subset.TRAIN)
        first, second = manifest[0], manifest[1]

        assert first.utterance_id == "LA_T_1138215"
        assert first.speaker_id == "LA_0079"
        assert first.label is Label.BONAFIDE
        assert first.attack_id is None
        assert first.subset is Subset.TRAIN

        assert second.utterance_id == "LA_T_1271820"
        assert second.label is Label.SPOOF
        assert second.attack_id == "A01"

    def test_builds_flac_paths_under_the_audio_dir(self):
        manifest = parse_asvspoof2019_cm_protocol(
            PROTOCOL_2019, Subset.TRAIN, Path("/data/LA/ASVspoof2019_LA_train/flac")
        )
        assert manifest[0].path == Path(
            "/data/LA/ASVspoof2019_LA_train/flac/LA_T_1138215.flac"
        )

    def test_finds_the_attack_id_in_either_middle_column(self):
        """The PA condition puts the environment id in column 3 and the attack
        in column 4; LA is documented the other way round. The parser scans
        both, so a swapped layout still parses."""

        swapped = "LA_0079 LA_T_1271820 A01 - spoof\n"
        manifest = parse_asvspoof2019_cm_protocol(swapped, Subset.TRAIN)
        assert manifest[0].attack_id == "A01"

    def test_spoof_without_an_attack_id_is_a_loud_failure(self):
        """A missing attack id on a spoof trial means the column layout is not
        what this module assumes, which must not be silently absorbed."""

        broken = "LA_0079 LA_T_1271820 - - spoof\n"
        with pytest.raises(ValueError, match="carries no attack id"):
            parse_asvspoof2019_cm_protocol(broken, Subset.TRAIN)

    def test_bonafide_with_an_attack_id_is_a_loud_failure(self):
        broken = "LA_0079 LA_T_1138215 - A01 bonafide\n"
        with pytest.raises(ValueError, match="carries attack id"):
            parse_asvspoof2019_cm_protocol(broken, Subset.TRAIN)

    def test_too_few_columns_is_rejected_with_a_line_number(self):
        with pytest.raises(ValueError, match="line 1"):
            parse_asvspoof2019_cm_protocol("LA_0079 LA_T_1 spoof\n", Subset.TRAIN)

    def test_blank_lines_comments_and_a_header_are_skipped(self):
        text = (
            "# ASVspoof 2019 LA\n"
            "\n"
            "SPEAKER_ID AUDIO_FILE_NAME - SYSTEM_ID KEY\n"
            "LA_0079 LA_T_1138215 - - bonafide\n"
        )
        manifest = parse_asvspoof2019_cm_protocol(text, Subset.TRAIN)
        assert len(manifest) == 1

    def test_loads_from_a_fake_unpacked_tree(self, tmp_path):
        root = tmp_path / "LA"
        protocols = root / "ASVspoof2019_LA_cm_protocols"
        protocols.mkdir(parents=True)
        (protocols / "ASVspoof2019.LA.cm.train.trn.txt").write_text(
            PROTOCOL_2019, encoding="utf-8"
        )
        (protocols / "ASVspoof2019.LA.cm.dev.trl.txt").write_text(
            "LA_0082 LA_D_1000001 - A02 spoof\n", encoding="utf-8"
        )

        manifest = load_asvspoof2019_la(root, (Subset.TRAIN, Subset.DEV))
        assert len(manifest) == 5
        assert manifest.counts == {
            "dev/spoof": 1,
            "train/bonafide": 2,
            "train/spoof": 2,
        }
        assert manifest[4].path == (
            root / "ASVspoof2019_LA_dev" / "flac" / "LA_D_1000001.flac"
        )

    def test_a_missing_protocol_file_names_the_path_and_the_fix(self, tmp_path):
        with pytest.raises(FileNotFoundError, match="Unpack"):
            load_asvspoof2019_la(tmp_path / "LA", (Subset.TRAIN,))

    def test_an_unknown_subset_is_rejected(self, tmp_path):
        with pytest.raises(ValueError, match="not an ASVspoof 2019 LA subset"):
            load_asvspoof2019_la(tmp_path, (Subset.EVAL_2021,))


# ---------------------------------------------------------------------------
# ASVspoof 2021 LA eval
# ---------------------------------------------------------------------------


class TestParse2021:
    def test_keeps_only_the_requested_phase_by_default(self):
        manifest = parse_asvspoof2021_cm_keys(KEYS_2021, phases=("eval",))
        assert len(manifest) == 2
        assert {u.utterance_id for u in manifest} == {
            "LA_E_1000147",
            "LA_E_5013670",
        }

    def test_keeps_everything_when_no_phase_filter_is_given(self):
        manifest = parse_asvspoof2021_cm_keys(KEYS_2021, phases=None)
        assert len(manifest) == 4

    def test_extracts_each_field(self):
        manifest = parse_asvspoof2021_cm_keys(KEYS_2021, phases=None)
        progress = manifest[0]
        assert progress.utterance_id == "LA_E_9332881"
        assert progress.speaker_id == "LA_0009"
        assert progress.label is Label.SPOOF
        assert progress.attack_id == "A07"
        assert progress.subset is Subset.EVAL_2021
        assert progress.meta["codec"] == "alaw"
        assert progress.meta["transmission"] == "ita_tx"
        assert progress.meta["phase"] == "progress"

    def test_bonafide_rows_carry_no_attack_id(self):
        manifest = parse_asvspoof2021_cm_keys(KEYS_2021, phases=("eval",))
        bonafide = next(u for u in manifest if u.label is Label.BONAFIDE)
        assert bonafide.attack_id is None

    def test_everything_lands_in_the_held_out_subset(self):
        manifest = parse_asvspoof2021_cm_keys(KEYS_2021, phases=None)
        assert all(u.subset.is_held_out for u in manifest)

    def test_locates_the_label_column_rather_than_indexing_blindly(self):
        """An extra column before the key must not shift the parse. The label
        is found by token and the attack id read from just before it."""

        shifted = "LA_0009 LA_E_9332881 alaw ita_tx extra A07 spoof notrim eval\n"
        manifest = parse_asvspoof2021_cm_keys(shifted, phases=("eval",))
        assert manifest[0].label is Label.SPOOF
        assert manifest[0].attack_id == "A07"

    def test_a_line_with_no_label_token_is_a_loud_failure(self):
        broken = "LA_0009 LA_E_9332881 alaw ita_tx A07 unknown notrim eval\n"
        with pytest.raises(ValueError, match="no 'bonafide' or 'spoof' token"):
            parse_asvspoof2021_cm_keys(broken, phases=None)

    def test_builds_paths_under_the_flac_dir(self):
        manifest = parse_asvspoof2021_cm_keys(
            KEYS_2021, Path("/data/ASVspoof2021_LA_eval/flac"), phases=("eval",)
        )
        assert manifest[0].path == Path(
            "/data/ASVspoof2021_LA_eval/flac/LA_E_1000147.flac"
        )

    def test_loads_from_a_fake_unpacked_keys_tree(self, tmp_path):
        cm_dir = tmp_path / "keys" / "LA" / "CM"
        cm_dir.mkdir(parents=True)
        (cm_dir / "trial_metadata.txt").write_text(KEYS_2021, encoding="utf-8")
        audio = tmp_path / "ASVspoof2021_LA_eval" / "flac"
        audio.mkdir(parents=True)

        manifest = load_asvspoof2021_la_eval(
            tmp_path / "keys", tmp_path / "ASVspoof2021_LA_eval"
        )
        assert len(manifest) == 2
        assert manifest[0].path.parent == audio

    def test_missing_keys_name_the_archive_to_unpack(self, tmp_path):
        with pytest.raises(FileNotFoundError, match=r"LA-keys-full\.tar\.gz"):
            load_asvspoof2021_la_eval(tmp_path / "nope")


# ---------------------------------------------------------------------------
# In-the-Wild
# ---------------------------------------------------------------------------


class TestInTheWild:
    def test_parses_a_headed_meta_csv(self, tmp_path):
        (tmp_path / "meta.csv").write_text(ITW_META, encoding="utf-8")
        manifest = load_in_the_wild(tmp_path)

        assert len(manifest) == 3
        assert manifest[0].utterance_id == "0"
        assert manifest[0].path == tmp_path / "0.wav"
        assert manifest[0].label is Label.SPOOF
        assert manifest[1].label is Label.BONAFIDE
        assert manifest[1].speaker_id == "Alec Guinness"
        assert all(u.subset is Subset.IN_THE_WILD for u in manifest)

    def test_parses_a_headerless_meta_csv(self, tmp_path):
        (tmp_path / "meta.csv").write_text(
            "0.wav,Someone,spoof\n1.wav,Someone,bona-fide\n", encoding="utf-8"
        )
        manifest = load_in_the_wild(tmp_path)
        assert len(manifest) == 2
        assert manifest[1].label is Label.BONAFIDE

    def test_accepts_filename_as_an_alias_for_file(self, tmp_path):
        (tmp_path / "meta.csv").write_text(
            "filename,label\na.wav,spoof\n", encoding="utf-8"
        )
        manifest = load_in_the_wild(tmp_path)
        assert manifest[0].utterance_id == "a"

    def test_missing_meta_names_the_archive_to_unpack(self, tmp_path):
        with pytest.raises(FileNotFoundError, match=r"release_in_the_wild\.zip"):
            load_in_the_wild(tmp_path)

    def test_a_short_row_is_rejected_with_its_line_number(self, tmp_path):
        (tmp_path / "meta.csv").write_text(
            "file,speaker,label\na.wav\n", encoding="utf-8"
        )
        with pytest.raises(ValueError, match=":2:"):
            load_in_the_wild(tmp_path)


# ---------------------------------------------------------------------------
# Manifest behaviour
# ---------------------------------------------------------------------------


def _manifest() -> Manifest:
    return parse_asvspoof2019_cm_protocol(PROTOCOL_2019, Subset.TRAIN)


class TestManifest:
    def test_counts_by_subset_and_label(self):
        assert _manifest().counts == {"train/bonafide": 2, "train/spoof": 2}

    def test_attack_ids_are_sorted_and_deduplicated(self):
        assert _manifest().attack_ids == ("A01", "A06")

    def test_filter_by_label(self):
        spoof = _manifest().filter(label=Label.SPOOF)
        assert len(spoof) == 2
        assert all(u.label is Label.SPOOF for u in spoof)

    def test_filter_by_attack_id(self):
        assert len(_manifest().filter(attack_ids=["A01"])) == 1

    def test_filter_by_several_subsets(self):
        combined = _manifest() + parse_asvspoof2021_cm_keys(KEYS_2021, phases=("eval",))
        assert len(combined.filter(subset=[Subset.TRAIN, Subset.EVAL_2021])) == 6
        assert len(combined.filter(subset=Subset.TRAIN)) == 4

    def test_for_training_refuses_a_manifest_holding_held_out_data(self):
        """The one mistake that cannot be caught by looking at the resulting
        numbers, so it is made structurally impossible."""

        leaky = _manifest() + parse_asvspoof2021_cm_keys(KEYS_2021, phases=("eval",))
        with pytest.raises(ValueError, match="refusing to train on held-out"):
            leaky.for_training()

    def test_for_training_keeps_only_the_train_split(self):
        both = _manifest() + parse_asvspoof2019_cm_protocol(
            "LA_0082 LA_D_1 - A02 spoof\n", Subset.DEV
        )
        assert len(both.for_training()) == 4

    def test_class_weights_are_inverse_frequency_in_target_order(self):
        """8 spoof, 2 bonafide: the weights are total / (2 * count), so
        (10/16, 10/4) = (0.625, 2.5), ordered (spoof, bonafide) to match
        Label.target."""

        utterances = [
            Utterance(f"s{i}", Path(f"s{i}.flac"), Label.SPOOF, Subset.TRAIN, "A01")
            for i in range(8)
        ] + [
            Utterance(f"b{i}", Path(f"b{i}.flac"), Label.BONAFIDE, Subset.TRAIN)
            for i in range(2)
        ]
        spoof_weight, bonafide_weight = Manifest(tuple(utterances)).class_weights()
        assert spoof_weight == pytest.approx(0.625)
        assert bonafide_weight == pytest.approx(2.5)
        assert bonafide_weight > spoof_weight

    def test_class_weights_degrade_to_unity_for_a_single_class(self):
        single = _manifest().filter(label=Label.SPOOF)
        assert single.class_weights() == (1.0, 1.0)

    def test_sample_weights_are_per_utterance(self):
        manifest = _manifest()
        weights = manifest.sample_weights()
        assert len(weights) == len(manifest)
        bonafide_weights = {
            w for w, u in zip(weights, manifest, strict=False) if u.is_bonafide
        }
        assert len(bonafide_weights) == 1

    def test_label_map_feeds_the_metrics_splitter(self):
        assert _manifest().label_map()["LA_T_1271820"] == "spoof"

    def test_missing_files_lists_paths_not_on_disk(self, tmp_path):
        manifest = _manifest().with_root(tmp_path)
        assert len(manifest.missing_files()) == 4
        assert len(manifest.missing_files(limit=2)) == 2

    def test_with_root_rebases_relative_paths_only(self, tmp_path):
        absolute = Manifest(
            (
                Utterance(
                    "x",
                    Path("/already/absolute.flac"),
                    Label.SPOOF,
                    Subset.TRAIN,
                    "A01",
                ),
                Utterance("y", Path("relative.flac"), Label.BONAFIDE, Subset.TRAIN),
            )
        ).with_root(tmp_path)
        assert absolute[0].path == Path("/already/absolute.flac")
        assert absolute[1].path == tmp_path / "relative.flac"

    def test_addition_concatenates_and_records_both_sources(self):
        combined = _manifest() + parse_asvspoof2021_cm_keys(KEYS_2021, phases=("eval",))
        assert len(combined) == 6
        assert "asvspoof2019_la/train" in combined.source
        assert "asvspoof2021_la/eval" in combined.source

    def test_indexing_and_iteration(self):
        manifest = _manifest()
        assert manifest[0].utterance_id == "LA_T_1138215"
        assert len(list(manifest)) == 4


# ---------------------------------------------------------------------------
# ASV scores and root discovery
# ---------------------------------------------------------------------------


class TestAsvScores:
    def test_splits_by_trial_type(self):
        text = (
            "LA_0001 target 1.5\n"
            "LA_0001 nontarget -2.0\n"
            "LA_0001 spoof 0.25\n"
            "LA_0002 target 3.0\n"
        )
        targets, nontargets, spoofs = parse_asv_scores(text)
        assert targets == [1.5, 3.0]
        assert nontargets == [-2.0]
        assert spoofs == [0.25]

    def test_tolerates_an_extra_utterance_id_column(self):
        text = "LA_0001 LA_E_1 target 1.5\n"
        targets, _, _ = parse_asv_scores(text)
        assert targets == [1.5]

    def test_an_unknown_trial_type_is_rejected(self):
        with pytest.raises(ValueError, match="no target/nontarget/spoof"):
            parse_asv_scores("LA_0001 mystery 1.0\n")


class TestDiscoverRoots:
    def test_everything_absent_is_a_normal_state(self, tmp_path):
        roots = discover_roots(tmp_path)
        assert roots.data_dir == tmp_path
        assert roots.asvspoof2019_la is None
        assert roots.asvspoof2021_la_eval is None
        assert roots.musan is None
        assert "not found" in roots.describe()

    def test_a_missing_data_dir_does_not_raise(self, tmp_path):
        roots = discover_roots(tmp_path / "does-not-exist")
        assert roots.asvspoof2019_la is None

    def test_finds_an_unpacked_2019_tree(self, tmp_path):
        (tmp_path / "LA" / "ASVspoof2019_LA_cm_protocols").mkdir(parents=True)
        roots = discover_roots(tmp_path)
        assert roots.asvspoof2019_la == tmp_path / "LA"

    def test_an_la_dir_without_protocols_is_not_the_2019_tree(self, tmp_path):
        """`keys/LA` and the 2021 tree both contain a directory called LA.
        Matching on the name alone would point the 2019 loader at the wrong
        place."""

        (tmp_path / "keys" / "LA" / "CM").mkdir(parents=True)
        roots = discover_roots(tmp_path)
        assert roots.asvspoof2019_la is None
        assert roots.asvspoof2021_keys == tmp_path / "keys"

    def test_finds_nested_augmentation_corpora(self, tmp_path):
        (tmp_path / "musan" / "noise").mkdir(parents=True)
        (tmp_path / "RIRS_NOISES" / "simulated_rirs").mkdir(parents=True)
        roots = discover_roots(tmp_path)
        assert roots.musan == tmp_path / "musan"
        assert roots.rirs_noises == tmp_path / "RIRS_NOISES"

    def test_default_data_dir_honours_the_download_env_var(self, monkeypatch):
        monkeypatch.setenv("CALLBACK_DATA_DIR", "/tmp/elsewhere")
        assert default_data_dir() == Path("/tmp/elsewhere")

    def test_default_data_dir_falls_back_to_the_download_default(self, monkeypatch):
        monkeypatch.delenv("CALLBACK_DATA_DIR", raising=False)
        assert default_data_dir() == Path("~/data/callback").expanduser()
