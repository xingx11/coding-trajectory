"""Unit tests for TOML parsing (score.extract_toml_section, score._parse_scored_toml)
and passrate calculation (toml_utils.calc_passrate, read/write round-trip, helpers)."""

from __future__ import annotations

import tempfile
import unittest
from pathlib import Path

from ctpipe.config import (
    REFERENCE_CRITERION_DESCRIPTIONS,
    MAX_CRITERIA_COUNT,
    MIN_CRITERIA_COUNT,
    REFERENCE_CRITERION_NAMES,
)
from ctpipe.score import extract_toml_section, _parse_scored_toml
from ctpipe.toml_utils import (
    Criterion,
    escape_toml_basic,
    escape_toml_multiline,
    calc_passrate,
    has_score_tiers,
    is_complete_rubric,
    is_unscored_template,
    read_quality_toml,
    write_quality_toml,
)

# Grab a usable subset of valid names for building test TOML
_NAMES = REFERENCE_CRITERION_NAMES[:MAX_CRITERIA_COUNT]


def _make_criterion(name: str, score: int = 3, rationale: str = "ok") -> Criterion:
    return Criterion(
        name=name,
        description=REFERENCE_CRITERION_DESCRIPTIONS[name],
        type="likert",
        points=5,
        weight=1.0,
        score=score,
        rationale=rationale,
    )


def _build_toml_block(
    name: str,
    score: int = 3,
    rationale: str = "evidence here",
    description: str | None = None,
    weight: float | None = None,
) -> str:
    desc = description or REFERENCE_CRITERION_DESCRIPTIONS[name]
    w = weight if weight is not None else 1.0
    return (
        f'[[criterion]]\n'
        f'name = "{name}"\n'
        f'description = "{desc}"\n'
        f'type = "likert"\n'
        f'points = 5\n'
        f'weight = {w}\n'
        f'score = {score}\n'
        f'rationale = "{rationale}"\n'
    )


def _build_valid_toml(count: int = 7, score: int = 3) -> str:
    blocks = [_build_toml_block(_NAMES[i], score=score) for i in range(count)]
    return "\n".join(blocks)


# ---------------------------------------------------------------------------
# extract_toml_section
# ---------------------------------------------------------------------------
class TestExtractTomlSection(unittest.TestCase):

    def test_plain_toml_unchanged(self):
        raw = _build_valid_toml(7)
        self.assertEqual(extract_toml_section(raw), raw.strip())

    def test_strips_bad_pattern_section(self):
        toml_part = _build_valid_toml(7)
        raw = toml_part + "\nBad Pattern 命中：\n- xxx：detail"
        result = extract_toml_section(raw)
        self.assertNotIn("Bad Pattern", result)
        self.assertIn("[[criterion]]", result)

    def test_strips_bad_pattern_variant(self):
        toml_part = _build_valid_toml(7)
        raw = toml_part + "\nBad Pattern命中：未发现"
        result = extract_toml_section(raw)
        self.assertNotIn("Bad Pattern", result)

    def test_strips_bad_pattern_lowercase(self):
        toml_part = _build_valid_toml(7)
        raw = toml_part + "\nbad pattern hits: none"
        result = extract_toml_section(raw)
        self.assertNotIn("bad pattern", result.lower().split("[[criterion]]")[0]
                         if "[[criterion]]" in result else result.lower())

    def test_strips_markdown_fences(self):
        toml_part = _build_valid_toml(7)
        raw = "```toml\n" + toml_part + "\n```"
        result = extract_toml_section(raw)
        self.assertIn("[[criterion]]", result)
        self.assertNotIn("```", result)

    def test_strips_claude_meta_prefix(self):
        toml_part = _build_valid_toml(7)
        raw = "[__claude_meta:version=1]\n" + toml_part
        result = extract_toml_section(raw)
        self.assertIn("[[criterion]]", result)
        self.assertNotIn("__claude_meta", result)

    def test_empty_input(self):
        result = extract_toml_section("")
        self.assertEqual(result.strip(), "")


# ---------------------------------------------------------------------------
# _parse_scored_toml
# ---------------------------------------------------------------------------
class TestParseScoreToml(unittest.TestCase):

    def test_valid_toml_parses(self):
        raw = _build_valid_toml(7)
        result = _parse_scored_toml(raw)
        self.assertIsNotNone(result)
        self.assertEqual(len(result), 7)
        for c in result:
            self.assertIn(c.name, REFERENCE_CRITERION_NAMES)
            self.assertEqual(c.score, 3)

    def test_min_criteria_count(self):
        raw = _build_valid_toml(MIN_CRITERIA_COUNT)
        result = _parse_scored_toml(raw)
        self.assertIsNotNone(result)
        self.assertEqual(len(result), MIN_CRITERIA_COUNT)

    def test_max_criteria_count(self):
        raw = _build_valid_toml(MAX_CRITERIA_COUNT)
        result = _parse_scored_toml(raw)
        self.assertIsNotNone(result)
        self.assertEqual(len(result), MAX_CRITERIA_COUNT)

    def test_too_few_criteria_rejected(self):
        raw = _build_valid_toml(MIN_CRITERIA_COUNT - 1)
        self.assertIsNone(_parse_scored_toml(raw))

    def test_too_many_criteria_rejected(self):
        names = REFERENCE_CRITERION_NAMES[: MAX_CRITERIA_COUNT + 1]
        blocks = [_build_toml_block(n) for n in names]
        raw = "\n".join(blocks)
        self.assertIsNone(_parse_scored_toml(raw))

    def test_invalid_criterion_name_rejected(self):
        blocks = [_build_toml_block(_NAMES[i]) for i in range(6)]
        blocks.append(
            '[[criterion]]\nname = "InvalidCamelCase"\ndescription = "x"\n'
            'type = "likert"\npoints = 5\nweight = 1.0\nscore = 3\nrationale = "ok"\n')
        self.assertIsNone(_parse_scored_toml("\n".join(blocks)))

    def test_duplicate_name_rejected(self):
        blocks = [_build_toml_block(_NAMES[0])] * 7
        self.assertIsNone(_parse_scored_toml("\n".join(blocks)))

    def test_score_zero_rejected(self):
        blocks = [_build_toml_block(_NAMES[i], score=(0 if i == 0 else 3)) for i in range(7)]
        self.assertIsNone(_parse_scored_toml("\n".join(blocks)))

    def test_score_six_rejected(self):
        blocks = [_build_toml_block(_NAMES[i], score=(6 if i == 0 else 3)) for i in range(7)]
        self.assertIsNone(_parse_scored_toml("\n".join(blocks)))

    def test_float_score_rejected(self):
        toml = _build_valid_toml(7).replace("score = 3", "score = 3.5", 1)
        self.assertIsNone(_parse_scored_toml(toml))

    def test_empty_rationale_rejected(self):
        blocks = [_build_toml_block(_NAMES[i], rationale=("" if i == 0 else "ok")) for i in range(7)]
        self.assertIsNone(_parse_scored_toml("\n".join(blocks)))

    def test_malformed_toml_rejected(self):
        self.assertIsNone(_parse_scored_toml("this is not toml at all"))

    def test_expected_names_match(self):
        names = _NAMES[:7]
        raw = _build_valid_toml(7)
        result = _parse_scored_toml(raw, expected_names=names)
        self.assertIsNotNone(result)

    def test_expected_names_mismatch_rejected(self):
        raw = _build_valid_toml(7)
        wrong_names = _NAMES[1:8]
        self.assertIsNone(_parse_scored_toml(raw, expected_names=wrong_names))

    def test_weight_preserved_from_toml(self):
        """Weight values in TOML are preserved as-is (no longer overridden by name)."""
        names_with_weights = [(_NAMES[0], 2.0)] + [(_NAMES[i], 1.0) for i in range(1, 7)]
        blocks = [_build_toml_block(n, weight=w) for n, w in names_with_weights]
        raw = "\n".join(blocks)
        result = _parse_scored_toml(raw)
        self.assertIsNotNone(result)
        self.assertEqual(result[0].weight, 2.0)
        for c in result[1:]:
            self.assertEqual(c.weight, 1.0)

    def test_description_preserved_from_toml(self):
        """Description is read from TOML as-is (no longer overridden by canonical)."""
        blocks = [_build_toml_block(_NAMES[i], description="AI wrote this") for i in range(7)]
        raw = "\n".join(blocks)
        result = _parse_scored_toml(raw, use_custom_descriptions=False)
        self.assertIsNotNone(result)
        for c in result:
            self.assertEqual(c.description, "AI wrote this")

    def test_custom_descriptions_missing_tiers_rejected(self):
        blocks = [_build_toml_block(_NAMES[i], description="没有评分层级的简短描述") for i in range(7)]
        raw = "\n".join(blocks)
        self.assertIsNone(_parse_scored_toml(raw, use_custom_descriptions=True))

    def test_custom_descriptions_with_tiers_accepted(self):
        tiered = (
            "自定义维度。1分：完全失败无有效产出。"
            "2分：部分完成但关键缺失大。"
            "3分：主路径完成存在明显遗漏。"
            "4分：大部分完成仅有轻微问题。"
            "5分：完整高质量有充分验证。"
        )
        blocks = [_build_toml_block(_NAMES[i], description=tiered) for i in range(7)]
        raw = "\n".join(blocks)
        result = _parse_scored_toml(raw, use_custom_descriptions=True)
        self.assertIsNotNone(result)
        for c in result:
            self.assertEqual(c.description, tiered)

    def test_strips_bad_pattern_before_parsing(self):
        toml_part = _build_valid_toml(7)
        raw = toml_part + "\n\nBad Pattern 命中：\n- loop：反复尝试"
        result = _parse_scored_toml(raw)
        self.assertIsNotNone(result)
        self.assertEqual(len(result), 7)

    def test_strips_markdown_fences_before_parsing(self):
        toml_part = _build_valid_toml(7)
        raw = "```toml\n" + toml_part + "\n```"
        result = _parse_scored_toml(raw)
        self.assertIsNotNone(result)

    def test_score_boundary_1_and_5(self):
        blocks_1 = [_build_toml_block(_NAMES[i], score=1) for i in range(7)]
        blocks_5 = [_build_toml_block(_NAMES[i], score=5) for i in range(7)]
        r1 = _parse_scored_toml("\n".join(blocks_1))
        r5 = _parse_scored_toml("\n".join(blocks_5))
        self.assertIsNotNone(r1)
        self.assertIsNotNone(r5)
        self.assertTrue(all(c.score == 1 for c in r1))
        self.assertTrue(all(c.score == 5 for c in r5))


# ---------------------------------------------------------------------------
# calc_passrate
# ---------------------------------------------------------------------------
class TestCalcPassrate(unittest.TestCase):

    def test_all_perfect(self):
        criteria = [_make_criterion(_NAMES[i], score=5) for i in range(7)]
        self.assertAlmostEqual(calc_passrate(criteria), 1.0)

    def test_all_lowest(self):
        criteria = [_make_criterion(_NAMES[i], score=1) for i in range(7)]
        self.assertAlmostEqual(calc_passrate(criteria), 0.2)

    def test_weighted_calculation(self):
        arch = "architecture_boundaries_and_security_compliance"
        other = [n for n in REFERENCE_CRITERION_NAMES if n != arch][:6]
        criteria = [_make_criterion(arch, score=5)]
        criteria += [_make_criterion(n, score=5) for n in other]
        total_score = sum(c.score * c.weight for c in criteria)
        total_points = sum(c.points * c.weight for c in criteria)
        self.assertAlmostEqual(calc_passrate(criteria), total_score / total_points)

    def test_mixed_scores(self):
        criteria = [_make_criterion(_NAMES[i], score=(i % 5) + 1) for i in range(7)]
        result = calc_passrate(criteria)
        self.assertGreater(result, 0.0)
        self.assertLessEqual(result, 1.0)

    def test_weight_matters(self):
        """Different weights should shift the passrate result."""
        c_high = [
            Criterion(name="dim_a", description="d", type="likert", points=5, weight=2.0, score=5, rationale="ok"),
            Criterion(name="dim_b", description="d", type="likert", points=5, weight=1.0, score=1, rationale="ok"),
        ]
        c_low = [
            Criterion(name="dim_a", description="d", type="likert", points=5, weight=2.0, score=1, rationale="ok"),
            Criterion(name="dim_b", description="d", type="likert", points=5, weight=1.0, score=5, rationale="ok"),
        ]
        # weight=2.0 dimension scored high vs low should shift the result
        self.assertGreater(calc_passrate(c_high), calc_passrate(c_low))

    def test_zero_total_points_raises(self):
        c = Criterion(name="x", description="", type="likert",
                      points=0, weight=0.0, score=0, rationale="")
        with self.assertRaises(ValueError):
            calc_passrate([c])

    def test_single_criterion(self):
        c = _make_criterion(_NAMES[0], score=4)
        self.assertAlmostEqual(calc_passrate([c]), 4.0 / 5.0)


# ---------------------------------------------------------------------------
# escape_toml_basic / escape_toml_multiline
# ---------------------------------------------------------------------------
class TestEscapeToml(unittest.TestCase):

    def test_basic_no_special(self):
        self.assertEqual(escape_toml_basic("hello"), "hello")

    def test_basic_backslash(self):
        self.assertEqual(escape_toml_basic("a\\b"), "a\\\\b")

    def test_basic_quotes(self):
        self.assertEqual(escape_toml_basic('say "hi"'), 'say \\"hi\\"')

    def test_basic_newline_and_tab(self):
        self.assertEqual(escape_toml_basic("a\nb\tc"), "a\\nb\\tc")

    def test_multiline_triple_quotes(self):
        result = escape_toml_multiline('has """ inside')
        self.assertNotIn('"""', result)

    def test_multiline_trailing_quote(self):
        result = escape_toml_multiline('ends with"')
        self.assertTrue(result.endswith('\\"'))


# ---------------------------------------------------------------------------
# read_quality_toml / write_quality_toml round-trip
# ---------------------------------------------------------------------------
class TestReadWriteRoundTrip(unittest.TestCase):

    def test_round_trip(self):
        criteria = [_make_criterion(_NAMES[i], score=i + 1, rationale=f"reason {i}") for i in range(5)]
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "test.quality.toml"
            write_quality_toml(path, criteria)
            loaded = read_quality_toml(path)
        self.assertEqual(len(loaded), len(criteria))
        for orig, loaded_c in zip(criteria, loaded):
            self.assertEqual(orig.name, loaded_c.name)
            self.assertEqual(orig.score, loaded_c.score)
            self.assertEqual(orig.weight, loaded_c.weight)
            self.assertEqual(orig.rationale, loaded_c.rationale)

    def test_round_trip_special_chars_in_rationale(self):
        c = _make_criterion(_NAMES[0], score=3, rationale='包含"引号"和\\反斜杠')
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "special.quality.toml"
            write_quality_toml(path, [c])
            loaded = read_quality_toml(path)
        self.assertEqual(len(loaded), 1)
        self.assertEqual(loaded[0].rationale, c.rationale)

    def test_written_file_is_valid_toml(self):
        import tomllib
        criteria = [_make_criterion(_NAMES[i], score=3) for i in range(7)]
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "valid.quality.toml"
            write_quality_toml(path, criteria)
            data = tomllib.loads(path.read_text(encoding="utf-8"))
        self.assertEqual(len(data["criterion"]), 7)


# ---------------------------------------------------------------------------
# is_unscored_template / is_complete_rubric
# ---------------------------------------------------------------------------
class TestTemplateAndRubricChecks(unittest.TestCase):

    def test_unscored_template_true(self):
        criteria = [
            Criterion(name=_NAMES[i], description="d", type="likert",
                      points=5, weight=1.0, score=0, rationale="")
            for i in range(7)
        ]
        self.assertTrue(is_unscored_template(criteria))

    def test_unscored_template_false_when_scored(self):
        criteria = [_make_criterion(_NAMES[i], score=3, rationale="ok") for i in range(7)]
        self.assertFalse(is_unscored_template(criteria))

    def test_unscored_template_false_partial(self):
        criteria = [
            Criterion(name=_NAMES[0], description="d", type="likert",
                      points=5, weight=1.0, score=3, rationale=""),
            Criterion(name=_NAMES[1], description="d", type="likert",
                      points=5, weight=1.0, score=0, rationale=""),
        ]
        self.assertFalse(is_unscored_template(criteria))

    def test_complete_rubric_true(self):
        criteria = [_make_criterion(_NAMES[i], score=3, rationale="reason") for i in range(7)]
        self.assertTrue(is_complete_rubric(criteria))

    def test_complete_rubric_false_too_few(self):
        criteria = [_make_criterion(_NAMES[i], score=3) for i in range(MIN_CRITERIA_COUNT - 1)]
        self.assertFalse(is_complete_rubric(criteria))

    def test_complete_rubric_false_no_rationale(self):
        criteria = [_make_criterion(_NAMES[i], score=3, rationale="") for i in range(7)]
        self.assertFalse(is_complete_rubric(criteria))


# ---------------------------------------------------------------------------
# has_score_tiers
# ---------------------------------------------------------------------------
class TestHasScoreTiers(unittest.TestCase):

    def test_valid_five_tiers(self):
        desc = (
            "评估维度。1分：完全失败无产出。"
            "2分：部分完成但关键缺失。"
            "3分：主路径完成有遗漏。"
            "4分：大部分完成轻微问题。"
            "5分：完整高质量充分验证。"
        )
        self.assertTrue(has_score_tiers(desc))

    def test_missing_tier_rejected(self):
        desc = "1分：a。2分：bb。3分：ccc。4分：dddd。"
        self.assertFalse(has_score_tiers(desc))

    def test_empty_tier_content_rejected(self):
        desc = "1分：。2分：bb。3分：ccc。4分：dddd。5分：eeeee。"
        self.assertFalse(has_score_tiers(desc))

    def test_colon_variants(self):
        desc = (
            "1分:完全失败无有效产出。"
            "2分:部分完成关键缺失大。"
            "3分:主路径完成存在遗漏。"
            "4分:大部分完成轻微问题。"
            "5分:完整高质量充分验证。"
        )
        self.assertTrue(has_score_tiers(desc))

    def test_no_tiers_at_all(self):
        self.assertFalse(has_score_tiers("这是一个普通描述，没有评分层级"))

    def test_real_criterion_description(self):
        desc = REFERENCE_CRITERION_DESCRIPTIONS[REFERENCE_CRITERION_NAMES[0]]
        self.assertTrue(has_score_tiers(desc))


if __name__ == "__main__":
    unittest.main()
