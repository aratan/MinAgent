"""The gate that decides whether a self-change is allowed to stand.

The claims under test are the ones that make this gate worth having, and they
came from measurements rather than taste: pass@1 moves by more than a point and a
half between runs of the *same* configuration, so a single before/after
comparison discovers improvements that do not exist; and an average hides the
specific case that went from working to broken, so a release can look better
while shipping a regression.
"""

from minagent.evidence import (
    FIX,
    MIN_RUNS_PER_ARM,
    NOISE_FLOOR_POINTS,
    REGRESSION,
    Arm,
    CaseOutcome,
    Holdout,
    compare_arms,
    find_flips,
    read_holdout,
    split_holdout,
)


def _arm(name: str, rows: list[tuple[str, bool, int]], *, holdout: bool = False) -> Arm:
    return Arm(name, tuple(CaseOutcome(*row) for row in rows), holdout)


# --- The noise floor ---------------------------------------------------------


def test_a_gain_smaller_than_the_noise_is_not_an_improvement():
    """The measurement this gate exists for: same config, two runs, a real gap."""
    cases = [f"case-{index}" for index in range(10)]
    baseline = _arm("base", [(case, True, 1) for case in cases])
    # Nine of ten pass, against ten of ten: +1.0 points, inside the noise.
    candidate = _arm(
        "cand",
        [(case, True, 1) for case in cases[:-1]] + [(cases[-1], False, 1)],
    )

    verdict = compare_arms(baseline, candidate)

    assert not verdict.promote
    assert not verdict.measurable


def test_a_gain_larger_than_the_noise_with_enough_runs_promotes():
    cases = [f"case-{index}" for index in range(10)]
    # One case broken in the baseline, fixed in the candidate, across every run:
    # 90.0 -> 100.0, which clears the floor and costs no regressions.
    baseline = _arm(
        "base",
        [
            (case, case != cases[-1], run)
            for case in cases
            for run in range(MIN_RUNS_PER_ARM)
        ],
    )
    candidate = _arm(
        "cand", [(case, True, run) for case in cases for run in range(MIN_RUNS_PER_ARM)]
    )

    verdict = compare_arms(baseline, candidate, cases=cases)

    assert verdict.promote
    assert verdict.delta_points is not None
    assert verdict.delta_points >= NOISE_FLOOR_POINTS
    assert verdict.regression_count == 0
    assert len(verdict.fixes) == 1


def test_too_few_runs_is_unmeasurable_rather_than_a_verdict():
    """Reporting "not enough runs" is more useful than reporting a refusal,
    because it tells a caller what to do next."""
    baseline = _arm("base", [("a", True, 1), ("b", False, 1)])
    candidate = _arm("cand", [("a", True, 1), ("b", True, 1)])

    verdict = compare_arms(baseline, candidate, min_runs=9)

    assert not verdict.promote
    assert not verdict.measurable
    assert "run" in verdict.reason.lower()


def test_ten_results_in_one_run_still_count_as_one_run():
    baseline = _arm("base", [(f"case-{index}", True, 1) for index in range(10)])
    candidate = _arm("cand", [(f"case-{index}", index != 0, 1) for index in range(10)])

    verdict = compare_arms(baseline, candidate, min_runs=9)

    assert not verdict.measurable
    assert baseline.runs() == 1


# --- Regressions outrank a better average ------------------------------------


def test_a_case_that_broke_blocks_a_better_mean():
    cases = [f"case-{index}" for index in range(10)]
    rows = [(case, True, 1) for case in cases]
    # Fix five, break one. The mean improves and the change is still wrong.
    candidate_rows = [
        (case, index not in (0, 1, 2, 3, 4), 1) for index, case in enumerate(cases)
    ]

    verdict = compare_arms(_arm("base", rows), _arm("cand", candidate_rows), cases=cases)

    assert not verdict.promote
    assert verdict.regression_count >= 1


def test_a_flip_is_named_with_its_direction():
    baseline = _arm("base", [("tooling", False, 1)])
    candidate = _arm("cand", [("tooling", True, 1)])

    flips = find_flips(baseline, candidate)

    assert len(flips) == 1
    assert flips[0].case == "tooling"
    assert flips[0].direction == FIX
    # The detail names the runs that decided it, not the case - the case is
    # already the field the caller looks in.
    assert "failed" in flips[0].detail and "passed" in flips[0].detail


def test_too_many_regressions_reject_even_a_large_gain():
    cases = [f"case-{index}" for index in range(10)]
    rows = [(case, True, 1) for case in cases]
    broken = [(case, index not in (0, 1), 1) for index, case in enumerate(cases)]

    verdict = compare_arms(_arm("base", rows), _arm("cand", broken), max_regressions=1)

    assert not verdict.promote
    assert verdict.regression_count == 2
    assert all(flip.direction == REGRESSION for flip in verdict.regressions)


def test_a_regression_budget_of_zero_forbids_breaking_anything():
    cases = ["a", "b", "c"]
    baseline = _arm("base", [(case, True, 1) for case in cases])
    candidate = _arm("cand", [("a", True, 1), ("b", False, 1), ("c", True, 1)])

    verdict = compare_arms(baseline, candidate, max_regressions=0)

    assert not verdict.promote


# --- The holdout -------------------------------------------------------------


def test_the_holdout_respects_the_fraction_it_is_given():
    """A configured fraction that silently reserved a different amount would
    make every downstream number a lie nobody could detect."""
    cases = [f"case-{index}" for index in range(40)]

    assert len(split_holdout(cases, 0.1).cases) == 4
    assert len(split_holdout(cases, 0.2).cases) == 8
    assert len(split_holdout(cases, 0.5).cases) == 20


def test_the_holdout_is_the_same_split_every_time():
    cases = [f"case-{index}" for index in range(40)]

    assert split_holdout(cases, 0.2).cases == split_holdout(cases, 0.2).cases
    assert split_holdout(cases, 0.2).cases == split_holdout(list(reversed(cases)), 0.2).cases


def test_a_holdout_of_one_case_leaves_something_to_gate_on():
    holdout = split_holdout(["only"], 0.5)

    assert len(holdout.cases) == 1
    assert holdout.gating_cases(["only"]) == []


def test_reserved_cases_are_never_gated_on():
    holdout = Holdout(cases=("secret",))

    assert holdout.gating_cases(["visible", "secret"]) == ["visible"]
    assert holdout.holds("secret")


def test_the_holdout_reading_covers_only_the_reserved_cases():
    """The bug this pins: averaging the whole arm here reports the full suite's
    improvement as if it were the reserved sample's - the exact leakage the
    holdout was created to prevent."""
    baseline = _arm(
        "base",
        [("a", True, 1), ("b", True, 1), ("secret", False, 1)],
    )
    candidate = _arm(
        "cand",
        [("a", True, 1), ("b", False, 1), ("secret", True, 1)],
    )

    reading = read_holdout(Holdout(cases=("secret",)), baseline, candidate)

    # The whole arm is 66.7 -> 66.7, which would read as "no movement" and hide
    # the fact that the reserved case is the one that moved.
    assert reading.before_rate == 0.0
    assert reading.after_rate == 100.0
    assert reading.cases == ("secret",)


def test_a_reserved_case_going_backwards_is_reported_as_a_surprise():
    baseline = _arm("base", [("a", True, 1), ("secret", True, 1)])
    candidate = _arm("cand", [("a", True, 1), ("secret", False, 1)])

    reading = read_holdout(Holdout(cases=("secret",)), baseline, candidate)

    assert not reading.agrees()
    assert "fitted" in reading.surprise


def test_reading_the_holdout_spends_it():
    holdout = Holdout(cases=("secret",))
    baseline = _arm("base", [("secret", True, 1)])
    candidate = _arm("cand", [("secret", True, 1)])

    read_holdout(holdout, baseline, candidate)

    assert holdout.spent


def test_the_holdout_survives_a_round_trip():
    holdout = split_holdout([f"case-{index}" for index in range(20)], 0.25)
    holdout.spend()

    restored = Holdout.from_json(holdout.to_json())

    assert restored.cases == holdout.cases
    assert restored.spent


def test_broken_holdout_json_reads_as_empty_rather_than_crashing():
    assert Holdout.from_json("not json at all").cases == ()
