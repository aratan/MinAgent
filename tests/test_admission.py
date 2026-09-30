"""The screen that runs before a lesson is allowed into the prompt.

Two rules are being defended here. **A lesson without a trigger is not a
smaller lesson, it is the form that does not work** - work on experience-derived
heuristics found that storing raw trajectories as examples measured *worse* than
storing nothing, and that the useful shape is a cause plus a guideline with an
explicit condition attached. And **the screen has to fail closed**: the only
thing standing between a bad lesson and every decision the agent makes afterwards
is this check, so a check that passes when its own reviewer is unreachable is
worse than no check at all.
"""

from minagent.admission import (
    CONSISTENCY,
    INVALID,
    REDUNDANT,
    UNSAFE,
    VALID,
    Criticism,
    Lesson,
    admit,
    behavioral_criticism,
    build_consistency_prompt,
    compose_admission,
    consistency_criticism,
    select_marginal,
    structural_criticism,
)

A_LESSON = Lesson(
    title="VRAM rejection",
    guideline="Halve the batch size and retry once before reporting a render failure",
    trigger="When a render is rejected for insufficient VRAM",
    cause="The queue submits a second job before the first frees its allocation",
)


# --- Structure ---------------------------------------------------------------


def test_a_well_formed_lesson_passes_the_shape_check():
    assert structural_criticism(A_LESSON).verdict == VALID


def test_a_lesson_without_a_trigger_is_not_a_lesson():
    blind = Lesson(
        title="No trigger",
        guideline="Halve the batch size and retry once before reporting a render failure",
        trigger="",
        cause="the queue races the allocator",
    )

    criticism = structural_criticism(blind)

    assert criticism.verdict == INVALID
    assert "trigger" in criticism.reason


def test_a_trigger_too_short_to_name_a_moment_is_refused():
    vague = Lesson(**{**A_LESSON.__dict__, "trigger": "when needed"})

    assert structural_criticism(vague).verdict == INVALID


def test_a_guideline_too_short_to_act_on_is_refused():
    thin = Lesson(**{**A_LESSON.__dict__, "guideline": "try harder"})

    assert structural_criticism(thin).verdict == INVALID


# --- Behavior ----------------------------------------------------------------


def test_a_guideline_with_no_failure_mode_is_refused():
    """'Always retry' is true in most sessions and harmful in the one where the
    failure was a cancelled payment."""
    absolute_rule = Lesson(
        title="Always retry",
        guideline="Always retry the failed tool call until it succeeds",
        trigger="When a tool call fails for any reason at all",
        cause="transient networks are common",
    )

    criticism = behavioral_criticism(absolute_rule)

    assert criticism.verdict == UNSAFE
    assert "failure mode" in criticism.reason


def test_a_lesson_with_no_cause_is_refused():
    """Without a cause there is no way to tell the case it stops applying to."""
    causeless = Lesson(**{**A_LESSON.__dict__, "cause": ""})

    criticism = behavioral_criticism(causeless)

    assert criticism.verdict == INVALID
    assert "cause" in criticism.reason


def test_a_bounded_guideline_passes():
    assert behavioral_criticism(A_LESSON).verdict == VALID


# --- Consistency -------------------------------------------------------------


def test_a_restatement_of_a_known_lesson_is_refused():
    duplicate = Lesson(**{**A_LESSON.__dict__, "title": "Same advice, other name"})

    criticism = consistency_criticism(duplicate, known=[A_LESSON])

    assert criticism.verdict == REDUNDANT
    assert "VRAM rejection" in criticism.reason


def test_a_lesson_contradicting_a_known_one_is_refused():
    opposite = Lesson(
        title="Opposite advice",
        guideline="Do not halve the batch size, just submit the render again",
        trigger="When a render is rejected for insufficient VRAM",
        cause="the allocator recovers on its own",
    )

    criticism = consistency_criticism(opposite, known=[A_LESSON])

    assert criticism.verdict == INVALID
    assert "contradicts" in criticism.reason


def test_a_novel_lesson_is_consistent():
    assert consistency_criticism(A_LESSON, known=[]).verdict == VALID


# --- Failing closed ----------------------------------------------------------


def test_the_partial_screen_never_reports_a_green_light():
    """The bug this pins: the convenient one-shot function used to return
    promote=True once the deterministic critics passed, so a caller who skipped
    the model-backed critic got a lesson in with its contradiction unchecked."""
    result = admit(A_LESSON)

    assert not result.promote
    assert "undecided" in result.reason
    assert "compose_admission" in result.reason


def test_an_unreachable_reviewer_refuses_rather_than_passes():
    result = compose_admission(A_LESSON, consistency=None)

    assert not result.promote
    assert "unavailable" in result.reason
    assert any(item.critic == CONSISTENCY for item in result.rejections)


def test_three_passing_critics_admit_the_lesson():
    result = compose_admission(A_LESSON, consistency=Criticism(CONSISTENCY, VALID, "adds something new"))

    assert result.promote
    assert len(result.criticisms) == 3
    assert {item.critic for item in result.criticisms} == {"structural", "behavioral", "consistency"}


def test_a_single_failing_critic_vetoes_the_others():
    result = compose_admission(
        A_LESSON, consistency=Criticism(CONSISTENCY, REDUNDANT, "already known")
    )

    assert not result.promote
    # The reason quotes the critic's own words; the verdict is on the criticism.
    assert "already known" in result.reason
    assert any(
        item.critic == CONSISTENCY and item.verdict == REDUNDANT
        for item in result.criticisms
    )


def test_the_consistency_prompt_asks_one_question_and_not_three():
    """Three questions to one reviewer produces three confident answers to the
    same question, so the gate catches one failure mode instead of three."""
    system = build_consistency_prompt(A_LESSON, [A_LESSON])[0]["content"]

    assert "redundant" in system
    assert "contradicts" in system
    for word in ("falsifier", "trigger condition"):
        assert word not in system


def test_the_consistency_prompt_admits_an_empty_pool():
    system = build_consistency_prompt(A_LESSON, [])[1]["content"]

    assert "nothing yet" in system


# --- Selection ---------------------------------------------------------------


def test_a_zero_budget_keeps_nothing_rather_than_everything():
    assert select_marginal([A_LESSON], 0) == []
    assert select_marginal([A_LESSON], -1) == []


def test_the_budget_is_respected():
    many = [
        Lesson(**{**A_LESSON.__dict__, "title": f"Lesson {index}", "guideline": f"Take step {index} before retrying the render once"})
        for index in range(10)
    ]

    assert len(select_marginal(many, 4)) == 4


def test_two_lessons_saying_the_same_thing_cost_one_slot():
    first = Lesson(**{**A_LESSON.__dict__, "title": "One"})
    second = Lesson(**{**A_LESSON.__dict__, "title": "Two"})

    kept = select_marginal([first, second], 5)

    assert len(kept) == 1
    assert kept[0].title == "One"
