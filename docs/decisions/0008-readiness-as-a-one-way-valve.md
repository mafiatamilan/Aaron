# 0008 — The readiness verdict is a typed field, consumed through a one-way valve

**Status:** accepted

## Context

Release 0.4's "the argument is generic, so ask" rule lived inside the agent's
argument builder. It worked, but it was invisible: nothing outside the agent
could tell whether a request had been judged complete, the rule was scattered
across three closures (`need`, `need_referent`, `need_content`), and a
classifier that wanted to know "is this request specific enough?" had no way to
ask without being able to build the arguments — which it cannot do, because
the dependency runs the other way.

Release 0.5 wanted a *second* classifier output answering that question, and
the first attempt at it produced a set of findings that changed the design:

* the shipped dataset contains exactly **one** usable example of an
  under-specified request, so the output had no supervision to learn from;
* the deterministic rule over-refused free-text arguments — "recall my name"
  was refused with "not what to look for", because the user *had* named the
  thing;
* the closed list of generic nouns is the only thing standing between the
  assistant and a class of requests the rule cannot see.

## Decision

1. **A typed verdict on the command.** `aaron_core::intent::Readiness` has
   four states — `Ready`, `NeedsClarification`, `NotApplicable`,
   `Undetermined` — and rides on `Command` as one field. `Undetermined` is the
   default, including for a payload serialised before the field existed: a
   missing verdict grants nothing, and `Ready` would be a claim the payload
   never made.

2. **The rule lives in `aaron-core`.** Both the classifier and the agent
   depend on `aaron-core` and neither depends on the other, so the answer to
   "which arguments does this intent need" lives there, in one table. The
   agent's argument builder and the classifier's verdict are checked against
   each other by a test that asserts they agree on every intent across 48
   utterances.

3. **The agent consumes it through a one-way valve.** The check sits after the
   argument builder and before the loop guard, so it can only turn "proceed"
   into "ask". It can never turn "refuse" into "proceed", and it never
   authorises anything: the policy engine, the allow-list, the confirmation
   gate and the loop guard all still decide what runs.

## Reasoning

* **One rule, one place.** The alternative — duplicating the argument table
  into `aaron-brain` — was the obvious risk of a two-crate dependency, and it
  is a drift risk that a test catches and a comment does not.

* **The one-way valve is the safety property, and it is structural.** A
  readiness verdict that could *authorise* would be a new authorisation
  boundary, and authorisation boundaries are the thing this project is most
  careful about. A valve that can only subtract permission cannot weaken
  anything, and it is checkable in one sentence.

* **`Undetermined` rather than `Ready`.** The default matters: a payload with
  no verdict, a backend with no head, and a verdict that failed to decode are
  all *no opinion*, and treating them as `Ready` would turn a missing field
  into a permission.

* **The argument builder's message wins.** It is more specific than anything
  a verdict can produce — "I understood `read_file` but not which file to
  read" beats "I need one more detail" — so the builder is checked first and
  the verdict only sees requests the builder *accepted*.

## Consequences

* A learned readiness head was built, measured, and **rejected**. It is a
  better classifier of the task than the rule (F1 0.683 against 0.471) and
  the wrong thing to ship: it is dominated by the rule at every threshold, and
  the dispatch-rate reduction it buys on adversarial input costs the refusal
  of 86 % of good requests. The head's code, checkpoint format and training
  path are kept, because the measurement is the point, and the promotion gate
  refuses a candidate over the 1,000,000-parameter budget — so a head-bearing
  checkpoint cannot be promoted by accident.

* Because the verdict is typed, the agent can now *report* it, and the
  application layer can show a caller whether a request was under-specified
  rather than guessing from the reply text.

* One known interaction is recorded rather than papered over: the check runs
  before the policy engine, so a verdict can preempt a policy *denial* and the
  user is asked for more detail instead of being told the request is refused.
  Nothing is dispatched either way and the boundary is untouched, but the
  message is less informative. It is one of the reasons a learned head was
  rejected rather than adopted, and it is the thing to revisit first if one
  ever is.
