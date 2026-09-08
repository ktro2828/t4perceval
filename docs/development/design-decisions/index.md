# Design decisions

Architecture Decision Records: what was decided, why, what else was considered, and what it costs.

While `t4perceval` is still evolving quickly this matters more than usual, because it preserves not
only _what_ the architecture is but _how it became that way_ -- which is what tells a future reader
whether a constraint is load-bearing or incidental.

| ADR                                                         | Status   | Decides                                                             |
| :---------------------------------------------------------- | :------- | :------------------------------------------------------------------ |
| [0001](0001-ecs-data-model.md) ECS data model               | Accepted | columns and systems instead of objects and an enum                  |
| [0002](0002-arrow-storage.md) Arrow storage                 | Accepted | Arrow as the on-disk and in-memory interchange format               |
| [0003](0003-coordinate-system.md) Coordinate frames as data | Accepted | frames recorded as rows, and a guard instead of a silent conversion |
| [0004](0004-persistent-recording.md) Persistent recordings  | Proposed | a directory format for a whole evaluation                           |

## Writing one

Copy this structure:

```markdown
# Title

## Status

Accepted / Proposed / Superseded

## Context

What problem are we solving?

## Decision

What did we decide?

## Rationale

Why was this approach chosen?

## Alternatives considered

What other approaches were considered?

## Consequences

What are the resulting trade-offs?
```

Number files sequentially and never renumber one. Superseding an ADR means adding a new one and
marking the old as `Superseded by NNNN`, not editing it -- the record of a decision that turned out
badly is more useful than its absence.

Add an ADR when a change would make a future reader ask "why is it like this?". Ordinary
implementation choices do not need one; anything that constrains what the library can do later does.
