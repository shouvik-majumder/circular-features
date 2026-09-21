"""Item sets and prompt templates.

The claim we are testing is that a language model represents cyclic categories, such as the days
of the week, on an actual **circle** in activation space, rather than as seven unrelated
directions. To test that we need:

  * a cyclic set   (weekdays, months) where a circle is predicted,
  * a linear set   (number words, ranked sizes) where an ordered line is predicted instead,
  * control sets   (unrelated common nouns) where neither is predicted.

Without the last two, "the points form a ring" is not evidence of anything: seven random
high-dimensional vectors projected to their own top two principal components often look
vaguely ring-shaped. The controls are the whole point.

Each item is embedded in several short templates and the activations are averaged, so that we
measure the representation of the *concept* rather than the quirks of one sentence.
"""
from __future__ import annotations

from dataclasses import dataclass, field


@dataclass
class ItemSet:
    name: str
    items: list[str]
    templates: list[str]
    kind: str = "cyclic"          # cyclic | ordered | control
    period: int | None = None     # length of the cycle, for cyclic sets

    def __post_init__(self) -> None:
        if self.kind == "cyclic" and self.period is None:
            self.period = len(self.items)

    def prompts(self) -> list[tuple[str, str]]:
        """(item, prompt) pairs. The item always appears verbatim in the prompt."""
        return [(item, t.format(item=item)) for item in self.items for t in self.templates]


DAY_TEMPLATES = [
    "Today is {item}",
    "The meeting is on {item}",
    "She always goes swimming on {item}",
    "I will see you next {item}",
    "The shop is closed on {item}",
]

MONTH_TEMPLATES = [
    "The month is {item}",
    "We are travelling in {item}",
    "His birthday is in {item}",
    "The report is due in {item}",
    "It rained every day last {item}",
]

GENERIC_TEMPLATES = [
    "The word is {item}",
    "He said the word {item}",
    "She wrote down {item}",
    "They talked about {item}",
    "I remember the {item}",
]

WEEKDAYS = ItemSet(
    "weekdays",
    ["Monday", "Tuesday", "Wednesday", "Thursday", "Friday", "Saturday", "Sunday"],
    DAY_TEMPLATES, kind="cyclic",
)

MONTHS = ItemSet(
    "months",
    ["January", "February", "March", "April", "May", "June",
     "July", "August", "September", "October", "November", "December"],
    MONTH_TEMPLATES, kind="cyclic",
)

# Ordered but not cyclic: a good intermediate case. A line, not a ring, is expected.
NUMBERS = ItemSet(
    "number words",
    ["one", "two", "three", "four", "five", "six", "seven", "eight", "nine", "ten"],
    ["The number is {item}", "He counted to {item}", "She has {item} apples",
     "Add {item} to the total", "Chapter {item} begins now"],
    kind="ordered",
)

# Controls: arbitrary sets with no cyclic or ordinal structure at all.
ANIMALS = ItemSet(
    "animals (control)",
    ["salmon", "tiger", "sparrow", "beetle", "dolphin", "rabbit", "python"],
    GENERIC_TEMPLATES, kind="control",
)

OBJECTS = ItemSet(
    "objects (control)",
    ["hammer", "pillow", "lantern", "bucket", "ladder", "kettle", "mirror"],
    GENERIC_TEMPLATES, kind="control",
)

SETS = {s.name: s for s in (WEEKDAYS, MONTHS, NUMBERS, ANIMALS, OBJECTS)}
DEFAULT_ORDER = ["weekdays", "months", "number words", "animals (control)", "objects (control)"]
