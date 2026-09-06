"""
Unrelated-text distractors for Stage 2.

Requirements these are written to meet:
- fluent, grammatical English
- clearly NOT a numeric answer to a math word problem
- unrelated to arithmetic and to typical GSM8K domains
- no digits and no counting words ("one", "second", "pair", ...) so that the
  number-overlap metric in analyze.py is not polluted
- neutral, non-directive content (nothing that reads like an instruction)

Mix of single sentences and short paragraphs so length is not a giveaway of
which option is the distractor.
"""

SENTENCES = [
    "The coastal fog usually burns off by mid-morning, leaving the afternoon clear and bright.",
    "Migratory storks return to the same rooftop nests each spring, repairing them with fresh twigs.",
    "The library's reading room smelled of old paper and floor polish, and the radiators ticked as they warmed.",
    "A bakery near the harbor is known for a dense rye loaf that keeps well for many days.",
    "During the festival, paper lanterns are strung between the balconies and left glowing until dawn.",
    "The museum reorganized its ceramics wing so that visitors move through it in rough chronological order.",
    "Cyclists on the ridge road often stop at the overlook to watch the valley fill with cloud after rain.",
    "The old ferry still runs on weekends, though most commuters now take the bridge.",
    "Her grandmother's recipe calls for letting the dough rest overnight in a cool pantry.",
    "The orchard hires extra pickers in autumn, many of whom return every season.",
    "Streetlights along the promenade switch to a warmer tone after midnight to reduce glare.",
    "The choir rehearses in the church hall on Thursdays, and the sound carries across the square.",
    "Herons have taken up residence along the reservoir's northern edge.",
    "The train slows near the tunnel so passengers can glimpse the mural painted on the embankment.",
    "Most of the shops on that street close for a long break in the early afternoon.",
    "The tailor keeps bolts of wool sorted by weight rather than color.",
    "After the renovation, the theater added a side staircase but kept the original brass railings.",
    "Rainwater from the courtyard drains into a cistern that feeds the building's garden taps.",
    (
        "The community garden began as a small plot behind the school and now covers most of the "
        "vacant lot. Volunteers keep a rotating schedule through the growing season, and surplus "
        "produce is left in a shaded crate by the gate for anyone to take."
    ),
    (
        "The lighthouse was automated long ago, but the keeper's cottage is still maintained as a "
        "small exhibit. Visitors can climb the spiral stair on clear days, when the platform at the "
        "top offers a view of the whole bay and the islands beyond."
    ),
]
