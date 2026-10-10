"""
Task 5: Question set -- 10 queries total: 6 factual (F), 2 boundary (B),
2 multi-hop/follow-up (M). Organized into conversational groups; memory
resets between groups. Each run (config) executes all groups in sequence.
"""

GROUPS = [
    {
        "label": "group_1_keyword_research",
        "turns": [
            {"q": "What are long-tail keywords?", "type": "F"},
            {"q": "Why are they especially useful for freelancers?", "type": "M"},
            {"q": "What tool could I use for keyword research?", "type": "F"},
        ],
    },
    {
        "label": "group_2_onpage_technical",
        "turns": [
            {"q": "What should a title tag include?", "type": "F"},
            {"q": "What is the recommended meta description length?", "type": "F"},
            {"q": "What do header tags help organize on a page?", "type": "F"},
            {"q": "Who is the current CEO of Google?", "type": "B"},
        ],
    },
    {
        "label": "group_3_local_seo",
        "turns": [
            {"q": "What is the most important local SEO asset for a small business?", "type": "F"},
            {"q": "How should a freelancer handle reviews on that profile?", "type": "M"},
            {"q": "What is the exact ranking formula Google's algorithm uses?", "type": "B"},
        ],
    },
]

ALL_QUESTIONS = [t for g in GROUPS for t in g["turns"]]
assert len(ALL_QUESTIONS) == 10
assert sum(1 for t in ALL_QUESTIONS if t["type"] == "F") == 6
assert sum(1 for t in ALL_QUESTIONS if t["type"] == "B") == 2
assert sum(1 for t in ALL_QUESTIONS if t["type"] == "M") == 2
