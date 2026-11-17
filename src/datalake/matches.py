"""The data lake: every external match, its source, label type and split.

Match IDs and split assignments for the scoring dataset.

    from datalake.matches import MATCHES, by_split
    by_split("test")        # the matches nobody trains or tunes on
"""

CONTRACT_VERSION = 2

# label_source: how the labels are made. See datalake/CONTRACT.md.
FLAGS, PROVIDER_MODEL = "flags", "provider_model"
# Possession from the provider's model, dead ball inferred from gaps between phases of play.
PROVIDER_PHASE_GAPS = "provider_model_phase_gaps"

TEST = {
    # DFL holdout with H / A / D provider flags.
    "dfl_J03WQQ", "dfl_J03WR9",
    # Spanish robustness check: the two 2019/20 Clasicos. H / A only, reported separately.
    "sc2019_2841", "sc2019_3442",
}


def _entry(key, source, native_id, label_source, labels, note=""):
    """One match record. `labels` is the letters its truth may contain besides X: "HAD", or "HA"
    when there is no dead-ball label. `split` is "test" for the keys in TEST, else "train"."""
    return {
        "key": key, "source": source, "native_id": native_id, "label_source": label_source,
        "labels": labels, "split": "test" if key in TEST else "train", "note": note,
    }


def _build():
    out = []
    for mid in ("J03WMX", "J03WN1", "J03WOH", "J03WOY", "J03WPY", "J03WQQ", "J03WR9"):
        out.append(_entry("dfl_" + mid, "dfl", mid, FLAGS, "HAD"))
    for mid in (2017461, 2016236, 2015213, 2013725, 2011166, 2010085, 2007721, 2007448, 2006363,
                2006229, 1996435, 1996436, 1986691, 1959846, 1953632, 1927964, 1925299, 1899585,
                1886347, 1874553):
        out.append(_entry("scal_{}".format(mid), "skillcorner_aleague", mid, PROVIDER_PHASE_GAPS, "HAD",
                          "D is inferred: frames outside every phase of play"))
    for mid in (4039, 3749, 3518, 3442, 2841, 2440, 2417, 2269, 2068):
        out.append(_entry("sc2019_{}".format(mid), "skillcorner_2019", mid, PROVIDER_MODEL, "HA",
                          "no dead-ball label; frames without a possessing team are X"))
    return out


MATCHES = _build()
BY_KEY = {m["key"]: m for m in MATCHES}

assert len(MATCHES) == 36 and len(BY_KEY) == 36
assert TEST <= set(BY_KEY), "a test key does not name a match"


def by_split(split):
    return [m for m in MATCHES if m["split"] == split]


def by_source(source):
    return [m for m in MATCHES if m["source"] == source]


# The shared development match, and the folds for a close call: leave one DFL training match out.
DEV_MATCH = "dfl_J03WMX"


def lomo_folds():
    """[(held-out key, [training keys])] over the five DFL training matches."""
    pool = [m["key"] for m in MATCHES if m["source"] == "dfl" and m["split"] == "train"]
    return [(held, [k for k in pool if k != held]) for held in pool]
