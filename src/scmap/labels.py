"""Harmonising the cell annotations of the two cohorts.

Both cohorts were annotated by the same group with the same two-level scheme
(`Cell_type` / `Cell_subtype`), but the fine labels drifted between releases. Evaluation is
only meaningful on labels that mean the same thing in both cohorts, so every query subtype
is assigned one of three roles:

* **shared**  – exists in the reference (possibly after renaming) and is scored normally;
* **novel**   – a real cell population that the reference never annotated; scored only as
  a novelty-detection target, never as a classification error;
* **excluded** – uninformative labels ("Unknown", unsubtyped umbrella labels).
"""

from __future__ import annotations

import pandas as pd

UNKNOWN = "Unknown"

# reference (SMC) renames: split subtypes that the query does not resolve are merged
REFERENCE_RENAME = {
    "Mature Enterocytes type 1": "Mature Enterocytes",
    "Mature Enterocytes type 2": "Mature Enterocytes",
}

# query (KUL3) renames onto reference vocabulary
QUERY_RENAME = {
    "SPP1+A": "SPP1+",
    "SPP1+B": "SPP1+",
}

# query subtypes absent from the reference annotation (natural open-set test)
QUERY_NOVEL = frozenset({"Anti-inflammatory", "BEST4+ Enterocytes", "Tuft cells"})

# query labels that cannot be scored at the fine level
QUERY_EXCLUDED = frozenset({UNKNOWN, "Epithelial cells", "Unspecified Plasma"})


def harmonise_reference(fine: pd.Series) -> pd.Series:
    return fine.replace(REFERENCE_RENAME)


def harmonise_query(fine: pd.Series) -> pd.Series:
    return fine.replace(QUERY_RENAME)


def query_role(fine_harmonised: pd.Series, reference_labels: set[str]) -> pd.Series:
    """Return 'shared', 'novel' or 'excluded' for every query cell."""

    def role(label: str) -> str:
        if label in QUERY_EXCLUDED:
            return "excluded"
        if label in QUERY_NOVEL:
            return "novel"
        if label in reference_labels:
            return "shared"
        raise ValueError(f"query label {label!r} has no role; update scmap.labels")

    return fine_harmonised.map(role)
