import re
import unicodedata
from typing import Dict, List, Tuple, Set

try:
    from rapidfuzz import fuzz, process
    HAVE_RAPIDFUZZ = True
except Exception:
    HAVE_RAPIDFUZZ = False

tt: Dict[str, List[str]] = {
    'Sozialdemokratischen Partei Deutschlands': ['SPD'],
    'Christlich-Demokratische Union': ['CDU'],
    'Bündnis 90/Die Grünen': ['Bündnis 90/Die Grünen','Bündis 90/Die Grünen', 'Bündnis 90/ Die Grünen', 'Bündis 90/Die Grünen ', 'Bündnis 90 /Die Grünen'],
    'Alternative für Deutschland': ['AfD'],
    'Junges Duisburg': ['Junges Duisburg', 'JUDU'],
    'Duisburger Alternative Liste': ['DAL'],
    'Freie Demokratische Partei': ['FDP'],
    'Fraktion Die Linke./Die PARTEI': ['Die Linke./Die PARTEI', 'Die Linke./ Die PARTEI'],
    'Bündnis Sahra Wagenknecht': ['Bündnis Sahra Wagenknecht'],
    'Sozial, Gerecht, Unabhängig': ['SGU'],
    'Die Linke.': ['Die Linke.'],
    'parteilos': ['parteilos', 'Parteilos'],
    'Fraktion Tierschutz/DAL': ['Tierschutz/DAL'],
    'Solidarität für Duisburg': ['SfD', 'SFD'],
    'Tierschutz': ['Tierschutz'],
    'Wir gestalten Duisburg': ['WGD'],
    'Fraktion TR/DAL': ['TR/DAL', 'TR / DAL'],
    'Aktive Bürgerinitiative': ['ABI'],
    'Müslüman Türkler Birligi': ['MTB'],
    'Tükische Repräsentanz': ['TR'],
    'Bündinis für Innovations und Gerechtigkeit': ['BIG-DERGAH'],
    'AG-Handicap': ['AG-Handicap'],
    'AG-Handicap / VKM-Duisburg e.V.': ['AG-Handicap / VKM-Duisburg e.V.'],
    'Bürgerlich-Liberale': ['BL'],
    'Die PARTEI': ['Die PARTEI']
}

CANON_TAG = {
    'Sozialdemokratischen Partei Deutschlands': 'SPD',
    'Christlich-Demokratische Union': 'CDU',
    'Bündnis 90/Die Grünen': 'B90/GRÜNE',
    'Alternative für Deutschland': 'AfD',
    'Junges Duisburg': 'JUDU',
    'Duisburger Alternative Liste': 'DAL',
    'Freie Demokratische Partei': 'FDP',
    'Fraktion Die Linke./Die PARTEI': 'LINKE/PARTEI',
    'Bündnis Sahra Wagenknecht': 'BSW',
    'Sozial, Gerecht, Unabhängig': 'SGU',
    'Die Linke.': 'DIE LINKE',
    'parteilos': 'PARTEILOS',
    'Fraktion Tierschutz/DAL': 'Tierschutz/DAL',
    'Solidarität für Duisburg': 'SFD',
    'Tierschutz': 'Tierschutz',
    'Wir gestalten Duisburg': 'WGD',
    'Fraktion TR/DAL': 'TR/DAL',
    'Aktive Bürgerinitiative': 'ABI',
    'Müslüman Türkler Birligi': 'MTB',
    'Tükische Repräsentanz': 'TR',
    'Bündinis für Innovations und Gerechtigkeit': 'BIG-DERGAH',
    'AG-Handicap': 'AG-Handicap',
    'AG-Handicap / VKM-Duisburg e.V.': 'AG-Handicap/VKM',
    'Bürgerlich-Liberale': 'BL',
    'Die PARTEI': 'Die PARTEI',
}


def fold_diacritics(s: str) -> str:
    s = unicodedata.normalize("NFKD", s)
    return "".join(c for c in s if not unicodedata.combining(c))

def std_spaces_and_slashes(s: str) -> str:
    s = s.replace("\xa0", " ")
    s = unicodedata.normalize("NFKC", s)
    s = re.sub(r"\s+\/\s+", "/", s)  # " / " -> "/"
    s = re.sub(r"\s+", " ", s).strip()
    return s

def normalize_for_match(s: str) -> str:
    s = std_spaces_and_slashes(s).lower()
    s = fold_diacritics(s)
    # remove benign punctuation except slash (we keep / for combos like TR/DAL)
    s = re.sub(r"[.,;:()\[\]\"“”„‚’'–—-]", " ", s)
    s = re.sub(r"\s+", " ", s).strip()
    return s


def alias_to_regex(alias: str) -> str:
    """
    Turn an alias into a relaxed regex:
      - escape all regex metacharacters
      - make spaces flexible (\s+)
      - allow optional dot where a literal '.' appears in the alias
    """
    a = alias.strip()
    esc = re.escape(a)
    esc = esc.replace(r"\.", r"\.?")
    esc = re.sub(r"\s+", r"\\s+", esc)
    return esc

def build_compiled_patterns(tt: Dict[str, List[str]]) -> List[Tuple[str, re.Pattern, List[str]]]:
    """
    Return list of (canonical_key, compiled_regex_on_folded_text, raw_aliases)
    We match on a folded, normalized version of the input — so we also fold alias.
    """
    compiled = []
    for canon, aliases in tt.items():
        pats = []
        for alias in aliases:
            folded = normalize_for_match(alias)
            pats.append(alias_to_regex(folded))
        # also try to match the canonical key words themselves (useful if the long name appears)
        folded_key = normalize_for_match(canon)
        pats.append(alias_to_regex(folded_key))
        # allow optional 'fraktion' around/near the alias (common in text)
        # We'll match via \b(pat)(?:\s+fraktion)?\b OR \bfraktion\s+(pat)\b
        pat_union = "(?:" + "|".join(pats) + ")"
        pat = rf"\b(?:{pat_union})(?:\s+fraktion)?\b|\bfraktion\s+(?:{pat_union})\b"
        compiled.append((canon, re.compile(pat, flags=re.IGNORECASE), aliases))
    return compiled

COMPILED = build_compiled_patterns(tt)


def detect_parties(text: str, return_tags: bool = True, fuzzy: bool = True) -> Set[str]:
    """
    Returns a set of canonical tags (default) or canonical keys.
    - Matches using folded+normalized text and flexible regex.
    - Optionally uses RapidFuzz to catch strong near-misses.
    """
    folded_text = normalize_for_match(text)

    found_keys: Set[str] = set()
    for canon, pat, _aliases in COMPILED:
        if pat.search(folded_text):
            found_keys.add(canon)

    # Fuzzy fallback (optional)
    if fuzzy and HAVE_RAPIDFUZZ:
        # Build alias universe (folded)
        alias_to_key = {}
        for canon, _pat, aliases in COMPILED:
            for a in aliases + [canon]:
                alias_to_key[normalize_for_match(a)] = canon

        # If nothing found or to add more, try fuzzy search on the whole string
        # We split text into windows of tokens to avoid silly matches
        tokens = folded_text.split()
        candidates = set()
        for n in range(1, min(6, len(tokens)) + 1):  # n-grams up to 6 tokens
            for i in range(len(tokens) - n + 1):
                candidates.add(" ".join(tokens[i:i+n]))

        # Score each candidate against alias list
        choices = list(alias_to_key.keys())
        for cand in candidates:
            match, score, _ = process.extractOne(
                cand, choices, scorer=fuzz.WRatio
            )
            if score >= 92:  # conservative threshold
                found_keys.add(alias_to_key[match])

    if return_tags:
        # map canonical keys -> your chosen canonical short tag
        return {CANON_TAG.get(k, k) for k in found_keys}
    return found_keys


MOTION_TYPES = ["Prüfantrag", "Antrag", "Anfrage", "Dringlichkeitsantrag"]
def extract_motion_type(s: str):
    for t in MOTION_TYPES:
        if re.search(rf"\b{re.escape(t)}\b", s, re.IGNORECASE):
            return t
    return None

def extract_subject(s: str):
    m = re.search(r"hier:\s*(.*)$", s, flags=re.IGNORECASE)
    if m:
        return m.group(1).strip(" .—–-")
    parts = s.split(";")
    return parts[-1].strip() if len(parts) > 1 else s

