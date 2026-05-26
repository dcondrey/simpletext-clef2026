"""
SimpleText Task 1 regex-only: Deterministic rule-based simplification.

No model needed. Applies exactly the transformations that Cochrane PLS references
make, maximizing SARI by targeting delete_score and add_score without hurting
keep_score.

Transformations (derived from 7,082 training pairs):
1. Remove statistical parentheticals (CI, RR, OR, HR, MD, SMD, I², IQR, P-values)
2. Replace "participants" → "people" (179 confirmed in training)
3. Replace "adverse events/effects" → "side effects" (84 confirmed)
4. Replace "mortality" → "death(s)" (106 confirmed)
5. Remove "(RCTs)" / "(RCT)" after study type mentions
6. Clean up residual formatting artifacts
"""

import json
import logging
import re
import sys

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
log = logging.getLogger(__name__)

TEAM_ID = "writerslogic"


def simplify_sentence(text: str, language: str = "en") -> str:
    """Apply rule-based simplifications to a single sentence."""
    if language != "en":
        return text

    out = text

    # 1. Remove statistical parentheticals
    # Pattern: (RR 0.76; 95% CI 0.64 to 0.91, P = 0.002; I² = 0%)
    # Pattern: (risk ratio (RR) 0.20, 95% confidence interval (CI) 0.12 to 0.32; ...)
    # Pattern: (mean difference (MD) -0.15 kg, 95% CI -0.55 to 0.24; P = 0.45, I² = 46%)
    # Pattern: (interquartile range (IQR): 0.8% to 18.8%)
    # Pattern: (odds ratio (OR) 1.23, 95% CI ...)

    # Remove nested stat parentheticals: (RR ...; 95% CI ...; I² = ...)
    # This handles most statistical measure patterns
    stat_markers = (
        r'risk\s+ratio\s*\(RR\)',
        r'odds\s+ratio\s*\(OR\)',
        r'hazard\s+ratio\s*\(HR\)',
        r'mean\s+difference\s*\(MD\)',
        r'standardised\s+mean\s+difference\s*\(SMD\)',
        r'standardized\s+mean\s+difference\s*\(SMD\)',
        r'confidence\s+interval\s*\(CI\)',
        r'interquartile\s+range\s*\(IQR\)',
        r'\bRR\s+\d',
        r'\bOR\s+\d',
        r'\bHR\s+\d',
        r'\bMD\s+[-\d]',
        r'\bSMD\s+[-\d]',
        r'95%\s*CI\b',
        r'I²\s*=',
        r'I2\s*=',
        r'\bP\s*[=<>]\s*0\.',
        r'\bp\s*[=<>]\s*0\.',
        r'\bIQR\b',
    )

    # Strategy: find parenthetical groups that contain stat markers and remove them
    # But preserve parentheticals that contain useful info (study counts, participant counts)
    def should_remove_paren(content: str) -> bool:
        """Decide if a parenthetical should be removed."""
        for marker in stat_markers:
            if re.search(marker, content):
                return True
        return False

    def strip_stat_content(content: str) -> str:
        """From a parenthetical that has stats mixed with useful info,
        try to keep useful parts like study counts."""
        # Extract useful fragments: "N studies", "N participants/people"
        useful = []
        parts = re.split(r';\s*', content)
        for part in parts:
            part = part.strip()
            # Keep parts with study/participant counts
            if re.search(r'\d+\s+(?:stud|trial|participant|people|RCT)', part, re.IGNORECASE):
                # But remove stat prefixes within this part
                cleaned = re.sub(r'^.*?(\d+\s+(?:stud|trial|participant|people|RCT))', r'\1', part, flags=re.IGNORECASE)
                useful.append(cleaned)
            # Keep certainty assessments
            elif re.search(r'(?:low|moderate|high|very\s+low)[\s-]*certainty', part, re.IGNORECASE):
                useful.append(part)
        return '; '.join(useful) if useful else ''

    # Process parenthetical expressions
    def process_parens(text: str) -> str:
        result = []
        i = 0
        while i < len(text):
            if text[i] == '(':
                # Find matching close paren
                depth = 1
                j = i + 1
                while j < len(text) and depth > 0:
                    if text[j] == '(':
                        depth += 1
                    elif text[j] == ')':
                        depth -= 1
                    j += 1
                paren_content = text[i+1:j-1]

                if should_remove_paren(paren_content):
                    kept = strip_stat_content(paren_content)
                    if kept:
                        result.append(f'({kept})')
                    # else: drop the entire parenthetical
                else:
                    result.append(text[i:j])
                i = j
            else:
                result.append(text[i])
                i += 1
        return ''.join(result)

    out = process_parens(out)

    # 2. Word replacements (case-preserving)
    def replace_word(text, old, new):
        """Replace whole-word, case-aware."""
        def repl(m):
            orig = m.group(0)
            if orig[0].isupper():
                return new[0].upper() + new[1:]
            return new
        return re.sub(r'\b' + re.escape(old) + r'\b', repl, text)

    # "participants" → "people"
    out = replace_word(out, 'participants', 'people')
    out = replace_word(out, 'Participants', 'People')

    # "adverse events" → "side effects"
    out = re.sub(r'\badverse\s+events?\b', 'side effects', out, flags=re.IGNORECASE)
    # "adverse effects" → "side effects"
    out = re.sub(r'\badverse\s+effects?\b', 'side effects', out, flags=re.IGNORECASE)

    # "mortality" → "death" (context-sensitive)
    out = re.sub(r'\ball-cause\s+mortality\b', 'death from any cause', out, flags=re.IGNORECASE)
    out = replace_word(out, 'mortality', 'death')

    # Remove "(RCTs)" or "(RCT)" standalone
    out = re.sub(r'\s*\(RCTs?\)', '', out)

    # "randomised controlled trials (RCTs)" → "randomised controlled trials"
    # Already handled by the above

    # 3. Clean up formatting artifacts
    # Double spaces
    out = re.sub(r'  +', ' ', out)
    # Space before punctuation
    out = re.sub(r'\s+([,;.])', r'\1', out)
    # Empty parentheses
    out = re.sub(r'\(\s*\)', '', out)
    # Trailing/leading whitespace
    out = out.strip()
    # Comma before closing paren with nothing after
    out = re.sub(r',\s*\)', ')', out)
    # Semicolon at start of parenthetical
    out = re.sub(r'\(\s*;\s*', '(', out)
    # Double semicolons
    out = re.sub(r';\s*;', ';', out)

    return out


def run_task(input_path, output_path, task):
    with open(input_path) as f:
        data = json.load(f)

    run_id = f"{TEAM_ID}_Task{task.replace('.', '')}_regex"
    log.info("Task %s: %d items, run_id=%s", task, len(data), run_id)

    results = []
    copies = 0
    for item in data:
        lang = item.get("language", "en")
        if task == "1.1":
            simplified = simplify_sentence(item["complex"], lang)
        else:
            # Document-level: apply sentence-level to each sentence in the abstract
            simplified = simplify_sentence(item["complex"], lang)

        if simplified.strip() == item["complex"].strip():
            copies += 1

        results.append({**item, "prediction": simplified, "run_id": run_id})

    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(results, f, ensure_ascii=False)

    changed = len(data) - copies
    log.info("Done: %d items, %d changed (%.1f%%), %d copies (%.1f%%)",
             len(data), changed, changed/len(data)*100, copies, copies/len(data)*100)

    # Compression stats
    ratios = []
    for item in results:
        cw = len(item["complex"].split())
        pw = len(item["prediction"].split())
        if cw > 0:
            ratios.append(pw / cw)
    import numpy as np
    r = np.array(ratios)
    log.info("Compression: mean=%.3f, median=%.3f", r.mean(), np.median(r))


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--task", choices=["1.1", "1.2"], required=True)
    parser.add_argument("--input", required=True)
    parser.add_argument("--output", required=True)
    args = parser.parse_args()

    run_task(args.input, args.output, args.task)
