from __future__ import annotations

from collections import Counter, defaultdict
from dataclasses import dataclass
import itertools
import math
import re
from typing import Dict, Iterable, List, Optional, Sequence

import pandas as pd


FAMILY_PATTERNS = {
    "bit": "secret bit manipulation rule transforms 8-bit binary numbers",
    "gravity": "the gravitational constant has been secretly changed",
    "unit": "a secret unit conversion is applied to measurements",
    "cipher": "secret encryption rules are used on text",
    "roman": "numbers are secretly converted into a different numeral system",
    "equation": "a secret set of transformation rules is applied to equations",
}

PAIR_RE = re.compile(r"^(.*?) -> (.*?)$", re.M)
EQUATION_PAIR_RE = re.compile(r"^(.+?) = (.+?)$", re.M)
CIPHER_TARGET_RE = re.compile(r"Now, decrypt the following text: (.*)$", re.S)
ROMAN_TARGET_RE = re.compile(r"Now, write the number (\d+) in the Wonderland numeral system\.", re.S)
GRAVITY_EXAMPLE_RE = re.compile(r"For t = ([0-9.]+)s, distance = ([0-9.]+) m")
GRAVITY_TARGET_RE = re.compile(r"Now, determine the falling distance for t = ([0-9.]+)s", re.S)
UNIT_EXAMPLE_RE = re.compile(r"([0-9.]+)\s*([a-zA-Z]+)\s*becomes\s*([0-9.]+)")
UNIT_TARGET_RE = re.compile(r"Now, convert the following measurement:\s*([0-9.]+)\s*([a-zA-Z]+)", re.S)
BIT_PAIR_RE = re.compile(r"([01]{8}) -> ([01]{8})")
BIT_TARGET_RE = re.compile(r"Now, determine the output for: ([01]{8})")
EQUATION_TARGET_RE = re.compile(r"Now, determine the result for: (.*)$", re.S)
NUMERIC_EQUATION_RE = re.compile(r"^(\d\d)(.)(\d\d)$")
BIT_MASK = 0xFF
SYMBOLIC_OTHER_INDEX = {0: 1, 1: 0, 3: 4, 4: 3}
SYMBOLIC_OPERATOR_CHARS = set("+-*/\\[]{}()@#$%^&|!?<>\":;`'")
SYMBOLIC_ATOMS: tuple[tuple[str, object], ...] = tuple([("slot", index) for index in range(5)] + [
    ("other", 0),
    ("other", 1),
    ("other", 3),
    ("other", 4),
    ("shared_first", None),
    ("shared_last", None),
    ("left_only_first", None),
    ("left_only_last", None),
    ("right_only_first", None),
    ("right_only_last", None),
    ("role", 0),
    ("role", 1),
])
MAX_SYMBOLIC_PROGRAM_LENGTH = 4
MAX_SYMBOLIC_POSITION_CANDIDATES = 12
MAX_SYMBOLIC_EXACT_PROGRAMS = 256
MAX_SYMBOLIC_CANDIDATE_PRODUCT = 5000


def detect_family(prompt: str) -> str:
    prompt_lower = prompt.lower()
    for family, needle in FAMILY_PATTERNS.items():
        if needle in prompt_lower:
            return family
    raise ValueError(f"Unknown prompt family: {prompt[:120]!r}")


def to_roman(number: int) -> str:
    numerals = [
        (1000, "M"),
        (900, "CM"),
        (500, "D"),
        (400, "CD"),
        (100, "C"),
        (90, "XC"),
        (50, "L"),
        (40, "XL"),
        (10, "X"),
        (9, "IX"),
        (5, "V"),
        (4, "IV"),
        (1, "I"),
    ]
    out: List[str] = []
    remaining = number
    for value, token in numerals:
        while remaining >= value:
            out.append(token)
            remaining -= value
    return "".join(out)


def word_pattern(word: str) -> tuple[int, ...]:
    token_ids: Dict[str, int] = {}
    out: List[int] = []
    next_id = 0
    for char in word:
        if char not in token_ids:
            token_ids[char] = next_id
            next_id += 1
        out.append(token_ids[char])
    return tuple(out)


def sentence_pattern(text: str) -> tuple[tuple[int, ...], ...]:
    return tuple(word_pattern(word) for word in text.split())


def intersect_parameter_intervals(intervals: Sequence[tuple[float, float]]) -> Optional[tuple[float, float]]:
    low = max(a for a, _ in intervals)
    high = min(b for _, b in intervals)
    if low <= high:
        return (low, high)
    return None


def rounded_candidates(interval: tuple[float, float]) -> List[str]:
    low, high = interval
    start = int((low - 0.02) * 100)
    end = int((high + 0.02) * 100) + 1
    candidates: List[str] = []
    for cent in range(start, end + 1):
        value = cent / 100.0
        bucket_low = value - 0.005
        bucket_high = value + 0.005
        if max(low, bucket_low) <= min(high, bucket_high):
            candidates.append(f"{value:.2f}")
    return sorted(set(candidates))


def format_decimal(value: float) -> str:
    rendered = f"{value:.2f}"
    if "." in rendered:
        rendered = rendered.rstrip("0").rstrip(".")
    return rendered


def least_squares_scale(xs: Sequence[float], ys: Sequence[float]) -> float:
    denominator = sum(x * x for x in xs)
    if denominator == 0:
        return 0.0
    return sum(x * y for x, y in zip(xs, ys)) / denominator


def rol8(value: int, shift: int) -> int:
    return ((value << shift) & BIT_MASK) | (value >> (8 - shift))


def ror8(value: int, shift: int) -> int:
    return (value >> shift) | ((value << (8 - shift)) & BIT_MASK)


def shl8(value: int, shift: int) -> int:
    return (value << shift) & BIT_MASK


def shr8(value: int, shift: int) -> int:
    return value >> shift


def popcount8(value: int) -> int:
    return bin(value & BIT_MASK).count("1")


def bit_term_score(values: Sequence[int], expected: Sequence[int]) -> int:
    return sum(8 - popcount8(observed ^ gold) for observed, gold in zip(values[:-1], expected))


@dataclass
class CipherResources:
    vocabulary: Dict[int, List[str]]
    unigram_counts: Counter[str]
    bigram_counts: Counter[tuple[str, str]]
    trigram_counts: Counter[tuple[str, str, str]]


@dataclass
class EquationResources:
    numeric_priors: Counter[str]
    numeric_op_priors: Dict[str, Counter[str]]
    symbolic_priors: Counter[tuple[int, ...]]
    symbolic_program_priors: Dict[str, Counter[tuple[tuple[str, object], ...]]]
    symbolic_global_program_priors: Counter[tuple[tuple[str, object], ...]]


@dataclass
class SymbolicRetrievalEntry:
    target_op: str
    prompt: str
    answer: str
    gram_counts: Counter[str]


class NemotronReasoningSolver:
    def __init__(self) -> None:
        self.cipher_resources = CipherResources(
            vocabulary=defaultdict(list),
            unigram_counts=Counter(),
            bigram_counts=Counter(),
            trigram_counts=Counter(),
        )
        self.equation_resources = EquationResources(
            numeric_priors=Counter(),
            numeric_op_priors={},
            symbolic_priors=Counter(),
            symbolic_program_priors={},
            symbolic_global_program_priors=Counter(),
        )
        self.symbolic_retrieval_entries: List[SymbolicRetrievalEntry] = []

    def fit(self, train_df: pd.DataFrame) -> "NemotronReasoningSolver":
        cipher_plain_sentences: List[List[str]] = []
        cipher_rows = train_df[train_df["prompt"].map(detect_family) == "cipher"]
        for _, row in cipher_rows.iterrows():
            for _, plain in PAIR_RE.findall(row["prompt"]):
                words = plain.split()
                cipher_plain_sentences.append(words)
            answer_words = str(row["answer"]).split()
            cipher_plain_sentences.append(answer_words)

        vocab = defaultdict(list)
        unigram_counts: Counter[str] = Counter()
        bigram_counts: Counter[tuple[str, str]] = Counter()
        trigram_counts: Counter[tuple[str, str, str]] = Counter()
        seen_words = set()

        for words in cipher_plain_sentences:
            unigram_counts.update(words)
            bigram_counts.update(zip(words, words[1:]))
            trigram_counts.update(zip(words, words[1:], words[2:]))
            for word in words:
                if word not in seen_words:
                    seen_words.add(word)
                    vocab[len(word)].append(word)

        for key in vocab:
            vocab[key].sort(key=lambda w: (-unigram_counts[w], w))

        self.cipher_resources = CipherResources(
            vocabulary=vocab,
            unigram_counts=unigram_counts,
            bigram_counts=bigram_counts,
            trigram_counts=trigram_counts,
        )
        self.fit_equation_resources(train_df)
        return self

    def fit_equation_resources(self, train_df: pd.DataFrame) -> None:
        numeric_priors: Counter[str] = Counter()
        numeric_op_priors: defaultdict[str, Counter[str]] = defaultdict(Counter)
        symbolic_priors: Counter[tuple[int, ...]] = Counter()
        symbolic_program_priors: defaultdict[str, Counter[tuple[tuple[str, object], ...]]] = defaultdict(Counter)
        symbolic_retrieval_entries: List[SymbolicRetrievalEntry] = []
        equation_rows = train_df[train_df["prompt"].map(detect_family) == "equation"]
        for _, row in equation_rows.iterrows():
            equation_kind = self.detect_equation_kind(row["prompt"])
            if equation_kind == "numeric":
                info = self.parse_numeric_equation_prompt(row["prompt"])
                target_op = info["target_op"]
                candidates = self.numeric_equation_candidates_for_op(info["examples"], target_op)
                target_a, target_b = info["target_numbers"]
                target_matches = {
                    name
                    for name, value in self.numeric_equation_operations(target_a, target_b).items()
                    if value == str(row["answer"])
                }
                for name in candidates & target_matches:
                    numeric_priors[name] += 1
                if target_matches:
                    weight = 1.0 / len(target_matches)
                    for name in target_matches:
                        numeric_op_priors[target_op][name] += weight
            else:
                target = self.parse_equation_target(row["prompt"])
                target_op = target[2]
                examples = [(lhs, rhs) for lhs, rhs in self.parse_equation_pairs(row["prompt"]) if lhs[2] == target_op]
                symbolic_retrieval_entries.append(
                    SymbolicRetrievalEntry(
                        target_op=target_op,
                        prompt=row["prompt"],
                        answer=str(row["answer"]),
                        gram_counts=self.char_wb_ngram_counts(self.canonicalize_symbolic_prompt(row["prompt"])),
                    )
                )
                templates = self.symbolic_subsequence_templates(examples)
                for template in templates:
                    if self.apply_symbolic_template(target, template) == str(row["answer"]):
                        symbolic_priors[template] += 1
                answer = str(row["answer"])
                if examples and len(answer) <= MAX_SYMBOLIC_PROGRAM_LENGTH:
                    matches: List[tuple[tuple[str, object], ...]] = []
                    for program in self.enumerate_exact_symbolic_programs(examples):
                        if self.fit_symbolic_program(program, examples, target) == answer:
                            matches.append(program)
                    if matches:
                        weight = 1.0 / len(matches)
                        for program in matches:
                            symbolic_program_priors[target_op][program] += weight

        self.equation_resources = EquationResources(
            numeric_priors=numeric_priors,
            numeric_op_priors=dict(numeric_op_priors),
            symbolic_priors=symbolic_priors,
            symbolic_program_priors=dict(symbolic_program_priors),
            symbolic_global_program_priors=sum(symbolic_program_priors.values(), Counter()),
        )
        self.symbolic_retrieval_entries = symbolic_retrieval_entries

    def solve(self, prompt: str) -> Optional[str]:
        family = detect_family(prompt)
        if family == "roman":
            return self.solve_roman(prompt)
        if family == "gravity":
            return self.solve_gravity(prompt)
        if family == "unit":
            return self.solve_unit(prompt)
        if family == "cipher":
            return self.solve_cipher(prompt)
        if family == "bit":
            return self.solve_bit(prompt)
        if family == "equation":
            return self.solve_equation(prompt)
        return None

    def solve_roman(self, prompt: str) -> str:
        match = ROMAN_TARGET_RE.search(prompt)
        if not match:
            raise ValueError("Could not parse Roman target")
        return to_roman(int(match.group(1)))

    def solve_gravity(self, prompt: str) -> str:
        examples = [(float(t), float(d)) for t, d in GRAVITY_EXAMPLE_RE.findall(prompt)]
        target_match = GRAVITY_TARGET_RE.search(prompt)
        if not examples or not target_match:
            raise ValueError("Could not parse gravity prompt")
        target_t = float(target_match.group(1))
        xs = [0.5 * t * t for t, _ in examples]
        ys = [d for _, d in examples]
        g_est = least_squares_scale(xs, ys)
        g_intervals = []
        for t, d in examples:
            lower = 2.0 * (d - 0.005) / (t * t)
            upper = 2.0 * (d + 0.005) / (t * t)
            g_intervals.append((lower, upper))
        overlap = intersect_parameter_intervals(g_intervals)
        target_x = 0.5 * target_t * target_t
        if overlap is None:
            return format_decimal(target_x * g_est)

        distance_interval = (target_x * overlap[0], target_x * overlap[1])
        candidates = rounded_candidates(distance_interval)
        if len(candidates) == 1:
            return candidates[0]
        if candidates:
            best = min((float(candidate) for candidate in candidates), key=lambda value: abs(value - (target_x * g_est)))
            return format_decimal(best)
        return format_decimal(target_x * g_est)

    def solve_unit(self, prompt: str) -> str:
        examples = [(float(src), float(dst)) for src, _, dst in UNIT_EXAMPLE_RE.findall(prompt)]
        target_match = UNIT_TARGET_RE.search(prompt)
        if not examples or not target_match:
            raise ValueError("Could not parse unit prompt")
        value = float(target_match.group(1))
        xs = [src for src, _ in examples if src != 0]
        ys = [dst for src, dst in examples if src != 0]
        ratio_est = least_squares_scale(xs, ys)
        ratio_intervals = []
        for src, dst in examples:
            if src == 0:
                continue
            ratio_intervals.append(((dst - 0.005) / src, (dst + 0.005) / src))
        overlap = intersect_parameter_intervals(ratio_intervals)
        if overlap is None:
            return format_decimal(value * ratio_est)

        converted_interval = (value * overlap[0], value * overlap[1])
        candidates = rounded_candidates(converted_interval)
        if len(candidates) == 1:
            return candidates[0]
        if candidates:
            best = min((float(candidate) for candidate in candidates), key=lambda candidate: abs(candidate - (value * ratio_est)))
            return format_decimal(best)
        return format_decimal(value * ratio_est)

    def solve_cipher(self, prompt: str) -> Optional[str]:
        pairs = PAIR_RE.findall(prompt)
        target_match = CIPHER_TARGET_RE.search(prompt)
        if not pairs or not target_match:
            raise ValueError("Could not parse cipher prompt")
        target = target_match.group(1).strip()
        target_words = target.split()

        cipher_to_plain: Dict[str, str] = {}
        plain_to_cipher: Dict[str, str] = {}
        fixed_word_map: Dict[str, str] = {}

        for cipher_sent, plain_sent in pairs:
            c_words = cipher_sent.split()
            p_words = plain_sent.split()
            for c_word, p_word in zip(c_words, p_words):
                fixed_word_map.setdefault(c_word, p_word)
                for c_char, p_char in zip(c_word, p_word):
                    existing = cipher_to_plain.get(c_char)
                    if existing is not None and existing != p_char:
                        return None
                    reverse = plain_to_cipher.get(p_char)
                    if reverse is not None and reverse != c_char:
                        return None
                    cipher_to_plain[c_char] = p_char
                    plain_to_cipher[p_char] = c_char

        variables = []
        assignment: Dict[str, str] = {}
        for word in target_words:
            if word in fixed_word_map:
                assignment[word] = fixed_word_map[word]
            elif word not in assignment:
                variables.append(word)

        candidates: Dict[str, List[str]] = {}
        for word in variables:
            options = self.candidate_plain_words(word, cipher_to_plain, plain_to_cipher)
            if not options:
                return None
            candidates[word] = options

        variables.sort(key=lambda word: len(candidates[word]))

        best_solution: Optional[Dict[str, str]] = None
        best_score = -10**18

        def score_sequence(words: Sequence[str]) -> int:
            score = 0
            for word in words:
                score += self.cipher_resources.unigram_counts[word]
            for bigram in zip(words, words[1:]):
                score += 4 * self.cipher_resources.bigram_counts[bigram]
            for trigram in zip(words, words[1:], words[2:]):
                score += 8 * self.cipher_resources.trigram_counts[trigram]
            return score

        def backtrack(
            index: int,
            local_assignment: Dict[str, str],
            local_c2p: Dict[str, str],
            local_p2c: Dict[str, str],
        ) -> None:
            nonlocal best_solution, best_score
            if index == len(variables):
                decoded = [local_assignment[word] for word in target_words]
                current_score = score_sequence(decoded)
                if current_score > best_score:
                    best_score = current_score
                    best_solution = dict(local_assignment)
                return

            cipher_word = variables[index]
            for candidate in candidates[cipher_word]:
                new_c2p = dict(local_c2p)
                new_p2c = dict(local_p2c)
                ok = True
                for c_char, p_char in zip(cipher_word, candidate):
                    existing = new_c2p.get(c_char)
                    if existing is not None and existing != p_char:
                        ok = False
                        break
                    reverse = new_p2c.get(p_char)
                    if reverse is not None and reverse != c_char:
                        ok = False
                        break
                    new_c2p[c_char] = p_char
                    new_p2c[p_char] = c_char
                if not ok:
                    continue
                local_assignment[cipher_word] = candidate
                backtrack(index + 1, local_assignment, new_c2p, new_p2c)
                del local_assignment[cipher_word]

        backtrack(0, dict(assignment), dict(cipher_to_plain), dict(plain_to_cipher))
        if best_solution is None:
            return None
        return " ".join(best_solution[word] for word in target_words)

    def solve_bit(self, prompt: str) -> Optional[str]:
        pairs = [(int(src, 2), int(dst, 2)) for src, dst in BIT_PAIR_RE.findall(prompt)]
        target_match = BIT_TARGET_RE.search(prompt)
        if not pairs or not target_match:
            raise ValueError("Could not parse bit prompt")

        target = int(target_match.group(1), 2)
        xs = [src for src, _ in pairs] + [target]
        expected = tuple(dst for _, dst in pairs)
        terms: Dict[tuple[int, ...], tuple[str, tuple[int, ...], int]] = {}
        shallow_terms: Dict[tuple[int, ...], tuple[str, tuple[int, ...]]] = {}
        fixed_basis_terms: List[tuple[str, tuple[int, ...]]] = []
        fixed_basis_names = {
            "x",
            "~x",
            "rol1(x)",
            "ror1(x)",
            "shl1(x)",
            "shr1(x)",
            "rol2(x)",
            "ror2(x)",
            "shl2(x)",
            "shr2(x)",
            "rol3(x)",
            "ror3(x)",
        }

        def add(name: str, values: Sequence[int]) -> Optional[int]:
            value_tuple = tuple(value & BIT_MASK for value in values)
            signature = value_tuple[:-1]
            score = bit_term_score(value_tuple, expected)
            existing = terms.get(signature)
            if existing is None or score > existing[2] or (score == existing[2] and len(name) < len(existing[0])):
                terms[signature] = (name, value_tuple, score)
            if signature == expected:
                return value_tuple[-1]
            return None

        def add_shallow(name: str, values: Sequence[int]) -> Optional[int]:
            value_tuple = tuple(value & BIT_MASK for value in values)
            signature = value_tuple[:-1]
            if signature not in shallow_terms:
                shallow_terms[signature] = (name, value_tuple)
            if signature == expected:
                return value_tuple[-1]
            return None

        base_builders = [
            ("x", lambda x: x),
            ("~x", lambda x: x ^ BIT_MASK),
            ("0", lambda _: 0),
            ("255", lambda _: BIT_MASK),
        ]
        for shift in range(1, 8):
            base_builders.extend(
                [
                    (f"rol{shift}(x)", lambda x, shift=shift: rol8(x, shift)),
                    (f"ror{shift}(x)", lambda x, shift=shift: ror8(x, shift)),
                    (f"shl{shift}(x)", lambda x, shift=shift: shl8(x, shift)),
                    (f"shr{shift}(x)", lambda x, shift=shift: shr8(x, shift)),
                ]
            )

        for name, fn in base_builders:
            values = [fn(x) for x in xs]
            if name in fixed_basis_names:
                fixed_basis_terms.append((name, tuple(value & BIT_MASK for value in values)))
            match = add(name, values)
            add_shallow(name, values)
            if match is not None:
                return f"{match:08b}"

        current_shallow_terms = list(shallow_terms.values())
        for name, values in current_shallow_terms:
            match = add_shallow(f"~({name})", [value ^ BIT_MASK for value in values])
            if match is not None:
                return f"{match:08b}"

        beam_width = 64
        ternary_width = 8
        search_rounds = 4

        for _ in range(search_rounds):
            ranked_terms = sorted(
                terms.values(),
                key=lambda item: (-item[2], len(item[0]), item[0]),
            )[:beam_width]
            if not ranked_terms:
                break

            for name, values, _ in ranked_terms:
                match = add(f"~({name})", [value ^ BIT_MASK for value in values])
                if match is not None:
                    return f"{match:08b}"

            for name_a, values_a, _ in ranked_terms:
                for name_b, values_b, _ in ranked_terms:
                    binary_candidates = [
                        (f"({name_a}^{name_b})", [a ^ b for a, b in zip(values_a, values_b)]),
                        (f"({name_a}&{name_b})", [a & b for a, b in zip(values_a, values_b)]),
                        (f"({name_a}|{name_b})", [a | b for a, b in zip(values_a, values_b)]),
                    ]
                    for name, values in binary_candidates:
                        match = add(name, values)
                        if match is not None:
                            return f"{match:08b}"

            top_terms = ranked_terms[:ternary_width]
            for name_a, values_a, _ in top_terms:
                for name_b, values_b, _ in top_terms:
                    for name_c, values_c, _ in top_terms:
                        majority_values = [
                            (a & b) | (a & c) | (b & c)
                            for a, b, c in zip(values_a, values_b, values_c)
                        ]
                        match = add(f"maj({name_a},{name_b},{name_c})", majority_values)
                        if match is not None:
                            return f"{match:08b}"
                        choice_values = [
                            (a & b) | ((a ^ BIT_MASK) & c)
                            for a, b, c in zip(values_a, values_b, values_c)
                        ]
                        match = add(f"ch({name_a},{name_b},{name_c})", choice_values)
                        if match is not None:
                            return f"{match:08b}"

        shallow_all_terms = list(shallow_terms.values())
        for index, (_, values_a) in enumerate(shallow_all_terms):
            for _, values_b in shallow_all_terms[index:]:
                binary_candidates = [
                    [a ^ b for a, b in zip(values_a, values_b)],
                    [a & b for a, b in zip(values_a, values_b)],
                    [a | b for a, b in zip(values_a, values_b)],
                ]
                for values in binary_candidates:
                    value_tuple = tuple(value & BIT_MASK for value in values)
                    if value_tuple[:-1] == expected:
                        return f"{value_tuple[-1]:08b}"

        for _, values_a in fixed_basis_terms:
            for _, values_b in fixed_basis_terms:
                for _, values_c in fixed_basis_terms:
                    majority_values = [
                        (a & b) | (a & c) | (b & c)
                        for a, b, c in zip(values_a, values_b, values_c)
                    ]
                    if tuple(majority_values[:-1]) == expected:
                        return f"{majority_values[-1]:08b}"
                    choice_values = [
                        (a & b) | ((a ^ BIT_MASK) & c)
                        for a, b, c in zip(values_a, values_b, values_c)
                    ]
                    if tuple(choice_values[:-1]) == expected:
                        return f"{choice_values[-1]:08b}"

        return None

    def solve_equation(self, prompt: str) -> Optional[str]:
        if self.detect_equation_kind(prompt) == "numeric":
            return self.solve_numeric_equation(prompt)
        return self.solve_symbolic_equation(prompt)

    def detect_equation_kind(self, prompt: str) -> str:
        pairs = self.parse_equation_pairs(prompt)
        if all(NUMERIC_EQUATION_RE.match(lhs) for lhs, _ in pairs):
            return "numeric"
        return "symbolic"

    def parse_equation_pairs(self, prompt: str) -> List[tuple[str, str]]:
        return [(lhs.strip(), rhs.strip()) for lhs, rhs in EQUATION_PAIR_RE.findall(prompt)]

    def parse_equation_target(self, prompt: str) -> str:
        match = EQUATION_TARGET_RE.search(prompt)
        if not match:
            raise ValueError("Could not parse equation target")
        return match.group(1).strip()

    def parse_numeric_equation_prompt(self, prompt: str) -> Dict[str, object]:
        pairs = self.parse_equation_pairs(prompt)
        examples = []
        for lhs, rhs in pairs:
            match = NUMERIC_EQUATION_RE.match(lhs)
            if not match:
                raise ValueError("Expected numeric equation prompt")
            examples.append(
                {
                    "lhs": lhs,
                    "rhs": rhs,
                    "a": int(match.group(1)),
                    "op": match.group(2),
                    "b": int(match.group(3)),
                }
            )

        target = self.parse_equation_target(prompt)
        target_match = NUMERIC_EQUATION_RE.match(target)
        if not target_match:
            raise ValueError("Expected numeric equation target")
        return {
            "examples": examples,
            "target_op": target_match.group(2),
            "target_numbers": (int(target_match.group(1)), int(target_match.group(3))),
        }

    def numeric_equation_operations(self, a: int, b: int) -> Dict[str, str]:
        a_tens, a_ones = divmod(a, 10)
        b_tens, b_ones = divmod(b, 10)
        reverse_a = int(f"{a:02d}"[::-1])
        reverse_b = int(f"{b:02d}"[::-1])
        outputs: Dict[str, str] = {}

        def add(name: str, value: object) -> None:
            outputs[name] = str(value)

        add("a+b", a + b)
        add("a-b", a - b)
        add("b-a", b - a)
        add("abs_diff_numbers", abs(a - b))
        add("a*b", a * b)
        if b != 0:
            add("a//b", a // b)
            add("a%b", a % b)
        if a != 0:
            add("b//a", b // a)
            add("b%a", b % a)
        add("concat_ab", f"{a:02d}{b:02d}")
        add("concat_ba", f"{b:02d}{a:02d}")
        add("digit_concat_sums", f"{a_tens + b_tens}{a_ones + b_ones}")
        add("digit_concat_absdiffs", f"{abs(a_tens - b_tens)}{abs(a_ones - b_ones)}")
        add("digit_concat_products", f"{a_tens * b_tens}{a_ones * b_ones}")
        add("digit_concat_cross_products", f"{a_tens * b_ones}{a_ones * b_tens}")
        add("sum_digits_total", a_tens + a_ones + b_tens + b_ones)
        add("prod_digits_total", a_tens * a_ones * b_tens * b_ones)
        add("gcd", math.gcd(a, b))
        add("lcm", abs(a * b) // math.gcd(a, b) if a and b else 0)
        add("xor", a ^ b)
        add("or", a | b)
        add("and", a & b)
        add("rev_a+rev_b", reverse_a + reverse_b)
        add("rev_a-rev_b", reverse_a - reverse_b)
        add("rev_b-rev_a", reverse_b - reverse_a)
        add("rev_a*rev_b", reverse_a * reverse_b)
        add("a_plus_rev_b", a + reverse_b)
        add("rev_a_plus_b", reverse_a + b)
        add("a_times_rev_b", a * reverse_b)
        add("rev_a_times_b", reverse_a * b)
        add("digit_concat_sorted", "".join(sorted(f"{a:02d}{b:02d}")))
        add("digit_concat_sorted_desc", "".join(sorted(f"{a:02d}{b:02d}", reverse=True)))
        add("sum_products_plus", a_tens * b_tens + a_ones * b_ones)
        add("sum_products_cross", a_tens * b_ones + a_ones * b_tens)
        add("concat_a_absdiff", f"{a:02d}{abs(a - b)}")
        add("concat_b_absdiff", f"{b:02d}{abs(a - b)}")
        add("concat_a_sum", f"{a:02d}{a + b}")
        add("concat_b_sum", f"{b:02d}{a + b}")
        add("concat_a_prod", f"{a:02d}{a * b}")
        add("concat_b_prod", f"{b:02d}{a * b}")
        return outputs

    def numeric_equation_candidates_for_op(
        self,
        examples: Sequence[Dict[str, object]],
        target_op: str,
    ) -> set[str]:
        matching_examples = [example for example in examples if example["op"] == target_op]
        candidates: Optional[set[str]] = None
        for example in matching_examples:
            operations = self.numeric_equation_operations(int(example["a"]), int(example["b"]))
            matches = {name for name, value in operations.items() if value == str(example["rhs"])}
            candidates = matches if candidates is None else (candidates & matches)
        return candidates or set()

    def solve_numeric_equation(self, prompt: str) -> Optional[str]:
        info = self.parse_numeric_equation_prompt(prompt)
        observed_ops = sorted({example["op"] for example in info["examples"]} | {info["target_op"]})
        all_functions = sorted(
            {
                function_name
                for counter in self.equation_resources.numeric_op_priors.values()
                for function_name in counter
            }
        )
        if not all_functions:
            return None
        ranked_candidates: Dict[str, List[str]] = {}
        for operator in observed_ops:
            candidates = self.numeric_equation_candidates_for_op(info["examples"], operator)
            op_priors = self.equation_resources.numeric_op_priors.get(operator, Counter())
            if candidates:
                ranked_candidates[operator] = sorted(
                    candidates,
                    key=lambda name: (-op_priors[name], -self.equation_resources.numeric_priors[name], name),
                )[:4]
            elif op_priors:
                ranked_candidates[operator] = sorted(
                    op_priors,
                    key=lambda name: (-op_priors[name], -self.equation_resources.numeric_priors[name], name),
                )[:4]
            else:
                ranked_candidates[operator] = sorted(
                    all_functions,
                    key=lambda name: (-self.equation_resources.numeric_priors[name], name),
                )[:4]

        best_assignment: Optional[tuple[float, Dict[str, str]]] = None
        for choice in itertools.product(*(ranked_candidates[operator] for operator in observed_ops)):
            if len(set(choice)) < len(choice):
                continue
            assignment = dict(zip(observed_ops, choice))
            score = 0.0
            for operator, function_name in assignment.items():
                op_priors = self.equation_resources.numeric_op_priors.get(operator, Counter())
                score += 10.0 * op_priors[function_name] + self.equation_resources.numeric_priors[function_name]
            candidate = (score, assignment)
            if best_assignment is None or candidate[0] > best_assignment[0]:
                best_assignment = candidate

        if best_assignment is None:
            return None

        best_name = best_assignment[1][info["target_op"]]
        target_a, target_b = info["target_numbers"]
        return self.numeric_equation_operations(target_a, target_b)[best_name]

    def symbolic_subsequence_templates(
        self,
        examples: Sequence[tuple[str, str]],
    ) -> List[tuple[int, ...]]:
        if not examples:
            return []
        output_lengths = {len(rhs) for _, rhs in examples}
        if len(output_lengths) != 1:
            return []
        output_length = next(iter(output_lengths))
        templates = []
        for template in itertools.product(range(5), repeat=output_length):
            if all(self.apply_symbolic_template(lhs, template) == rhs for lhs, rhs in examples):
                templates.append(template)
        return templates

    def apply_symbolic_template(self, lhs: str, template: tuple[int, ...]) -> str:
        return "".join(lhs[index] for index in template)

    def solve_symbolic_equation(self, prompt: str) -> Optional[str]:
        target = self.parse_equation_target(prompt)
        target_op = target[2]
        examples = [(lhs, rhs) for lhs, rhs in self.parse_equation_pairs(prompt) if lhs[2] == target_op]
        templates = self.symbolic_subsequence_templates(examples)
        if templates:
            best_template = sorted(
                templates,
                key=lambda template: (-self.equation_resources.symbolic_priors[template], template),
            )[0]
            prediction = self.apply_symbolic_template(target, best_template)
        else:
            prediction = self.solve_symbolic_program_prior(target, target_op, examples)
        if prediction is None:
            return None
        shortened = self.solve_symbolic_retrieval_shorten_fallback(prompt, prediction)
        return shortened or prediction

    def canonicalize_symbolic_prompt(self, prompt: str) -> str:
        pairs = self.parse_equation_pairs(prompt)
        target = self.parse_equation_target(prompt)
        mapping: Dict[str, str] = {}
        next_id = 0

        def norm_char(char: str) -> str:
            nonlocal next_id
            if char == " ":
                return "_"
            if char in SYMBOLIC_OPERATOR_CHARS:
                return f"[{char}]"
            if char not in mapping:
                mapping[char] = f"v{next_id}"
                next_id += 1
            return mapping[char]

        chunks: List[str] = []
        for lhs, rhs in pairs:
            chunks.append("L:" + " ".join(norm_char(char) for char in lhs))
            chunks.append("R:" + " ".join(norm_char(char) for char in rhs))
        chunks.append("T:" + " ".join(norm_char(char) for char in target))
        return " || ".join(chunks)

    def visible_symbol_order(self, prompt: str) -> List[str]:
        pairs = self.parse_equation_pairs(prompt)
        target = self.parse_equation_target(prompt)
        seen: List[str] = []
        for text in [*(lhs for lhs, _ in pairs), *(rhs for _, rhs in pairs), target]:
            for char in text:
                if char in SYMBOLIC_OPERATOR_CHARS:
                    continue
                if char not in seen:
                    seen.append(char)
        return seen

    def transfer_symbolic_answer_by_rank(
        self,
        train_prompt: str,
        train_answer: str,
        eval_prompt: str,
    ) -> Optional[str]:
        train_visible = self.visible_symbol_order(train_prompt)
        eval_visible = self.visible_symbol_order(eval_prompt)
        rank_map = dict(zip(train_visible, eval_visible))
        out: List[str] = []
        for char in train_answer:
            if char in rank_map:
                out.append(rank_map[char])
            else:
                out.append(char)
        return "".join(out)

    def char_wb_ngram_counts(self, text: str, low: int = 3, high: int = 6) -> Counter[str]:
        grams: Counter[str] = Counter()
        for token in text.split():
            padded = f" {token} "
            for ngram_size in range(low, high + 1):
                if len(padded) < ngram_size:
                    continue
                for index in range(len(padded) - ngram_size + 1):
                    grams[padded[index : index + ngram_size]] += 1
        return grams

    def cosine_counter_similarity(self, left: Counter[str], right: Counter[str]) -> float:
        if not left or not right:
            return 0.0
        dot = sum(value * right.get(token, 0) for token, value in left.items())
        left_norm = math.sqrt(sum(value * value for value in left.values()))
        right_norm = math.sqrt(sum(value * value for value in right.values()))
        if left_norm == 0.0 or right_norm == 0.0:
            return 0.0
        return dot / (left_norm * right_norm)

    def solve_symbolic_retrieval_shorten_fallback(
        self,
        prompt: str,
        current_prediction: str,
    ) -> Optional[str]:
        target = self.parse_equation_target(prompt)
        query_grams = self.char_wb_ngram_counts(self.canonicalize_symbolic_prompt(prompt))
        best: Optional[tuple[float, str]] = None
        for entry in self.symbolic_retrieval_entries:
            if entry.target_op != target[2]:
                continue
            similarity = self.cosine_counter_similarity(query_grams, entry.gram_counts)
            if similarity < 0.80:
                continue
            prediction = self.transfer_symbolic_answer_by_rank(entry.prompt, entry.answer, prompt)
            if prediction is None:
                continue
            if len(prediction) >= len(current_prediction):
                continue
            if not current_prediction.startswith(prediction):
                continue
            if best is None or similarity > best[0]:
                best = (similarity, prediction)
        return None if best is None else best[1]

    def emit_symbolic_atom(self, lhs: str, atom: tuple[str, object]) -> Optional[str | tuple[str, int]]:
        kind, arg = atom
        left = lhs[:2]
        right = lhs[3:]
        if kind == "slot":
            return lhs[arg]
        if kind == "other":
            return lhs[SYMBOLIC_OTHER_INDEX[arg]]
        if kind == "shared_first":
            for char in left:
                if char in right:
                    return char
            return None
        if kind == "shared_last":
            for char in reversed(left):
                if char in right:
                    return char
            return None
        if kind == "left_only_first":
            for char in left:
                if char not in right:
                    return char
            return None
        if kind == "left_only_last":
            for char in reversed(left):
                if char not in right:
                    return char
            return None
        if kind == "right_only_first":
            for char in right:
                if char not in left:
                    return char
            return None
        if kind == "right_only_last":
            for char in reversed(right):
                if char not in left:
                    return char
            return None
        if kind == "role":
            return ("role", arg)
        raise ValueError(f"Unknown symbolic atom: {atom!r}")

    def symbolic_atom_complexity(self, atom: tuple[str, object]) -> int:
        kind = atom[0]
        if kind == "role":
            return 3
        return 1

    def symbolic_atom_sort_key(self, atom: tuple[str, object]) -> tuple:
        kind, arg = atom
        rank = {
            "slot": 0,
            "other": 1,
            "shared_first": 2,
            "shared_last": 3,
            "left_only_first": 4,
            "left_only_last": 5,
            "right_only_first": 6,
            "right_only_last": 7,
            "role": 8,
        }[kind]
        return (rank, arg)

    def symbolic_program_complexity(self, program: tuple[tuple[str, object], ...]) -> int:
        return sum(self.symbolic_atom_complexity(atom) for atom in program)

    def fit_symbolic_atom_position(
        self,
        atom: tuple[str, object],
        examples: Sequence[tuple[str, str]],
        out_pos: int,
        role_bindings: Dict[int, str],
    ) -> Optional[Dict[int, str]]:
        new_bindings: Dict[int, str] = {}
        for lhs, rhs in examples:
            if len(rhs) <= out_pos:
                return None
            token = self.emit_symbolic_atom(lhs, atom)
            if token is None:
                return None
            gold = rhs[out_pos]
            if isinstance(token, tuple):
                _, role_id = token
                existing = new_bindings.get(role_id, role_bindings.get(role_id))
                if existing is None:
                    new_bindings[role_id] = gold
                elif existing != gold:
                    return None
            elif token != gold:
                return None
        return new_bindings

    def candidate_symbolic_atoms_for_position(
        self,
        examples: Sequence[tuple[str, str]],
        out_pos: int,
    ) -> List[tuple[str, object]]:
        best_non_role_atoms: Dict[tuple[str, ...], tuple[int, tuple, tuple[str, object]]] = {}
        role_atoms: List[tuple[str, object]] = []
        for atom in SYMBOLIC_ATOMS:
            if self.fit_symbolic_atom_position(atom, examples, out_pos, {}) is None:
                continue
            if atom[0] == "role":
                role_atoms.append(atom)
                continue
            signature = tuple(self.emit_symbolic_atom(lhs, atom) for lhs, _ in examples)
            score = (self.symbolic_atom_complexity(atom), self.symbolic_atom_sort_key(atom))
            existing = best_non_role_atoms.get(signature)
            if existing is None or score < existing[:2]:
                best_non_role_atoms[signature] = (score[0], score[1], atom)

        candidates = [entry[2] for entry in best_non_role_atoms.values()] + role_atoms
        candidates.sort(key=lambda atom: (self.symbolic_atom_complexity(atom), self.symbolic_atom_sort_key(atom)))
        return candidates[:MAX_SYMBOLIC_POSITION_CANDIDATES]

    def enumerate_exact_symbolic_programs(
        self,
        examples: Sequence[tuple[str, str]],
    ) -> List[tuple[tuple[str, object], ...]]:
        if not examples:
            return []
        output_lengths = {len(rhs) for _, rhs in examples}
        if len(output_lengths) != 1:
            return []
        output_length = next(iter(output_lengths))
        if output_length > MAX_SYMBOLIC_PROGRAM_LENGTH:
            return []

        position_candidates = [
            self.candidate_symbolic_atoms_for_position(examples, out_pos)
            for out_pos in range(output_length)
        ]
        if any(not candidates for candidates in position_candidates):
            return []

        candidate_product = 1
        for candidates in position_candidates:
            candidate_product *= len(candidates)
        if candidate_product > MAX_SYMBOLIC_CANDIDATE_PRODUCT:
            return []

        position_order = sorted(range(output_length), key=lambda out_pos: (len(position_candidates[out_pos]), out_pos))
        assignments: List[Optional[tuple[str, object]]] = [None] * output_length
        programs: List[tuple[tuple[str, object], ...]] = []
        best_cost: Optional[int] = None

        def search(column_index: int, role_bindings: Dict[int, str], running_cost: int) -> None:
            nonlocal best_cost
            if best_cost is not None and running_cost > best_cost + 1:
                return
            if len(programs) >= MAX_SYMBOLIC_EXACT_PROGRAMS and best_cost is not None and running_cost > best_cost:
                return
            if column_index == len(position_order):
                program = tuple(assignments)
                if best_cost is None or running_cost < best_cost:
                    best_cost = running_cost
                programs.append(program)
                return

            out_pos = position_order[column_index]
            for atom in position_candidates[out_pos]:
                atom_cost = self.symbolic_atom_complexity(atom)
                next_cost = running_cost + atom_cost
                if best_cost is not None and next_cost > best_cost + 1:
                    continue
                new_bindings = self.fit_symbolic_atom_position(atom, examples, out_pos, role_bindings)
                if new_bindings is None:
                    continue
                assignments[out_pos] = atom
                merged_bindings = dict(role_bindings)
                merged_bindings.update(new_bindings)
                search(column_index + 1, merged_bindings, next_cost)
                assignments[out_pos] = None

        search(0, {}, 0)
        if not programs:
            return []

        deduped_programs = sorted(
            set(programs),
            key=lambda program: (
                self.symbolic_program_complexity(program),
                tuple(self.symbolic_atom_sort_key(atom) for atom in program),
            ),
        )
        if best_cost is not None:
            deduped_programs = [
                program
                for program in deduped_programs
                if self.symbolic_program_complexity(program) <= best_cost + 1
            ]
        return deduped_programs[:MAX_SYMBOLIC_EXACT_PROGRAMS]

    def fit_symbolic_program(
        self,
        program: tuple[tuple[str, object], ...],
        examples: Sequence[tuple[str, str]],
        target: Optional[str] = None,
    ) -> Optional[str]:
        roles: Dict[int, str] = {}
        for lhs, rhs in examples:
            if len(rhs) != len(program):
                return None
            for atom, gold in zip(program, rhs):
                token = self.emit_symbolic_atom(lhs, atom)
                if token is None:
                    return None
                if isinstance(token, tuple):
                    _, role_id = token
                    existing = roles.get(role_id)
                    if existing is None:
                        roles[role_id] = gold
                    elif existing != gold:
                        return None
                elif token != gold:
                    return None

        if target is None:
            return ""

        output: List[str] = []
        for atom in program:
            token = self.emit_symbolic_atom(target, atom)
            if token is None:
                return None
            if isinstance(token, tuple):
                _, role_id = token
                if role_id not in roles:
                    return None
                output.append(roles[role_id])
            else:
                output.append(token)
        return "".join(output)

    def solve_symbolic_program_prior(
        self,
        target: str,
        target_op: str,
        examples: Sequence[tuple[str, str]],
    ) -> Optional[str]:
        program_priority_lists = [
            self.equation_resources.symbolic_program_priors.get(target_op, Counter()).most_common(200),
            self.equation_resources.symbolic_global_program_priors.most_common(300),
        ]
        for ranked_programs in program_priority_lists:
            for program, _ in ranked_programs:
                prediction = self.fit_symbolic_program(program, examples, target)
                if prediction is not None:
                    return prediction
        return None

    def candidate_plain_words(
        self,
        cipher_word: str,
        cipher_to_plain: Dict[str, str],
        plain_to_cipher: Dict[str, str],
    ) -> List[str]:
        options = []
        expected_pattern = word_pattern(cipher_word)
        for word in self.cipher_resources.vocabulary[len(cipher_word)]:
            if word_pattern(word) != expected_pattern:
                continue
            ok = True
            for c_char, p_char in zip(cipher_word, word):
                existing = cipher_to_plain.get(c_char)
                if existing is not None and existing != p_char:
                    ok = False
                    break
                reverse = plain_to_cipher.get(p_char)
                if reverse is not None and reverse != c_char:
                    ok = False
                    break
            if ok:
                options.append(word)
        return options


def solve_dataframe(train_df: pd.DataFrame, prompt_df: pd.DataFrame) -> pd.DataFrame:
    solver = NemotronReasoningSolver().fit(train_df)
    outputs = []
    for _, row in prompt_df.iterrows():
        prediction = solver.solve(row["prompt"])
        outputs.append({"id": row["id"], "answer": prediction})
    return pd.DataFrame(outputs)
