from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd
from faker import Faker


SEED = 20260405
OUTPUT_PATH = Path("data/generated/nepotism_synthetic_data.xlsx")

CANDIDATE_COLUMNS = [
    "company_id",
    "scenario",
    "candidate_id",
    "first_name",
    "last_name",
    "full_name",
    "gender",
    "city_name",
    "high_school_name",
    "college_name",
    "age",
    "years_experience",
    "education_level",
    "high_school_gpa",
    "college_gpa",
    "interview_score",
    "test_score",
    "referral_flag",
    "family_link_flag",
    "close_family_relation_flag",
    "same_high_school_flag",
    "same_city_flag",
    "same_college_flag",
    "same_last_name_flag",
    "connection_strength",
    "merit_score",
    "discretionary_channel",
    "hired_flag",
]

EMPLOYEE_COLUMNS = [
    "company_id",
    "scenario",
    "employee_id",
    "candidate_id",
    "first_name",
    "last_name",
    "full_name",
    "gender",
    "city_name",
    "high_school_name",
    "college_name",
    "department_id",
    "manager_id",
    "role_level",
    "age",
    "years_experience",
    "education_level",
    "high_school_gpa",
    "college_gpa",
    "referral_flag",
    "family_link_flag",
    "close_family_relation_flag",
    "same_high_school_flag",
    "same_city_flag",
    "same_college_flag",
    "same_last_name_flag",
    "connection_strength",
    "merit_score",
    "discretionary_channel",
    "tenure_months",
    "performance_score",
    "salary",
    "promoted_flag",
    "months_to_promotion",
]

EDUCATION_GRAD_AGE = {
    "High School": 18,
    "Bachelor": 22,
    "Master": 24,
    "Doctorate": 28,
}

EDUCATION_RANK = {
    "High School": 0,
    "Bachelor": 1,
    "Master": 2,
    "Doctorate": 3,
}

MALE_FIRST_NAMES = [
    ("Noam", 16),
    ("David", 15),
    ("Ori", 12),
    ("Yonatan", 11),
    ("Eitan", 10),
    ("Omer", 10),
    ("Ariel", 9),
    ("Daniel", 9),
    ("Yair", 8),
    ("Roee", 7),
    ("Nadav", 7),
    ("Ido", 7),
    ("Amit", 6),
    ("Alon", 6),
    ("Lior", 6),
    ("Yuval", 6),
    ("Itay", 5),
    ("Shahar", 5),
    ("Assaf", 4),
    ("Tomer", 4),
]

FEMALE_FIRST_NAMES = [
    ("Noa", 17),
    ("Yael", 14),
    ("Maya", 12),
    ("Tamar", 11),
    ("Shira", 11),
    ("Amit", 9),
    ("Dana", 9),
    ("Roni", 9),
    ("Adi", 8),
    ("Neta", 7),
    ("Lihi", 7),
    ("Gal", 7),
    ("Michal", 6),
    ("Hila", 6),
    ("Ofir", 6),
    ("Yarden", 5),
    ("Rotem", 5),
    ("Tal", 5),
    ("Moran", 4),
    ("Einav", 4),
]

SURNAMES = [
    ("Cohen", 28),
    ("Levi", 22),
    ("Mizrahi", 15),
    ("Peretz", 13),
    ("Biton", 12),
    ("Friedman", 11),
    ("Azoulay", 10),
    ("Malka", 10),
    ("Dahan", 9),
    ("Ben-David", 9),
    ("Katz", 8),
    ("Avraham", 7),
    ("Sharabi", 7),
    ("Ohayon", 7),
    ("Shapiro", 6),
    ("Berkowitz", 6),
    ("Gabay", 6),
    ("Haddad", 6),
    ("Nasser", 5),
    ("Amar", 5),
    ("Halabi", 4),
    ("Toledano", 4),
    ("Sasson", 4),
    ("Abutbul", 4),
    ("Ben-Ami", 4),
    ("Buzaglo", 3),
    ("Turgeman", 3),
    ("Moyal", 3),
    ("Bitan", 3),
    ("Vaknin", 3),
    ("Mor", 3),
    ("Kadosh", 2),
    ("Sadeh", 2),
    ("Naim", 2),
    ("Harari", 2),
    ("Dayan", 2),
    ("Swisa", 2),
    ("Hassan", 2),
    ("Zoabi", 1),
    ("Kariv", 1),
]

CITIES = [
    ("Jerusalem", 17),
    ("Tel Aviv", 15),
    ("Haifa", 11),
    ("Beersheba", 9),
    ("Rishon LeZion", 8),
    ("Petah Tikva", 7),
    ("Ashdod", 7),
    ("Netanya", 7),
    ("Holon", 6),
    ("Bnei Brak", 5),
    ("Ramat Gan", 5),
    ("Rehovot", 5),
    ("Ashkelon", 5),
    ("Bat Yam", 4),
    ("Herzliya", 4),
    ("Kfar Saba", 4),
    ("Ra'anana", 4),
    ("Modi'in-Maccabim-Re'ut", 4),
    ("Eilat", 2),
    ("Nazareth", 2),
]

HIGH_SCHOOLS_BY_CITY = {
    "Jerusalem": [
        ("Gymnasia Rehavia", 7),
        ("Leyada High School", 5),
        ("Boyer High School", 4),
    ],
    "Tel Aviv": [
        ("Herzliya Hebrew Gymnasium", 7),
        ("Ironi Aleph High School", 5),
        ("Alliance High School Tel Aviv", 4),
    ],
    "Haifa": [
        ("The Hebrew Reali School", 8),
        ("Hogim High School", 4),
        ("Ironi Hey Haifa", 3),
    ],
    "Beersheba": [
        ("Makif Aleph Beersheba", 6),
        ("Makif Vav Beersheba", 4),
        ("Comprehensive High School Beersheba", 3),
    ],
    "Rishon LeZion": [
        ("Gymnasia Realit Rishon LeZion", 5),
        ("Makif Zayin Rishon LeZion", 4),
        ("Ironi Gimel Rishon LeZion", 3),
    ],
    "Petah Tikva": [
        ("Golda High School", 5),
        ("Brener High School Petah Tikva", 4),
        ("Amit Bar Ilan Petah Tikva", 3),
    ],
    "Ashdod": [
        ("Makif Aleph Ashdod", 6),
        ("Makif Tet Ashdod", 4),
        ("ORT Yad Leibovich Ashdod", 3),
    ],
    "Netanya": [
        ("Shapira High School Netanya", 5),
        ("ORT Yad Leibovitch Netanya", 4),
        ("Rabin High School Netanya", 3),
    ],
    "Holon": [
        ("Kugel High School", 6),
        ("ORT Holon", 4),
        ("Ironi Gimel Holon", 3),
    ],
    "Bnei Brak": [
        ("Bnei Brak Municipal High School", 5),
        ("Ohel Shem Religious High School", 4),
        ("Amit Bnei Brak", 3),
    ],
    "Ramat Gan": [
        ("Ohel Shem", 8),
        ("Blich High School", 5),
        ("Amit Ramat Gan", 2),
    ],
    "Rehovot": [
        ("De Shalit High School", 7),
        ("Amit Rehovot", 3),
        ("Katzir High School Rehovot", 2),
    ],
    "Ashkelon": [
        ("Makif Aleph Ashkelon", 6),
        ("ORT Afridar Ashkelon", 4),
        ("Rabin High School Ashkelon", 3),
    ],
    "Bat Yam": [
        ("Ramot High School Bat Yam", 5),
        ("ORT Ramat Yosef", 4),
        ("Ironi High School Bat Yam", 3),
    ],
    "Herzliya": [
        ("The New High School Herzliya", 6),
        ("Yovel High School Herzliya", 4),
        ("Rabin High School Herzliya", 3),
    ],
    "Kfar Saba": [
        ("Katzir High School Kfar Saba", 6),
        ("Rabin High School Kfar Saba", 4),
        ("Galili High School", 3),
    ],
    "Ra'anana": [
        ("MetroWest High School", 7),
        ("Ostrovsky High School", 5),
        ("Amichai High School", 2),
    ],
    "Modi'in-Maccabim-Re'ut": [
        ("Ironi Aleph Modi'in", 6),
        ("Moriah High School Modi'in", 4),
        ("Yachad High School Modi'in", 3),
    ],
    "Eilat": [
        ("Begin High School Eilat", 6),
        ("Rabin High School Eilat", 4),
        ("Goldwater High School", 3),
    ],
    "Nazareth": [
        ("St. Joseph High School Nazareth", 6),
        ("Baptist High School Nazareth", 4),
        ("Orthodox School Nazareth", 3),
    ],
}

COLLEGES = [
    ("Tel Aviv University", 15),
    ("Hebrew University of Jerusalem", 13),
    ("Technion", 12),
    ("Ben-Gurion University of the Negev", 11),
    ("Bar-Ilan University", 10),
    ("University of Haifa", 8),
    ("Reichman University", 6),
    ("The Open University of Israel", 6),
    ("Ariel University", 5),
    ("Sapir College", 4),
    ("Hadassah Academic College", 4),
    ("Ono Academic College", 4),
]

COLLEGES_BY_REGION = {
    "Jerusalem": [
        ("Hebrew University of Jerusalem", 8),
        ("Hadassah Academic College", 4),
        ("The Open University of Israel", 3),
        ("Bar-Ilan University", 3),
    ],
    "Tel Aviv": [
        ("Tel Aviv University", 8),
        ("Reichman University", 5),
        ("Bar-Ilan University", 4),
        ("Ono Academic College", 3),
    ],
    "Haifa": [
        ("Technion", 8),
        ("University of Haifa", 7),
        ("The Open University of Israel", 3),
    ],
    "Beersheba": [
        ("Ben-Gurion University of the Negev", 9),
        ("Sapir College", 5),
        ("The Open University of Israel", 3),
    ],
    "Rishon LeZion": [
        ("Tel Aviv University", 7),
        ("Bar-Ilan University", 5),
        ("Ono Academic College", 4),
    ],
    "Petah Tikva": [
        ("Bar-Ilan University", 7),
        ("Tel Aviv University", 6),
        ("Ono Academic College", 4),
    ],
    "Ashdod": [
        ("Ben-Gurion University of the Negev", 6),
        ("Sapir College", 5),
        ("The Open University of Israel", 3),
    ],
    "Netanya": [
        ("Reichman University", 5),
        ("Tel Aviv University", 5),
        ("The Open University of Israel", 4),
    ],
    "Holon": [
        ("Tel Aviv University", 6),
        ("Bar-Ilan University", 5),
        ("Ono Academic College", 4),
    ],
    "Bnei Brak": [
        ("Bar-Ilan University", 7),
        ("Ono Academic College", 5),
        ("The Open University of Israel", 4),
    ],
    "Ramat Gan": [
        ("Bar-Ilan University", 8),
        ("Tel Aviv University", 5),
        ("Ono Academic College", 4),
    ],
    "Rehovot": [
        ("Tel Aviv University", 5),
        ("Bar-Ilan University", 4),
        ("The Open University of Israel", 4),
    ],
    "Ashkelon": [
        ("Sapir College", 6),
        ("Ben-Gurion University of the Negev", 5),
        ("The Open University of Israel", 3),
    ],
    "Bat Yam": [
        ("Tel Aviv University", 6),
        ("Bar-Ilan University", 4),
        ("Ono Academic College", 4),
    ],
    "Herzliya": [
        ("Reichman University", 7),
        ("Tel Aviv University", 6),
        ("Bar-Ilan University", 3),
    ],
    "Kfar Saba": [
        ("Reichman University", 5),
        ("Tel Aviv University", 5),
        ("The Open University of Israel", 4),
    ],
    "Ra'anana": [
        ("Reichman University", 6),
        ("Tel Aviv University", 5),
        ("Bar-Ilan University", 4),
    ],
    "Modi'in-Maccabim-Re'ut": [
        ("Hebrew University of Jerusalem", 5),
        ("Bar-Ilan University", 5),
        ("The Open University of Israel", 4),
    ],
    "Eilat": [
        ("Ben-Gurion University of the Negev", 5),
        ("The Open University of Israel", 4),
        ("Sapir College", 3),
    ],
    "Nazareth": [
        ("University of Haifa", 6),
        ("Technion", 5),
        ("The Open University of Israel", 3),
    ],
}


@dataclass(frozen=True)
class ScenarioConfig:
    name: str
    company_prefix: str
    companies: int
    base_candidates_per_company: int
    family_link_rate: float
    referral_rate: float
    connection_weight: float
    family_weight: float
    referral_weight: float
    discretionary_weight: float
    merit_weight: float
    hire_intercept: float
    negative_selection_bonus: float
    cluster_bias: float
    management_connection_weight: float
    promotion_connection_weight: float
    salary_connection_premium: float
    nepotism_dominance_target: float
    target_hire_rate: float
    target_promotion_rate: float
    promotion_dominance_target: float


@dataclass(frozen=True)
class HiringContext:
    hiring_context_id: str
    city_name: str
    high_school_name: str
    college_name: float | str
    last_name: str
    bias_strength: float
    family_profile_index: int | None


SCENARIOS = [
    ScenarioConfig(
        name="Merit-based",
        company_prefix="MER",
        companies=16,
        base_candidates_per_company=260,
        family_link_rate=0.06,
        referral_rate=0.18,
        connection_weight=0.40,
        family_weight=0.25,
        referral_weight=0.15,
        discretionary_weight=0.10,
        merit_weight=1.80,
        hire_intercept=-5.10,
        negative_selection_bonus=0.00,
        cluster_bias=0.15,
        management_connection_weight=0.10,
        promotion_connection_weight=0.10,
        salary_connection_premium=0.01,
        nepotism_dominance_target=0.12,
        target_hire_rate=0.022,
        target_promotion_rate=0.070,
        promotion_dominance_target=0.12,
    ),
    ScenarioConfig(
        name="Moderate favoritism",
        company_prefix="MOD",
        companies=16,
        base_candidates_per_company=290,
        family_link_rate=0.13,
        referral_rate=0.31,
        connection_weight=0.85,
        family_weight=0.70,
        referral_weight=0.45,
        discretionary_weight=0.35,
        merit_weight=1.25,
        hire_intercept=-4.95,
        negative_selection_bonus=0.35,
        cluster_bias=0.40,
        management_connection_weight=0.24,
        promotion_connection_weight=0.24,
        salary_connection_premium=0.025,
        nepotism_dominance_target=0.32,
        target_hire_rate=0.030,
        target_promotion_rate=0.078,
        promotion_dominance_target=0.34,
    ),
    ScenarioConfig(
        name="High nepotism risk",
        company_prefix="HNR",
        companies=16,
        base_candidates_per_company=330,
        family_link_rate=0.22,
        referral_rate=0.43,
        connection_weight=1.42,
        family_weight=1.18,
        referral_weight=0.76,
        discretionary_weight=0.78,
        merit_weight=0.85,
        hire_intercept=-5.00,
        negative_selection_bonus=0.75,
        cluster_bias=0.78,
        management_connection_weight=0.48,
        promotion_connection_weight=0.46,
        salary_connection_premium=0.04,
        nepotism_dominance_target=0.56,
        target_hire_rate=0.032,
        target_promotion_rate=0.095,
        promotion_dominance_target=0.64,
    ),
]


def sigmoid(value: float | np.ndarray) -> float | np.ndarray:
    return 1.0 / (1.0 + np.exp(-value))


def clip_round(value: float, lower: float, upper: float, digits: int = 1) -> float:
    return round(float(np.clip(value, lower, upper)), digits)


def weighted_choice(options: list[tuple[str, float]], rng: np.random.Generator) -> str:
    labels = [label for label, _ in options]
    weights = np.array([weight for _, weight in options], dtype=float)
    probabilities = weights / weights.sum()
    return str(rng.choice(labels, p=probabilities))


def generate_name(gender: str, rng: np.random.Generator) -> tuple[str, str]:
    first_name_pool = MALE_FIRST_NAMES if gender == "Male" else FEMALE_FIRST_NAMES
    first_name = weighted_choice(first_name_pool, rng)
    last_name = weighted_choice(SURNAMES, rng)
    return first_name, last_name


def generate_city(
    rng: np.random.Generator,
    preferred_city: str | None = None,
    preferred_weight: float = 0.0,
) -> str:
    if preferred_city and rng.random() < preferred_weight:
        return preferred_city
    return weighted_choice(CITIES, rng)


def generate_high_school(city_name: str, rng: np.random.Generator) -> str:
    options = HIGH_SCHOOLS_BY_CITY.get(city_name)
    if options:
        return weighted_choice(options, rng)
    flattened = [item for schools in HIGH_SCHOOLS_BY_CITY.values() for item in schools]
    return weighted_choice(flattened, rng)


def generate_college(
    city_name: str,
    education_level: str,
    academic_index: float,
    rng: np.random.Generator,
) -> float | str:
    if EDUCATION_RANK[education_level] < EDUCATION_RANK["Bachelor"]:
        return np.nan

    regional_options = COLLEGES_BY_REGION.get(city_name, COLLEGES)
    labels = [label for label, _ in regional_options]
    weights = np.array([weight for _, weight in regional_options], dtype=float)

    if academic_index > 1.2:
        weights *= np.array(
            [
                1.15
                if label
                in {
                    "Tel Aviv University",
                    "Hebrew University of Jerusalem",
                    "Technion",
                    "Ben-Gurion University of the Negev",
                    "Bar-Ilan University",
                }
                else 0.90
                for label in labels
            ]
        )
    elif academic_index < -0.8:
        weights *= np.array(
            [
                0.85
                if label in {"Tel Aviv University", "Hebrew University of Jerusalem", "Technion"}
                else 1.10
                for label in labels
            ]
        )

    probabilities = weights / weights.sum()
    return str(rng.choice(labels, p=probabilities))


def generate_gpas(
    education_level: str,
    academic_index: float,
    rng: np.random.Generator,
) -> tuple[float, float]:
    high_school_mean = 84.0 + academic_index * 4.8
    high_school_gpa = clip_round(rng.normal(high_school_mean, 6.5), 56.0, 100.0)

    if EDUCATION_RANK[education_level] < EDUCATION_RANK["Bachelor"]:
        return high_school_gpa, np.nan

    college_mean = 81.5 + academic_index * 4.6 + EDUCATION_RANK[education_level] * 1.2
    college_gpa = clip_round(rng.normal(college_mean, 5.8), 58.0, 99.0)
    return high_school_gpa, college_gpa


def choose_education_level(
    age: int,
    academic_index: float,
    rng: np.random.Generator,
) -> str:
    choices: list[tuple[str, float]] = []

    if age >= 18:
        choices.append(("High School", 0.12 if age < 26 else 0.08))
    if age >= 21:
        choices.append(("Bachelor", 0.62))
    if age >= 24:
        master_weight = 0.18 + (0.05 if academic_index > 0.7 else 0.0)
        choices.append(("Master", master_weight))
    if age >= 28:
        doctorate_weight = 0.03 + (0.03 if academic_index > 1.1 else 0.0)
        choices.append(("Doctorate", doctorate_weight))

    if academic_index < -0.8:
        choices = [
            (label, weight * (1.35 if label == "High School" else 0.90))
            for label, weight in choices
        ]

    return weighted_choice(choices, rng)


def build_family_profiles(
    total_candidates: int,
    company_city: str,
    config: ScenarioConfig,
    rng: np.random.Generator,
) -> list[dict[str, object]]:
    # Family profiles create repeated surname/city/school patterns for connected applicants.
    estimated_family_candidates = max(4, int(total_candidates * config.family_link_rate))
    profile_count = max(2, estimated_family_candidates // 3)

    profiles = []
    for _ in range(profile_count):
        family_city = generate_city(rng, preferred_city=company_city, preferred_weight=0.72)
        family_college = generate_college(family_city, "Bachelor", rng.normal(0.1, 0.7), rng)
        profiles.append(
            {
                "last_name": weighted_choice(SURNAMES[:18], rng),
                "city_name": family_city,
                "high_school_name": generate_high_school(family_city, rng),
                "college_name": family_college,
                "connection_strength": clip_round(rng.uniform(0.65, 0.97), 0.0, 1.0, 2),
                "discretionary_channel": str(
                    rng.choice(
                        ["family", "executive_sponsor", "legacy_referral"],
                        p=[0.58, 0.24, 0.18],
                    )
                ),
            }
        )
    return profiles


def build_hiring_contexts(
    company_id: str,
    company_city: str,
    config: ScenarioConfig,
    family_profiles: list[dict[str, object]],
    rng: np.random.Generator,
) -> list[HiringContext]:
    context_count = max(5, min(10, config.base_candidates_per_company // 40))
    contexts = []

    for index in range(context_count):
        context_city = generate_city(rng, preferred_city=company_city, preferred_weight=0.64)
        context_college = generate_college(
            context_city,
            "Bachelor",
            rng.normal(0.25, 0.65),
            rng,
        )
        family_profile_index = None
        bias_strength = rng.uniform(0.08, 0.24)

        if family_profiles and rng.random() < config.family_link_rate * (1.4 if config.name == "High nepotism risk" else 1.0):
            family_profile_index = int(rng.integers(0, len(family_profiles)))
            profile = family_profiles[family_profile_index]
            if rng.random() < 0.56:
                context_city = str(profile["city_name"])
            if rng.random() < 0.52:
                context_college = profile["college_name"]
            bias_strength += rng.uniform(0.10, 0.28)
            last_name = str(profile["last_name"])
        else:
            last_name = weighted_choice(SURNAMES[:24], rng)

        contexts.append(
            HiringContext(
                hiring_context_id=f"{company_id}-HC{index + 1:02d}",
                city_name=context_city,
                high_school_name=generate_high_school(context_city, rng),
                college_name=context_college,
                last_name=last_name,
                bias_strength=float(np.clip(bias_strength, 0.05, 0.72)),
                family_profile_index=family_profile_index,
            )
        )

    return contexts


def choose_hiring_context(
    contexts: list[HiringContext],
    family_profile_index: int | None,
    config: ScenarioConfig,
    rng: np.random.Generator,
) -> HiringContext:
    if family_profile_index is not None:
        aligned_contexts = [context for context in contexts if context.family_profile_index == family_profile_index]
        if aligned_contexts and rng.random() < (0.68 if config.name == "High nepotism risk" else 0.42):
            return aligned_contexts[int(rng.integers(0, len(aligned_contexts)))]

    weights = np.array([context.bias_strength for context in contexts], dtype=float)
    weights = 0.8 + weights * (1.2 + config.connection_weight * 0.35)
    probabilities = weights / weights.sum()
    return contexts[int(rng.choice(len(contexts), p=probabilities))]


def get_discretionary_channel(
    close_family_relation_flag: int,
    same_high_school_flag: int,
    same_college_flag: int,
    same_city_flag: int,
    referral_flag: int,
    same_last_name_flag: int,
) -> str:
    if close_family_relation_flag:
        return "family"
    if same_high_school_flag:
        return "legacy_referral"
    if same_college_flag:
        return "alumni_network"
    if referral_flag and same_city_flag:
        return "manager_endorsement"
    if same_city_flag:
        return "internal_source"
    if same_last_name_flag or referral_flag:
        return "referral"
    return "none"


def get_discretionary_signal(discretionary_channel: str) -> float:
    return {
        "none": 0.0,
        "alumni_network": 0.15,
        "internal_source": 0.22,
        "referral": 0.24,
        "manager_endorsement": 0.28,
        "legacy_referral": 0.35,
        "family": 0.45,
        "executive_sponsor": 0.50,
    }.get(discretionary_channel, 0.0)


def generate_candidate(
    company_id: str,
    company_city: str,
    scenario_name: str,
    config: ScenarioConfig,
    family_profiles: list[dict[str, object]],
    hiring_contexts: list[HiringContext],
    nepotism_adjustment: float,
    hire_intercept_adjustment: float,
    rng: np.random.Generator,
    fake: Faker,
) -> dict[str, object]:
    # Education, GPA, testing, and experience create merit. Hiring-context overlap
    # then drives a separate internal nepotism score used only during generation.
    gender = str(rng.choice(["Male", "Female"], p=[0.51, 0.49]))
    academic_index = float(rng.normal(0.0, 1.0))
    age = int(np.clip(np.round(rng.normal(32.0, 7.2)), 21, 59))
    education_level = choose_education_level(age, academic_index, rng)
    grad_age = EDUCATION_GRAD_AGE[education_level]
    max_experience = max(0, age - grad_age - 1)
    experience_center = max_experience - rng.uniform(0.0, 2.5)
    years_experience = int(np.clip(np.round(rng.normal(experience_center, 2.0)), 0, max_experience))

    family_link_flag = int(rng.random() < config.family_link_rate)
    family_profile = None
    family_profile_index = None
    if family_link_flag and family_profiles:
        family_profile_index = int(rng.integers(0, len(family_profiles)))
        family_profile = family_profiles[family_profile_index]

    if family_profile:
        first_name, _ = generate_name(gender, rng)
        last_name = str(family_profile["last_name"])
        city_name = (
            str(family_profile["city_name"])
            if rng.random() < 0.82
            else generate_city(rng, preferred_city=company_city, preferred_weight=0.46)
        )
    else:
        first_name, last_name = generate_name(gender, rng)
        city_name = generate_city(rng, preferred_city=company_city, preferred_weight=0.38)

    high_school_name = (
        str(family_profile["high_school_name"])
        if family_profile and rng.random() < 0.72
        else generate_high_school(city_name, rng)
    )

    if EDUCATION_RANK[education_level] >= EDUCATION_RANK["Bachelor"]:
        if family_profile and family_profile["college_name"] and rng.random() < 0.54:
            college_name = str(family_profile["college_name"])
        else:
            college_name = generate_college(city_name, education_level, academic_index, rng)
    else:
        college_name = np.nan

    high_school_gpa, college_gpa = generate_gpas(education_level, academic_index, rng)
    test_mean = 68.0 + academic_index * 10.0 + min(years_experience, 12) * 0.8
    test_score = clip_round(rng.normal(test_mean, 8.5), 35.0, 100.0)

    interview_mean = 66.0 + academic_index * 7.0 + min(years_experience, 14) * 0.9
    interview_score = clip_round(rng.normal(interview_mean, 9.0), 35.0, 100.0)

    gpa_component = college_gpa if not pd.isna(college_gpa) else high_school_gpa
    experience_component = min(years_experience, 15) / 15 * 100
    high_school_bonus = 0.005 * high_school_gpa
    merit_score = clip_round(
        0.35 * test_score
        + 0.33 * interview_score
        + 0.20 * gpa_component
        + 0.12 * experience_component
        + high_school_bonus,
        35.0,
        100.0,
    )
    qualification_match = int(
        (EDUCATION_RANK[education_level] >= EDUCATION_RANK["Bachelor"] and years_experience >= 1)
        or years_experience >= 4
        or merit_score >= 78.0
    )

    hiring_context = choose_hiring_context(hiring_contexts, family_profile_index, config, rng)
    same_city_flag = int(city_name == hiring_context.city_name)
    same_high_school_flag = int(high_school_name == hiring_context.high_school_name)
    same_college_flag = int(
        pd.notna(college_name)
        and pd.notna(hiring_context.college_name)
        and str(college_name) == str(hiring_context.college_name)
    )
    same_last_name_flag = int(last_name == hiring_context.last_name)

    close_family_base = (
        family_profile_index is not None
        and hiring_context.family_profile_index == family_profile_index
        and same_last_name_flag == 1
    )
    close_family_relation_flag = int(
        close_family_base
        and rng.random()
        < (
            0.18
            + 0.20 * same_city_flag
            + 0.12 * same_high_school_flag
            + 0.08 * same_college_flag
            + 0.18 * hiring_context.bias_strength
            + 0.10 * nepotism_adjustment
        )
    )

    if close_family_relation_flag:
        family_link_flag = 1

    referral_probability = (
        config.referral_rate
        + 0.32 * close_family_relation_flag
        + 0.18 * same_high_school_flag
        + 0.14 * same_college_flag
        + 0.05 * same_city_flag
        + 0.04 * same_last_name_flag
    )
    referral_flag = int(rng.random() < min(referral_probability, 0.96))

    observed_nepotism_signal = (
        1.25 * close_family_relation_flag
        + 0.62 * same_high_school_flag
        + 0.30 * same_city_flag
        + 0.20 * same_college_flag
        + 0.12 * same_last_name_flag
        + 0.20 * family_link_flag
        + 0.10 * referral_flag
        + 0.42 * hiring_context.bias_strength
        + 0.24 * nepotism_adjustment
    )
    internal_nepotism_score = clip_round(observed_nepotism_signal * 31.0 + rng.normal(0.0, 3.5), 0.0, 100.0, 2)
    connection_strength = clip_round(
        0.07
        + internal_nepotism_score / 112.0
        + 0.05 * referral_flag
        + 0.03 * family_link_flag
        + rng.normal(0.0, 0.05),
        0.0,
        1.0,
        2,
    )
    discretionary_channel = get_discretionary_channel(
        close_family_relation_flag=close_family_relation_flag,
        same_high_school_flag=same_high_school_flag,
        same_college_flag=same_college_flag,
        same_city_flag=same_city_flag,
        referral_flag=referral_flag,
        same_last_name_flag=same_last_name_flag,
    )

    discretionary_signal = get_discretionary_signal(discretionary_channel)
    discretionary_multiplier = 1.0 + discretionary_signal * (0.55 + 0.25 * config.connection_weight)
    connected_competitive_edge = {
        "Merit-based": 0.03,
        "Moderate favoritism": 0.22,
        "High nepotism risk": 0.42,
    }[config.name] * int(
        connection_strength >= 0.38
        and 66.0 <= merit_score <= 84.0
        and (same_high_school_flag == 1 or same_city_flag == 1 or same_college_flag == 1 or referral_flag == 1)
    )
    connected_override_edge = {
        "Merit-based": 0.00,
        "Moderate favoritism": 0.10,
        "High nepotism risk": 0.24,
    }[config.name] * int(
        connection_strength >= 0.55
        and merit_score < 74.0
        and (close_family_relation_flag == 1 or same_high_school_flag == 1 or discretionary_signal >= 0.24)
    )

    merit_centered = (merit_score - 70.0) / 9.5
    merit_component = (
        1.18 * config.merit_weight * merit_centered
        + 0.24 * max((merit_score - 80.0) / 8.0, 0.0)
        + 0.10 * ((experience_component - 45.0) / 25.0)
        + 0.16 * qualification_match
    )
    nepotism_centered = (internal_nepotism_score - 22.0) / 16.0
    nepotism_component = (
        0.58 * config.connection_weight * nepotism_centered
        + config.family_weight * close_family_relation_flag
        + 0.34 * same_high_school_flag
        + 0.16 * same_city_flag
        + 0.12 * same_college_flag
        + 0.07 * same_last_name_flag
        + 0.12 * family_link_flag
        + 0.42 * config.referral_weight * referral_flag
        + 0.55 * config.discretionary_weight * discretionary_signal
        + 0.12 * hiring_context.bias_strength
        + connected_competitive_edge * (1.0 + 0.55 * discretionary_signal)
        + connected_override_edge * (1.0 + 0.80 * discretionary_signal)
        + 0.28 * nepotism_adjustment
    ) * discretionary_multiplier
    low_merit_connected = int(internal_nepotism_score > 56.0 and merit_score < 63.0)
    internal_connection_dominant = int(nepotism_component > merit_component)

    hire_logit = (
        config.hire_intercept
        + hire_intercept_adjustment
        + merit_component
        + nepotism_component
        + config.negative_selection_bonus * low_merit_connected
        + rng.normal(0.0, 0.12)
    )
    hire_prob = clip_round(float(sigmoid(hire_logit)), 0.001, 0.35, 4)
    hired_flag = int(rng.random() < hire_prob)

    full_name = f"{first_name} {last_name}"

    return {
        "company_id": company_id,
        "scenario": scenario_name,
        "candidate_id": "",
        "first_name": first_name,
        "last_name": last_name,
        "full_name": full_name,
        "gender": gender,
        "city_name": city_name,
        "high_school_name": high_school_name,
        "college_name": college_name,
        "age": age,
        "years_experience": years_experience,
        "education_level": education_level,
        "high_school_gpa": high_school_gpa,
        "college_gpa": college_gpa,
        "interview_score": interview_score,
        "test_score": test_score,
        "referral_flag": referral_flag,
        "family_link_flag": family_link_flag,
        "close_family_relation_flag": close_family_relation_flag,
        "same_high_school_flag": same_high_school_flag,
        "same_city_flag": same_city_flag,
        "same_college_flag": same_college_flag,
        "same_last_name_flag": same_last_name_flag,
        "connection_strength": connection_strength,
        "merit_score": merit_score,
        "discretionary_channel": discretionary_channel,
        "hired_flag": hired_flag,
        "_internal_nepotism_score": internal_nepotism_score,
        "_internal_connection_dominant": internal_connection_dominant,
        "_internal_merit_component": round(float(merit_component), 4),
        "_internal_nepotism_component": round(float(nepotism_component), 4),
        "_internal_hiring_context_id": hiring_context.hiring_context_id,
    }


def assign_role_levels(
    employees: pd.DataFrame,
    config: ScenarioConfig,
    rng: np.random.Generator,
) -> pd.Series:
    # Leadership roles mostly follow experience and merit, with connection effects
    # becoming stronger in favoritism-heavy scenarios.
    base_levels = np.select(
        [
            employees["years_experience"] <= 2,
            employees["years_experience"] <= 6,
            employees["years_experience"] <= 12,
            employees["years_experience"] <= 20,
        ],
        [1, 2, 3, 4],
        default=5,
    ).astype(int)

    leadership_score = (
        employees["merit_score"] * 0.55
        + employees["years_experience"] * 2.3
        + employees["connection_strength"] * 28 * config.management_connection_weight
        + rng.normal(0.0, 2.5, size=len(employees))
    )

    sorted_index = employees.assign(_leadership_score=leadership_score).sort_values(
        "_leadership_score", ascending=False
    ).index.tolist()

    director_count = max(1, min(2, len(employees) // 85 + 1))
    manager_count = max(2, min(max(3, len(employees) // 18), max(len(employees) - director_count, 2)))

    role_levels = pd.Series(base_levels, index=employees.index, dtype="int64")
    director_index = sorted_index[:director_count]
    manager_index = sorted_index[director_count : director_count + manager_count]

    role_levels.loc[director_index] = 5
    role_levels.loc[manager_index] = 4

    non_management_index = [idx for idx in employees.index if idx not in director_index + manager_index]
    role_levels.loc[non_management_index] = np.clip(role_levels.loc[non_management_index], 1, 3)
    return role_levels


def compute_promotion_components(
    row,
    config: ScenarioConfig,
    tenure: int,
    performance: float,
    manager_favoritism_boost: float,
    rng: np.random.Generator,
) -> tuple[float, float, int, float]:
    discretionary_signal = get_discretionary_signal(row.discretionary_channel)
    discretionary_multiplier = 1.0 + discretionary_signal * (
        0.26 + 0.58 * config.promotion_connection_weight
    )
    scenario_connection_boost = 0.82 + 1.10 * config.promotion_connection_weight
    merit_scale = {
        "Merit-based": 1.02,
        "Moderate favoritism": 0.93,
        "High nepotism risk": 0.80,
    }[config.name]

    tenure_years = min(tenure / 12.0, 8.0)
    promotion_merit_component = (
        merit_scale
        * (
            0.082 * (performance - 70.0)
            + 0.020 * min(tenure, 84)
            - 0.32 * (row.role_level - 1)
            + 0.010 * (row.merit_score - 72.0)
        )
        + 0.15 * int(performance >= 80.0)
        + 0.08 * int(tenure >= 24)
        - 0.12 * int(performance < 62.0)
        + rng.normal(0.0, 0.10)
    )
    weak_connection_proxy = max(row.same_last_name_flag, row.referral_flag)
    connected_promotion_edge = {
        "Merit-based": 0.02,
        "Moderate favoritism": 0.18,
        "High nepotism risk": 0.25,
    }[config.name] * int(
        (row.connection_strength >= 0.40 or row.same_high_school_flag == 1 or row.close_family_relation_flag == 1)
        and 66.0 <= performance <= 80.0
        and tenure >= 12
    )
    low_performance_override = {
        "Merit-based": 0.00,
        "Moderate favoritism": 0.10,
        "High nepotism risk": 0.15,
    }[config.name] * int(
        (row.connection_strength >= 0.55 or row.close_family_relation_flag == 1 or row.same_high_school_flag == 1)
        and performance < 74.0
        and tenure >= 6
    )
    promotion_nepotism_component = (
        1.30 * config.promotion_connection_weight * row.close_family_relation_flag
        + 0.86 * config.promotion_connection_weight * row.same_high_school_flag
        + 0.54 * config.promotion_connection_weight * row.same_city_flag
        + 0.34 * config.promotion_connection_weight * row.same_college_flag
        + 0.12 * config.promotion_connection_weight * weak_connection_proxy
        + 0.14 * config.promotion_connection_weight * row.family_link_flag
        + 0.84 * config.promotion_connection_weight * row.connection_strength
        + 0.30 * config.promotion_connection_weight * discretionary_signal
        + manager_favoritism_boost
        + 0.20 * config.promotion_connection_weight * int(
            row.connection_strength >= 0.55 and performance <= 77.0
        )
        + 0.10 * config.promotion_connection_weight * int(
            row.same_high_school_flag == 1 and row.role_level >= 3
        )
        + connected_promotion_edge * (1.0 + 0.65 * discretionary_signal)
        + low_performance_override * (1.0 + 0.85 * discretionary_signal)
    ) * discretionary_multiplier * scenario_connection_boost

    promotion_connection_dominant = int(promotion_nepotism_component > promotion_merit_component)
    promotion_logit = (
        -2.95
        + promotion_merit_component
        + promotion_nepotism_component
        + 0.10 * int(performance >= 78.0)
        - 0.08 * int(tenure < 12)
    )
    return (
        round(float(promotion_merit_component), 4),
        round(float(promotion_nepotism_component), 4),
        promotion_connection_dominant,
        float(promotion_logit),
    )


def generate_employee_records(
    hired_candidates: pd.DataFrame,
    config: ScenarioConfig,
    rng: np.random.Generator,
    fake: Faker,
    promotion_intercept_adjustment: float = 0.0,
) -> pd.DataFrame:
    if hired_candidates.empty:
        return pd.DataFrame(columns=EMPLOYEE_COLUMNS)

    employees = hired_candidates.copy().reset_index(drop=True)
    employees["employee_id"] = employees["candidate_id"].astype(str).str.replace("C", "E", n=1, regex=False)
    employees["role_level"] = assign_role_levels(employees, config, rng)

    departments = [
        f"D{dept_id:02d}" for dept_id in range(1, min(7, max(4, len(employees) // 45 + 4)) + 1)
    ]
    employees["department_id"] = pd.NA
    employees["manager_id"] = pd.NA

    directors = employees[employees["role_level"] == 5].index.tolist()
    managers = employees[employees["role_level"] == 4].index.tolist()
    if not managers and directors:
        managers = directors[:1]

    for idx, manager_idx in enumerate(managers):
        employees.at[manager_idx, "department_id"] = departments[idx % len(departments)]

    for idx, director_idx in enumerate(directors):
        employees.at[director_idx, "department_id"] = "EXEC"
        if idx > 0:
            employees.at[director_idx, "manager_id"] = employees.at[directors[0], "employee_id"]

    if directors:
        for idx, manager_idx in enumerate(managers):
            assigned_director = directors[idx % len(directors)]
            employees.at[manager_idx, "manager_id"] = employees.at[assigned_director, "employee_id"]

    favored_managers = []
    if managers:
        manager_frame = employees.loc[managers, ["employee_id", "department_id", "connection_strength"]].copy()
        favored_count = max(1, len(managers) // 3)
        favored_managers = (
            manager_frame.sort_values("connection_strength", ascending=False).head(favored_count).index.tolist()
        )
    favored_manager_ids = {
        str(employees.at[manager_idx, "employee_id"])
        for manager_idx in favored_managers
        if pd.notna(employees.at[manager_idx, "employee_id"])
    }

    non_managers = employees[employees["role_level"] <= 3].index.tolist()
    for employee_idx in non_managers:
        # In the highest-risk setting, connected hires are intentionally clustered
        # under a small set of well-connected managers.
        connected = (
            employees.at[employee_idx, "connection_strength"] > 0.60
            or employees.at[employee_idx, "close_family_relation_flag"] == 1
            or employees.at[employee_idx, "same_high_school_flag"] == 1
        )
        if connected and favored_managers and rng.random() < config.cluster_bias:
            chosen_manager = int(rng.choice(favored_managers))
        else:
            chosen_manager = int(rng.choice(managers))

        employees.at[employee_idx, "manager_id"] = employees.at[chosen_manager, "employee_id"]
        employees.at[employee_idx, "department_id"] = employees.at[chosen_manager, "department_id"]

    role_tenure_center = {1: 14, 2: 26, 3: 44, 4: 58, 5: 74}
    role_salary_base = {1: 9800, 2: 14500, 3: 21500, 4: 31500, 5: 47000}

    tenure_months = []
    performance_scores = []
    salaries = []
    promoted_flags = []
    months_to_promotion = []
    promotion_merit_components = []
    promotion_nepotism_components = []
    promotion_connection_dominant_flags = []

    for row in employees.itertuples(index=False):
        max_tenure = max(3, min(int(row.years_experience * 12 + 18), 120))
        center = min(role_tenure_center[row.role_level], max_tenure)
        tenure = int(np.clip(np.round(rng.normal(center, max(6, center * 0.30))), 1, max_tenure))

        performance = (
            56.0
            + 0.42 * (row.merit_score - 60.0)
            + 0.09 * min(tenure, 72)
            + {
                "Merit-based": 1.8,
                "Moderate favoritism": -0.5,
                "High nepotism risk": -3.4,
            }[config.name]
            + rng.normal(0.0, 5.5)
        )
        if config.name == "High nepotism risk" and row.connection_strength > 0.70 and row.merit_score < 63.0:
            performance -= rng.uniform(6.0, 12.0)
        elif config.name == "Moderate favoritism" and row.connection_strength > 0.65 and row.merit_score < 66.0:
            performance -= rng.uniform(2.0, 5.0)
        performance = clip_round(performance, 38.0, 98.0)

        salary_multiplier = (
            1.0
            + (performance - 72.0) / 230.0
            + min(tenure, 84) / 900.0
            + row.connection_strength * config.salary_connection_premium
        )
        salary = int(round(role_salary_base[row.role_level] * salary_multiplier / 100.0) * 100)

        if row.role_level >= 5:
            promotion_prob = 0.0
            promoted_flag = 0
            promotion_months = np.nan
            promotion_merit_component = 0.0
            promotion_nepotism_component = 0.0
            promotion_connection_dominant = 0
        else:
            manager_favoritism_boost = 0.0
            if pd.notna(row.manager_id) and str(row.manager_id) in favored_manager_ids:
                manager_favoritism_boost = (
                    (0.12 + 0.18 * config.management_connection_weight)
                    * (
                        0.35
                        + 0.65
                        * int(
                            row.connection_strength >= 0.38
                            or row.close_family_relation_flag == 1
                            or row.same_high_school_flag == 1
                        )
                    )
                )
            (
                promotion_merit_component,
                promotion_nepotism_component,
                promotion_connection_dominant,
                promotion_logit,
            ) = compute_promotion_components(
                row=row,
                config=config,
                tenure=tenure,
                performance=performance,
                manager_favoritism_boost=manager_favoritism_boost,
                rng=rng,
            )
            promotion_prob = clip_round(
                float(sigmoid(promotion_logit + promotion_intercept_adjustment)),
                0.001,
                0.45,
                4,
            )
            promoted_flag = int(rng.random() < promotion_prob)
            promotion_months = int(rng.integers(6, tenure + 1)) if promoted_flag and tenure >= 6 else np.nan

        tenure_months.append(tenure)
        performance_scores.append(performance)
        salaries.append(salary)
        promoted_flags.append(promoted_flag)
        months_to_promotion.append(promotion_months)
        promotion_merit_components.append(promotion_merit_component)
        promotion_nepotism_components.append(promotion_nepotism_component)
        promotion_connection_dominant_flags.append(promotion_connection_dominant)

    employees["tenure_months"] = tenure_months
    employees["performance_score"] = performance_scores
    employees["salary"] = salaries
    employees["promoted_flag"] = promoted_flags
    employees["months_to_promotion"] = months_to_promotion
    employees["_internal_promotion_merit_component"] = promotion_merit_components
    employees["_internal_promotion_nepotism_component"] = promotion_nepotism_components
    employees["_internal_promotion_connection_dominant"] = promotion_connection_dominant_flags

    employee_frame = employees[
        [
            "company_id",
            "scenario",
            "employee_id",
            "candidate_id",
            "first_name",
            "last_name",
            "full_name",
            "gender",
            "city_name",
            "high_school_name",
            "college_name",
            "department_id",
            "manager_id",
            "role_level",
            "age",
            "years_experience",
            "education_level",
            "high_school_gpa",
            "college_gpa",
            "referral_flag",
            "family_link_flag",
            "close_family_relation_flag",
            "same_high_school_flag",
            "same_city_flag",
            "same_college_flag",
            "same_last_name_flag",
            "connection_strength",
            "merit_score",
            "discretionary_channel",
            "tenure_months",
            "performance_score",
            "salary",
            "promoted_flag",
            "months_to_promotion",
            "_internal_promotion_merit_component",
            "_internal_promotion_nepotism_component",
            "_internal_promotion_connection_dominant",
        ]
    ].copy()

    return employee_frame


def generate_scenario_data(
    config: ScenarioConfig,
    rng: np.random.Generator,
    fake: Faker,
    nepotism_adjustment: float = 0.0,
    hire_intercept_adjustment: float = 0.0,
    promotion_intercept_adjustment: float = 0.0,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    candidate_frames = []
    employee_frames = []

    for company_number in range(1, config.companies + 1):
        company_id = f"{config.company_prefix}-{company_number:02d}"
        company_city = generate_city(rng)
        company_size = int(
            np.clip(
                np.round(rng.normal(config.base_candidates_per_company, 18)),
                config.base_candidates_per_company - 20,
                config.base_candidates_per_company + 30,
            )
        )

        family_profiles = build_family_profiles(company_size, company_city, config, rng)
        hiring_contexts = build_hiring_contexts(company_id, company_city, config, family_profiles, rng)
        candidate_rows = [
            generate_candidate(
                company_id=company_id,
                company_city=company_city,
                scenario_name=config.name,
                config=config,
                family_profiles=family_profiles,
                hiring_contexts=hiring_contexts,
                nepotism_adjustment=nepotism_adjustment,
                hire_intercept_adjustment=hire_intercept_adjustment,
                rng=rng,
                fake=fake,
            )
            for _ in range(company_size)
        ]

        candidate_frame = pd.DataFrame(candidate_rows)
        candidate_frame["candidate_id"] = [
            f"C{config.company_prefix}{company_number:02d}{candidate_number:05d}"
            for candidate_number in range(1, len(candidate_frame) + 1)
        ]
        candidate_frames.append(candidate_frame)

        hired_candidates = candidate_frame[candidate_frame["hired_flag"] == 1].copy()
        employee_frame = generate_employee_records(
            hired_candidates,
            config,
            rng,
            fake,
            promotion_intercept_adjustment=promotion_intercept_adjustment,
        )
        employee_frames.append(employee_frame)

    return (
        pd.concat(candidate_frames, ignore_index=True),
        pd.concat(employee_frames, ignore_index=True),
    )


def get_connection_dominant_share(candidates: pd.DataFrame) -> float:
    hired_candidates = candidates[candidates["hired_flag"] == 1]
    if hired_candidates.empty:
        return 0.0
    return float(hired_candidates["_internal_connection_dominant"].mean())


def get_hire_rate(candidates: pd.DataFrame) -> float:
    if candidates.empty:
        return 0.0
    return float(candidates["hired_flag"].mean())


def get_promotion_rate(employees: pd.DataFrame) -> float:
    if employees.empty:
        return 0.0
    return float(employees["promoted_flag"].mean())


def get_promoted_connection_dominant_share(employees: pd.DataFrame) -> float:
    promoted_employees = employees[employees["promoted_flag"] == 1]
    if promoted_employees.empty:
        return 0.0
    return float(promoted_employees["_internal_promotion_connection_dominant"].mean())


def build_fake(seed: int) -> Faker:
    fake = Faker("he_IL")
    fake.seed_instance(seed)
    return fake


def calibrate_scenario_generation(
    config: ScenarioConfig,
    scenario_seed: int,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    if config.name == "High nepotism risk":
        adjustment_grid = [0.04, 0.08, 0.12, 0.16, 0.20]
        intercept_grid = [-0.40, -0.10, 0.15, 0.40]
        promotion_intercept_grid = [-1.10, -0.80, -0.50]
    elif config.name == "Moderate favoritism":
        adjustment_grid = [-0.06, -0.02, 0.02]
        intercept_grid = [-0.55, -0.25, 0.05]
        promotion_intercept_grid = [-1.15, -0.90, -0.65]
    else:
        adjustment_grid = [-0.04, 0.0]
        intercept_grid = [-0.65, -0.40, -0.15]
        promotion_intercept_grid = [-0.95, -0.75, -0.55]
    best_result: tuple[float, float, float, float, pd.DataFrame, pd.DataFrame] | None = None
    best_score = float("inf")

    for intercept_index, intercept_adjustment in enumerate(intercept_grid):
        for adjustment_index, nepotism_adjustment in enumerate(adjustment_grid):
            for promotion_index, promotion_adjustment in enumerate(promotion_intercept_grid):
                local_seed = scenario_seed + intercept_index * 100 + adjustment_index * 7 + promotion_index * 13
                local_rng = np.random.default_rng(local_seed)
                local_fake = build_fake(local_seed)
                candidates, employees = generate_scenario_data(
                    config=config,
                    rng=local_rng,
                    fake=local_fake,
                    nepotism_adjustment=nepotism_adjustment,
                    hire_intercept_adjustment=intercept_adjustment,
                    promotion_intercept_adjustment=promotion_adjustment,
                )

                hire_rate = get_hire_rate(candidates)
                hire_dominance_share = get_connection_dominant_share(candidates)
                promotion_rate = get_promotion_rate(employees)
                promotion_dominance_share = get_promoted_connection_dominant_share(employees)
                promotion_count = int(employees["promoted_flag"].sum()) if not employees.empty else 0

                hire_dom_weight = {
                    "Merit-based": 0.35,
                    "Moderate favoritism": 0.70,
                    "High nepotism risk": 1.50,
                }[config.name]
                promotion_dom_weight = {
                    "Merit-based": 0.35,
                    "Moderate favoritism": 0.75,
                    "High nepotism risk": 1.10,
                }[config.name]

                score = abs(hire_rate - config.target_hire_rate) + 0.90 * abs(
                    promotion_rate - config.target_promotion_rate
                )
                score += hire_dom_weight * abs(
                    hire_dominance_share - config.nepotism_dominance_target
                )
                score += promotion_dom_weight * abs(
                    promotion_dominance_share - config.promotion_dominance_target
                )

                if config.name == "High nepotism risk":
                    if hire_dominance_share < 0.50 or hire_dominance_share > 0.62:
                        score += 0.22
                    if promotion_dominance_share < 0.55 or promotion_dominance_share > 0.90:
                        score += 0.22
                    if len(employees) < 34:
                        score += 0.16
                elif config.name == "Moderate favoritism":
                    if hire_dominance_share < 0.18 or hire_dominance_share > 0.48:
                        score += 0.12
                    if promotion_dominance_share < 0.22 or promotion_dominance_share > 0.50:
                        score += 0.12
                else:
                    if hire_dominance_share > 0.22:
                        score += 0.12
                    if promotion_dominance_share > 0.28:
                        score += 0.12

                if not 0.01 <= hire_rate <= 0.05:
                    score += 0.50 + abs(hire_rate - config.target_hire_rate)
                if not 0.05 <= promotion_rate <= 0.12:
                    score += 0.40 + abs(promotion_rate - config.target_promotion_rate)
                if promotion_count < 3:
                    score += 0.18

                if score < best_score:
                    best_score = score
                    best_result = (
                        hire_rate,
                        hire_dominance_share,
                        promotion_rate,
                        promotion_dominance_share,
                        candidates,
                        employees,
                    )

    if best_result is None:
        raise RuntimeError(f"Calibration failed for {config.name}.")

    hire_rate, dominance_share, promotion_rate, promotion_dominance_share, candidates, employees = best_result
    if not 0.01 <= hire_rate <= 0.05:
        raise RuntimeError(
            f"Failed to calibrate {config.name} to a 1%-5% hire rate. Observed {hire_rate:.3f}."
        )
    if not 0.05 <= promotion_rate <= 0.12:
        raise RuntimeError(
            f"Failed to calibrate {config.name} to a 5%-12% promotion rate. Observed {promotion_rate:.3f}."
        )
    if config.name == "High nepotism risk" and not 0.50 <= dominance_share <= 0.62:
        raise RuntimeError(
            f"Failed to calibrate {config.name} to about 52% connection-dominant hires. "
            f"Observed {dominance_share:.3f}."
        )
    if config.name == "High nepotism risk" and not 0.55 <= promotion_dominance_share <= 0.90:
        raise RuntimeError(
            f"Failed to calibrate {config.name} to clearly connection-driven but not absolute promotions. "
            f"Observed {promotion_dominance_share:.3f}."
        )

    return candidates, employees


def export_to_excel(candidates: pd.DataFrame, employees: pd.DataFrame, output_path: Path) -> None:
    output_path.parent.mkdir(parents=True, exist_ok=True)

    with pd.ExcelWriter(output_path, engine="openpyxl") as writer:
        candidates.to_excel(writer, sheet_name="candidates", index=False)
        employees.to_excel(writer, sheet_name="employees", index=False)


def print_summary(candidates: pd.DataFrame, employees: pd.DataFrame, output_path: Path) -> None:
    candidate_counts = candidates["scenario"].value_counts().reindex(
        [config.name for config in SCENARIOS], fill_value=0
    )
    employee_counts = employees["scenario"].value_counts().reindex(
        [config.name for config in SCENARIOS], fill_value=0
    )
    top_surnames = candidates["last_name"].value_counts().head(10)
    top_cities = candidates["city_name"].value_counts().head(10)
    hired_candidates = candidates[candidates["hired_flag"] == 1].copy()
    hire_counts = hired_candidates["scenario"].value_counts().reindex(
        [config.name for config in SCENARIOS], fill_value=0
    )
    hire_rates = candidates.groupby("scenario")["hired_flag"].mean().reindex(
        [config.name for config in SCENARIOS], fill_value=0.0
    )
    promotion_rates = employees.groupby("scenario")["promoted_flag"].mean().reindex(
        [config.name for config in SCENARIOS], fill_value=0.0
    )
    connection_dominant_share = (
        hired_candidates.groupby("scenario")["_internal_connection_dominant"].mean().reindex(
            [config.name for config in SCENARIOS], fill_value=0.0
        )
    )
    promoted_employees = employees[employees["promoted_flag"] == 1].copy()
    promoted_connection_dominant_share = (
        promoted_employees.groupby("scenario")["_internal_promotion_connection_dominant"].mean().reindex(
            [config.name for config in SCENARIOS], fill_value=0.0
        )
    )
    avg_hired_merit = hired_candidates.groupby("scenario")["merit_score"].mean().reindex(
        [config.name for config in SCENARIOS], fill_value=0.0
    )
    avg_promoted_merit = promoted_employees.groupby("scenario")["merit_score"].mean().reindex(
        [config.name for config in SCENARIOS], fill_value=0.0
    )
    avg_hired_nepotism = hired_candidates.groupby("scenario")["_internal_nepotism_score"].mean().reindex(
        [config.name for config in SCENARIOS], fill_value=0.0
    )
    avg_hidden_merit_component = hired_candidates.groupby("scenario")["_internal_merit_component"].mean().reindex(
        [config.name for config in SCENARIOS], fill_value=0.0
    )
    avg_hidden_nepotism_component = (
        hired_candidates.groupby("scenario")["_internal_nepotism_component"].mean().reindex(
            [config.name for config in SCENARIOS], fill_value=0.0
        )
    )

    print(f"output file path: {output_path.as_posix()}")
    print(f"total candidate rows: {len(candidates)}")
    print(f"total employee rows: {len(employees)}")
    print("candidate rows per scenario:")
    for scenario_name, count in candidate_counts.items():
        print(f"  - {scenario_name}: {count}")
    print("employee rows per scenario:")
    for scenario_name, count in employee_counts.items():
        print(f"  - {scenario_name}: {count}")
    print("top 10 most common surnames in the generated data:")
    for surname, count in top_surnames.items():
        print(f"  - {surname}: {count}")
    print("top 10 most common cities in the generated data:")
    for city, count in top_cities.items():
        print(f"  - {city}: {count}")
    print("total hires by scenario:")
    for scenario_name, count in hire_counts.items():
        print(f"  - {scenario_name}: {count}")
    print("total hire rate by scenario:")
    for scenario_name, value in hire_rates.items():
        print(f"  - {scenario_name}: {value:.3%}")
    print("total promotion rate by scenario:")
    for scenario_name, value in promotion_rates.items():
        print(f"  - {scenario_name}: {value:.3%}")
    print("average hidden merit component among hired candidates by scenario:")
    for scenario_name, value in avg_hidden_merit_component.items():
        print(f"  - {scenario_name}: {value:.3f}")
    print("average hidden nepotism component among hired candidates by scenario:")
    for scenario_name, value in avg_hidden_nepotism_component.items():
        print(f"  - {scenario_name}: {value:.3f}")
    print("share of hired candidates that are connection-dominant by scenario:")
    for scenario_name, share in connection_dominant_share.items():
        print(f"  - {scenario_name}: {share:.3f}")
    print("share of promoted employees that are connection-dominant internally by scenario:")
    for scenario_name, share in promoted_connection_dominant_share.items():
        print(f"  - {scenario_name}: {share:.3f}")
    print("average merit score of hired candidates by scenario:")
    for scenario_name, value in avg_hired_merit.items():
        print(f"  - {scenario_name}: {value:.2f}")
    print("average merit score of promoted employees by scenario:")
    for scenario_name, value in avg_promoted_merit.items():
        print(f"  - {scenario_name}: {value:.2f}")
    print("average nepotism score of hired candidates by scenario:")
    for scenario_name, value in avg_hired_nepotism.items():
        print(f"  - {scenario_name}: {value:.2f}")


def main() -> None:
    rng = np.random.default_rng(SEED)
    Faker.seed(SEED)

    candidate_frames = []
    employee_frames = []

    for scenario_index, scenario in enumerate(SCENARIOS, start=1):
        scenario_seed = SEED + scenario_index * 1000 + int(rng.integers(0, 500))
        candidates, employees = calibrate_scenario_generation(scenario, scenario_seed)
        candidate_frames.append(candidates)
        employee_frames.append(employees)

    candidates_internal_df = pd.concat(candidate_frames, ignore_index=True)
    employees_internal_df = pd.concat(employee_frames, ignore_index=True)
    candidates_df = candidates_internal_df[CANDIDATE_COLUMNS]
    employees_df = employees_internal_df[EMPLOYEE_COLUMNS]

    export_to_excel(candidates_df, employees_df, OUTPUT_PATH)
    print_summary(candidates_internal_df, employees_internal_df, OUTPUT_PATH)


if __name__ == "__main__":
    main()
