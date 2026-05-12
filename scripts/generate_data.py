"""
Generate a realistic synthetic loan dataset with clear decision patterns.
1000 rows with meaningful feature-target relationships so XGBoost
learns proper approval logic.
"""
import random
import csv
import os

random.seed(42)

ROWS = 1000
OUTPUT = os.path.join(os.path.dirname(__file__), "..", "data", "loan_data.csv")

AREAS = ["rural", "suburban", "urban"]
EDUCATIONS = ["high_school", "graduate", "postgraduate"]

def generate_row():
    age = random.randint(20, 60)
    education = random.choice(EDUCATIONS)
    self_employed = random.choice([0, 1])
    area_type = random.choice(AREAS)

    # Income correlated with age/education
    base_income = 15000 + age * 800
    if education == "graduate":
        base_income *= 1.3
    elif education == "postgraduate":
        base_income *= 1.7
    if area_type == "urban":
        base_income *= 1.2
    income = int(base_income * random.uniform(0.6, 1.5))

    credit_score = random.randint(300, 850)
    employment_years = max(0, age - 20 - random.randint(0, 10))
    loan_amount = random.randint(50000, 600000)
    property_value = int(loan_amount * random.uniform(1.0, 3.0))
    debt_to_income = round(random.uniform(0.05, 0.85), 2)

    # --- Approval logic (realistic rules with noise) ---
    score = 0.0

    # Credit score is the strongest signal
    if credit_score >= 750:
        score += 3.0
    elif credit_score >= 700:
        score += 2.0
    elif credit_score >= 650:
        score += 1.0
    elif credit_score >= 600:
        score += 0.0
    elif credit_score >= 500:
        score -= 1.5
    else:
        score -= 3.0  # Very bad credit = strong reject

    # Income vs loan amount ratio
    income_ratio = income / max(loan_amount, 1)
    if income_ratio > 0.5:
        score += 2.0
    elif income_ratio > 0.3:
        score += 1.0
    elif income_ratio > 0.15:
        score += 0.0
    else:
        score -= 1.5

    # Debt-to-income
    if debt_to_income < 0.2:
        score += 1.5
    elif debt_to_income < 0.35:
        score += 0.5
    elif debt_to_income < 0.5:
        score -= 0.5
    else:
        score -= 2.0

    # Employment stability
    if employment_years >= 10:
        score += 1.0
    elif employment_years >= 5:
        score += 0.5
    elif employment_years >= 2:
        score += 0.0
    else:
        score -= 1.0

    # Education
    if education == "postgraduate":
        score += 0.5
    elif education == "high_school":
        score -= 0.5

    # Property value as collateral
    if property_value > loan_amount * 2:
        score += 0.5
    elif property_value < loan_amount:
        score -= 0.5

    # Self employed slight penalty
    if self_employed:
        score -= 0.3

    # Add noise
    score += random.gauss(0, 0.8)

    # Threshold
    approved = 1 if score > 0.5 else 0

    return [
        approved, age, income, loan_amount, credit_score, employment_years,
        debt_to_income, area_type, education, self_employed, property_value,
    ]


rows = [generate_row() for _ in range(ROWS)]

# Check balance
approved_count = sum(r[0] for r in rows)
print(f"Generated {ROWS} rows: {approved_count} approved ({approved_count*100//ROWS}%), "
      f"{ROWS - approved_count} rejected ({(ROWS-approved_count)*100//ROWS}%)")

header = [
    "approved", "age", "income", "loan_amount", "credit_score",
    "employment_years", "debt_to_income", "area_type", "education",
    "self_employed", "property_value",
]

with open(OUTPUT, "w", newline="") as f:
    writer = csv.writer(f)
    writer.writerow(header)
    writer.writerows(rows)

print(f"✓ Saved to {OUTPUT}")
