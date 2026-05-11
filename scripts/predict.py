"""
Step 4: Predict — Send a loan application to the endpoint.

REFRESHER — How Real-Time Inference Works:
  1. You POST a CSV row (or JSON) to the endpoint via invoke_endpoint()
  2. The container deserializes it, feeds it to the XGBoost model
  3. Model outputs a probability (0.0 to 1.0)
  4. If probability > 0.5 → Approved, else → Rejected

  The features MUST be in the same order as training:
  age, income, loan_amount, credit_score, employment_years,
  debt_to_income, area_type(encoded), education(encoded),
  self_employed, property_value
"""
import json
import sys
from config import get_session, ENDPOINT_NAME


# Feature encoding maps (same as training)
AREA_MAP = {"rural": 0, "suburban": 1, "urban": 2}
EDU_MAP = {"high_school": 0, "graduate": 1, "postgraduate": 2}


def predict(application: dict) -> dict:
    """
    Send a loan application to the SageMaker endpoint.

    Args:
        application: dict with keys:
            age, income, loan_amount, credit_score, employment_years,
            debt_to_income, area_type, education, self_employed, property_value

    Returns:
        dict with: approved (bool), probability (float), raw_score (float)
    """
    session = get_session()
    runtime = session.client("sagemaker-runtime")

    # Encode categoricals
    area = AREA_MAP.get(application.get("area_type", ""), application.get("area_type", 0))
    edu = EDU_MAP.get(application.get("education", ""), application.get("education", 0))

    # Build CSV row (same order as training features)
    row = ",".join(str(x) for x in [
        application["age"],
        application["income"],
        application["loan_amount"],
        application["credit_score"],
        application["employment_years"],
        application["debt_to_income"],
        area,
        edu,
        application["self_employed"],
        application["property_value"],
    ])

    # Invoke endpoint
    response = runtime.invoke_endpoint(
        EndpointName=ENDPOINT_NAME,
        ContentType="text/csv",
        Body=row,
    )

    score = float(response["Body"].read().decode("utf-8").strip())
    approved = score > 0.5

    return {
        "approved": approved,
        "probability": round(score, 4),
        "decision": "APPROVED ✓" if approved else "REJECTED ✗",
    }


def main():
    # ── Example: Likely to be approved ────────────────────────
    good_applicant = {
        "age": 35,
        "income": 85000,
        "loan_amount": 200000,
        "credit_score": 720,
        "employment_years": 10,
        "debt_to_income": 0.25,
        "area_type": "urban",
        "education": "graduate",
        "self_employed": 0,
        "property_value": 350000,
    }

    # ── Example: Likely to be rejected ────────────────────────
    risky_applicant = {
        "age": 22,
        "income": 18000,
        "loan_amount": 150000,
        "credit_score": 520,
        "employment_years": 0,
        "debt_to_income": 0.70,
        "area_type": "rural",
        "education": "high_school",
        "self_employed": 1,
        "property_value": 80000,
    }

    print("=" * 55)
    print("  LOAN PREDICTION — SageMaker Endpoint")
    print("=" * 55)

    for label, app in [("Good Applicant", good_applicant), ("Risky Applicant", risky_applicant)]:
        print(f"\n── {label} ──")
        for k, v in app.items():
            print(f"  {k:20s}: {v}")
        result = predict(app)
        print(f"\n  → Decision:    {result['decision']}")
        print(f"  → Probability: {result['probability']}")

    print("\n" + "=" * 55)


if __name__ == "__main__":
    main()
