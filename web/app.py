"""
Loan Prediction Demo — Flask Web App
Provides a beautiful UI to submit loan applications and see predictions.
"""
import os
import sys
import json
from flask import Flask, request, jsonify, render_template

# Add scripts dir to path for config import
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "scripts"))
from config import get_session, ENDPOINT_NAME, REGION

app = Flask(__name__, template_folder="templates", static_folder="static")

AREA_MAP = {"rural": 0, "suburban": 1, "urban": 2}
EDU_MAP = {"high_school": 0, "graduate": 1, "postgraduate": 2}


@app.route("/")
def index():
    return render_template("index.html")


@app.route("/predict", methods=["POST"])
def predict():
    try:
        data = request.get_json()

        # Validate & extract fields
        age = int(data["age"])
        income = float(data["income"])
        loan_amount = float(data["loan_amount"])
        credit_score = int(data["credit_score"])
        employment_years = int(data["employment_years"])
        debt_to_income = float(data["debt_to_income"])
        area_type = AREA_MAP[data["area_type"]]
        education = EDU_MAP[data["education"]]
        self_employed = int(data["self_employed"])
        property_value = float(data["property_value"])

        # Build CSV row
        row = ",".join(str(x) for x in [
            age, income, loan_amount, credit_score, employment_years,
            debt_to_income, area_type, education, self_employed, property_value,
        ])

        # Call SageMaker
        session = get_session()
        runtime = session.client("sagemaker-runtime")
        response = runtime.invoke_endpoint(
            EndpointName=ENDPOINT_NAME,
            ContentType="text/csv",
            Body=row,
        )
        score = float(response["Body"].read().decode("utf-8").strip())
        approved = score > 0.5

        return jsonify({
            "approved": approved,
            "probability": round(score * 100, 2),
            "decision": "APPROVED" if approved else "REJECTED",
            "risk_level": "Low" if score > 0.7 else "Medium" if score > 0.5 else "High",
            "details": {
                "age": age,
                "income": income,
                "loan_amount": loan_amount,
                "credit_score": credit_score,
                "employment_years": employment_years,
                "debt_to_income": round(debt_to_income * 100, 1),
                "area_type": data["area_type"],
                "education": data["education"],
                "self_employed": bool(self_employed),
                "property_value": property_value,
            }
        })
    except KeyError as e:
        return jsonify({"error": f"Missing field: {e}"}), 400
    except Exception as e:
        return jsonify({"error": str(e)}), 500


@app.route("/health")
def health():
    return jsonify({"status": "ok", "endpoint": ENDPOINT_NAME, "region": REGION})


if __name__ == "__main__":
    app.run(host="0.0.0.0", port=5050, debug=True)
