from flask import Flask, render_template, request, jsonify
import joblib
import pandas as pd
import os
from datetime import datetime
# Attempt to import flask_cors; allow app to run without it
try:
    from flask_cors import CORS
    CORS_AVAILABLE = True
except Exception:
    CORS_AVAILABLE = False

app = Flask(__name__)
if CORS_AVAILABLE:
    CORS(app)

# Load trained model (gracefully)
MODEL_PATH = "best_bike_price_model.pkl"
model = None
model_loaded = False
load_error = None
if os.path.exists(MODEL_PATH):
    try:
        model = joblib.load(MODEL_PATH)
        model_loaded = True
    except Exception as e:
        load_error = str(e)
        model = None
        model_loaded = False
else:
    load_error = f"Model file not found at {MODEL_PATH}. Run the training notebook to create it."

CURRENT_YEAR = datetime.now().year


def apply_heuristics(base_pred, owner=None, seller_type=None, model_name=None, km_driven=None, apply_adjustments=False):
    """Apply simple heuristic multipliers to approximate OLX-like behavior.
    These are configurable, lightweight adjustments (NOT a replacement for retraining).
    Returns: adjusted_prediction, breakdown(dict)
    """
    breakdown = {"base": float(base_pred)}

    if not apply_adjustments:
        return float(base_pred), breakdown

    # Base multipliers
    owner_map = {
        '1st owner': 1.05,
        '2nd owner': 0.98,
        '3rd owner': 0.94,
        '4th owner': 0.90
    }
    seller_map = {
        'individual': 1.00,
        'dealer': 1.03,
        'trustmark dealer': 1.05
    }
    model_map = {
        'royal': 1.15,     # Royal Enfield-like premium
        'honda': 1.06,
        'yamaha': 1.05,
        'bajaj': 1.00,
        'hero': 1.00,
        'suzuki': 1.04
    }

    multiplier = 1.0

    if owner:
        owner_key = owner.lower()
        mult = owner_map.get(owner_key, 1.0)
        multiplier *= mult
        breakdown['owner_multiplier'] = mult

    if seller_type:
        seller_key = seller_type.lower()
        mult = seller_map.get(seller_key, 1.0)
        multiplier *= mult
        breakdown['seller_multiplier'] = mult

    if km_driven is not None:
        # km-driven adjustment: lower price when higher kms
        if km_driven > 50000:
            mult = 0.88
        elif km_driven > 30000:
            mult = 0.95
        elif km_driven < 15000:
            mult = 1.03
        else:
            mult = 1.0
        multiplier *= mult
        breakdown['km_multiplier'] = mult

    if model_name:
        lowered = model_name.lower()
        found = False
        for key, val in model_map.items():
            if key in lowered:
                multiplier *= val
                breakdown['model_multiplier'] = val
                found = True
                break
        if not found:
            breakdown['model_multiplier'] = 1.0

    adjusted = float(base_pred * multiplier)
    breakdown['adjusted'] = adjusted
    breakdown['total_multiplier'] = multiplier

    return adjusted, breakdown


@app.route("/", methods=["GET", "POST"])
def index():
    prediction = None
    adjusted_prediction = None
    breakdown = None
    error = None

    # Quick check: if model not loaded, show friendly message
    if not model_loaded:
        if request.method == "POST":
            error = f"Model not loaded: {load_error}"
        return render_template("index.html", prediction=None, adjusted_prediction=None, breakdown=None, error=error)

    if request.method == "POST":
        try:
            # read inputs from form
            year = int(request.form.get("year"))
            km_driven = float(request.form.get("km_driven"))
            ex_showroom_price = float(request.form.get("ex_showroom_price"))

            # optional fields for OLX-like adjustments
            owner = request.form.get("owner") or None
            seller_type = request.form.get("seller_type") or None
            model_name = request.form.get("model_name") or None
            apply_adjustments = request.form.get("apply_adjustments") == 'on'

            # Basic validation
            if year < 1900 or year > CURRENT_YEAR:
                raise ValueError("Please enter a valid manufacturing year.")
            if km_driven < 0:
                raise ValueError("Kilometers driven cannot be negative.")
            if ex_showroom_price <= 0:
                raise ValueError("Ex-showroom price must be positive.")

            # Compute age (match training features)
            age = CURRENT_YEAR - year

            # Build input DataFrame with same feature order used in training
            X = pd.DataFrame([[year, km_driven, ex_showroom_price, age]],
                             columns=["year", "km_driven", "ex_showroom_price", "age"])

            pred = model.predict(X)[0]
            prediction = float(pred)

            # Apply heuristic adjustments if requested
            adjusted_prediction, breakdown = apply_heuristics(prediction, owner=owner,
                                                              seller_type=seller_type,
                                                              model_name=model_name,
                                                              km_driven=km_driven,
                                                              apply_adjustments=apply_adjustments)

        except Exception as e:
            error = str(e)

    return render_template("index.html", prediction=prediction, adjusted_prediction=adjusted_prediction,
                           breakdown=breakdown, error=error)


@app.route('/health', methods=['GET'])
def health():
    """Health endpoint to verify model is loaded and show model info."""
    # Make sure everything returned is JSON serializable
    feature_names = getattr(model, 'feature_names_in_', None)
    if feature_names is not None:
        try:
            # convert numpy arrays to lists if needed
            feature_names = list(feature_names)
        except Exception:
            feature_names = str(feature_names)

    return jsonify({
        'model_loaded': model_loaded,
        'model_path': MODEL_PATH,
        'load_error': load_error if not model_loaded else None,
        'feature_names': feature_names,
        'cors_available': CORS_AVAILABLE
    })


@app.route("/predict", methods=["POST"])
def predict_api():
    """JSON API. Expect JSON payload with keys: year, km_driven, ex_showroom_price
       Optional keys to approximate OLX-like behavior: owner, seller_type, model_name, apply_adjustments (bool)
       Example: {"year":2018, "km_driven":15000, "ex_showroom_price":85000, "owner":"1st owner", "apply_adjustments": true}
    """
    data = request.get_json(force=True)
    required = ["year", "km_driven", "ex_showroom_price"]

    if not data:
        return jsonify({"error": "No JSON payload provided"}), 400

    missing = [k for k in required if k not in data]
    if missing:
        return jsonify({"error": f"Missing fields: {missing}"}), 400

    try:
        year = int(data["year"])
        km_driven = float(data["km_driven"])
        ex_showroom_price = float(data["ex_showroom_price"])

        if year < 1900 or year > CURRENT_YEAR:
            return jsonify({"error": "Invalid year"}), 400
        if km_driven < 0:
            return jsonify({"error": "Invalid km_driven"}), 400
        if ex_showroom_price <= 0:
            return jsonify({"error": "Invalid ex_showroom_price"}), 400

        age = CURRENT_YEAR - year

        X = pd.DataFrame([[year, km_driven, ex_showroom_price, age]],
                         columns=["year", "km_driven", "ex_showroom_price", "age"])

        pred = float(model.predict(X)[0])

        # Optional heuristic adjustments
        owner = data.get("owner")
        seller_type = data.get("seller_type")
        model_name = data.get("model_name")
        apply_adjustments = bool(data.get("apply_adjustments", False))

        adjusted, breakdown = apply_heuristics(pred, owner=owner, seller_type=seller_type,
                                              model_name=model_name, km_driven=km_driven,
                                              apply_adjustments=apply_adjustments)

        resp = {"predicted_selling_price": pred, "adjusted_prediction": adjusted, "breakdown": breakdown}
        return jsonify(resp)
    except Exception as e:
        return jsonify({"error": str(e)}), 500


if __name__ == "__main__":
    # For local development only. Use a WSGI server for production.
    app.run(host="0.0.0.0", port=5000, debug=True)
