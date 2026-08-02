"""Built-in deterministic tools: BMI, unit conversion, drug-info lookup.

These are intentionally offline and dependency-free so they always work and
never cost an LLM call. Each returns a Markdown string, or None when it cannot
confidently handle the input.
"""

import re

from .registry import register_tool


# --------------------------------------------------------------------------- #
# BMI calculator
# --------------------------------------------------------------------------- #
@register_tool(
    name="bmi_calculator",
    description="Compute Body Mass Index from weight (kg) and height (cm or m).",
    keywords=["bmi", "body mass index"],
    timeout=3, retries=0, idempotent=True, read_only=True,
)
def bmi_calculator(text):
    """Parse weight/height from text and compute BMI + category."""
    low = text.lower()
    # weight in kg
    w = re.search(r"(\d+(?:\.\d+)?)\s*(?:kg|kgs|kilograms?)", low)
    # height in cm or m
    h_cm = re.search(r"(\d+(?:\.\d+)?)\s*(?:cm|centimet)", low)
    h_m = re.search(r"(\d+(?:\.\d+)?)\s*(?:m|meters?)\b", low)
    if not w or not (h_cm or h_m):
        return None
    weight = float(w.group(1))
    height_m = float(h_cm.group(1)) / 100.0 if h_cm else float(h_m.group(1))
    if height_m <= 0:
        return None
    bmi = weight / (height_m ** 2)
    if bmi < 18.5:
        cat = "Underweight"
    elif bmi < 25:
        cat = "Normal weight"
    elif bmi < 30:
        cat = "Overweight"
    else:
        cat = "Obese"
    return (
        f"**BMI Result**\n\n"
        f"- Weight: {weight:g} kg\n- Height: {height_m:g} m\n"
        f"- **BMI: {bmi:.1f}** ({cat})\n\n"
        f"_BMI is a screening tool, not a diagnosis. Consult a clinician for "
        f"a full assessment._"
    )


# --------------------------------------------------------------------------- #
# Simple medical unit converter
# --------------------------------------------------------------------------- #
@register_tool(
    name="unit_converter",
    description="Convert common clinical units (kg<->lb, cm<->in, °C<->°F).",
    keywords=["convert", "conversion"],
    timeout=3, retries=0, idempotent=True, read_only=True,
)
def unit_converter(text):
    low = text.lower()
    m = re.search(r"(\d+(?:\.\d+)?)\s*(kg|lb|lbs|cm|in|inch|inches|c|f|°c|°f)\b", low)
    if not m:
        return None
    val = float(m.group(1))
    unit = m.group(2)
    conversions = {
        "kg": (val * 2.20462, "lb"),
        "lb": (val / 2.20462, "kg"), "lbs": (val / 2.20462, "kg"),
        "cm": (val / 2.54, "in"),
        "in": (val * 2.54, "cm"), "inch": (val * 2.54, "cm"), "inches": (val * 2.54, "cm"),
        "c": (val * 9 / 5 + 32, "°F"), "°c": (val * 9 / 5 + 32, "°F"),
        "f": ((val - 32) * 5 / 9, "°C"), "°f": ((val - 32) * 5 / 9, "°C"),
    }
    if unit not in conversions:
        return None
    out_val, out_unit = conversions[unit]
    return f"**Unit conversion**: {val:g} {unit} = **{out_val:.2f} {out_unit}**"


# --------------------------------------------------------------------------- #
# eGFR calculator (CKD-EPI 2021 creatinine equation)
# --------------------------------------------------------------------------- #
@register_tool(
    name="egfr_calculator",
    description="Estimate eGFR using the CKD-EPI 2021 creatinine equation.",
    keywords=["egfr", "ckd-epi", "creatinine", "kidney function"],
    timeout=3, retries=0, idempotent=True, read_only=True,
)
def egfr_calculator(text):
    low = text.lower()
    if "egfr" not in low and "creatinine" not in low:
        return None

    age_m = re.search(r"age\s*(\d{1,3})|(?:\b(\d{1,3})\s*(?:years? old|yo)\b)", low)
    cr_m = re.search(r"(?:creatinine|scr|cr)\s*(\d+(?:\.\d+)?)", low)
    if not age_m or not cr_m:
        return None

    age = int(age_m.group(1) or age_m.group(2))
    scr = float(cr_m.group(1))
    female = bool(re.search(r"\b(female|woman|women|女|女性)\b", low))
    male = bool(re.search(r"\b(male|man|men|男|男性)\b", low))
    if not (female or male) or age <= 0 or scr <= 0:
        return None

    k = 0.7 if female else 0.9
    alpha = -0.241 if female else -0.302
    egfr = 142 * (min(scr / k, 1) ** alpha) * (max(scr / k, 1) ** -1.200) * (0.9938 ** age)
    if female:
        egfr *= 1.012

    stage = "G1/G2 (normal or mildly decreased)" if egfr >= 60 else "G3+ (decreased; needs clinical context)"
    return (
        f"**eGFR Result (CKD-EPI 2021 creatinine)**\n\n"
        f"- Age: {age}\n- Sex: {'female' if female else 'male'}\n- Serum creatinine: {scr:g} mg/dL\n"
        f"- **eGFR: {egfr:.1f} mL/min/1.73m²** ({stage})\n\n"
        f"_For adults only. eGFR depends on lab calibration, clinical context, medications, pregnancy status, and acute illness. Discuss abnormal results with a clinician._"
    )


# --------------------------------------------------------------------------- #
# CHA2DS2-VASc stroke-risk score for atrial fibrillation
# --------------------------------------------------------------------------- #
@register_tool(
    name="cha2ds2_vasc",
    description="Compute CHA2DS2-VASc stroke-risk score for atrial fibrillation.",
    keywords=["cha2ds2", "cha2ds2-vasc", "vasc", "atrial fibrillation", "afib"],
    timeout=3, retries=0, idempotent=True, read_only=True,
)
def cha2ds2_vasc(text):
    low = text.lower()
    if not any(k in low for k in ("cha2ds2", "vasc", "atrial fibrillation", "afib")):
        return None

    score = 0
    reasons = []
    age_m = re.search(r"age\s*(\d{1,3})|(?:\b(\d{1,3})\s*(?:years? old|yo)\b)", low)
    age = int(age_m.group(1) or age_m.group(2)) if age_m else 0
    if age >= 75:
        score += 2; reasons.append("Age ≥75 (+2)")
    elif 65 <= age <= 74:
        score += 1; reasons.append("Age 65–74 (+1)")
    if re.search(r"\b(female|woman|women|女|女性)\b", low):
        score += 1; reasons.append("Female sex (+1)")
    if re.search(r"\b(chf|heart failure|congestive)\b", low):
        score += 1; reasons.append("Heart failure (+1)")
    if re.search(r"\b(hypertension|high blood pressure|htn|高血压)\b", low):
        score += 1; reasons.append("Hypertension (+1)")
    if re.search(r"\b(diabetes|dm|糖尿病)\b", low):
        score += 1; reasons.append("Diabetes (+1)")
    if re.search(r"\b(stroke|tia|thromboembolism|中风|卒中)\b", low):
        score += 2; reasons.append("Stroke/TIA/thromboembolism (+2)")
    if re.search(r"\b(vascular|mi|myocardial infarction|pad|aortic plaque|心梗)\b", low):
        score += 1; reasons.append("Vascular disease (+1)")

    return (
        f"**CHA2DS2-VASc Result**\n\n"
        f"- **Score: {score}**\n"
        f"- Factors: {', '.join(reasons) if reasons else 'No factors detected from the query'}\n\n"
        f"_This score is intended for atrial fibrillation stroke-risk discussion. Anticoagulation decisions require bleeding-risk review and clinician judgment._"
    )


# --------------------------------------------------------------------------- #
# Local drug-info lookup (small curated table; extend as needed)
# --------------------------------------------------------------------------- #
_DRUG_DB = {
    "paracetamol": {
        "aka": "acetaminophen / Tylenol",
        "class": "Analgesic / antipyretic",
        "typical_adult_dose": "500–1000 mg every 4–6 h; max 3–4 g/day",
        "cautions": "Liver impairment; avoid alcohol; watch cumulative dose in combo products.",
    },
    "acetaminophen": {
        "aka": "paracetamol / Tylenol",
        "class": "Analgesic / antipyretic",
        "typical_adult_dose": "500–1000 mg every 4–6 h; max 3–4 g/day",
        "cautions": "Liver impairment; avoid alcohol; watch cumulative dose in combo products.",
    },
    "ibuprofen": {
        "aka": "Advil / Nurofen",
        "class": "NSAID",
        "typical_adult_dose": "200–400 mg every 4–6 h; max 1200 mg/day OTC",
        "cautions": "GI bleeding, renal impairment, cardiovascular risk; take with food.",
    },
    "amoxicillin": {
        "aka": "Amoxil",
        "class": "Penicillin antibiotic",
        "typical_adult_dose": "250–500 mg every 8 h (regimen-dependent)",
        "cautions": "Penicillin allergy; complete the full prescribed course.",
    },
    "metformin": {
        "aka": "Glucophage",
        "class": "Biguanide (antidiabetic)",
        "typical_adult_dose": "500 mg once–twice daily, titrated; max ~2000–2550 mg/day",
        "cautions": "Renal function; risk of lactic acidosis; GI upset.",
    },
    "aspirin": {
        "aka": "acetylsalicylic acid",
        "class": "NSAID / antiplatelet",
        "typical_adult_dose": "75–325 mg/day (indication-dependent)",
        "cautions": "Bleeding risk; avoid in children (Reye's syndrome); GI irritation.",
    },
}


@register_tool(
    name="drug_info",
    description="Look up basic reference info for common medications.",
    keywords=["drug", "medication", "medicine", "dose", "dosage"] + list(_DRUG_DB.keys()),
    timeout=5, retries=1, idempotent=True, read_only=True,
)
def drug_info(text):
    low = text.lower()
    for name, info in _DRUG_DB.items():
        if re.search(r"\b" + re.escape(name) + r"\b", low):
            return (
                f"**{name.capitalize()}** ({info['aka']})\n\n"
                f"- **Class**: {info['class']}\n"
                f"- **Typical adult dose**: {info['typical_adult_dose']}\n"
                f"- **Cautions**: {info['cautions']}\n\n"
                f"_Reference information only — not a prescription. Dosing must be "
                f"individualised by a licensed clinician/pharmacist._"
            )
    return None
