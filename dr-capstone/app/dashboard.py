"""Streamlit client.  Start the API first, then:  streamlit run app/dashboard.py"""
import base64
import os

import pandas as pd
import requests
import streamlit as st

API = os.environ.get("DR_API", "http://127.0.0.1:8000")
GRADES = ["0 No DR", "1 Mild", "2 Moderate", "3 Severe", "4 Proliferative"]

st.set_page_config(page_title="DR screening prototype", layout="wide")
st.title("Diabetic retinopathy grading: research prototype")
st.warning("Research prototype for a university capstone. Not a medical device and not for clinical use. "
           "A qualified clinician must grade real patient images.")

f = st.file_uploader("Upload a retinal fundus photograph (PNG or JPEG)", type=["png", "jpg", "jpeg"])
if f is not None:
    with st.spinner("Running the 3-model ensemble and Grad-CAM..."):
        try:
            r = requests.post(f"{API}/predict", files={"file": (f.name, f.getvalue())}, timeout=180)
        except requests.ConnectionError:
            st.error(f"Cannot reach the API at {API}. Start it with: uvicorn app.api:app --port 8000")
            st.stop()
    if r.status_code != 200:
        st.error(r.json().get("detail", r.text))
        st.stop()
    d = r.json()

    c1, c2 = st.columns(2)
    c1.image(base64.b64decode(d["image_png_b64"]), caption="Preprocessed input (cropped, square, 512 px)")
    c2.image(base64.b64decode(d["cam_png_b64"]), caption="Grad-CAM: regions that drove the predicted grade")

    a, b = st.columns(2)
    a.metric("Predicted grade (ICDR)", GRADES[d["grade"]], f"confidence {d['confidence']:.0%}", delta_color="off")
    if d["referable"]:
        b.error(f"Refer for review: P(moderate or worse) = {d['p_referable']:.2f} >= cut-off {d['cutoff']:.2f}")
    else:
        b.success(f"Below referral cut-off: P(moderate or worse) = {d['p_referable']:.2f} < {d['cutoff']:.2f}")
    if d["low_confidence"]:
        st.info("Low confidence (top probability under 0.60): treat this grade as uncertain. "
                "This flag is a simple display heuristic and has not been validated.")

    st.bar_chart(pd.DataFrame({"probability": d["probs"]}, index=GRADES))

    with st.expander("What this model can and cannot do"):
        st.markdown(
            "- EfficientNet-B0 ensemble (3 seeds, 512 px), trained on APTOS 2019 only; no external validation yet.\n"
            "- Held-out test set (526 images): QWK 0.916 (95% CI 0.891 to 0.938), accuracy 0.852.\n"
            "- The referral cut-off was chosen for 95% sensitivity on validation; on test it caught 92.7% "
            "of referable cases (95% CI 0.889 to 0.960) with 95.0% specificity.\n"
            "- Weak spots: severe NPDR (grade 3) recall was 27% and proliferative 63%; "
            "mild vs moderate is the commonest confusion.\n"
            "- Grad-CAM shows where the network looked, not why it decided; it can highlight image borders and artifacts."
        )
