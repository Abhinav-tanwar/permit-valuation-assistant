"""Local Streamlit frontend for the permit valuation model."""

from __future__ import annotations

from datetime import date

import streamlit as st

from predictor import (
    explain_prediction,
    load_metadata,
    load_metrics,
    predict_permit_valuation,
)


st.set_page_config(page_title="Permit Valuation Predictor", page_icon="🏗️")
st.title("Building Permit Valuation Predictor")
st.caption(
    "Estimate construction valuation from permit details. "
    "Results reflect the supplied training sample."
)

try:
    metadata = load_metadata()
    metrics = load_metrics()
except FileNotFoundError:
    st.error("Model artifacts are missing. Run `python train_model.py` first.")
    st.stop()

categories = metadata["categories"]
ranges = metadata["observed_ranges"]

with st.form("permit_form"):
    left, right = st.columns(2)
    with left:
        permit_type = st.selectbox("Permit type", categories["Permit Type"])
        work_class = st.selectbox("Work class", categories["Work Class"])
        permit_class = st.selectbox("Permit class", categories["Permit Class Mapped"])
        applied_date = st.date_input(
            "Applied date",
            value=date.fromisoformat(ranges["Applied Date"]["max"]),
        )
        existing_sqft = st.number_input(
            "Total existing building SQFT", min_value=0.0, value=0.0
        )
        remodel_sqft = st.number_input("Remodel/repair SQFT", min_value=0.0, value=0.0)
    with right:
        new_sqft = st.number_input("Total new/addition SQFT", min_value=0.0, value=0.0)
        floors = st.selectbox("Number of floors", categories["Number Of Floors"])
        housing_units = st.selectbox("Housing units", categories["Housing Units"])
        latitude = st.number_input(
            "Latitude",
            min_value=-90.0,
            max_value=90.0,
            value=float(metadata["lat_mean"]),
            format="%.6f",
        )
        longitude = st.number_input(
            "Longitude",
            min_value=-180.0,
            max_value=180.0,
            value=float(metadata["lon_mean"]),
            format="%.6f",
        )
    submitted = st.form_submit_button("Predict valuation")

if submitted:
    record = {
        "Permit Type": permit_type,
        "Work Class": work_class,
        "Permit Class Mapped": permit_class,
        "Applied Date": applied_date.isoformat(),
        "Total Existing Bldg SQFT": existing_sqft,
        "Remodel Repair SQFT": remodel_sqft,
        "Total New Add SQFT": new_sqft,
        "Number Of Floors": float(floors),
        "Housing Units": float(housing_units),
        "Latitude": latitude,
        "Longitude": longitude,
    }
    try:
        result = predict_permit_valuation(record, metadata=metadata)
        valuation = result["dollar_predictions"][0]
        st.metric("Model-predicted valuation", f"${valuation:,.2f}")
        if result["capped_fields"][0]:
            st.warning(
                "Training caps were applied to: "
                + ", ".join(result["capped_fields"][0])
            )
        explanation = explain_prediction(record, result["capped_fields"][0])
        st.subheader("Explanation")
        st.write(explanation["text"])
        st.caption(f"Explanation source: {explanation['source']}")
    except (ValueError, FileNotFoundError) as exc:
        st.error(str(exc))

with st.expander("Model evaluation and limits"):
    test_metrics = metrics["test"]
    metric_columns = st.columns(3)
    metric_columns[0].metric("Test R² (log target)", f"{test_metrics['r2']:.4f}")
    metric_columns[1].metric(
        "Average dollar error",
        f"${test_metrics['dollar_mae_against_capped_target']:,.0f}",
    )
    metric_columns[2].metric("Test permits", f"{metrics['split']['test_rows']:,}")
    st.write(
        "Dollar errors compare held-out predictions with values capped at the training sample's upper percentile. Results come from one random split and do not establish future appraisal accuracy."
    )
    st.caption(
        f"Model trained: {metrics['trained_at_utc'][:10]}. Log RMSE: {test_metrics['log_rmse']:.4f}."
    )
