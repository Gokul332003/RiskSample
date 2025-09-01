import streamlit as st
import mlflow
from mlflow.tracking import MlflowClient
import pandas as pd
import datetime

# -------------------------
# Helper
# -------------------------
def to_datetime(ts):
    return datetime.datetime.fromtimestamp(ts/1000).strftime("%Y-%m-%d %H:%M:%S") if ts else None

# -------------------------
# MLflow Configuration
# -------------------------
MLFLOW_TRACKING_URI = "https://262facaa0094.ngrok-free.app/"
mlflow.set_tracking_uri(MLFLOW_TRACKING_URI)
client = MlflowClient()

# -------------------------
# Sidebar
# -------------------------
st.sidebar.title("Developer Dashboard")
section = st.sidebar.radio("Navigate", ["Experiments", "Runs", "Registered Models", "Metrics Overview"])

# -------------------------
# Experiments Section
# -------------------------
if section == "Experiments":
    st.title("Experiments Overview")
    
    experiments = client.search_experiments(order_by=["name ASC"])
    if not experiments:
        st.warning("No experiments found in MLflow.")
    else:
        exp_data = [
            {
                "Experiment ID": exp.experiment_id,
                "Name": exp.name,
             #   "Artifact Location": exp.artifact_location,
                "Lifecycle Stage": exp.lifecycle_stage
            }
            for exp in experiments
        ]
        st.dataframe(pd.DataFrame(exp_data))

# -------------------------
# Runs Section
# -------------------------
elif section == "Runs":
    st.title("Model Runs")
    
    experiments = client.search_experiments(order_by=["name ASC"])
    exp_dict = {exp.name: exp.experiment_id for exp in experiments}
    
    selected_exp = st.selectbox("Select Experiment", list(exp_dict.keys()))
    if selected_exp:
        runs = client.search_runs([exp_dict[selected_exp]])
        
        if runs:
            run_data = []
            for run in runs:
                metrics = run.data.metrics
                params = run.data.params
                run_data.append({
                    "Run ID": run.info.run_id,
                    "Status": run.info.status,
                    "Start Time": to_datetime(run.info.start_time),
                    "End Time": to_datetime(run.info.end_time),
                    **params,
                    **metrics
                })
            st.dataframe(pd.DataFrame(run_data))
        else:
            st.warning("No runs found for this experiment.")

# -------------------------
# Registered Models Section
# -------------------------
elif section == "Registered Models":
    st.title("Registered Models in MLflow Registry")
    
    models = client.search_registered_models()
    if not models:
        st.warning("No registered models found in the MLflow Model Registry.")
    else:
        for model in models:
            st.subheader(f"📌 {model.name}")
            st.write(f"**Description:** {model.description or 'No description'}")
            
            if model.latest_versions:
                for version in model.latest_versions:
                    st.write(f"- Version {version.version} | Stage: `{version.current_stage}` | Run ID: {version.run_id}")
            else:
                st.write("_No versions available_")

# -------------------------
# Metrics Overview Section
# -------------------------
elif section == "Metrics Overview":
    st.title("Metrics Dashboard")
    
    experiments = client.search_experiments(order_by=["name ASC"])
    exp_dict = {exp.name: exp.experiment_id for exp in experiments}
    
    selected_exp = st.selectbox("Select Experiment for Metrics", list(exp_dict.keys()))
    
    if selected_exp:
        runs = client.search_runs([exp_dict[selected_exp]])
        if runs:
            # Collect all metric keys across runs
            all_metrics = set()
            for run in runs:
                all_metrics.update(run.data.metrics.keys())
            
            if all_metrics:
                metric_key = st.selectbox("Select Metric to Visualize", list(all_metrics))
                if metric_key:
                    df = pd.DataFrame([
                        {"Run ID": run.info.run_id, metric_key: run.data.metrics.get(metric_key, None)}
                        for run in runs if metric_key in run.data.metrics
                    ])
                    st.line_chart(df.set_index("Run ID"))
            else:
                st.warning("No metrics found for this experiment.")
        else:
            st.warning("No runs found to display metrics.")




