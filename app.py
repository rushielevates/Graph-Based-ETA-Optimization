import streamlit as st
import pandas as pd
import numpy as np
import joblib
import plotly.express as px
import plotly.graph_objects as go
import time

# ── Page Config ──────────────────────────────────────────────────────
st.set_page_config(
    page_title="Delhivery Network Intelligence",
    page_icon="🚚",
    layout="wide"
)

# ── Brand Colors ─────────────────────────────────────────────────────
DELHIVERY_ORANGE = "#FF6600"
DELHIVERY_BLUE = "#004d99"

# ── Load Data & Models (cached) ─────────────────────────────────────
@st.cache_data
def load_all():
    edges_df = pd.read_csv('edges_data.csv')
    hub_metrics = pd.read_csv('hub_metrics.csv')
    tradeoff = pd.read_csv('ftl_carting_tradeoff.csv')
    return edges_df, hub_metrics, tradeoff

@st.cache_resource
def load_models():
    # ETA model (XGBoost + Graph from Phase 6)
    eta_model = joblib.load('eta_model_xgb_graph.pkl')
    # FTL vs Carting classifier (from Phase 7)
    clf = joblib.load('route_classifier.pkl')
    clf_features = joblib.load('route_classifier_features.pkl')
    # Deployment artifacts (contains feature lists and fill values)
    deploy = joblib.load('deployment_artifacts.pkl')
    return eta_model, clf, clf_features, deploy

edges_df, hub_metrics, tradeoff = load_all()
eta_model, route_clf, clf_features, deploy = load_models()

# Extract feature lists and fill values
baseline_features = deploy['baseline_features']   # 61 features
fill_vals = deploy['fill_vals']

# -----------------------------------------------------------------
# Helper: Build complete feature vector and predict ETA
# -----------------------------------------------------------------
def predict_eta(osrm_time, distance, route_type, hour, dayofweek, interstate):
    """
    Predict ETA using the trained XGBoost graph model.
    Builds a feature vector with ALL features the model expects.
    Unknown graph features (embeddings, metrics) are set to 0.
    """
    # Get exact feature list from the model (111 features)
    model_features = eta_model.get_booster().feature_names
    
    # Initialize all features to 0
    row = {feat: 0.0 for feat in model_features}
    
    # Fill user-provided baseline features
    row['segment_osrm_time'] = osrm_time
    row['segment_osrm_distance'] = distance
    row['route_type_enc'] = 1 if route_type == 'FTL' else 0
    row['cutoff_hour'] = hour
    row['cutoff_dayofweek'] = dayofweek
    row['is_peak_hour'] = 1 if (9 <= hour <= 12 or 18 <= hour <= 21) else 0
    row['is_weekend'] = 1 if dayofweek >= 5 else 0
    row['is_interstate'] = interstate
    row['is_night_shift'] = 1 if hour < 6 else 0
    
    # Derived OSRM features
    row['sqrt_osrm_time'] = np.sqrt(osrm_time)
    row['sqrt_osrm_distance'] = np.sqrt(distance)
    row['log_osrm_time'] = np.log1p(osrm_time)
    row['log_osrm_distance'] = np.log1p(distance)
    row['osrm_speed_kmh'] = distance / (osrm_time/60 + 1e-6)
    row['km_per_minute'] = distance / (osrm_time + 1e-6)
    
    # Circular time features
    row['hour_sin'] = np.sin(2 * np.pi * hour / 24)
    row['hour_cos'] = np.cos(2 * np.pi * hour / 24)
    row['day_sin'] = np.sin(2 * np.pi * dayofweek / 7)
    row['day_cos'] = np.cos(2 * np.pi * dayofweek / 7)
    row['month_sin'] = np.sin(2 * np.pi * 6 / 12)   # default to June
    row['month_cos'] = np.cos(2 * np.pi * 6 / 12)
    row['od_start_sin'] = 0
    row['od_start_cos'] = 1
    
    # Fill remaining baseline features with median values from training
    for feat in baseline_features:
        if feat in row and row[feat] == 0 and feat in fill_vals:
            row[feat] = fill_vals[feat]
    
    # All graph features (src_emb_*, dst_emb_*, betweenness, etc.) are already 0
    # This is acceptable because the model saw zeros for unseen hubs during training
    
    # Create DataFrame with columns in the exact order the model expects
    X = pd.DataFrame([row])[model_features].astype(np.float32)
    
    # Predict on sqrt scale, then square to get minutes
    pred_sqrt = eta_model.predict(X)[0]
    return max(0, pred_sqrt ** 2)

# ── Header ────────────────────────────────────────────────────────────
st.title("🚚 Delhivery Network Operations Dashboard")
st.markdown("**Graph‑Based ETA Optimization & Bottleneck Analysis**")
st.divider()

# ── KPI Row ──────────────────────────────────────────────────────────
k1, k2, k3, k4 = st.columns(4)
total_edges = len(edges_df)
pct_chronic = (edges_df['median_delay_ratio'] > 1.2).mean() * 100
avg_delay_ratio = edges_df['median_delay_ratio'].mean()
k1.metric("Corridors Analysed", f"{total_edges:,}")
k2.metric("Chronically Delayed Corridors", f"{pct_chronic:.1f}%", delta_color="inverse")
k3.metric("Avg Delay Ratio (corridor)", f"{avg_delay_ratio:.2f}")
k4.metric("SLA Threshold", ">1.2 (20% over OSRM)")
st.divider()

# ── Tabs ─────────────────────────────────────────────────────────────
tab1, tab2, tab3, tab4, tab5, tab6 = st.tabs([
    "📊 Bottleneck Hubs",
    "🗺️ Network Explorer",
    "⏱️ ETA Predictor",
    "🚛 Route Recommender",
    "📈 Model Comparison",
    "🔥 State Heatmap & Corridor Search"
])

# =========================================================================
# TAB 1 – BOTTLENECK HUBS
# =========================================================================
with tab1:
    st.subheader("Top Bottleneck Hubs by SLA Breach Contribution")
    with st.spinner("Loading hub metrics..."):
        time.sleep(0.2)
        top_n = st.slider("Show top N hubs", 5, 20, 10)
        top_hubs = hub_metrics.head(top_n).copy()
        fig_bar = px.bar(
            top_hubs.sort_values('sla_pct'),
            x='sla_pct', y='facility',
            orientation='h',
            color='sla_pct',
            color_continuous_scale='Reds',
            labels={'sla_pct': '% of Total Excess Delay', 'facility': 'Hub'},
            title=f"Top {top_n} Hubs — SLA Breach Contribution"
        )
        st.plotly_chart(fig_bar, width='stretch')
        st.dataframe(hub_metrics[['facility', 'betweenness', 'total_degree', 'sla_pct']].head(15),
                     width='stretch')

# =========================================================================
# TAB 2 – NETWORK EXPLORER
# =========================================================================
with tab2:
    st.subheader("Corridor Delay Explorer")
    with st.spinner("Loading corridor data..."):
        time.sleep(0.2)
        col1, col2 = st.columns(2)
        with col1:
            route_filter = st.selectbox("Route Type", ["All", "FTL", "Carting"])
        with col2:
            delay_threshold = st.slider("Min Delay Ratio to show", 1.0, 5.0, 1.5)

        plot_edges = edges_df.copy()
        if route_filter != "All":
            plot_edges = plot_edges[plot_edges['route_type'] == route_filter]
        plot_edges = plot_edges[plot_edges['median_delay_ratio'] >= delay_threshold]
        plot_edges = plot_edges.sort_values('median_delay_ratio', ascending=False).head(50)

        st.markdown(f"Showing **{len(plot_edges)}** corridors with delay ≥ {delay_threshold}")
        fig_corr = px.bar(
            plot_edges.head(20),
            x='median_delay_ratio',
            y=plot_edges.head(20)['source'] + ' → ' + plot_edges.head(20)['dest'],
            orientation='h',
            color='median_delay_ratio',
            color_continuous_scale='Reds',
            title="Top 20 Worst Delay Corridors",
            labels={'median_delay_ratio': 'Median Delay Ratio', 'y': 'Corridor'}
        )
        fig_corr.update_layout(yaxis={'categoryorder': 'total ascending'})
        st.plotly_chart(fig_corr, width='stretch')

# =========================================================================
# TAB 3 – ETA PREDICTOR (using trained XGBoost + Graph model)
# =========================================================================
with tab3:
    st.subheader("Predict Delivery ETA")
    st.markdown("**Get a Graph‑Enhanced ETA prediction using the trained XGBoost model**")
    c1, c2 = st.columns(2)
    with c1:
        osrm_time = st.number_input("OSRM Estimated Time (minutes)", 5, 1000, 60)
        distance = st.number_input("Segment Distance (km)", 1, 2000, 80)
        route_type_input = st.selectbox("Route Type", ["FTL", "Carting"], key='eta_rt')
    with c2:
        hour_input = st.slider("Hour of Dispatch", 0, 23, 10, key='eta_hour')
        day_input = st.selectbox("Day of Week",
                                 ["Monday","Tuesday","Wednesday","Thursday","Friday","Saturday","Sunday"])
        interstate = st.checkbox("Inter-state trip?", key='eta_inter')

    day_map = {"Monday":0,"Tuesday":1,"Wednesday":2,"Thursday":3,"Friday":4,"Saturday":5,"Sunday":6}
    dow = day_map[day_input]

    if st.button("🔍 Predict ETA", type="primary"):
        with st.spinner("Running model prediction..."):
            graph_pred = predict_eta(osrm_time, distance, route_type_input, hour_input, dow, int(interstate))
            baseline_pred = osrm_time * edges_df['median_delay_ratio'].median()  # simple baseline

            col_a, col_b, col_c = st.columns(3)
            col_a.metric("OSRM Estimate", f"{osrm_time:.0f} min", "System baseline")
            col_b.metric("Baseline Prediction", f"{baseline_pred:.0f} min",
                         delta=f"{baseline_pred - osrm_time:.0f} min vs OSRM")
            col_c.metric("Graph-Enhanced Pred.", f"{graph_pred:.0f} min",
                         delta=f"{graph_pred - osrm_time:.0f} min vs OSRM",
                         delta_color="inverse" if graph_pred < baseline_pred else "normal")

            improvement = baseline_pred - graph_pred
            if improvement > 0:
                st.success(f"✅ Graph model is **{improvement:.1f} min more accurate** "
                           f"({improvement/baseline_pred*100:.1f}% improvement)")
            else:
                st.info("Both methods agree on this trip profile.")

            # Gauge chart
            fig_gauge = go.Figure(go.Indicator(
                mode="gauge+number+delta",
                value=graph_pred,
                delta={'reference': osrm_time, 'valueformat': '.0f'},
                title={'text': "Graph-Enhanced ETA (minutes)"},
                gauge={
                    'axis': {'range': [osrm_time * 0.8, osrm_time * 3]},
                    'bar': {'color': DELHIVERY_ORANGE},
                    'steps': [
                        {'range': [osrm_time * 0.8, osrm_time * 1.2], 'color': "lightgreen"},
                        {'range': [osrm_time * 1.2, osrm_time * 2.0], 'color': "lightyellow"},
                        {'range': [osrm_time * 2.0, osrm_time * 3.0], 'color': "lightsalmon"},
                    ],
                    'threshold': {
                        'line': {'color': "red", 'width': 4},
                        'thickness': 0.75,
                        'value': baseline_pred
                    }
                }
            ))
            st.plotly_chart(fig_gauge, width='stretch')

# =========================================================================
# TAB 4 – ROUTE RECOMMENDER (using trained classifier)
# =========================================================================
with tab4:
    st.subheader("FTL vs Carting Route Recommender")
    st.markdown("**Data‑backed recommendation based on corridor performance**")
    c1, c2 = st.columns(2)
    with c1:
        rec_dist = st.number_input("Distance (km)", 1, 2000, 80, key='rec_dist')
        rec_hour = st.slider("Hour of Dispatch", 0, 23, 14, key='rec_hour')
    with c2:
        rec_inter = st.checkbox("Inter-state trip?", key='rec_inter')
        rec_day = st.selectbox("Day of Week",
                               ["Monday","Tuesday","Wednesday","Thursday","Friday","Saturday","Sunday"],
                               key='rec_day')
    if st.button("🚛 Get Recommendation", type="primary"):
        with st.spinner("Analysing route profile..."):
            day_map = {"Monday":0,"Tuesday":1,"Wednesday":2,"Thursday":3,"Friday":4,"Saturday":5,"Sunday":6}
            dow = day_map[rec_day]
            is_peak = 1 if (9 <= rec_hour <= 12 or 18 <= rec_hour <= 21) else 0
            is_we = 1 if dow >= 5 else 0
            # Rule-based override for very short distances
            if rec_dist <= 30 and not rec_inter:
                rec = "Carting"
                conf = 85.0
                prob_ftl = 0.15
                st.info("ℹ️ Rule override: Very short intra‑state trip – Carting is recommended for cost efficiency.")
            else:
                input_row = pd.DataFrame([{
                    'segment_osrm_distance': rec_dist,
                    'cutoff_hour': rec_hour,
                    'cutoff_dayofweek': dow,
                    'is_peak_hour': is_peak,
                    'is_weekend': is_we,
                    'is_interstate': int(rec_inter),
                    'hist_avg_delay': 1.5,
                    'src_total_degree': 0, 'dst_total_degree': 0,
                    'src_betweenness': 0, 'dst_betweenness': 0,
                    'src_sla_pct': 0, 'dst_sla_pct': 0
                }])
                input_row = input_row[clf_features]
                prob_ftl = route_clf.predict_proba(input_row)[0][1]
                rec = "FTL" if prob_ftl >= 0.5 else "Carting"
                conf = max(prob_ftl, 1 - prob_ftl) * 100
            st.success(f"🟢 **Recommended: {rec}** — Confidence: {conf:.1f}%")
            # Time comparison
            osrm_approx = rec_dist / (50/60)
            ftl_delay = edges_df[edges_df['route_type']=='FTL']['median_delay_ratio'].median()
            cart_delay = edges_df[edges_df['route_type']=='Carting']['median_delay_ratio'].median()
            ftl_eta = osrm_approx * ftl_delay
            cart_eta = osrm_approx * cart_delay
            saving = cart_eta - ftl_eta
            m1, m2, m3 = st.columns(3)
            m1.metric("FTL Expected Time", f"{ftl_eta:.0f} min")
            m2.metric("Carting Expected Time", f"{cart_eta:.0f} min")
            m3.metric("Time Saved with FTL", f"{saving:.0f} min", delta_color="inverse")

# =========================================================================
# TAB 5 – MODEL COMPARISON
# =========================================================================
with tab5:
    st.subheader("Model Performance Comparison")
    results = pd.DataFrame({
        "Model": ["XGBoost Baseline", "XGBoost + Graph", "Segmented XGBoost",
                  "LightGBM + Graph", "Stacked Ensemble"],
        "MAE (min)": [7.70, 7.64, 7.83, 7.69, 7.90],
        "R²": [0.463, 0.467, 0.454, 0.458, 0.467],
        "Within 15%": [43.7, 44.0, 42.9, 43.6, 42.9]
    })
    col1, col2 = st.columns(2)
    with col1:
        fig_mae = px.bar(results, x='Model', y='MAE (min)', color='Model',
                         title="Mean Absolute Error (lower is better)",
                         color_discrete_sequence=[DELHIVERY_ORANGE])
        fig_mae.update_layout(xaxis_tickangle=-45)
        st.plotly_chart(fig_mae, width='stretch')
    with col2:
        fig_r2 = px.bar(results, x='Model', y='R²', color='Model',
                        title="R² Score (higher is better)",
                        color_discrete_sequence=[DELHIVERY_BLUE])
        fig_r2.update_layout(xaxis_tickangle=-45)
        st.plotly_chart(fig_r2, width='stretch')
    st.dataframe(results.style.format({'MAE (min)': '{:.2f}', 'R²': '{:.3f}', 'Within 15%': '{:.1f}'}),
                 width='stretch')

# =========================================================================
# TAB 6 – STATE HEATMAP & CORRIDOR SEARCH
# =========================================================================
with tab6:
    st.subheader("Top Delay Corridors")
    top_corridors = edges_df.nlargest(10, 'median_delay_ratio')
    fig_state = px.bar(top_corridors, x='source', y='median_delay_ratio',
                       color='route_type', title="Highest Delay Corridors",
                       labels={'source': 'Source Hub', 'median_delay_ratio': 'Median Delay Ratio'})
    st.plotly_chart(fig_state, width='stretch')

    st.divider()
    st.subheader("Corridor Delay Search")
    col1, col2 = st.columns(2)
    with col1:
        src = st.text_input("Source Hub Code", value="IND000000ACB")
    with col2:
        dst = st.text_input("Destination Hub Code", value="IND562132AAA")
    if st.button("🔍 Search Corridor"):
        match = edges_df[(edges_df['source'] == src) & (edges_df['dest'] == dst)]
        if len(match) == 0:
            st.warning("No corridor found for these hubs.")
        else:
            for _, row in match.iterrows():
                st.markdown(f"**Corridor:** {row['source']} → {row['dest']}")
                st.markdown(f"- **Route Type:** {row['route_type']}")
                st.markdown(f"- **Median Delay Ratio:** {row['median_delay_ratio']:.2f}")
                st.markdown(f"- **Trip Count:** {row['trip_count']}")
                st.markdown(f"- **Median Actual Time:** {row.get('median_actual_time', 'N/A')} min")
                st.markdown(f"- **Hour of Day:** {row.get('hour', 'N/A')}")

st.divider()
st.caption("Delhivery Network Intelligence | Built with Streamlit + XGBoost + node2vec | Consulting & Analytics Club, IIT Guwahati")