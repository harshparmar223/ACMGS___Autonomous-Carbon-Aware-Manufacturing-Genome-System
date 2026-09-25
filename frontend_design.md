# ACMGS Frontend Design & UI Specification
**Document ID:** `FRONTEND_DESIGN_SPEC_V9`  
**Application:** Autonomous Carbon-Aware Manufacturing Genome System (ACMGS)  
**Primary UI File:** `src/dashboard/app.py`  
**Framework:** Streamlit (Python) with custom embedded CSS, Plotly Graph Objects/Express, and SVG graphics  

---

## 🎯 Purpose & Design Philosophy
This document preserves the **exact frontend design, layout, styling tokens, widgets, custom CSS, and interactive behaviors** of the ACMGS Control Center. 

When building or replicating the frontend in any new directory or workspace, **this design specification must be preserved with 100% fidelity**:
1. **Never simplify or flatten the UI** to basic Streamlit defaults.
2. **Preserve the dark cyberpunk/industrial aesthetic** with curated neon accents.
3. **Preserve the exact 6 core tabs + Edge/Actuation tabs** and all component arrangements.
4. **Preserve the custom SVG animated branding** and glassmorphic card styling.

---

## 🎨 1. Core Visual Tokens & Typography

### 1.1 Color Palette
| Token Name | Hex / RGBA | Role / Usage |
|:---|:---|:---|
| **Deep Space Canvas** | `#0a0e1a` to `#0d1426` | Global page background gradient (`135deg`) |
| **Sidebar Canvas** | `#0d1426` to `#090d1a` | Vertical sidebar background gradient (`180deg`) |
| **Electric Cyan (`_CYAN`)** | `#00d4ff` | Primary system accent, active tabs, header gradient, process metrics |
| **Emerald Green (`_GREEN`)** | `#00ff88` | Clean carbon zone, high yield, material metrics, online factory state |
| **Amber Gold (`_YELLOW`)** | `#ffd600` | Mixed carbon zone, warnings, predicted quality, degraded factory state |
| **Crimson Red (`_RED`)** | `#ff4b4b` | Dirty carbon zone, anomalies, energy warnings, critical factory state |
| **Deep Purple (`_PURPLE`)** | `#a855f7` | Energy DNA (EdNA 00-15) latent features, quality pills |
| **Tangerine Orange (`_ORANGE`)** | `#f97316` | Energy consumption (kWh) indicators |
| **Card Glass Surface** | `rgba(255,255,255,0.038)` | Metric tiles, parameter containers, sidebar cards |
| **Glass Border** | `rgba(255,255,255,0.09)` | Subtle borders defining cards and inputs |

### 1.2 Typography
- **Primary Body & Headings**: `'Inter', sans-serif` (weights: 300, 400, 500, 600, 700, 800) imported from Google Fonts.
- **Engineering Readouts & Telemetry**: `'JetBrains Mono', monospace` (weights: 400, 600) used for all numeric metrics, batch IDs, z-scores, and telemetry values.

---

## 🧱 2. Custom CSS Architecture

The dashboard injects a master `<style>` block via `st.markdown(..., unsafe_allow_html=True)`. The exact rules are:

```css
@import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;500;600;700;800&family=JetBrains+Mono:wght@400;600&display=swap');

/* ── Global Canvas ─────────────────────────────────────────── */
.stApp {
    background: linear-gradient(135deg, #0a0e1a 0%, #0d1426 100%) !important;
    font-family: 'Inter', sans-serif !important;
}
.main .block-container {
    padding-top: 0.8rem;
    padding-bottom: 2rem;
    max-width: 1500px;
}

/* ── Header Banner ─────────────────────────────────────────── */
.acmgs-header {
    background: linear-gradient(135deg,
        rgba(0,212,255,0.1) 0%,
        rgba(0,255,136,0.06) 50%,
        rgba(0,80,200,0.08) 100%
    );
    border: 1px solid rgba(0,212,255,0.22);
    border-radius: 16px;
    padding: 20px 32px;
    margin-bottom: 16px;
}
.acmgs-header h1 {
    font-size: 2rem;
    font-weight: 800;
    background: linear-gradient(90deg, #00d4ff 0%, #00ff88 70%);
    -webkit-background-clip: text;
    -webkit-text-fill-color: transparent;
    background-clip: text;
    margin: 0;
    letter-spacing: -0.02em;
}
.acmgs-header p {
    color: rgba(255,255,255,0.5);
    font-size: 0.875rem;
    margin: 6px 0 10px 0;
}

/* ── Badges ────────────────────────────────────────────────── */
.hbadge {
    display: inline-block;
    background: rgba(0,212,255,0.12);
    border: 1px solid rgba(0,212,255,0.3);
    border-radius: 20px;
    padding: 2px 12px;
    font-size: 0.7rem;
    font-weight: 600;
    color: #00d4ff;
    letter-spacing: 0.1em;
    text-transform: uppercase;
    margin-right: 6px;
}
.hbadge-green  { background: rgba(0,255,136,0.12); border-color: rgba(0,255,136,0.3); color: #00ff88; }
.hbadge-yellow { background: rgba(255,214,0,0.1);  border-color: rgba(255,214,0,0.3);  color: #ffd600; }
.hbadge-red    { background: rgba(255,75,75,0.12); border-color: rgba(255,75,75,0.3);  color: #ff6b6b; }

/* ── Metrics ───────────────────────────────────────────────── */
[data-testid="stMetricValue"] {
    font-size: 1.8rem !important;
    font-weight: 700 !important;
    color: #00d4ff !important;
    font-family: 'JetBrains Mono', monospace !important;
}
[data-testid="stMetricLabel"] {
    font-size: 0.76rem !important;
    font-weight: 600 !important;
    color: rgba(255,255,255,0.45) !important;
    text-transform: uppercase !important;
    letter-spacing: 0.06em !important;
}
[data-testid="metric-container"] {
    background: rgba(255,255,255,0.038) !important;
    border: 1px solid rgba(255,255,255,0.09) !important;
    border-radius: 12px !important;
    padding: 1rem 1.2rem !important;
    transition: border-color 0.2s;
}
[data-testid="metric-container"]:hover {
    border-color: rgba(0,212,255,0.35) !important;
}

/* ── Tabs Navigation ───────────────────────────────────────── */
.stTabs [data-baseweb="tab-list"] {
    background: rgba(255,255,255,0.025);
    border: 1px solid rgba(255,255,255,0.07);
    border-radius: 12px;
    padding: 5px;
    gap: 3px;
    width: 100%;
    box-sizing: border-box;
}
.stTabs [data-baseweb="tab"] {
    background: transparent;
    border-radius: 8px;
    color: rgba(255,255,255,0.5);
    font-weight: 500;
    font-size: 0.83rem;
    padding: 7px 16px;
    transition: all 0.2s;
    flex: 1 1 0;
    justify-content: center;
    text-align: center;
}
.stTabs [aria-selected="true"] {
    background: rgba(0,212,255,0.16) !important;
    color: #00d4ff !important;
    font-weight: 600 !important;
}

/* ── Sidebar ───────────────────────────────────────────────── */
[data-testid="stSidebar"] {
    background: linear-gradient(180deg, #0d1426 0%, #090d1a 100%) !important;
    border-right: 1px solid rgba(255,255,255,0.06) !important;
}

/* ── Section Labels ────────────────────────────────────────── */
.slabel {
    font-size: 0.72rem;
    font-weight: 600;
    color: rgba(255,255,255,0.3);
    text-transform: uppercase;
    letter-spacing: 0.12em;
    border-bottom: 1px solid rgba(255,255,255,0.07);
    padding-bottom: 5px;
    margin: 18px 0 10px 0;
}

/* ── Scrollbars & Form Elements ────────────────────────────── */
::-webkit-scrollbar { width: 5px; height: 5px; }
::-webkit-scrollbar-track { background: transparent; }
::-webkit-scrollbar-thumb { background: rgba(0,212,255,0.22); border-radius: 10px; }
::-webkit-scrollbar-thumb:hover { background: rgba(0,212,255,0.4); }

.stSlider > div > div > div { background: rgba(0,212,255,0.18) !important; }
.stTextInput > div > div > input,
.stSelectbox > div > div {
    background: rgba(255,255,255,0.04) !important;
    border: 1px solid rgba(255,255,255,0.1) !important;
    color: #ffffff !important;
    border-radius: 8px !important;
}
[data-testid="stDataFrame"] {
    border-radius: 10px !important;
    border: 1px solid rgba(255,255,255,0.08) !important;
}
```

---

## 🧬 3. Custom SVG Brand Logo Specification
The sidebar features a bespoke animated SVG vector representing the cyber-physical genome architecture:

- **Hexagon (`<polygon>`)**: Outer ring representing Carbon ($C$) with a linear gradient stroke (`#00d4ff` to `#00ff88`).
- **6 Vertex Dots (`<circle>`)**: Glowing dots symbolizing Manufacturing ($M$) gear nodes.
- **Double Helix Strands (`<path>`)**: Dual sinusoidal curves with cross-rungs symbolizing the Batch Genome ($G$).
- **Orbital Ellipse (`<ellipse>`)**: 35-degree rotated ellipse representing System Integration ($S$).
- **Central Core (`<circle>`)**: Glowing concentric circle representing the Autonomous ($A$) closed-loop decision engine.

---

## 🎛️ 4. Sidebar Control Architecture

The sidebar remains active across all tabs and provides global monitoring & control:
1. **Brand Identity Block**: Animated SVG logo, gradient text `ACMGS`, and version label `CONTROL CENTER v9.0`.
2. **Live Carbon Monitor Slider**:
   - Range: `0` to `600` `gCO₂/kWh`, default `220`, step `5`.
   - Real-time classification into 3 dynamic zones:
     - `LOW` (≤150): Clean green card (`00ff88`), description: *"Renewable energy dominant — Maximize production output at full capacity."*
     - `MEDIUM` (150-400): Mixed yellow card (`ffd600`), description: *"Balanced energy mix — Optimize for efficiency and sustainability."*
     - `HIGH` (>400): Conservation red card (`ff4b4b`), description: *"Heavy fossil fuel load — Activate conservation mode, minimize energy."*
3. **AI Recommendation Mini-Panel**:
   - Displays real-time optimal predicted targets from the dispatch state machine:
     - Predicted Yield (Cyan, JetBrains Mono)
     - Predicted kWh (Emerald Green)
     - Predicted kg CO₂ (Crimson Red)
4. **Database Status Counters**:
   - Table row count tiles with custom icons (`📦 Batches`, `🔋 Embeddings`, `🧬 Genomes`, `🔮 Predictions`, `⚖️ Pareto`, `🌍 Schedules`).
   - Database file size readout (`DB Size: X.X MB`).
   - Live timestamp and `🔄 Refresh All Data` cache clearing button.

---

## 📑 5. Tab-by-Tab Detailed Architecture

### Tab 1: 🎛️ Command Center
- **KPI Summary Strip (4 Columns)**:
  - Total Batches (`2,000 loaded`)
  - Average Batch Yield (`σ = 0.0842`)
  - Average Carbon Intensity (`gCO₂/kWh`)
  - Best Pareto Yield (`100 solutions`)
- **Carbon Radial Gauge (Left Column, Width 4/10)**:
  - Plotly radial speedometer ($0 - 600\text{ gCO}_2/\text{kWh}$) with 3 colored step bands.
  - Large 54px JetBrains Mono readout and dynamic zone banner strip below gauge.
- **Optimal Manufacturing Schedule (Right Column, Width 6/10)**:
  - 3-tier parameter grid formatted in 4-column cards:
    - *Row 1 (Process - Cyan)*: Temperature (°C), Pressure (bar), Speed (rpm), Feed Rate (kg/h).
    - *Row 2 (Material - Green)*: Density (g/cm³), Hardness (HV), Material Grade, Humidity (%).
    - *Row 3 (Outcomes - Yellow/Orange/Red)*: Pred Yield, Pred Quality, Pred Energy (kWh), Pred Carbon (kg).
- **Historical Decisions Table**:
  - Full-width table displaying past dispatch decisions and carbon zone records.

### Tab 2: 📈 Production Analytics
- **Summary Metrics**: Avg Yield, Avg Quality, Avg Energy, Avg Carbon.
- **Fleet Distribution (2 Columns)**:
  - Left: 60-bin Yield Distribution Histogram with cyan borders.
  - Right: Carbon Zone Distribution Donut Chart (Plotly Pie with 50% center hole).
- **Fleet-Wide Scatter Plot**:
  - Carbon Intensity vs. Energy Consumption across all 2,000 batches.
  - Vertical dashed boundary lines at $150$ and $400\text{ gCO}_2/\text{kWh}$.
- **Dual Correlation Scatters**:
  - Temperature vs. Yield scatter colored by zone.
  - Production Speed vs. Energy Consumption scatter colored by zone.
- **Feature Correlation Matrix & Batch Lookup**:
  - Left: $9 \times 9$ interactive heatmap (`#ff4b4b` negative, dark zero, `#00d4ff` positive).
  - Right: Batch Search Explorer with text prefix filter (`BATCH_00`).

### Tab 3: ⚖️ Pareto Intelligence
- **Filter Controls**:
  - Sliders for Minimum Predicted Yield and Maximum Predicted Carbon.
  - Dropdown to select color dimension (Yield, Quality, Energy, Carbon).
- **3D Pareto Frontier Space**:
  - 3-Axis scatter plot (Yield vs. Energy vs. Carbon) rendered with dark scene background (`rgb(13,17,23)`).
- **2D Pareto Trade-Off Front**:
  - Yield vs. Energy scatter with bubble size scaled by Product Quality and colored by Carbon.
- **Top 10 Pareto Solutions**:
  - Ranked horizontal bar chart highlighting the top 10 highest-yielding configurations.
- **Filtered Solutions Table**:
  - Clean exportable dataframe of optimal multi-variable parameters.

### Tab 4: 🧬 Genome Explorer
- **Individual Batch Selector**: Dropdown to inspect any batch from the dataset.
- **Process Profile Radar Chart**:
  - 5-Axis polar chart (Temperature, Pressure, Speed, Feed Rate, Humidity) normalized to percentile ranges ($0-100\%$).
- **25-Dimension Genome Vector Bar Chart**:
  - Bar chart plotting individual Z-scores for all 25 features.
  - Color-coded segments with vertical divider lines:
    - *Dimensions 0–4*: Process Parameters (Cyan)
    - *Dimensions 5–7*: Material Properties (Green)
    - *Dimensions 8–23*: Energy DNA Embeddings (Purple)
    - *Dimension 24*: Carbon Intensity (Yellow)
- **Population Heatmap**:
  - $25 \times N$ matrix displaying full genome feature distribution across batches.

### Tab 5: 🩺 System Health & Persistence
- **Relational Table Tiles**: 7 metric cards displaying live counts for all relational tables.
- **Database Storage Breakdown**: Horizontal bar chart comparing table volumes and storage consumption ($MB$).
- **Surrogate Accuracy Scatter**: Predicted vs. Actual yield scatter with $45^\circ$ reference line validating model calibration.
- **Pareto Objective Statistics**: Summary cards for Best Yield, Best Quality, Min Energy, and Min Carbon.
- **Pipeline Execution Audit Log**: Table recording execution timestamps, phase IDs, and completion statuses.

### Tab 6: 🤖 Digital Twin & Live Factory Mirror
- **Live Factory Health Banner**:
  - Dynamic status indicator (`FACTORY ONLINE 🟢`, `FACTORY DEGRADED 🟡`, or `FACTORY CRITICAL 🔴`).
  - `@keyframes dt_pulse` glowing animation based on anomaly rates ($>15\%$ Critical, $>7\%$ Degraded).
  - KPI pills: Total Batches, Anomalies count & %, Avg Yield, Avg Quality, Avg Energy.
- **`+ Simulate New Batch` Button**: Interactive button inserting a new synthetic batch into SQLite with immediate page re-render.
- **Live Machine Dials**: Speedometer gauges for Current Batch Temperature, Pressure, and Speed with 3-tier safe/warning/critical zones.
- **Rolling Multi-Batch Trend Lines**: Continuous time-series curves over the last 300 batches.

### Tabs 7 & 8: 🔌 Cyber-Physical Edge & Closed-Loop Control
- **Dual-Mode Hardware Selector**: Radio toggle between Physical ESP32 Hardware (WiFi/Serial) and Offline Synthetic Simulator.
- **Rolling Telemetry Buffer (300 Samples)**: Real-time time-series plots of Current ($A$), Active Power ($kW$), and Machine Temperature ($^\circ C$).
- **TreeSHAP Root Cause Attribution**: Waterfall plot displaying the exact parameter contributions driving batch anomalies.
- **Golden Signature Recipe Deltas**: Counterfactual parameter adjustments ($P_{\text{current}} \to P_{\text{optimal}}$).
- **MOSFET PWM Actuator Monitor**:
  - Real-time gauge of GPIO 18 gate drive duty cycle ($0-255$).
  - DC Cooling Fan RPM indicator and load-shedding status.
  - Live A/B impact metrics: $+2.8\%$ Yield, $-14.2\%$ Energy, $-28.5\%$ Carbon.

---

## 🚀 Execution & Verification
To launch this exact frontend dashboard:
```bash
# 1. Via Master CLI
python main.py --dashboard

# 2. Directly via Streamlit
streamlit run src/dashboard/app.py --server.port 8501

# 3. Concurrently with FastAPI Microservice
python main.py --serve
```
