-- =====================================================================
-- ACMGS Relational Schema (PostgreSQL & SQLite Compatible)
-- Defines 7 relational operational tables for end-to-end auditability
-- =====================================================================

-- Table 1: Manufacturing Batches
CREATE TABLE IF NOT EXISTS batches (
    batch_id VARCHAR(64) PRIMARY KEY,
    temp_c REAL NOT NULL,
    pressure_bar REAL NOT NULL,
    cycle_time_s REAL NOT NULL,
    motor_speed_rpm REAL NOT NULL,
    material_density REAL,
    hardness_hrc REAL,
    feedstock_purity REAL,
    grid_carbon_intensity REAL,
    yield_pct REAL NOT NULL,
    quality_score REAL NOT NULL,
    energy_kwh REAL NOT NULL,
    carbon_kg REAL NOT NULL,
    status VARCHAR(32) DEFAULT 'COMPLETED',
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
);

-- Table 2: 25-D Batch Genomes
CREATE TABLE IF NOT EXISTS genomes (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    batch_id VARCHAR(64) NOT NULL UNIQUE,
    genome_vector TEXT NOT NULL, -- JSON-encoded 25-D float array
    dim_count INTEGER DEFAULT 25,
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    FOREIGN KEY (batch_id) REFERENCES batches(batch_id) ON DELETE CASCADE
);

-- Table 3: Pareto Optimal Recipes
CREATE TABLE IF NOT EXISTS recipes (
    recipe_id VARCHAR(64) PRIMARY KEY,
    temp_c REAL NOT NULL,
    pressure_bar REAL NOT NULL,
    cycle_time_s REAL NOT NULL,
    motor_speed_rpm REAL NOT NULL,
    predicted_yield_pct REAL,
    predicted_quality_score REAL,
    predicted_energy_kwh REAL,
    predicted_carbon_kg REAL,
    pareto_rank INTEGER DEFAULT 1,
    is_active BOOLEAN DEFAULT 1,
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
);

-- Table 4: Edge Sensor Telemetry Logs (ESP32 / Arduino / Simulator)
CREATE TABLE IF NOT EXISTS telemetry_logs (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    node_id VARCHAR(64) NOT NULL,
    timestamp TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    temp_c REAL NOT NULL,
    current_amps REAL NOT NULL,
    voltage_volts REAL DEFAULT 24.0,
    power_watts REAL NOT NULL,
    humidity_pct REAL DEFAULT 45.0,
    grid_carbon REAL,
    source VARCHAR(32) DEFAULT 'ESP32' -- 'ESP32', 'ARDUINO', 'SIMULATOR'
);

-- Table 5: Actuator Events & Closed-Loop Control Dispatch (MOSFET PWM)
CREATE TABLE IF NOT EXISTS actuator_events (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    node_id VARCHAR(64) NOT NULL,
    timestamp TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    mosfet_pwm_duty INTEGER NOT NULL, -- 0 to 255
    fan_speed_pct REAL NOT NULL,      -- 0.0 to 100.0%
    temp_c REAL NOT NULL,
    grid_zone VARCHAR(32) NOT NULL,
    dispatch_reason TEXT,
    command_ack BOOLEAN DEFAULT 1
);

-- Table 6: Machine Health & Prognostics Records (RUL)
CREATE TABLE IF NOT EXISTS machine_health_records (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    timestamp TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    health_index REAL NOT NULL,       -- 0.0 to 100.0%
    rul_hours REAL NOT NULL,          -- Estimated Remaining Useful Life in operating hours
    status_tier VARCHAR(32) NOT NULL, -- 'OPTIMAL', 'DEGRADED', 'CRITICAL'
    dominant_stress_factor VARCHAR(64),
    vibration_anomaly_score REAL DEFAULT 0.0
);

-- Table 7: System Operational Alerts & Incident Log
CREATE TABLE IF NOT EXISTS system_alerts (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    timestamp TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    alert_level VARCHAR(32) NOT NULL, -- 'INFO', 'WARNING', 'CRITICAL'
    component VARCHAR(64) NOT NULL,   -- 'MOSFET', 'GRID_DISPATCH', 'SENSOR_NODE', 'PREDICTOR'
    message TEXT NOT NULL,
    is_resolved BOOLEAN DEFAULT 0,
    resolved_at TIMESTAMP
);

-- Indexes for ultra-fast time-series and batch lookups
CREATE INDEX IF NOT EXISTS idx_batches_created_at ON batches(created_at);
CREATE INDEX IF NOT EXISTS idx_telemetry_node_time ON telemetry_logs(node_id, timestamp);
CREATE INDEX IF NOT EXISTS idx_actuator_time ON actuator_events(timestamp);
CREATE INDEX IF NOT EXISTS idx_health_time ON machine_health_records(timestamp);
CREATE INDEX IF NOT EXISTS idx_alerts_level ON system_alerts(alert_level, is_resolved);
