-- Rise Finder Agent Swarm — Supabase schema
-- Run this in the Supabase SQL Editor (Dashboard → SQL Editor → New query)

-- =============================================================
-- 1. signals — raw data from github, social, and enrichment agents
-- =============================================================
CREATE TABLE IF NOT EXISTS signals (
    id              BIGINT GENERATED ALWAYS AS IDENTITY PRIMARY KEY,
    project_name    TEXT NOT NULL,
    source          TEXT NOT NULL CHECK (source IN ('github', 'social', 'enrichment')),
    data            JSONB NOT NULL DEFAULT '{}',
    star_velocity_7d  DOUBLE PRECISION,
    stars             INTEGER,
    social_score      DOUBLE PRECISION,
    enrichment_score  DOUBLE PRECISION,
    created_at      TIMESTAMPTZ NOT NULL DEFAULT now(),
    updated_at      TIMESTAMPTZ NOT NULL DEFAULT now(),
    UNIQUE (project_name, source)
);

CREATE INDEX idx_signals_project     ON signals (project_name);
CREATE INDEX idx_signals_source      ON signals (source);
CREATE INDEX idx_signals_created     ON signals (created_at DESC);
CREATE INDEX idx_signals_star_vel    ON signals (star_velocity_7d DESC) WHERE source = 'github';

-- =============================================================
-- 2. composite_scores — scored results per query
-- =============================================================
CREATE TABLE IF NOT EXISTS composite_scores (
    id                BIGINT GENERATED ALWAYS AS IDENTITY PRIMARY KEY,
    project_name      TEXT NOT NULL,
    composite_score   DOUBLE PRECISION NOT NULL DEFAULT 0,
    detailed_scores   JSONB NOT NULL DEFAULT '{}',
    has_partial_data  BOOLEAN NOT NULL DEFAULT false,
    query_id          TEXT,
    created_at        TIMESTAMPTZ NOT NULL DEFAULT now(),
    updated_at        TIMESTAMPTZ NOT NULL DEFAULT now(),
    UNIQUE (project_name)
);

CREATE INDEX idx_comp_score       ON composite_scores (composite_score DESC);
CREATE INDEX idx_comp_query       ON composite_scores (query_id);
CREATE INDEX idx_comp_created     ON composite_scores (created_at DESC);

-- =============================================================
-- 3. investor_query_log — every query that enters the pipeline
-- =============================================================
CREATE TABLE IF NOT EXISTS investor_query_log (
    id                BIGINT GENERATED ALWAYS AS IDENTITY PRIMARY KEY,
    query_id          TEXT NOT NULL UNIQUE,
    categories        JSONB NOT NULL DEFAULT '[]',
    "limit"           INTEGER NOT NULL DEFAULT 20,
    risk_profile      TEXT NOT NULL DEFAULT 'moderate',
    stage_preference  TEXT NOT NULL DEFAULT 'seed',
    bot_profile_name  TEXT,
    status            TEXT NOT NULL DEFAULT 'pending',
    results_count     INTEGER,
    processing_time_ms DOUBLE PRECISION,
    created_at        TIMESTAMPTZ NOT NULL DEFAULT now()
);

CREATE INDEX idx_query_status     ON investor_query_log (status);
CREATE INDEX idx_query_created    ON investor_query_log (created_at DESC);
CREATE INDEX idx_query_bot        ON investor_query_log (bot_profile_name) WHERE bot_profile_name IS NOT NULL;

-- =============================================================
-- 4. memos — AI-generated investment memos
-- =============================================================
CREATE TABLE IF NOT EXISTS memos (
    id                BIGINT GENERATED ALWAYS AS IDENTITY PRIMARY KEY,
    project_name      TEXT NOT NULL UNIQUE,
    summary           TEXT NOT NULL DEFAULT '',
    recommendation    TEXT NOT NULL DEFAULT '',
    risk_factors      JSONB NOT NULL DEFAULT '[]',
    bull_case         TEXT NOT NULL DEFAULT '',
    bear_case         TEXT NOT NULL DEFAULT '',
    raw_text          TEXT NOT NULL DEFAULT '',
    inference_time_ms DOUBLE PRECISION,
    tokens_used       INTEGER,
    gpu_util_during   DOUBLE PRECISION,
    model_used        TEXT,
    created_at        TIMESTAMPTZ NOT NULL DEFAULT now(),
    updated_at        TIMESTAMPTZ NOT NULL DEFAULT now()
);

CREATE INDEX idx_memos_project    ON memos (project_name);
CREATE INDEX idx_memos_created    ON memos (created_at DESC);

-- =============================================================
-- 5. bot_runs — log of every synthetic bot query
-- =============================================================
CREATE TABLE IF NOT EXISTS bot_runs (
    id                BIGINT GENERATED ALWAYS AS IDENTITY PRIMARY KEY,
    bot_profile_name  TEXT NOT NULL,
    query_params      JSONB NOT NULL DEFAULT '{}',
    response_time_ms  DOUBLE PRECISION,
    success           BOOLEAN NOT NULL DEFAULT true,
    results_count     INTEGER NOT NULL DEFAULT 0,
    started_at        TIMESTAMPTZ NOT NULL DEFAULT now(),
    completed_at      TIMESTAMPTZ
);

CREATE INDEX idx_bot_profile      ON bot_runs (bot_profile_name);
CREATE INDEX idx_bot_completed    ON bot_runs (completed_at DESC);
CREATE INDEX idx_bot_success      ON bot_runs (success);

-- =============================================================
-- 6. gpu_metrics — GPU utilization time series
-- =============================================================
CREATE TABLE IF NOT EXISTS gpu_metrics (
    id              BIGINT GENERATED ALWAYS AS IDENTITY PRIMARY KEY,
    gpu_util_pct    DOUBLE PRECISION NOT NULL DEFAULT 0,
    vram_used_mb    DOUBLE PRECISION NOT NULL DEFAULT 0,
    vram_total_mb   DOUBLE PRECISION NOT NULL DEFAULT 0,
    power_draw_w    DOUBLE PRECISION NOT NULL DEFAULT 0,
    temperature_c   DOUBLE PRECISION NOT NULL DEFAULT 0,
    is_mock         BOOLEAN NOT NULL DEFAULT false,
    recorded_at     TIMESTAMPTZ NOT NULL DEFAULT now()
);

CREATE INDEX idx_gpu_recorded     ON gpu_metrics (recorded_at DESC);

-- =============================================================
-- 7. queue_metrics — pipeline throughput time series
-- =============================================================
CREATE TABLE IF NOT EXISTS queue_metrics (
    id                BIGINT GENERATED ALWAYS AS IDENTITY PRIMARY KEY,
    queue_depth       INTEGER NOT NULL DEFAULT 0,
    total_processed   INTEGER NOT NULL DEFAULT 0,
    total_failed      INTEGER NOT NULL DEFAULT 0,
    runs_per_minute   INTEGER NOT NULL DEFAULT 0,
    avg_latency_ms    DOUBLE PRECISION NOT NULL DEFAULT 0,
    success_rate_pct  DOUBLE PRECISION NOT NULL DEFAULT 100,
    cumulative_runs   INTEGER NOT NULL DEFAULT 0,
    "timestamp"       TIMESTAMPTZ NOT NULL DEFAULT now()
);

CREATE INDEX idx_queue_ts         ON queue_metrics ("timestamp" DESC);

-- =============================================================
-- Auto-update updated_at triggers
-- =============================================================
CREATE OR REPLACE FUNCTION update_updated_at()
RETURNS TRIGGER AS $$
BEGIN
    NEW.updated_at = now();
    RETURN NEW;
END;
$$ LANGUAGE plpgsql;

CREATE TRIGGER trg_signals_updated
    BEFORE UPDATE ON signals
    FOR EACH ROW EXECUTE FUNCTION update_updated_at();

CREATE TRIGGER trg_composite_scores_updated
    BEFORE UPDATE ON composite_scores
    FOR EACH ROW EXECUTE FUNCTION update_updated_at();

CREATE TRIGGER trg_memos_updated
    BEFORE UPDATE ON memos
    FOR EACH ROW EXECUTE FUNCTION update_updated_at();

-- =============================================================
-- Row Level Security — service key bypasses RLS,
-- anon/authenticated can read scores, memos, and metrics
-- =============================================================
ALTER TABLE signals             ENABLE ROW LEVEL SECURITY;
ALTER TABLE composite_scores    ENABLE ROW LEVEL SECURITY;
ALTER TABLE investor_query_log  ENABLE ROW LEVEL SECURITY;
ALTER TABLE memos               ENABLE ROW LEVEL SECURITY;
ALTER TABLE bot_runs            ENABLE ROW LEVEL SECURITY;
ALTER TABLE gpu_metrics         ENABLE ROW LEVEL SECURITY;
ALTER TABLE queue_metrics       ENABLE ROW LEVEL SECURITY;

-- Public read for frontend-facing tables
CREATE POLICY "Public read composite_scores"  ON composite_scores  FOR SELECT USING (true);
CREATE POLICY "Public read memos"             ON memos             FOR SELECT USING (true);
CREATE POLICY "Public read gpu_metrics"       ON gpu_metrics       FOR SELECT USING (true);
CREATE POLICY "Public read queue_metrics"     ON queue_metrics     FOR SELECT USING (true);
CREATE POLICY "Public read signals"           ON signals           FOR SELECT USING (true);

-- Service-role full access (inserts/updates come from backend with service key)
CREATE POLICY "Service insert signals"            ON signals            FOR INSERT WITH CHECK (true);
CREATE POLICY "Service update signals"            ON signals            FOR UPDATE USING (true);
CREATE POLICY "Service insert composite_scores"   ON composite_scores   FOR INSERT WITH CHECK (true);
CREATE POLICY "Service update composite_scores"   ON composite_scores   FOR UPDATE USING (true);
CREATE POLICY "Service insert investor_query_log" ON investor_query_log FOR INSERT WITH CHECK (true);
CREATE POLICY "Service update investor_query_log" ON investor_query_log FOR UPDATE USING (true);
CREATE POLICY "Service insert memos"              ON memos              FOR INSERT WITH CHECK (true);
CREATE POLICY "Service update memos"              ON memos              FOR UPDATE USING (true);
CREATE POLICY "Service insert bot_runs"           ON bot_runs           FOR INSERT WITH CHECK (true);
CREATE POLICY "Service insert gpu_metrics"        ON gpu_metrics        FOR INSERT WITH CHECK (true);
CREATE POLICY "Service insert queue_metrics"      ON queue_metrics      FOR INSERT WITH CHECK (true);

-- =============================================================
-- Realtime — enable for tables the Lovable frontend subscribes to
-- =============================================================
ALTER PUBLICATION supabase_realtime ADD TABLE composite_scores;
ALTER PUBLICATION supabase_realtime ADD TABLE memos;
ALTER PUBLICATION supabase_realtime ADD TABLE gpu_metrics;
ALTER PUBLICATION supabase_realtime ADD TABLE queue_metrics;

-- =============================================================
-- Cleanup policy helper (optional — run on cron via pg_cron)
-- Keeps gpu_metrics and queue_metrics to last 24h
-- =============================================================
CREATE OR REPLACE FUNCTION cleanup_old_metrics()
RETURNS void AS $$
BEGIN
    DELETE FROM gpu_metrics   WHERE recorded_at  < now() - INTERVAL '24 hours';
    DELETE FROM queue_metrics WHERE "timestamp"  < now() - INTERVAL '24 hours';
END;
$$ LANGUAGE plpgsql;
