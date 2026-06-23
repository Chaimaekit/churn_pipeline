-- Supabase Schema for Telecom Churn Pipeline
-- Run this in Supabase SQL Editor

-- Enable UUID extension
CREATE EXTENSION IF NOT EXISTS "uuid-ossp";

-- ============================================
-- 1. RAW SCRAPED COMMENTS (from scrape_facebook.py)
-- ============================================
CREATE TABLE IF NOT EXISTS scraped_comments (
    id UUID PRIMARY KEY DEFAULT uuid_generate_v4(),
    customer_id TEXT NOT NULL,
    post_url TEXT NOT NULL,
    username TEXT NOT NULL,
    text TEXT NOT NULL,
    scraped_at TIMESTAMPTZ NOT NULL DEFAULT NOW(),
    created_at TIMESTAMPTZ NOT NULL DEFAULT NOW()
);

-- Index for fast lookups by customer
CREATE INDEX IF NOT EXISTS idx_scraped_comments_customer 
ON scraped_comments(customer_id);

-- Index for time-based queries
CREATE INDEX IF NOT EXISTS idx_scraped_comments_scraped_at 
ON scraped_comments(scraped_at DESC);

-- ============================================
-- 2. PROCESSED FEEDBACK (from feedback_analyse.py / LLM output)
-- ============================================
CREATE TABLE IF NOT EXISTS processed_feedback (
    id UUID PRIMARY KEY DEFAULT uuid_generate_v4(),
    customer_id TEXT NOT NULL,
    feedback_category TEXT NOT NULL CHECK (feedback_category IN (
        'pricing', 'service_quality', 'customer_support', 'churn_intent'
    )),
    sentiment TEXT NOT NULL CHECK (sentiment IN ('positive', 'neutral', 'negative')),
    complaint_intensity INTEGER NOT NULL CHECK (complaint_intensity BETWEEN 1 AND 5),
    raw_text TEXT NOT NULL,
    analyzed_at TIMESTAMPTZ NOT NULL DEFAULT NOW(),
    created_at TIMESTAMPTZ NOT NULL DEFAULT NOW()
);

CREATE INDEX IF NOT EXISTS idx_processed_feedback_customer 
ON processed_feedback(customer_id);

CREATE INDEX IF NOT EXISTS idx_processed_feedback_category 
ON processed_feedback(feedback_category);

CREATE INDEX IF NOT EXISTS idx_processed_feedback_sentiment 
ON processed_feedback(sentiment);

-- ============================================
-- 3. TELECOM SUBSCRIBERS (from CDR/billing data)
-- ============================================
CREATE TABLE IF NOT EXISTS subscribers (
    id UUID PRIMARY KEY DEFAULT uuid_generate_v4(),
    customer_id TEXT UNIQUE NOT NULL,
    state TEXT,
    account_length INTEGER,
    area_code INTEGER,
    international_plan TEXT,
    voice_mail_plan TEXT,
    number_vmail_messages INTEGER,
    total_day_minutes REAL,
    total_day_calls INTEGER,
    total_day_charge REAL,
    total_eve_minutes REAL,
    total_eve_calls INTEGER,
    total_eve_charge REAL,
    total_night_minutes REAL,
    total_night_calls INTEGER,
    total_night_charge REAL,
    total_intl_minutes REAL,
    total_intl_calls INTEGER,
    total_intl_charge REAL,
    customer_service_calls INTEGER,
    churn INTEGER DEFAULT 0,
    feedback_text TEXT,
    feedback_category TEXT,
    sentiment TEXT,
    complaint_intensity INTEGER,
    created_at TIMESTAMPTZ NOT NULL DEFAULT NOW(),
    updated_at TIMESTAMPTZ NOT NULL DEFAULT NOW()
);

CREATE INDEX IF NOT EXISTS idx_subscribers_customer_id 
ON subscribers(customer_id);

CREATE INDEX IF NOT EXISTS idx_subscribers_churn 
ON subscribers(churn);

-- ============================================
-- 4. PREDICTIONS (from model inference)
-- ============================================
CREATE TABLE IF NOT EXISTS predictions (
    id UUID PRIMARY KEY DEFAULT uuid_generate_v4(),
    customer_id TEXT NOT NULL REFERENCES subscribers(customer_id),
    churn_probability REAL NOT NULL,
    churn_risk_level TEXT NOT NULL CHECK (churn_risk_level IN ('High', 'Low')),
    action_required BOOLEAN NOT NULL,
    model_version TEXT DEFAULT 'champion_catboost',
    predicted_at TIMESTAMPTZ NOT NULL DEFAULT NOW()
);

CREATE INDEX IF NOT EXISTS idx_predictions_customer 
ON predictions(customer_id);

CREATE INDEX IF NOT EXISTS idx_predictions_risk 
ON predictions(churn_risk_level) 
WHERE churn_risk_level = 'High';

-- ============================================
-- 5. BRAND REPUTATION SNAPSHOTS (daily aggregates)
-- ============================================
CREATE TABLE IF NOT EXISTS reputation_snapshots (
    id UUID PRIMARY KEY DEFAULT uuid_generate_v4(),
    snapshot_date DATE NOT NULL DEFAULT CURRENT_DATE,
    total_comments INTEGER NOT NULL,
    net_reputation_score REAL NOT NULL,
    brand_health_status TEXT NOT NULL,
    avg_complaint_intensity REAL NOT NULL,
    positive_count INTEGER NOT NULL,
    neutral_count INTEGER NOT NULL,
    negative_count INTEGER NOT NULL,
    top_complaint_category TEXT,
    created_at TIMESTAMPTZ NOT NULL DEFAULT NOW()
);

CREATE UNIQUE INDEX IF NOT EXISTS idx_reputation_snapshot_date 
ON reputation_snapshots(snapshot_date);

-- ============================================
-- 6. ROW LEVEL SECURITY (enable after creating tables)
-- ============================================
ALTER TABLE scraped_comments ENABLE ROW LEVEL SECURITY;
ALTER TABLE processed_feedback ENABLE ROW LEVEL SECURITY;
ALTER TABLE subscribers ENABLE ROW LEVEL SECURITY;
ALTER TABLE predictions ENABLE ROW LEVEL SECURITY;
ALTER TABLE reputation_snapshots ENABLE ROW LEVEL SECURITY;

-- Policy: analysts can read all, only system can write
CREATE POLICY "analysts_read_all" ON scraped_comments
    FOR SELECT USING (true);

CREATE POLICY "analysts_read_all" ON processed_feedback
    FOR SELECT USING (true);

CREATE POLICY "analysts_read_all" ON subscribers
    FOR SELECT USING (true);

CREATE POLICY "analysts_read_all" ON predictions
    FOR SELECT USING (true);

CREATE POLICY "analysts_read_all" ON reputation_snapshots
    FOR SELECT USING (true);

-- ============================================
-- 7. FUNCTIONS FOR /reputation API
-- ============================================
CREATE OR REPLACE FUNCTION get_reputation_summary()
RETURNS JSON AS $$
DECLARE
    total_comments INTEGER;
    pos INTEGER;
    neu INTEGER;
    neg INTEGER;
    avg_intensity REAL;
    net_score REAL;
    top_cat TEXT;
BEGIN
    SELECT COUNT(*) INTO total_comments FROM processed_feedback;

    SELECT COUNT(*) INTO pos FROM processed_feedback WHERE sentiment = 'positive';
    SELECT COUNT(*) INTO neu FROM processed_feedback WHERE sentiment = 'neutral';
    SELECT COUNT(*) INTO neg FROM processed_feedback WHERE sentiment = 'negative';

    SELECT AVG(complaint_intensity) INTO avg_intensity FROM processed_feedback;

    net_score := ROUND(((pos - neg)::REAL / NULLIF(total_comments, 0)) * 100, 2);

    SELECT feedback_category INTO top_cat
    FROM processed_feedback
    GROUP BY feedback_category
    ORDER BY COUNT(*) DESC
    LIMIT 1;

    RETURN json_build_object(
        'total_social_records_evaluated', total_comments,
        'net_reputation_score', COALESCE(net_score, 0),
        'brand_health_status', CASE 
            WHEN net_score >= 30 THEN 'Excellent / Highly Positive Brand Equity'
            WHEN net_score >= 0 THEN 'Stable / Generally Neutral'
            WHEN net_score >= -30 THEN 'At Risk / Negative Operational Friction'
            ELSE 'Critical Alert / High Churn and Public Backlash'
        END,
        'average_complaint_intensity', ROUND(COALESCE(avg_intensity, 0), 2),
        'sentiment_distribution', json_build_object(
            'positive', COALESCE(pos, 0),
            'neutral', COALESCE(neu, 0),
            'negative', COALESCE(neg, 0)
        ),
        'top_operational_complaints', top_cat
    );
END;
$$ LANGUAGE plpgsql;

-- ============================================
-- 8. REALTIME: auto-update reputation on new feedback
-- ============================================
CREATE OR REPLACE FUNCTION update_reputation_on_insert()
RETURNS TRIGGER AS $$
BEGIN
    -- Insert or update today's snapshot
    INSERT INTO reputation_snapshots (
        snapshot_date, total_comments, net_reputation_score, 
        brand_health_status, avg_complaint_intensity,
        positive_count, neutral_count, negative_count, top_complaint_category
    )
    SELECT 
        CURRENT_DATE,
        COUNT(*),
        ROUND(((SUM(CASE WHEN sentiment = 'positive' THEN 1 ELSE 0 END) - 
                SUM(CASE WHEN sentiment = 'negative' THEN 1 ELSE 0 END))::REAL / 
                NULLIF(COUNT(*), 0)) * 100, 2),
        'auto',
        ROUND(AVG(complaint_intensity), 2),
        SUM(CASE WHEN sentiment = 'positive' THEN 1 ELSE 0 END),
        SUM(CASE WHEN sentiment = 'neutral' THEN 1 ELSE 0 END),
        SUM(CASE WHEN sentiment = 'negative' THEN 1 ELSE 0 END),
        (SELECT feedback_category FROM processed_feedback 
         GROUP BY feedback_category ORDER BY COUNT(*) DESC LIMIT 1)
    FROM processed_feedback
    ON CONFLICT (snapshot_date) DO UPDATE SET
        total_comments = EXCLUDED.total_comments,
        net_reputation_score = EXCLUDED.net_reputation_score,
        brand_health_status = EXCLUDED.brand_health_status,
        avg_complaint_intensity = EXCLUDED.avg_complaint_intensity,
        positive_count = EXCLUDED.positive_count,
        neutral_count = EXCLUDED.neutral_count,
        negative_count = EXCLUDED.negative_count,
        top_complaint_category = EXCLUDED.top_complaint_category;

    RETURN NEW;
END;
$$ LANGUAGE plpgsql;

-- Trigger: update reputation snapshot on every new feedback
DROP TRIGGER IF EXISTS trg_update_reputation ON processed_feedback;
CREATE TRIGGER trg_update_reputation
    AFTER INSERT ON processed_feedback
    FOR EACH STATEMENT
    EXECUTE FUNCTION update_reputation_on_insert();