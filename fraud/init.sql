-- Initialize database for fraud detection system
CREATE DATABASE IF NOT EXISTS fraud_detection;

-- Use the database
\c fraud_detection;

-- Create tables
CREATE TABLE IF NOT EXISTS transactions (
    id SERIAL PRIMARY KEY,
    transaction_id VARCHAR(50) UNIQUE NOT NULL,
    account_id VARCHAR(50) NOT NULL,
    amount DECIMAL(15,2) NOT NULL,
    transaction_date TIMESTAMP NOT NULL,
    transaction_type VARCHAR(20),
    location VARCHAR(100),
    device_id VARCHAR(50),
    merchant_id VARCHAR(50),
    channel VARCHAR(20),
    risk_score DECIMAL(5,4),
    status VARCHAR(20) DEFAULT 'pending',
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
);

CREATE TABLE IF NOT EXISTS customers (
    id SERIAL PRIMARY KEY,
    customer_id VARCHAR(50) UNIQUE NOT NULL,
    name VARCHAR(100),
    email VARCHAR(100),
    phone VARCHAR(20),
    occupation VARCHAR(50),
    risk_profile JSONB,
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
);

CREATE TABLE IF NOT EXISTS fraud_alerts (
    id SERIAL PRIMARY KEY,
    transaction_id VARCHAR(50) REFERENCES transactions(transaction_id),
    alert_type VARCHAR(50),
    severity VARCHAR(20),
    description TEXT,
    status VARCHAR(20) DEFAULT 'open',
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
);

CREATE TABLE IF NOT EXISTS model_performance (
    id SERIAL PRIMARY KEY,
    model_name VARCHAR(50),
    version VARCHAR(20),
    accuracy DECIMAL(5,4),
    precision DECIMAL(5,4),
    recall DECIMAL(5,4),
    f1_score DECIMAL(5,4),
    auc_score DECIMAL(5,4),
    training_date TIMESTAMP,
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
);

-- Create indexes for better performance
CREATE INDEX IF NOT EXISTS idx_transactions_account_id ON transactions(account_id);
CREATE INDEX IF NOT EXISTS idx_transactions_date ON transactions(transaction_date);
CREATE INDEX IF NOT EXISTS idx_transactions_status ON transactions(status);
CREATE INDEX IF NOT EXISTS idx_fraud_alerts_transaction_id ON fraud_alerts(transaction_id);
CREATE INDEX IF NOT EXISTS idx_customers_customer_id ON customers(customer_id);

-- Create views for analytics
CREATE OR REPLACE VIEW daily_transaction_summary AS
SELECT 
    DATE(transaction_date) as transaction_date,
    COUNT(*) as transaction_count,
    SUM(amount) as total_amount,
    AVG(risk_score) as avg_risk_score,
    COUNT(CASE WHEN status = 'flagged' THEN 1 END) as flagged_count
FROM transactions
GROUP BY DATE(transaction_date)
ORDER BY transaction_date DESC;

CREATE OR REPLACE VIEW customer_risk_summary AS
SELECT 
    c.customer_id,
    c.name,
    COUNT(t.id) as total_transactions,
    AVG(t.risk_score) as avg_risk_score,
    COUNT(CASE WHEN t.status = 'flagged' THEN 1 END) as flagged_transactions,
    MAX(t.transaction_date) as last_transaction_date
FROM customers c
LEFT JOIN transactions t ON c.customer_id = t.account_id
GROUP BY c.customer_id, c.name
ORDER BY avg_risk_score DESC;
