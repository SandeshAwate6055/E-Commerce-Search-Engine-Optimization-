-- E-Commerce Product Search: PostgreSQL Schema

CREATE TABLE IF NOT EXISTS products (
    id               SERIAL PRIMARY KEY,
    pid              TEXT NOT NULL UNIQUE,
    image_path       TEXT NOT NULL,
    product_name     TEXT,
    main_category    TEXT,
    brand            TEXT,
    retail_price     NUMERIC,
    discounted_price NUMERIC,
    raw_caption      TEXT,
    norm_text        TEXT NOT NULL,
    created_at       TIMESTAMP DEFAULT CURRENT_TIMESTAMP
);

CREATE INDEX IF NOT EXISTS idx_pid ON products(pid);
CREATE INDEX IF NOT EXISTS idx_category ON products(main_category);
CREATE INDEX IF NOT EXISTS idx_image_path ON products(image_path);
