-- Run once on first container boot (idempotent).
-- Alembic manages the actual schema migrations.

-- Enable UUID generation
CREATE EXTENSION IF NOT EXISTS "uuid-ossp";

-- Enable pg_trgm for fuzzy prompt-hash search (useful in Phase 5 analytics)
CREATE EXTENSION IF NOT EXISTS pg_trgm;

-- Enable btree_gist for range queries on timestamps (dashboard queries)
CREATE EXTENSION IF NOT EXISTS btree_gist;

-- Ensure the default search path includes public
ALTER DATABASE autopilot SET search_path TO public;

-- Grant privileges (user already exists from POSTGRES_USER env var)
GRANT ALL PRIVILEGES ON DATABASE autopilot TO autopilot;
