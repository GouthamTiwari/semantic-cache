-- Semantic cache schema (SPEC section 10.2). {dim} is replaced with the embedding dimension.
-- One table per dimension, because an HNSW index needs a fixed vector size.
CREATE EXTENSION IF NOT EXISTS vector;

CREATE TABLE IF NOT EXISTS semcache_entries_{dim} (
    id            uuid PRIMARY KEY,
    scope_key     text             NOT NULL,
    namespace     text             NOT NULL,
    model_id      text             NOT NULL,
    exact_key     text             NOT NULL,
    query_text    text             NOT NULL,
    query_norm    text             NOT NULL,
    embedding     vector({dim})    NOT NULL,
    answer_text   text             NOT NULL,
    stop_reason   text             NOT NULL,
    input_tokens  integer          NOT NULL,
    output_tokens integer          NOT NULL,
    cost_usd      double precision NOT NULL,
    ttl_class     text             NOT NULL,
    created_at    timestamptz      NOT NULL,
    expires_at    timestamptz      NOT NULL,
    hit_count     integer          NOT NULL DEFAULT 0,
    last_hit_at   timestamptz,
    extra         jsonb            NOT NULL DEFAULT '{}'::jsonb
);

CREATE UNIQUE INDEX IF NOT EXISTS semcache_entries_{dim}_exact
    ON semcache_entries_{dim} (exact_key);
CREATE INDEX IF NOT EXISTS semcache_entries_{dim}_scope
    ON semcache_entries_{dim} (scope_key, expires_at);
CREATE INDEX IF NOT EXISTS semcache_entries_{dim}_ns
    ON semcache_entries_{dim} (namespace);
CREATE INDEX IF NOT EXISTS semcache_entries_{dim}_hnsw
    ON semcache_entries_{dim} USING hnsw (embedding vector_cosine_ops);
