"""llm_call_log - LLM 调用逐次统计(治理层)

Revision ID: 0010
Revises: 0009
Create Date: 2026-07-17
"""
from alembic import op

revision = "0010"
down_revision = "0009"
branch_labels = None
depends_on = None


def upgrade() -> None:
    op.execute("""
        CREATE TABLE IF NOT EXISTS llm_call_log (
            id            BIGSERIAL PRIMARY KEY,
            created_at    timestamptz NOT NULL DEFAULT now(),
            call_type     text NOT NULL,
            model         text NOT NULL,
            source        text NOT NULL,
            session_id    text,
            status        text NOT NULL,
            error_type    text,
            attempts      int NOT NULL,
            latency_ms    int NOT NULL,
            input_tokens  int,
            output_tokens int,
            cost          numeric(12, 6)
        )
    """)
    op.execute(
        "CREATE INDEX IF NOT EXISTS idx_llm_call_log_created_at ON llm_call_log (created_at)"
    )
    op.execute(
        "CREATE INDEX IF NOT EXISTS idx_llm_call_log_model_created"
        " ON llm_call_log (model, created_at)"
    )


def downgrade() -> None:
    op.execute("DROP TABLE IF EXISTS llm_call_log")
