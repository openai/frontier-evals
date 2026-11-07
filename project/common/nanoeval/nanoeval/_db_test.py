from uuid import uuid4

import pytest

from nanoeval._db import open_run_set_db


@pytest.mark.asyncio
async def test_reopening_run_set_preserves_max_concurrency() -> None:
    run_set_id = f"test-{uuid4()}"

    async with open_run_set_db(backup=False, run_set_id=run_set_id) as db:
        with db.conn() as conn:
            conn.execute(
                "UPDATE metadata SET value = ? WHERE key = 'max_concurrency'",
                (7,),
            )
            conn.commit()

    async with open_run_set_db(backup=False, run_set_id=run_set_id) as db:
        with db.conn() as conn:
            value = conn.execute(
                "SELECT value FROM metadata WHERE key = 'max_concurrency'"
            ).fetchone()[0]

    assert int(value) == 7
