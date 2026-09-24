"""BullMQ consumer for long jobs (gang-sheet compose) so they don't hold HTTP connections."""

import asyncio

from bullmq import Worker

from app.config import settings


async def process(job, job_token):
    # TODO: dispatch on job.name: "compose-sheet", "render-batch"
    return {"ok": True}


async def main() -> None:
    worker = Worker("render", process, {"connection": settings.redis_url})
    try:
        await asyncio.Event().wait()
    finally:
        await worker.close()


if __name__ == "__main__":
    asyncio.run(main())
