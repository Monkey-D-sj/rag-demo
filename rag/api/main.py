from contextlib import asynccontextmanager

from fastapi import FastAPI

from rag.api.modules import register_modules

@asynccontextmanager
async def lifespan(app: FastAPI):
	from rag.common.logging import setup_logging
	setup_logging()
	
	from rag.db import create_pg_pool, create_redis_client
	# ------ 初始化pg -------
	pool = await create_pg_pool()
	app.state.pg = pool
	
	# ------ 初始化redis -------
	resis = await create_redis_client()
	app.state.redis = resis


app = FastAPI(lifespan=lifespan)
register_modules(app)



