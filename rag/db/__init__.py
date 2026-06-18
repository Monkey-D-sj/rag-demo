# db 层已迁移至异步实现（psycopg3 AsyncConnectionPool + redis.asyncio）
# 使用 create_pg_pool / get_cursor / create_redis_client 替代旧同步接口
