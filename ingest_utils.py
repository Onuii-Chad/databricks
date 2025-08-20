# ingest_utils.py
from typing import Dict, List, Optional, Tuple
from pyspark.sql import DataFrame, SparkSession
from pyspark.sql.functions import col, lit


# scylladb 연결 및 현재 df 추출
def read_scylla(
    spark: SparkSession,
    hosts: str,
    user: str,
    password: str,
    keyspace: str,
    table: str,
    local_dc: Optional[str] = None,
    ssl_enabled: Optional[bool] = None,
    extra_options: Optional[Dict[str, str]] = None,
) -> DataFrame:
    """
    Scylla/Cassandra 소스에서 읽기.
    keyspace / table 필수.
    """
    reader = (spark.read.format("org.apache.spark.sql.cassandra")
              .option("spark.cassandra.connection.host", hosts)
              .option("spark.cassandra.auth.username", user)
              .option("spark.cassandra.auth.password", password)
              .options(table=table, keyspace=keyspace))
    if local_dc:
        reader = reader.option("spark.cassandra.connection.localDC", local_dc)
    if ssl_enabled is not None:
        reader = reader.option("spark.cassandra.connection.ssl.enabled", "true" if ssl_enabled else "false")
    if extra_options:
        for k, v in extra_options.items():
            reader = reader.option(k, v)
    return reader.load()


# mongodb 연결 및 현재 df 추출
def read_mongo(
    spark: SparkSession,
    connection_uri: str,
    database: str,
    collection: str,
    pipeline: Optional[str] = None,
    extra_options: Optional[Dict[str, str]] = None,
) -> DataFrame:
    """
    MongoDB Spark Connector v10.x (DataSource V2) 기준.
    pipeline 은 JSON 배열 문자열('[{...}, {...}]') 형태 가능.
    """
    reader = (spark.read.format("mongodb")
              .option("connection.uri", connection_uri)
              .option("database", database)
              .option("collection", collection))
    if pipeline:
        reader = reader.option("aggregation.pipeline", pipeline)
    if extra_options:
        for k, v in extra_options.items():
            reader = reader.option(k, v)
    return reader.load()



# bronze 테이블 기준 마지막 업데이트 이후 row 추출
def apply_incremental(df: DataFrame, incr_col: Optional[str], last_ts: Optional[str]) -> DataFrame:
    """
    증분 컬럼(incr_col)과 마지막 적재 시각(last_ts)이 있으면 필터 적용.
    """
    if incr_col and last_ts:
        return df.filter(col(incr_col) > lit(last_ts))
    return df


# 타겟 테이블 생성
def ensure_target_table(
    spark: SparkSession,
    df: DataFrame,
    target_table: str,
    partition_by: Optional[List[str]] = None,
    overwrite_schema: bool = True,
) -> None:
    """
    타겟 Delta 테이블이 없으면 스키마를 시딩해서 생성.
    """
    jcatalog = spark._jsparkSession.catalog()
    if not jcatalog.tableExists(target_table):
        writer = (df.limit(0).write
                  .format("delta")
                  .mode("overwrite")
                  .option("overwriteSchema", "true" if overwrite_schema else "false"))
        if partition_by:
            writer = writer.partitionBy(*partition_by)
        writer.saveAsTable(target_table)


# upsert
def merge_upsert_all_columns(
    spark: SparkSession,
    src: DataFrame,
    target_table: str,
    pk_cols: List[str],
) -> None:
    """
    Delta MERGE: PK 기준으로 존재하면 UPDATE, 없으면 INSERT (모든 컬럼).
    """
    from delta.tables import DeltaTable
    cond = " AND ".join([f"t.{k}=s.{k}" for k in pk_cols])
    delta_tbl = DeltaTable.forName(spark, target_table)
    (delta_tbl.alias("t")
     .merge(src.alias("s"), cond)
     .whenMatchedUpdateAll()
     .whenNotMatchedInsertAll()
     .execute())