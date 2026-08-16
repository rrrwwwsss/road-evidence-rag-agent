"""
SQL 查询服务：把自然语言查询问题转成只读 SQL，查询案件数据库（data/wupin_tanwei_dabt.db），返回可读结果。
"""
import re
import sqlite3
from pathlib import Path
from utils.logger_handler import logger
from utils.config_handler import agent_conf
from utils.path_tool import get_abs_path
from model.factory import chat_model
from langchain_core.prompts import PromptTemplate
from langchain_core.output_parsers import StrOutputParser


SQL_PROMPT_TEXT = """你是SQL专家，根据下面的数据库结构，把用户问题转换为一条只读的 SELECT 查询SQL。

数据库结构：
{schema}

注意事项：
1. 只能输出一条SELECT查询（可含WITH子句），禁止UPDATE/DELETE/INSERT等写操作，禁止多条语句。
2. 发生时间为文本格式"YYYY-MM-DD HH:MM:SS"，按日期过滤请用 substr(发生时间,1,10)='2026-07-23' 或 发生时间 LIKE '2026-07-23%'。
3. 若可能返回大量行，请用 LIMIT 100 限制。
4. 表名与列名必须来自数据库结构，不要臆造。

用户问题：{question}

只输出SQL语句本身，不要输出任何解释或代码块标记。"""


class SqlQueryService:
    def __init__(self):
        self.db_path = get_abs_path(agent_conf["sqlite_db_path"])
        self.model = chat_model
        self.prompt_template = PromptTemplate.from_template(SQL_PROMPT_TEXT)
        self.chain = self.prompt_template | self.model | StrOutputParser()
        self._schema_cache: str = ""

    def _connect(self) -> sqlite3.Connection:
        uri = Path(self.db_path).as_uri() + "?mode=ro"
        conn = sqlite3.connect(uri, uri=True, timeout=10)
        conn.row_factory = sqlite3.Row
        return conn

    def schema_text(self) -> str:
        """生成数据库结构描述，供模型编写SQL时参考。"""
        if self._schema_cache:
            return self._schema_cache

        lines = []
        conn = self._connect()
        try:
            tables = [r[0] for r in conn.execute(
                "SELECT name FROM sqlite_master WHERE type='table' AND name NOT LIKE 'sqlite_%'"
            )]
            for table in tables:
                cols = conn.execute(f"PRAGMA table_info({table})").fetchall()
                count = conn.execute(f"SELECT COUNT(*) FROM {table}").fetchone()[0]
                lines.append(f"表 {table}（{count} 行）")
                for col in cols:
                    lines.append(f"  - {col[1]} ({col[2]})")

                try:
                    types = [r[0] for r in conn.execute(
                        f"SELECT DISTINCT 违法类型 FROM {table} LIMIT 8"
                    )]
                    lines.append("  违法类型样例：" + "、".join(str(t) for t in types))
                except sqlite3.Error:
                    pass

                try:
                    mn, mx = conn.execute(
                        f"SELECT MIN(发生时间), MAX(发生时间) FROM {table}"
                    ).fetchone()
                    lines.append(f"  发生时间范围：{mn} ~ {mx}")
                except sqlite3.Error:
                    pass

            lines.append("字段说明：以上为数据库实际存在的表；results 为AI审核后的案件表（含所属支队、图片URL、is_committed：0未确认违法/1违法，所属支队可能为空，is_committed为0表示未确认违法、1表示违法）。")
        finally:
            conn.close()

        self._schema_cache = "\n".join(lines)
        return self._schema_cache

    def _to_sql(self, question: str, error_hint: str = "") -> str:
        full_question = question
        if error_hint:
            full_question = f"{question}\n（上次生成的SQL执行失败，错误：{error_hint}，请修正后重新输出）"

        sql = self.chain.invoke({"question": full_question, "schema": self.schema_text()}).strip()
        sql = re.sub(r"^```(?:sql)?\s*|\s*```$", "", sql, flags=re.M).strip()
        return sql

    def _execute(self, sql: str):
        sql = sql.strip().rstrip(";")
        if not re.match(r"^(SELECT|WITH)\b", sql, re.IGNORECASE):
            raise ValueError("仅支持SELECT查询")

        if ";" in sql:
            raise ValueError("仅支持单条SQL语句")

        conn = self._connect()
        try:
            cur = conn.execute(sql)
            rows = cur.fetchmany(200)
            columns = [desc[0] for desc in (cur.description or [])]
            return rows, columns
        finally:
            conn.close()

    def query(self, question: str) -> str:
        """自然语言问题 -> SQL -> 执行 -> 可读结果。"""
        try:
            sql = self._to_sql(question)
        except Exception as e:
            logger.error(f"[sql_query]SQL生成失败：{e}", exc_info=True)
            return f"SQL生成失败：{e}"

        try:
            rows, columns = self._execute(sql)
        except Exception as e:
            logger.warning(f"[sql_query]SQL执行失败，尝试修正重试：{e}")
            try:
                sql = self._to_sql(question, str(e))
                rows, columns = self._execute(sql)
            except Exception as e2:
                logger.error(f"[sql_query]SQL修正后仍失败：{e2}", exc_info=True)
                return f"查询失败：{e2}"

        if not rows:
            return f"查询SQL：{sql}\n查询结果：无匹配记录"

        result_lines = [f"查询SQL：{sql}", f"查询结果共 {len(rows)} 条："]
        for index, row in enumerate(rows[:50], 1):
            items = ", ".join(f"{col}={row[col]}" for col in columns)
            result_lines.append(f"{index}. {items}")

        if len(rows) > 50:
            result_lines.append(f"……（共{len(rows)}条，仅展示前50条）")

        return "\n".join(result_lines)


if __name__ == '__main__':
    svc = SqlQueryService()
    print(svc.schema_text())
    print("=" * 30)
    print(svc.query("擅自占用公路一共有多少条？"))