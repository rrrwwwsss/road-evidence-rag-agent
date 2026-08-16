import os
import shutil
import sqlite3
import pandas as pd

DB_PATH = os.path.join("data", "wupin_tanwei_dabt.db")
BACKUP_PATH = DB_PATH + ".bak_readable"
TARGET_COL = "发生时间"
TARGET_FMT = "%Y-%m-%d %H:%M:%S"

if not os.path.exists(DB_PATH):
    print(f"ERROR: 数据库文件不存在: {DB_PATH}")
    raise SystemExit(1)

# 备份
shutil.copy2(DB_PATH, BACKUP_PATH)
print(f"已备份数据库到: {BACKUP_PATH}")

conn = sqlite3.connect(DB_PATH)
cur = conn.cursor()

# 获取所有表名
cur.execute("SELECT name FROM sqlite_master WHERE type='table'")
tables = [r[0] for r in cur.fetchall()]

updated_tables = []

for t in tables:
    try:
        df = pd.read_sql_query(f"SELECT * FROM '{t}'", conn)
    except Exception as e:
        print(f"读取表 {t} 失败: {e}")
        continue

    if TARGET_COL not in df.columns:
        continue

    print(f"处理表: {t}, 行数={len(df)}")

    def parse_to_target(s):
        if pd.isna(s):
            return None
        s = str(s).strip()
        if not s:
            return None
        # 已经是目标格式：YYYY-MM-DD HH:MM:SS
        if len(s) == 19 and s[4] == '-' and s[7] == '-' and s[10] == ' ' and s[13] == ':' and s[16] == ':':
            return s
        # 常见已存在格式：YYYYMMDD_HHMMSS 或 YYYYMMDDHHMMSS
        candidates = ["%Y%m%d_%H%M%S", "%Y%m%d%H%M%S", "%Y-%m-%d %H:%M:%S", "%Y/%m/%d %H:%M:%S", "%Y-%m-%d", "%Y/%m/%d"]
        for fmt in candidates:
            try:
                dt = pd.to_datetime(s, format=fmt)
                return dt.strftime(TARGET_FMT)
            except Exception:
                pass
        # 泛解析
        try:
            dt = pd.to_datetime(s, errors='coerce')
            if pd.isna(dt):
                return None
            return dt.strftime(TARGET_FMT)
        except Exception:
            return None

    df[TARGET_COL] = df[TARGET_COL].apply(parse_to_target)
    df[TARGET_COL] = df[TARGET_COL].fillna("")

    try:
        df.to_sql(t, conn, if_exists='replace', index=False)
        updated_tables.append(t)
        print(f"已更新表 {t} 的 `{TARGET_COL}` 字段格式并写回数据库")
    except Exception as e:
        print(f"写回表 {t} 失败: {e}")

conn.close()

print("处理完成。被更新的表:")
for ut in updated_tables:
    print(" - ", ut)
print(f"已备份原始数据库为: {BACKUP_PATH}")
