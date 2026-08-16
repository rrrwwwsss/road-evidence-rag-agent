import os
import shutil
import sqlite3
import pandas as pd

DB_PATH = os.path.join("data", "wupin_tanwei_dabt.db")
BACKUP_PATH = DB_PATH + ".bak"
TARGET_COL = "发生时间"
TARGET_FMT = "%Y%m%d_%H%M%S"

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
        # 已经符合目标格式例如 20250816_123045
        if len(s) == 15 and s[8] == '_' and s.replace('_', '').isdigit():
            return s
        # 尝试一些常见格式
        fmts = ["%Y%m%d_%H%M%S", "%Y-%m-%d %H:%M:%S", "%Y/%m/%d %H:%M:%S", "%Y%m%d%H%M%S", "%Y-%m-%d", "%Y/%m/%d"]
        for fmt in fmts:
            try:
                dt = pd.to_datetime(s, format=fmt)
                return dt.strftime(TARGET_FMT)
            except Exception:
                pass
        # 使用 pandas 的泛解析
        try:
            dt = pd.to_datetime(s, errors='coerce')
            if pd.isna(dt):
                return None
            return dt.strftime(TARGET_FMT)
        except Exception:
            return None

    df[TARGET_COL] = df[TARGET_COL].apply(parse_to_target)

    # 将 None 转为空字符串，避免 SQL 写入 NULL 时影响下游
    df[TARGET_COL] = df[TARGET_COL].fillna("")

    # 覆盖写回表（注意：会替换表结构、索引和约束）
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
print(f"原始数据库文件已备份为: {BACKUP_PATH}")
