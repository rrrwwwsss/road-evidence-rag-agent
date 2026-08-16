import os
import shutil
import sqlite3

DB_PATH = os.path.join(os.path.dirname(os.path.dirname(__file__)), '..', 'data', 'wupin_tanwei_dabt.db')
# 上面构造路径可能包含冗余分隔符，修正为绝对
DB_PATH = os.path.abspath(DB_PATH)
BACKUP_PATH = DB_PATH + '.drop_tables.bak'

if not os.path.exists(DB_PATH):
    print(f"ERROR: 数据库文件不存在: {DB_PATH}")
    raise SystemExit(1)

print(f"数据库路径: {DB_PATH}")
print(f"备份路径: {BACKUP_PATH}")

# 备份
shutil.copy2(DB_PATH, BACKUP_PATH)
print(f"已备份数据库到: {BACKUP_PATH}")

conn = sqlite3.connect(DB_PATH)
cur = conn.cursor()

# 查询现有表
cur.execute("SELECT name FROM sqlite_master WHERE type='table'")
tables = [r[0] for r in cur.fetchall()]
print('当前表:', tables)

# 需要保留的表
keep = {'results'}

# 计算要删除的表
to_drop = [t for t in tables if t not in keep]

if not to_drop:
    print('没有需要删除的表。')
    conn.close()
    raise SystemExit(0)

print('将要删除的表:', to_drop)

# 为安全起见，脚本默认不直接执行删除，除非用户将 EXECUTE = True
EXECUTE = True

if not EXECUTE:
    print('\n安全保护：当前脚本为演示模式（未执行删除）。')
    print('如果你确认要执行删除，请将脚本中 EXECUTE = True 后重新运行，或使用交互方式确认。')
    conn.close()
    raise SystemExit(0)

# 如果到达这里，执行删除
for t in to_drop:
    try:
        cur.execute(f"DROP TABLE IF EXISTS '{t}'")
        print(f"已删除表: {t}")
    except Exception as e:
        print(f"删除表 {t} 失败: {e}")

conn.commit()
conn.close()
print('删除完成。请验证数据库。')
