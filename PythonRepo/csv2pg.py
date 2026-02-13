import os
import re
import csv
import time
import hashlib
import logging
from dataclasses import dataclass
from datetime import datetime, date
from typing import Optional, List, Dict, Tuple

import psycopg2
import psycopg2.extras


# ----------------------------
# Config
# ----------------------------
@dataclass
class Config:
    watch_dir: str
    scan_interval_sec: int = 10          # 每轮扫描间隔
    stable_window_sec: int = 15          # 文件大小稳定窗口
    stable_checks: int = 2               # 连续检查次数（2次即可）
    filename_regex: str = r".+テーブル_(\d{1,2})月(\d{1,2})日\d+便目\.csv$"
    current_year: int = datetime.now().year

    # DB
    pg_dsn: str = "host=localhost port=5432 dbname=xxx user=xxx password=xxx"
    ingest_table: str = "ingest_file"
    target_table: str = "shipment_record"  # 你的目标表名
    # 目标表唯一约束： (reg_time, skid_id)


# ----------------------------
# Logging
# ----------------------------
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s %(levelname)s %(message)s"
)
logger = logging.getLogger(__name__)


# ----------------------------
# Helpers
# ----------------------------
def parse_reg_date_from_filename(filename: str, cfg: Config) -> Optional[date]:
    """
    从文件名解析 月/日，年份用 cfg.current_year（你说只跑几个月，够用）
    """
    m = re.match(cfg.filename_regex, filename)
    if not m:
        return None
    month = int(m.group(1))
    day = int(m.group(2))
    return date(cfg.current_year, month, day)


def file_signature(path: str) -> Tuple[int, float]:
    st = os.stat(path)
    return st.st_size, st.st_mtime


def wait_until_stable(path: str, cfg: Config) -> bool:
    """
    文件大小在 stable_window_sec 内保持稳定（连续 stable_checks 次）
    """
    try:
        last_size, last_mtime = file_signature(path)
    except FileNotFoundError:
        return False

    stable_count = 0
    for _ in range(cfg.stable_checks):
        time.sleep(cfg.stable_window_sec / cfg.stable_checks)
        try:
            size, mtime = file_signature(path)
        except FileNotFoundError:
            return False

        if size == last_size:
            stable_count += 1
        else:
            stable_count = 0

        last_size, last_mtime = size, mtime

    return stable_count >= 1  # 2次中至少1次保持不变通常就够；你也可以改成 == cfg.stable_checks


def compute_hash(path: str) -> str:
    """
    用于 DONE 后变更检测。文件不大（<=100行），全量hash OK
    """
    h = hashlib.sha256()
    with open(path, "rb") as f:
        while True:
            chunk = f.read(1024 * 1024)
            if not chunk:
                break
            h.update(chunk)
    return h.hexdigest()


def read_and_validate_csv(path: str) -> Tuple[str, Optional[List[Dict]], Optional[str]]:
    """
    按你的规则读取：
      - 编码: UTF-8 with BOM -> utf-8-sig
      - 分隔符: ,
      - A列: index（模板自带，可能一直有）
      - B列: 完成/未完成/空
      - C列: skid_id（主键）
    规则：
      - 若出现：C非空 且 B==未完成 -> 文件整体不可转送 (return "PENDING")
      - 末尾判断：B空 且 C空 -> break
      - 严格模式：列数不够 / 必要字段缺失 -> FAILED
    return:
      status: "READY" | "PENDING" | "FAILED"
      rows: READY时返回要入库的行列表（每行dict）
      error: FAILED时返回错误信息
    """
    rows_to_insert: List[Dict] = []

    try:
        with open(path, "r", encoding="utf-8-sig", newline="") as f:
            reader = csv.reader(f, delimiter=",")
            for i, row in enumerate(reader, start=1):
                # 严格模式：最少要有 A,B,C 三列
                if len(row) < 3:
                    # 允许纯空行（某些工具会输出末尾空行）
                    if len(row) == 0 or all((c or "").strip() == "" for c in row):
                        continue
                    return "FAILED", None, f"row {i}: column count < 3, row={row}"

                a = (row[0] or "").strip()
                b = (row[1] or "").strip()
                c = (row[2] or "").strip()  # skid_id

                # 末尾：B空 且 C空（A可能仍有模板数字）
                if b == "" and c == "":
                    break

                # 只要 skid_id 有值，B 必须是 完成/未完成 之一（严格）
                if c != "" and b not in ("完成", "未完成"):
                    return "FAILED", None, f"row {i}: invalid status B='{b}' for skid_id='{c}'"

                # 文件级阻断条件：存在未完成
                if c != "" and b == "未完成":
                    return "PENDING", None, None

                # 收集可入库行：c非空 且 b==完成
                if c != "" and b == "完成":
                    # TODO: 这里把你需要的其他列按固定索引取出来
                    # 例如：dest = row[3].strip() if len(row)>3 else ""
                    rows_to_insert.append({
                        "skid_id": c,
                        # "col_x": ...,
                        # "col_y": ...,
                    })

    except PermissionError as e:
        return "FAILED", None, f"permission/lock error: {e}"
    except UnicodeDecodeError as e:
        return "FAILED", None, f"encoding error: {e}"
    except Exception as e:
        return "FAILED", None, f"unexpected csv error: {e}"

    if not rows_to_insert:
        # 文件里全是空/没有完成行：按严格模式你也可以视为FAILED或PENDING
        return "FAILED", None, "no completed rows found"
    return "READY", rows_to_insert, None


# ----------------------------
# DB Operations
# ----------------------------
def ensure_ingest_row(conn, cfg: Config, file_id: str, path: str, reg_date: date, size: int, mtime: float):
    """
    台账 upsert：新文件插入PENDING，已有则更新路径/size/mtime
    """
    sql = f"""
    INSERT INTO {cfg.ingest_table}(file_id, file_path, reg_date, file_size, file_mtime, status, updated_at)
    VALUES (%s, %s, %s, %s, to_timestamp(%s), 'PENDING', now())
    ON CONFLICT (file_id) DO UPDATE
      SET file_path = EXCLUDED.file_path,
          reg_date  = EXCLUDED.reg_date,
          file_size = EXCLUDED.file_size,
          file_mtime= EXCLUDED.file_mtime,
          updated_at= now()
    """
    with conn.cursor() as cur:
        cur.execute(sql, (file_id, path, reg_date, size, mtime))


def get_ingest_status(conn, cfg: Config, file_id: str) -> Optional[Dict]:
    sql = f"""
    SELECT file_id, status, content_hash, file_size, extract(epoch from file_mtime) AS file_mtime_epoch
    FROM {cfg.ingest_table}
    WHERE file_id = %s
    """
    with conn.cursor(cursor_factory=psycopg2.extras.RealDictCursor) as cur:
        cur.execute(sql, (file_id,))
        return cur.fetchone()


def mark_status(conn, cfg: Config, file_id: str, status: str, last_error: Optional[str] = None, content_hash: Optional[str] = None):
    sql = f"""
    UPDATE {cfg.ingest_table}
       SET status = %s,
           last_error = %s,
           content_hash = COALESCE(%s, content_hash),
           updated_at = now(),
           processed_at = CASE WHEN %s='DONE' THEN now() ELSE processed_at END,
           retry_count = CASE WHEN %s='FAILED' THEN retry_count+1 ELSE retry_count END
     WHERE file_id = %s
    """
    with conn.cursor() as cur:
        cur.execute(sql, (status, last_error, content_hash, status, status, file_id))


def upsert_business_rows(conn, cfg: Config, reg_date: date, rows: List[Dict]):
    """
    目标表 UPSERT 示例：唯一键 (reg_time, skid_id)
    reg_time = reg_date 00:00:00
    """
    reg_time = datetime(reg_date.year, reg_date.month, reg_date.day, 0, 0, 0)

    # 你需要的列名，按实际表结构补齐
    # 例： (reg_time, skid_id, col_x, col_y)
    sql = f"""
    INSERT INTO {cfg.target_table}(reg_time, skid_id)
    VALUES %s
    ON CONFLICT (reg_time, skid_id)
    DO UPDATE SET
      skid_id = EXCLUDED.skid_id
    """

    values = [(reg_time, r["skid_id"]) for r in rows]

    with conn.cursor() as cur:
        psycopg2.extras.execute_values(cur, sql, values, page_size=200)


# ----------------------------
# Main loop
# ----------------------------
def run(cfg: Config):
    logger.info("Starting watcher: %s", cfg.watch_dir)

    while True:
        try:
            files = [f for f in os.listdir(cfg.watch_dir) if f.lower().endswith(".csv")]
        except Exception as e:
            logger.error("listdir failed: %s", e)
            time.sleep(cfg.scan_interval_sec)
            continue

        # 可选：排序让处理顺序稳定（按文件名）
        files.sort()

        with psycopg2.connect(cfg.pg_dsn) as conn:
            conn.autocommit = False

            for filename in files:
                reg_date = parse_reg_date_from_filename(filename, cfg)
                if reg_date is None:
                    continue

                path = os.path.join(cfg.watch_dir, filename)
                file_id = filename  # 最简单：直接用文件名

                # 文件可能在扫描到后被移动
                if not os.path.exists(path):
                    continue

                try:
                    size, mtime = file_signature(path)
                except FileNotFoundError:
                    continue

                # 台账确保存在
                ensure_ingest_row(conn, cfg, file_id, path, reg_date, size, mtime)
                conn.commit()

                st = get_ingest_status(conn, cfg, file_id)
                if not st:
                    continue

                # DONE 后变更检测（mtime/size变化 或 hash变化）
                if st["status"] == "DONE":
                    # 快速判断：size/mtime 变化就再算 hash
                    if st["file_size"] != size or float(st["file_mtime_epoch"] or 0) != float(mtime):
                        try:
                            new_hash = compute_hash(path)
                        except Exception as e:
                            mark_status(conn, cfg, file_id, "FAILED", f"hash compute failed: {e}")
                            conn.commit()
                            continue

                        if st["content_hash"] and st["content_hash"] != new_hash:
                            mark_status(conn, cfg, file_id, "CHANGED_AFTER_DONE", "file changed after DONE", new_hash)
                            conn.commit()
                            continue
                        else:
                            # mtime/size变了但hash没变，更新台账信息即可
                            mark_status(conn, cfg, file_id, "DONE", None, new_hash)
                            conn.commit()
                            continue
                    else:
                        continue  # DONE 且没变 -> 跳过

                # 稳定窗口
                if not wait_until_stable(path, cfg):
                    continue

                # 尽可能独占：这里只能做“尝试读取”，读失败就记录FAILED下轮重试
                status, rows, err = read_and_validate_csv(path)

                if status == "PENDING":
                    # 未完成：保持PENDING（不算失败）
                    mark_status(conn, cfg, file_id, "PENDING", None)
                    conn.commit()
                    continue

                if status == "FAILED":
                    mark_status(conn, cfg, file_id, "FAILED", err)
                    conn.commit()
                    continue

                # READY: 入库
                try:
                    mark_status(conn, cfg, file_id, "PROCESSING", None)
                    conn.commit()

                    # 计算 hash（记录DONE时的内容指纹）
                    new_hash = compute_hash(path)

                    # 事务：业务入库 + 标记DONE
                    upsert_business_rows(conn, cfg, reg_date, rows)
                    mark_status(conn, cfg, file_id, "DONE", None, new_hash)
                    conn.commit()

                    logger.info("DONE: %s rows=%d reg_date=%s", filename, len(rows), reg_date)

                except Exception as e:
                    conn.rollback()
                    mark_status(conn, cfg, file_id, "FAILED", f"db insert failed: {e}")
                    conn.commit()

        time.sleep(cfg.scan_interval_sec)


if __name__ == "__main__":
    cfg = Config(
        watch_dir=r"C:\path\to\csv_folder",
        pg_dsn="host=127.0.0.1 port=5432 dbname=xxx user=xxx password=xxx"
    )
    run(cfg)
