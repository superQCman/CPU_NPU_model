#!/bin/bash
# ============================================================
# SSD 性能扫描脚本
# 内容：3 个锚点 + 两条扫描线，用于找 size* 和 QD*
#   锚点 1 : 基础时延  4K  QD=1
#   扫描 A : 固定 QD=32 扫 bs    -> 找 IOPS→带宽拐点 size*
#   扫描 B : 固定 bs=4k 扫 QD    -> 找时延→饱和拐点 QD*
#   扫描 B2: 固定 bs=256k 扫 QD  -> 大块读的饱和行为
#
# 用法：
#   1. 先准备测试文件（务必落在被测 SSD 上）：
#        fallocate -l 20G ~/testfile
#   2. 运行：
#        bash fio_ssd_benchmark.sh
#   3. 脚本末尾自动输出汇总表（bs/QD → IOPS、带宽、时延）
# ============================================================
set -u

# ---------- 配置 ----------
FILE=~/testfile           # 测试文件路径（被测 SSD 上）
RESULT_DIR=./fio_results  # 结果目录（每个测试一个 JSON）
ROW_BS=256k               # 大块读扫描用的 bs（可按需改成 128k/512k）

mkdir -p "$RESULT_DIR"

# # ---------- 环境检查 ----------
# echo "==== 环境检查 ===="
# if [ ! -f "$FILE" ]; then
#   echo "[错误] $FILE 不存在。请先执行： fallocate -l 20G $FILE"
#   exit 1
# fi
# echo "测试文件: $FILE"
# df -h "$FILE"
# echo

# ============================================================
# 1. 基础时延（锚点 α）—— 4K QD=1 + 分位数
# ============================================================
echo "==== [1/4] 基础时延 4K QD=1 ===="
fio --name=lat_base --rw=randread --bs=4k --size=1G \
    --iodepth=1 --numjobs=1 --runtime=30 --time_based \
    --filename="$FILE" --direct=1 \
    --clat_percentiles=1 --percentile_list=50:90:99:99.9:99.99 \
    --group_reporting=1 \
    --output-format=json --output="$RESULT_DIR/lat_base.json"

# ============================================================
# 2. 扫描 A：固定 QD=32，扫 bs → 找 size*（IOPS→带宽拐点）
# ============================================================
echo "==== [2/4] bs 扫描（QD=32）===="
for bs in 4k 8k 16k 32k 64k 128k 256k 512k 1m; do
  echo "  bs=$bs ..."
  fio --name="bs_${bs}" --rw=randread --bs=$bs --size=8G \
      --iodepth=32 --numjobs=1 --runtime=20 --time_based \
      --filename="$FILE" --direct=1 --group_reporting=1 \
      --output-format=json --output="$RESULT_DIR/bs_${bs}.json"
done

# ============================================================
# 3. 扫描 B：固定 bs=4k，扫 QD → 找 QD*（时延→饱和拐点）
# ============================================================
echo "==== [3/4] QD 扫描（bs=4k）===="
for qd in 1 2 4 8 16 32 64 128 256; do
  echo "  qd=$qd ..."
  fio --name="qd4k_${qd}" --rw=randread --bs=4k --size=4G \
      --iodepth=$qd --numjobs=1 --runtime=20 --time_based \
      --filename="$FILE" --direct=1 --group_reporting=1 \
      --output-format=json --output="$RESULT_DIR/qd4k_${qd}.json"
done

# ============================================================
# 4. 扫描 B2：固定 bs=256k，扫 QD → 大块读的饱和行为
# ============================================================
echo "==== [4/4] QD 扫描（bs=${ROW_BS}）===="
for qd in 1 2 4 8 16 32 64; do
  echo "  qd=$qd ..."
  fio --name="qd${ROW_BS}_${qd}" --rw=randread --bs=$ROW_BS --size=8G \
      --iodepth=$qd --numjobs=1 --runtime=20 --time_based \
      --filename="$FILE" --direct=1 --group_reporting=1 \
      --output-format=json --output="$RESULT_DIR/qd${ROW_BS}_${qd}.json"
done

echo
echo "全部测试完成，结果在 $RESULT_DIR/"

# ============================================================
# 汇总：解析 JSON，打印表格
# ============================================================
ROW_BS="$ROW_BS" python3 - <<'PY'
import json, os

RESULT_DIR = "fio_results"
ROW_BS = os.environ.get("ROW_BS", "256k")

def load(name):
    with open(os.path.join(RESULT_DIR, name + ".json")) as f:
        return json.load(f)

def read_obj(d):
    for j in d.get("jobs", []):
        if "read" in j:
            return j["read"]
    return None

def clat_mean_us(r):
    return r["clat_ns"]["mean"] / 1000.0

def clat_pct_us(r, target):
    p = r.get("clat_ns", {}).get("percentile", {})
    if not p:
        return None
    k = min(p, key=lambda x: abs(float(x) - target))
    return p[k] / 1000.0

def iops(r):
    return r.get("iops", 0)

def bw_mb_s(r):
    return r.get("bw", 0) / 1000.0

# ---- 基础时延 ----
print("\n========== 基础时延 (4K QD=1) ==========")
try:
    r = read_obj(load("lat_base"))
    if r:
        print(f"  clat mean : {clat_mean_us(r):.1f} us")
        for t in (50, 90, 99, 99.9, 99.99):
            v = clat_pct_us(r, t)
            if v is not None:
                print(f"  clat p{t:<5}: {v:.1f} us")
except FileNotFoundError:
    print("  (lat_base.json 未找到)")

# ---- 扫描 A ----
print("\n========== 扫描 A: bs 扫描 (QD=32) ==========")
print(f"{'bs':>8s} {'IOPS':>10s} {'BW(MB/s)':>10s} {'clat(us)':>10s}")
print("-" * 42)
for bs in ["4k","8k","16k","32k","64k","128k","256k","512k","1m"]:
    try:
        r = read_obj(load(f"bs_{bs}"))
        if r:
            print(f"{bs:>8s} {iops(r):>10.0f} {bw_mb_s(r):>10.0f} {clat_mean_us(r):>10.1f}")
    except FileNotFoundError:
        print(f"{bs:>8s}  (未找到)")

# ---- 扫描 B ----
print("\n========== 扫描 B: QD 扫描 (bs=4k) ==========")
print(f"{'qd':>6s} {'IOPS':>10s} {'clat(us)':>10s}")
print("-" * 30)
for qd in [1,2,4,8,16,32,64,128,256]:
    try:
        r = read_obj(load(f"qd4k_{qd}"))
        if r:
            print(f"{qd:>6d} {iops(r):>10.0f} {clat_mean_us(r):>10.1f}")
    except FileNotFoundError:
        print(f"{qd:>6d}  (未找到)")

# ---- 扫描 B2 ----
print(f"\n========== 扫描 B2: QD 扫描 (bs={ROW_BS}) ==========")
print(f"{'qd':>6s} {'IOPS':>10s} {'BW(MB/s)':>10s} {'clat(us)':>10s}")
print("-" * 42)
for qd in [1,2,4,8,16,32,64]:
    try:
        r = read_obj(load(f"qd{ROW_BS}_{qd}"))
        if r:
            print(f"{qd:>6d} {iops(r):>10.0f} {bw_mb_s(r):>10.0f} {clat_mean_us(r):>10.1f}")
    except FileNotFoundError:
        print(f"{qd:>6d}  (未找到)")

print("\n怎么读表：")
print("  - 扫描 A：BW(MB/s) 走平对应的 bs = size*（IOPS→带宽拐点）")
print("  - 扫描 B：IOPS 走平对应的 qd = QD*（时延→饱和拐点）")
PY
