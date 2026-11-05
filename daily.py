import fcntl, os, sys, subprocess, json, shutil, re
from pathlib import Path
from datetime import datetime
from zoneinfo import ZoneInfo
import pandas as pd
from threading import Event, Thread, Timer
import signal
root=Path(__file__).resolve().parent
os.chdir(root)
lock=open("Logs/run.lock","a")
try:
    fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
except BlockingIOError:
    sys.exit("已有掃描執行中")
et=datetime.now(ZoneInfo("America/New_York"))
stamp=et.strftime("%Y%m%d-%H%M%S")+f"-{os.getpid()}"
log=Path("Logs")/(stamp+".log")
print("執行日誌",log.resolve(),flush=True)
with log.open("x") as stream:
    print("ET",et,"HK",et.astimezone(ZoneInfo("Asia/Hong_Kong")),"ET時段",("盤前" if et.hour<9 or (et.hour==9 and et.minute<30) else "正常盤時段" if et.hour<16 else "收盤後"),file=stream,flush=True)
    os.environ["BIGLIST_STARTED_AT"] = et.isoformat()
    with subprocess.Popen(["./run_daily.sh"], pass_fds=(lock.fileno(),), stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True, start_new_session=True) as process:
        done = Event()
        def expire():
            if process.poll() is None:
                print("TIMEOUT: daily subprocess deadline; preserving checkpoints and Latest", flush=True)
                try:
                    os.killpg(process.pid, signal.SIGKILL)
                except ProcessLookupError:
                    pass
        watchdog = Timer(float(os.environ.get("BIGLIST_RUN_TIMEOUT", "7200")), expire)
        watchdog.daemon = True
        watchdog.start()
        def heartbeat():
            while not done.wait(60):
                print("子流程仍在執行；最近日誌更新", datetime.fromtimestamp(log.stat().st_mtime).astimezone(), flush=True)
        Thread(target=heartbeat, daemon=True).start()
        for line in process.stdout:
            stream.write(line)
            stream.flush()
            if not line.startswith("Financial cache fallback"):
                print(line.rstrip(), flush=True)
        watchdog.cancel()
        done.set()
        process.wait()
        result = process
    if result.returncode:
        print("掃描失敗，日誌",log.resolve(),file=sys.stderr)
        previous = root / "Reports" / "Latest" / "run_summary_latest.json"
        if previous.exists():
            try:
                previous_day = json.loads(previous.read_text())["as_of"]
                print("本次未發布。Latest 是歷史報告，資料日期", previous_day, file=sys.stderr)
            except (OSError, KeyError, ValueError):
                print("本次未發布。無法確認歷史報告日期。", file=sys.stderr)
        sys.exit(result.returncode)
out=Path("outputs")
s=json.loads((out/"run_summary_latest.json").read_text())
a=pd.read_csv(out/"audit_latest.csv",low_memory=False)
if s["scope"]!="FULL" or not s["scoring_complete"] or s["scanned"]!=len(a):
    sys.exit("掃描結果不完整")
dest=Path("Reports")/stamp
dest.mkdir()
for p in out.glob("*latest*"):
    if p.is_file(): shutil.copy2(p,dest/p.name)
from two_lists import render
from src.core.pipeline_io import session_close
as_of = pd.Timestamp(s["as_of"])
close = session_close(as_of)
receipt = log.read_text()
if (close is None or close + pd.Timedelta(minutes=30) > et or
    f"Market as-of: {as_of.date()}" not in receipt or
    f"Completed session: {as_of.date()};" not in receipt):
    sys.exit("EOD 收據不符，保留 Latest")
print("報告輸出", dest, flush=True)
render(dest, log)
previous = dest.parent / "Latest"
publish = True
if previous.exists():
    from compare_runs import compare
    previous = previous.resolve()
    changes = compare(previous, dest)
    changes.to_csv(dest / "名單變化.csv")
    prior_date = json.loads((previous / "run_summary_latest.json").read_text())["as_of"]
    publish = pd.Timestamp(s["as_of"]) > pd.Timestamp(prior_date)
for artifact in dest.iterdir():
    artifact.chmod(0o444)
dest.chmod(0o555)
if not publish:
    print("同日版本已保存，保留 Latest", dest)
else:
    tmp=dest.parent/(".latest-"+stamp)
    tmp.symlink_to(dest.name)
    os.replace(tmp,dest.parent/"Latest")
report = (dest / "研究報告.html").resolve()
print("研究報告已完成", report, flush=True)
try:
    subprocess.run(["/usr/bin/open", str(report)], check=True, timeout=15)
except (OSError, subprocess.SubprocessError) as exc:
    print("瀏覽器未能開啟", exc, report)
