import io, os, uuid, csv
from pathlib import Path
import pandas as pd
from fastapi import FastAPI, UploadFile, File, BackgroundTasks, HTTPException, Header, Depends
from fastapi.responses import FileResponse, StreamingResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel
from .agents import PipelineOrchestrator, CONFIG, deliver, now

app = FastAPI(title="Agentic Churn Platform")
JOBS: dict = {}; ORCH = PipelineOrchestrator(); STATIC = Path(__file__).parent / "static"

def auth(x_api_key: str = Header(default=None)):  # optional: set API_KEY env var to require it
    if os.getenv("API_KEY") and x_api_key != os.getenv("API_KEY"): raise HTTPException(401, "Invalid API key")
def job_or_404(jid):
    if jid not in JOBS: raise HTTPException(404, "Job not found")
    return JOBS[jid]

@app.post("/api/jobs", dependencies=[Depends(auth)])
async def create_job(bg: BackgroundTasks, file: UploadFile = File(...)):
    if not file.filename.lower().endswith(".csv"): raise HTTPException(400, "Please upload a .csv file")
    try: df = pd.read_csv(io.BytesIO(await file.read()))
    except Exception as e: raise HTTPException(400, f"Cannot parse CSV: {e}")
    jid = uuid.uuid4().hex[:10]
    JOBS[jid] = dict(id=jid, state="Queued", created=now(), filename=file.filename, raw=df, audit=[], deliveries={})
    bg.add_task(ORCH.run_workflow, JOBS[jid]); return {"job_id": jid}

@app.get("/api/jobs/{jid}", dependencies=[Depends(auth)])
def job_status(jid: str):
    j = job_or_404(jid); return {k: j.get(k) for k in ["id", "state", "reason", "filename", "created", "finished", "dashboard"]}

@app.get("/api/jobs/{jid}/customers", dependencies=[Depends(auth)])
def customers(jid: str, level: str = None):
    j = job_or_404(jid); out = []
    for a in j.get("assessments", []):
        if level and a["risk_level"] != level: continue
        out.append({**a, "offer": j["offers"].get(a["customer_id"])})  # contact details never exposed here
    return sorted(out, key=lambda a: -a["churn_score"])

class Approve(BaseModel):
    customer_ids: list[str]; approver: str = "manager"
@app.post("/api/jobs/{jid}/approve", dependencies=[Depends(auth)])
def approve(jid: str, body: Approve):
    j = job_or_404(jid); n = 0
    for c in body.customer_ids:
        o = j["offers"].get(c)
        if o and o["status"] == "Pending": o.update(status="Approved", approval_id=f"APR-{uuid.uuid4().hex[:6]}"); n += 1
    ORCH.audit(j, "offers_approved", count=n, approver=body.approver); return {"approved": n}

@app.post("/api/jobs/{jid}/deliver", dependencies=[Depends(auth)])
def do_deliver(jid: str):
    j = job_or_404(jid); res = [deliver(j, c) for c, o in j["offers"].items() if o["status"] == "Approved"]
    if j["state"] == "Awaiting approval" and res: j["state"] = "Completed"; j["finished"] = now()
    return {"processed": len(res), "note": "Delivery is SIMULATED; no real message was sent."}

@app.post("/api/jobs/{jid}/send-test", dependencies=[Depends(auth)])
def send_test(jid: str): job_or_404(jid); return {"status": "Simulated test message generated (nothing was sent)."}

@app.post("/api/jobs/{jid}/cancel", dependencies=[Depends(auth)])
def cancel(jid: str):
    j = job_or_404(jid); j.update(state="Cancelled", reason="Cancelled by user", finished=now()); return {"state": "Cancelled"}

@app.get("/api/jobs/{jid}/audit", dependencies=[Depends(auth)])
def audit(jid: str): return job_or_404(jid)["audit"]

@app.get("/api/jobs/{jid}/export.csv", dependencies=[Depends(auth)])
def export(jid: str):
    j = job_or_404(jid); buf = io.StringIO(); w = csv.writer(buf)
    w.writerow(["customer_id", "plan", "churn_score", "risk_level", "top_factors", "offer", "status", "model_version"])
    for a in j.get("assessments", []):
        o = j["offers"].get(a["customer_id"], {})
        w.writerow([a["customer_id"], a["plan"], a["churn_score"], a["risk_level"], "; ".join(f"{f['direction']} {f['feature']}" for f in a["top_factors"]), o.get("offer_type", ""), o.get("status", ""), a["model_version"]])
    return StreamingResponse(iter([buf.getvalue()]), media_type="text/csv", headers={"Content-Disposition": f"attachment; filename=churn_{jid}.csv"})

@app.get("/api/config", dependencies=[Depends(auth)])
def get_cfg(): return CONFIG
@app.post("/api/config", dependencies=[Depends(auth)])
def set_cfg(cfg: dict):
    for k in CONFIG:
        if k in cfg: CONFIG[k] = type(CONFIG[k])(cfg[k])
    return CONFIG

@app.get("/api/health")
def health(): return {"ok": True}
@app.get("/")
def index(): return FileResponse(STATIC / "index.html")
app.mount("/static", StaticFiles(directory=STATIC), name="static")
