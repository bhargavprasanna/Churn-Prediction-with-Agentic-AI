import sys, time, pathlib, pytest
sys.path.insert(0, str(pathlib.Path(__file__).parents[1] / "backend"))
from fastapi.testclient import TestClient
from app.main import app
c = TestClient(app); D = pathlib.Path(__file__).parents[1] / "sample_data"

def run(name):
    jid = c.post("/api/jobs", files={"file": (name, open(D / name, "rb"), "text/csv")}).json()["job_id"]
    for _ in range(120):
        j = c.get(f"/api/jobs/{jid}").json()
        if j["state"] in ("Awaiting approval", "Failed"): return jid, j
        time.sleep(.5)

@pytest.mark.parametrize("f,dom", [("customers.csv", None), ("spotify_users.csv", "music_streaming"), ("netflix_users.csv", "video_streaming"), ("amazon_customers.csv", "e_commerce")])
def test_domains(f, dom):
    jid, j = run(f); assert j["state"] == "Awaiting approval", j
    d = j["dashboard"]; m = d["model"]["metrics"]
    assert d["model"]["mode"] == "trained_model" and {"random_forest", "logistic_regression", "naive_bayes", "svm", "ensemble_voting", "ensemble_stacking"} <= set(m["held_out"])
    if dom: assert d["domain"] == dom
    print(f, d["domain"], "best:", m["selected"], m["held_out"][m["selected"]], "| roles:", d["quality"]["roles"])
    rows = c.get(f"/api/jobs/{jid}/customers").json(); assert "Email" not in rows[0] and "email" not in rows[0]
    off = [r for r in rows if r["offer"]]; print("  sample offer:", off[0]["plan"], off[0]["risk_level"], "->", off[0]["offer"]["offer_type"]); print("  insights:", d["insights"][:2])

def test_flow_and_delivery():
    jid, _ = run("customers.csv"); rows = c.get(f"/api/jobs/{jid}/customers").json()
    ids = [r["customer_id"] for r in rows if r["offer"]][:5]
    assert c.post(f"/api/jobs/{jid}/approve", json={"customer_ids": ids}).json()["approved"] == 5
    assert c.post(f"/api/jobs/{jid}/deliver").json()["processed"] == 5 and c.get(f"/api/jobs/{jid}/export.csv").status_code == 200

def test_cold_start():
    _, j = run("customers_no_labels.csv"); assert j["dashboard"]["model"]["mode"] == "rule_based_fallback"

def test_bad_data():
    jid = c.post("/api/jobs", files={"file": ("x.csv", b"a,b\n1,2\n", "text/csv")}).json()["job_id"]; time.sleep(1)
    assert c.get(f"/api/jobs/{jid}").json()["state"] == "Failed"
