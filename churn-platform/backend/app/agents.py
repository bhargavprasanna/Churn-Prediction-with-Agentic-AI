"""Agentic pipeline: Orchestrator + Data/ML/Risk/Offer/Visual agents (deck slides 11-14)."""
import os, re, time, hashlib
from abc import ABC, abstractmethod
from datetime import datetime, timezone
import numpy as np, pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import recall_score, precision_score, f1_score, roc_auc_score, accuracy_score
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.naive_bayes import GaussianNB
from sklearn.svm import SVC
from sklearn.ensemble import VotingClassifier, StackingClassifier
from sklearn.inspection import permutation_importance
from sklearn.base import clone

now = lambda: datetime.now(timezone.utc).isoformat(timespec="seconds")
CONFIG = {"high_threshold": 0.60, "medium_threshold": 0.35, "offer_cost_pct": 0.10, "max_retries": 3}

class ValidationError(Exception): """Bad data: actionable error, never retried."""
class TransientError(Exception): """Temporary failure: bounded retry."""

class Agent(ABC):
    name = "Agent"
    @abstractmethod
    def execute(self, ctx: dict) -> dict: ...
    def validate_output(self, ctx: dict) -> None: pass

# ---------------- Domain-adaptive helpers ----------------
def toks(c): return [t.lower() for t in re.findall(r"[A-Z]+(?![a-z])|[A-Z]?[a-z]+|\d+", str(c))]
POS_T = {"ad","ads","advert","adverts","complaint","complaints","ticket","tickets","late","failed","fail","issue","issues","cancel","cancelled","refund","refunds","return","returns","decline","skip","skips","skipped","inactive","overdue","dispute","downgrade","since","lapse","abandon","abandoned","error","errors","support"}
NEG_P = ("listen","watch","song","track","movie","show","episode","login","session","open","order","purchase","device","tenure","month","activity","usage","visit","stream","play","spend","engage","playlist","download","cart","review","hour","minute","duration","frequency","profile","subscri")
def infer_sign(name):
    t = toks(name)
    if any(x in POS_T for x in t): return 1
    if any(x.startswith(NEG_P) for x in t): return -1
    return 0
DOMAINS = {"music_streaming": {"listen","song","songs","track","playlist","ads","ad","music","artist"},
           "video_streaming": {"movie","movies","show","episode","watch","device","devices","profile","series"},
           "e_commerce": {"order","orders","cart","purchase","basket","returns","shipping","delivery"},
           "telecom/saas": {"tenure","charges","data","calls","sms","contract"}}
def detect_domain(cols):
    bag = {t for c in cols for t in toks(c)}; sc = {d: len(bag & k) for d, k in DOMAINS.items()}
    d = max(sc, key=sc.get); return d if sc[d] > 0 else "generic_subscription"
LABEL_T = {"churn","churned","exited","attrition","cancelled","canceled","left","target","label"}
PLAN_T = {"plan","subscription","tier","membership","package","contract"}
REV_T = {"monthly","charges","charge","mrr","arpu","revenue","fee","price","amount","spend","value"}
BIN = {"1":1,"1.0":1,"yes":1,"true":1,"y":1,"churned":1,"left":1,"cancelled":1,"0":0,"0.0":0,"no":0,"false":0,"n":0,"active":0,"stayed":0}

def detect_roles(df):
    roles, n = {}, len(df)
    for c in df.columns:
        t, s = set(toks(c)), df[c]; nu = s.nunique(dropna=True); low = str(c).lower()
        if t & {"email","mail"}: roles[c] = "email"
        elif t & {"phone","mobile","msisdn"}: roles[c] = "phone"
        elif t & {"consent","optin","opt"}: roles[c] = "consent"
        elif t & LABEL_T and nu <= 3: roles[c] = "label"
        elif (low == "id" or low.endswith("id")) and nu > .9 * n: roles[c] = "id"
        elif t & PLAN_T and nu <= 20: roles[c] = "plan"
        elif t & REV_T and pd.api.types.is_numeric_dtype(s) and not pd.api.types.is_bool_dtype(s): roles[c] = "revenue"
        else: roles[c] = "feature"
    return roles

def llm_schema(df):
    """Agentic step: Claude interprets the schema. Only column names/dtypes/cardinality are sent (no cell values)."""
    if not os.getenv("ANTHROPIC_API_KEY"): return None
    try:
        import anthropic, json
        desc = [{"col": str(c), "dtype": str(df[c].dtype), "unique": int(df[c].nunique())} for c in df.columns]
        msg = anthropic.Anthropic().messages.create(model=os.getenv("CLAUDE_MODEL", "claude-sonnet-4-6"), max_tokens=900, messages=[{"role": "user", "content":
            "Churn dataset schema: " + json.dumps(desc) + '\nReturn ONLY JSON: {"domain":"<business domain>","roles":{col:"id|label|plan|revenue|email|phone|consent|feature|drop"},"risk_direction":{numeric_feature_col: 1 if higher value means MORE churn risk, -1 if LESS, 0 unknown}}'}])
        txt = msg.content[0].text; return json.loads(txt[txt.index("{"): txt.rindex("}") + 1])
    except Exception: return None

class DataAgent(Agent):
    name = "DataAgent"
    def execute(self, ctx):
        import warnings; df = ctx["raw"].copy(); log = []
        df.columns = [str(c).strip() for c in df.columns]
        roles = detect_roles(df); domain = detect_domain(df.columns); llm = llm_schema(df); dirs = {}
        if llm:
            for c, r in (llm.get("roles") or {}).items():
                if c in df.columns and r in {"id","label","plan","revenue","email","phone","consent","feature","drop"}: roles[c] = r
            dirs = {c: int(v) for c, v in (llm.get("risk_direction") or {}).items() if c in df.columns and v in (-1, 0, 1)}
            domain = llm.get("domain") or domain; log.append("Schema interpreted with LLM assistance (names/dtypes only)")
        by = lambda r: [c for c, v in roles.items() if v == r]
        idc = by("id")[:1]
        if idc: df = df.drop_duplicates(idc[0]).reset_index(drop=True); ids = df[idc[0]].astype(str).values
        else: df = df.reset_index(drop=True); ids = np.array([f"row{i}" for i in range(len(df))]); log.append("No ID column found: generated row IDs")
        if len(df) < 5: raise ValidationError("Need at least 5 customer rows.")
        contacts = {}
        for i, cid in enumerate(ids):
            contacts[cid] = {k: df.at[i, by(k)[0]] for k in ("email","phone","consent") if by(k)}
        y = None
        if by("label"):
            y = df[by("label")[0]].astype(str).str.strip().str.lower().map(BIN)
            if y.notna().sum() == 0: y = None; log.append("Label column values not recognised as binary; using cold-start rules")
        plan = df[by("plan")[0]].astype(str).values if by("plan") else np.array(["-"] * len(df))
        revc = by("revenue")[:1]; rev = pd.to_numeric(df[revc[0]], errors="coerce").values if revc else None
        skip = set(idc + by("label") + by("email") + by("phone") + by("consent") + by("drop")); feat = {}; dropped = []
        for c in df.columns:
            if c in skip: continue
            s = df[c]
            if pd.api.types.is_bool_dtype(s): feat[c] = s.astype(float); continue
            if not pd.api.types.is_numeric_dtype(s):
                num = pd.to_numeric(s, errors="coerce")
                if num.notna().mean() > .9: s = num
                else:
                    with warnings.catch_warnings():
                        warnings.simplefilter("ignore"); dt = pd.to_datetime(s, errors="coerce")
                    if dt.notna().mean() > .9: feat[c] = (dt.max() - dt).dt.days.astype(float); log.append(f"'{c}' converted to days-since"); continue
                    if s.nunique() <= 20: feat[c] = s.fillna("unknown").astype(str); continue
                    dropped.append(c); continue
            feat[c] = s
        F = pd.DataFrame(feat); miss = int(F.isna().sum().sum()); catc = [c for c in F if not pd.api.types.is_numeric_dtype(F[c])]
        num = F.drop(columns=catc); num = num.fillna(num.median()).fillna(0)
        X = pd.concat([num, pd.get_dummies(F[catc], dtype=float)], axis=1); X = X.loc[:, X.nunique() > 1]
        if X.shape[1] == 0: raise ValidationError("No usable feature columns found.")
        if dropped: log.append(f"Dropped high-cardinality text columns (likely names/PII): {dropped}")
        signs = {c: dirs.get(c, infer_sign(c)) for c in num.columns if c in X.columns}
        ctx.update(ids=ids, X=X, y=y, plan=plan, revenue=rev, revenue_col=revc[0] if revc else None, contacts=contacts, domain=domain, signs=signs,
                   quality={"records": len(df), "missing_values": miss, "schema_status": "Valid", "log": log, "roles": {c: r for c, r in roles.items() if r != "feature"}, "domain": domain, "n_features": X.shape[1]})
        return ctx
    def validate_output(self, ctx): assert len(ctx["ids"]) == len(ctx["X"])

# ---------------- ML Agent: 4 base models + 2 ensembles, best selected ----------------
def build_candidates(n):
    base = {"random_forest": RandomForestClassifier(300, class_weight="balanced", random_state=42, n_jobs=-1),
            "logistic_regression": make_pipeline(StandardScaler(), LogisticRegression(max_iter=2000, class_weight="balanced")),
            "naive_bayes": make_pipeline(StandardScaler(), GaussianNB()),
            "svm": make_pipeline(StandardScaler(), SVC(probability=True, class_weight="balanced", random_state=42))}
    if n > 6000: base.pop("svm")  # SVM scales poorly; skip on very large uploads
    est = list(base.items())
    base["ensemble_voting"] = VotingClassifier(est, voting="soft")
    base["ensemble_stacking"] = StackingClassifier(est, final_estimator=LogisticRegression(max_iter=1000), cv=3)
    return base

class MLAgent(Agent):
    name = "MLAgent"
    def execute(self, ctx):
        X, y = ctx["X"], ctx.get("y"); Z = (X - X.mean()) / X.std().replace(0, 1)
        if y is not None and y.notna().sum() >= 50 and y.nunique() == 2 and y.value_counts().min() >= 10:
            m = y.notna(); Xl, yl = X[m], y[m].astype(int)
            Xtr, Xte, ytr, yte = train_test_split(Xl, yl, test_size=.25, stratify=yl, random_state=42)
            fitted, res = {}, {}
            for k, mdl in build_candidates(len(Xl)).items():
                mdl.fit(Xtr, ytr); p = mdl.predict_proba(Xte)[:, 1]; pred = (p >= .5).astype(int); fitted[k] = mdl
                res[k] = {a: round(float(b), 3) for a, b in dict(accuracy=accuracy_score(yte, pred), recall=recall_score(yte, pred), precision=precision_score(yte, pred, zero_division=0),
                          f1=f1_score(yte, pred), roc_auc=roc_auc_score(yte, p)).items()}
            best = max(res, key=lambda k: (res[k]["roc_auc"], res[k]["f1"]))
            pi = permutation_importance(fitted[best], Xte, yte, scoring="roc_auc", n_repeats=3, random_state=42, n_jobs=1).importances_mean
            imp = np.clip(pi, 0, None); imp = imp / imp.sum() if imp.sum() > 0 else np.ones(X.shape[1]) / X.shape[1]
            final = clone(fitted[best]).fit(Xl, yl); probs = final.predict_proba(X)[:, 1]   # refit on all labelled data
            sign = np.array([np.sign(np.corrcoef(Xl[c], yl)[0, 1]) if Xl[c].std() > 0 else 0 for c in X.columns])
            mode = "trained_model"; metrics = {"selected": best, "selection_metric": "roc_auc (held-out 25%)", "held_out": res}
        else:  # cold start: transparent rule-based fallback
            sg = ctx.get("signs", {}); sign = np.array([sg.get(c, 0) for c in X.columns], float); w = np.abs(sign)
            if w.sum() == 0: raise ValidationError("No churn labels and no recognisable engagement/risk columns for rule-based scoring.")
            imp = w / w.sum(); probs = np.clip((X.rank(pct=True).values * (imp * sign)).sum(axis=1) + imp[sign < 0].sum(), 0, 1)
            mode = "rule_based_fallback"; metrics = {"selected": "rule_based_fallback", "note": "No labelled churn outcomes; scores are heuristic and not validated."}
        ctx.update(probs=np.asarray(probs), contrib=Z.values * (imp * sign), Zvals=Z.values,
                   model={"mode": mode, "version": "m-" + hashlib.md5((mode + str(list(X.columns))).encode()).hexdigest()[:8] + time.strftime("-%Y%m%d"),
                          "metrics": metrics, "trained_at": now(), "train_mean_score": float(np.mean(probs))})
        return ctx
    def validate_output(self, ctx): assert np.all((ctx["probs"] >= 0) & (ctx["probs"] <= 1))

class RiskAgent(Agent):
    name = "RiskAgent"
    def execute(self, ctx):
        cols, out = list(ctx["X"].columns), []
        for i, cid in enumerate(ctx["ids"]):
            s = float(ctx["probs"][i]); lvl = "High" if s >= CONFIG["high_threshold"] else "Medium" if s >= CONFIG["medium_threshold"] else "Low"
            order = np.argsort(-ctx["contrib"][i])[:3]
            factors = [{"feature": cols[j], "direction": "high" if ctx["Zvals"][i][j] > 0 else "low", "impact": round(float(ctx["contrib"][i][j]), 4)} for j in order if ctx["contrib"][i][j] > 0]
            rev = None if ctx["revenue"] is None or np.isnan(ctx["revenue"][i]) else float(ctx["revenue"][i])
            out.append(dict(customer_id=cid, plan=str(ctx["plan"][i]), churn_score=round(s, 3), risk_level=lvl, top_factors=factors, monthly_charges=rev, model_version=ctx["model"]["version"]))
        ctx["assessments"] = out; return ctx

# ---------------- Offer Agent: plan tier x risk level x churn driver ----------------
LOW_T, HIGH_T = {"free","trial","basic","starter","lite","student","standard","monthly"}, {"premium","pro","plus","family","annual","yearly","gold","platinum","enterprise","business","vip","unlimited","prime","duo"}
def plan_tier(p):
    t = set(toks(p)); return 2 if t & HIGH_T else 0 if t & LOW_T else 1
INCENTIVE = {("High", 0): ("1-month premium upgrade trial", .15), ("High", 1): ("20% off next 3 renewals", .20), ("High", 2): ("25% loyalty credit + dedicated success contact", .25),
             ("Medium", 0): ("bonus content / perks pack", .05), ("Medium", 1): ("10% renewal discount", .10), ("Medium", 2): ("early access + loyalty points", .08)}
def driver_offer(feature, domain):
    t = set(toks(feature))
    if t & {"ad","ads","advert","adverts"}: return "Ad-free week"
    if t & {"payment","payments","late","failed","overdue","billing"}: return "Flexible payment plan"
    if t & {"support","ticket","tickets","complaint","complaints"}: return "Priority support callback"
    if t & {"device","devices","profile","profiles"}: return "Extra device/profile slot"
    if t & {"since","inactive","recency","abandon","abandoned"} or any(x.startswith(("listen","watch","song","movie","order","login","open","session","purchase","cart")) for x in t):
        return {"music_streaming": "Personalised playlist refresh", "video_streaming": "Curated watchlist + recommendations", "e_commerce": "Free-shipping coupon on next order"}.get(domain, "Guided feature walkthrough")
    return "Account health check-in"

class OfferAgent(Agent):
    name = "OfferAgent"
    def _rationale(self, a, offer, tier):
        facts = ", ".join(f"{f['direction']} {f['feature']}" for f in a["top_factors"]) or "general risk"
        if os.getenv("ANTHROPIC_API_KEY"):  # optional LLM step; no contact data is sent
            try:
                import anthropic
                msg = anthropic.Anthropic().messages.create(model=os.getenv("CLAUDE_MODEL", "claude-sonnet-4-6"), max_tokens=120, messages=[{"role": "user", "content":
                    f"In 2 sentences, justify offering '{offer}' to a '{a['plan']}' plan customer at {a['risk_level']} churn risk (score {a['churn_score']}). Drivers: {facts}. Do not promise retention."}])
                return msg.content[0].text.strip()
            except Exception: pass
        return f"{a['risk_level']} risk (score {a['churn_score']}) on the '{a['plan']}' plan, driven by {facts}. The offer targets the main driver and scales incentive to plan value. It is not a guarantee of retention."
    def execute(self, ctx):
        offers = {}; dom = ctx.get("domain", "")
        for a in ctx["assessments"]:
            if a["risk_level"] == "Low": a["clv_band"] = "n/a"; continue
            tier = plan_tier(a["plan"]); inc, pct = INCENTIVE[(a["risk_level"], tier)]
            main = next((driver_offer(f["feature"], dom) for f in a["top_factors"]), "Account health check-in"); m = a["monthly_charges"]
            offers[a["customer_id"]] = dict(offer_type=f"{main} + {inc}", incentive_pct=pct, plan_tier=["entry","standard","premium"][tier],
                estimated_cost=round(m * pct, 2) if m else None, rationale=self._rationale(a, f"{main} + {inc}", tier), status="Pending", approval_id=None)
        rev = [a["monthly_charges"] for a in ctx["assessments"] if a["monthly_charges"]]
        if len(rev) > 3:
            t1, t2 = np.percentile(rev, [33, 66])
            for a in ctx["assessments"]:
                m = a["monthly_charges"]; a["clv_band"] = a.get("clv_band") if a["risk_level"] == "Low" and m is None else ("n/a" if m is None else "Low" if m <= t1 else "Medium" if m <= t2 else "High")
        ctx["offers"] = offers; return ctx

# ---------------- Visual Agent: business insights ----------------
class VisualAgent(Agent):
    name = "VisualAgent"
    def execute(self, ctx):
        A = ctx["assessments"]; n = len(A); lv = lambda l: sum(a["risk_level"] == l for a in A); fc, plans = {}, {}
        for a in A:
            for f in a["top_factors"]:
                if a["risk_level"] == "High": fc[f["feature"]] = fc.get(f["feature"], 0) + 1
            p = plans.setdefault(a["plan"], dict(plan=a["plan"], customers=0, score=0.0, high=0)); p["customers"] += 1; p["score"] += a["churn_score"]; p["high"] += a["risk_level"] == "High"
        by_plan = sorted([dict(plan=p["plan"], customers=p["customers"], avg_score=round(p["score"] / p["customers"], 3), high_share=round(p["high"] / p["customers"], 3)) for p in plans.values()], key=lambda x: -x["avg_score"])
        hi = [a["monthly_charges"] for a in A if a["risk_level"] == "High" and a["monthly_charges"]]; allrev = [a["monthly_charges"] for a in A if a["monthly_charges"]]
        rc = ctx.get("revenue_col"); topf = sorted(fc.items(), key=lambda x: -x[1])[:5]; ins = []
        ins.append(f"{lv('High')} of {n} customers ({lv('High')/n:.0%}) are high risk and {lv('Medium')} are medium risk.")
        if len(by_plan) > 1: ins.append(f"The '{by_plan[0]['plan']}' plan has the highest average churn score ({by_plan[0]['avg_score']}); '{by_plan[-1]['plan']}' has the lowest ({by_plan[-1]['avg_score']}).")
        if topf: ins.append(f"Most common driver among high-risk customers: {topf[0][0]} ({topf[0][1]} customers). Target retention campaigns at it first.")
        if hi and allrev: ins.append(f"High-risk customers hold {sum(hi)/sum(allrev):.0%} of total '{rc}' (estimate, not measured revenue loss).")
        m = ctx["model"]["metrics"]
        ins.append(f"Scores come from {'the ' + m['selected'].replace('_', ' ') + ' model (best of 6 on held-out ROC-AUC ' + str(m['held_out'][m['selected']]['roc_auc']) + ')' if 'held_out' in m else 'transparent rules (no labels, unvalidated)'}.")
        cost = [o["estimated_cost"] for o in ctx["offers"].values() if o["estimated_cost"]]
        if cost: ins.append(f"Offering all proposed incentives would cost about {sum(cost):,.0f} per billing period at the configured rates; approve selectively.")
        hist = np.histogram([a["churn_score"] for a in A], bins=10, range=(0, 1))[0].tolist()
        ctx["dashboard"] = dict(customers_scored=n, high=lv("High"), medium=lv("Medium"), low=lv("Low"), avg_score=round(float(np.mean([a["churn_score"] for a in A])), 3),
            revenue_at_risk=(f"{sum(hi):,.0f} (estimate: sum of '{rc}' for High-risk customers)" if hi else "Estimate unavailable: no revenue field"),
            top_factors=topf, by_plan=by_plan, score_hist=hist, insights=ins, domain=ctx.get("domain"), model=ctx["model"], quality=ctx["quality"],
            drift=dict(status="baseline recorded", note="Compare future mean scores with train_mean_score; alert if shift > 0.15."))
        return ctx

class PipelineOrchestrator:
    AGENTS = [DataAgent(), MLAgent(), RiskAgent(), OfferAgent(), VisualAgent()]
    def run_workflow(self, job):
        job["state"] = "Processing"; self.audit(job, "workflow_started"); ctx = {"raw": job.pop("raw")}
        try:
            for ag in self.AGENTS:
                if job["state"] == "Cancelled": return
                ctx = self.handle_retry(ag, ctx, job)
            job.update(assessments=ctx["assessments"], offers=ctx["offers"], dashboard=ctx["dashboard"], contacts=ctx["contacts"])
            self.audit(job, "scored", model=ctx["model"]["version"]); job["state"] = "Awaiting approval"
        except ValidationError as e: self.fail(job, str(e))
        except Exception as e: self.fail(job, f"{type(e).__name__}: {e}")
    def handle_retry(self, ag, ctx, job):
        for attempt in range(1, CONFIG["max_retries"] + 1):
            try:
                ctx = ag.execute(ctx); ag.validate_output(ctx); self.audit(job, f"{ag.name}_ok"); return ctx
            except TransientError:
                self.audit(job, f"{ag.name}_retry_{attempt}"); time.sleep(0.2 * 2 ** attempt)
        raise RuntimeError(f"{ag.name} failed after {CONFIG['max_retries']} attempts")
    def fail(self, job, reason): job.update(state="Failed", reason=reason, finished=now()); self.audit(job, "failed", reason=reason)
    @staticmethod
    def audit(job, event, **kw): job["audit"].append({"ts": now(), "event": event, **kw})

# --- FR-05 channel adapter: SIMULATED unless a real provider is wired in ---
EMAIL_RE = re.compile(r"^[^@\s]+@[^@\s]+\.[^@\s]+$"); PHONE_RE = re.compile(r"^\+?[0-9][0-9\- ]{7,14}$")
def deliver(job, cid):
    o, c = job["offers"][cid], job["contacts"].get(cid, {})
    key = f"{job['id']}:{cid}"                      # idempotency key
    if key in job["deliveries"]: return job["deliveries"][key]
    consent = str(c.get("consent", "")).strip().lower() in {"1", "yes", "true", "y"}
    email = "" if pd.isna(c.get("email")) else str(c.get("email") or ""); phone = "" if pd.isna(c.get("phone")) else str(c.get("phone") or "")
    if o["status"] != "Approved": r = {"status": "Blocked: not approved"}
    elif not consent: r = {"status": "Skipped: no consent"}
    elif not (EMAIL_RE.match(email) or PHONE_RE.match(phone)): r = {"status": "Skipped: no valid contact"}
    else: r = {"status": "Simulated-sent", "channel": "email" if EMAIL_RE.match(email) else "sms", "simulated": True}
    r["ts"] = now(); job["deliveries"][key] = r; o["status"] = "Sent (simulated)" if r["status"] == "Simulated-sent" else r["status"]
    PipelineOrchestrator.audit(job, "delivery", customer=cid, result=r["status"]); return r
