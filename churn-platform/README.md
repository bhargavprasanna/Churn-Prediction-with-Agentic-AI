# Agentic Churn Platform
Backend (FastAPI) + agents (Data, ML, Risk, Offer, Visual) + web UI + Android WebView shell. See Setup_and_Run_Guide.pdf.
```
pip install -r backend/requirements.txt && python sample_data/make_sample.py
cd backend && uvicorn app.main:app --host 0.0.0.0 --port 8000     # open http://localhost:8000
python -m pytest tests -q                                          # from project root
```
APK: push to GitHub and run the "Build APK" workflow (enter your backend URL), or see the PDF for a local Gradle build.
