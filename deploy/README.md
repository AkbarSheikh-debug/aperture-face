# Hosting the web app

The web app (`api/static/index.html` + the FastAPI service) can be hosted in two ways. Both serve the same page and the same model (INT8 ResNet18, CPU only).

| | **Stateless / serverless** (`deploy/vercel/`) | **Stateful container** (`api/Dockerfile`) |
|---|---|---|
| Good for | a free public demo link | edge boxes (Raspberry Pi / Jetson), a company server |
| Where enrolled faces live | the visitor's browser only | browser (web page) **and** SQLite via `/enroll` `/identify` |
| Needs | a Vercel account | any host that runs Docker |

## Option A &mdash; Vercel (free Hobby plan)

`deploy/vercel/` is a self-contained project: `app.py` (stateless FastAPI), `requirements.txt` (5 pinned packages, no PyTorch), the 23 MB INT8 model, the face detector, the page and figures. It writes nothing to disk, which is what serverless hosting requires.

```bash
npm i -g vercel            # or use `npx vercel` instead of `vercel` below
vercel login               # opens the browser; sign in with GitHub / e-mail
cd deploy/vercel
vercel                     # first run: answer the prompts (framework: Other / FastAPI is auto-detected) -> preview URL
vercel --prod              # production URL, https://<project>.vercel.app
```

Or without the CLI: push this repo to GitHub, then in the Vercel dashboard choose **Add New &rarr; Project**, import the repo, and set **Root Directory** to `deploy/vercel`.

The camera needs HTTPS, which Vercel provides on every deployment.

If you edit anything under `api/`, `common/` or `results/`, refresh the deployable copy first:

```bash
python deploy/build_vercel.py
```

### Things to know (Vercel)
* **Deployed 2026-11-22** to https://aperture-face.vercel.app (project `aperture-face`, Hobby plan, public). The browser test suite (24 checks: enrol, identify, live mode, guided enrol, compare, results tab, stranger rejected) passed against the live URL at desktop (1200x1000) and phone (390x844) sizes with a fake camera. Not tested: a real phone, a real head turning, or heavy traffic. Warm `/analyze` calls took about 0.7 s from this PC. To redeploy: `python deploy/build_vercel.py`, then `cd deploy/vercel && vercel deploy --prod --yes`. Delete `deploy/vercel/.vercel` and `.env.local` (a Vercel token) before sharing the code.
* **Free-tier budget.** Hobby includes 4 CPU-hours/month of function time; exceeding it pauses functions for up to 30 days. The page therefore stops live mode after 2 minutes.
* **Cold starts.** After idle time the first request loads onnxruntime and the model, so it can take a few seconds; later requests are fast.
* **Hobby plan is for personal, non-commercial use** ([fair-use guidelines](https://vercel.com/docs/limits/fair-use-guidelines)). A portfolio / assessment demo fits; a product that serves paying customers does not. The model weights are also research-only (DigiFace-1M licence), so keep it non-commercial either way.
* **Request size.** Vercel rejects request bodies over 4.5 MB; the page shrinks uploaded photos in the browser first, and the API answers `413` for anything larger.
* **Privacy.** Frames are analysed in memory and discarded; nothing is stored on the server. Enrolled embeddings sit in the visitor's own `localStorage` (the page has a *Delete all* button). Anyone who opens the public link can use the demo, but cannot see anybody else's enrolments.

## Option B &mdash; Docker (any host)

```bash
docker build -f api/Dockerfile -t face-verify .      # from submission/
docker run -p 8000:8000 face-verify                   # http://localhost:8000
```

Free container hosts exist (Render's free web service sleeps when idle; Google Cloud Run has a free tier but asks for a card; Hugging Face Spaces supports Docker, but community reports say the free CPU tier may now require a PRO account, so check before relying on it). Browsers only allow camera access on `https://` or `localhost`, so use the host's HTTPS URL.

The container additionally exposes the stateful endpoints (`POST /enroll`, `POST /identify`, `DELETE /enroll/{name}`) backed by SQLite. These keep **one shared gallery for everyone who can reach the service**, so do not expose them on a public URL without adding authentication; the web page itself does not use them.
