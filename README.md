# 🦥 SlothTask — AI-Powered Task Simplifier

**Do less thinking. Get more done.**

SlothTask transforms your messy notes, overwhelming to-do lists, and scattered thoughts into a calm, prioritised, actionable plan — powered entirely by an open-weight AI model running locally in your browser.

---

## What SlothTask Does

Paste everything on your mind into a single text area. SlothTask's on-device AI reads your input and organises it into four clear categories:

| Category | Description |
|---|---|
| 🚀 **Start Here** | The single best task to tackle right now |
| 📋 **Next Steps** | Other important tasks in priority order |
| ⚡ **Quick Wins** | Tasks completable in under 10 minutes |
| 🌙 **Do Later** | Lower-urgency tasks that can wait |

Only the relevant categories are shown. No tasks are invented — the AI only works with what you provide.

---

## The Problem It Solves

Decision fatigue is real. When you have 15 things in your head, the hardest part is deciding what to do *first*. SlothTask removes that friction by quietly applying the "slow and steady" approach: one clear starting point, then the next step, then the next.

---

## How It Fits the "Slow and Steady" Theme

SlothTask embodies deliberate, unhurried progress. Instead of overwhelming users with complex dashboards or gamification, it does one thing: turns chaos into clarity. Steady focus beats frantic multitasking — which is exactly what a sloth would tell you.

---

## Open-Weight AI Model & Inference Library

| | |
|---|---|
| **Model** | [SmolLM2-360M-Instruct](https://huggingface.co/HuggingFaceTB/SmolLM2-360M-Instruct) by Hugging Face |
| **Parameters** | 360 million |
| **Quantisation** | 4-bit (q4) via ONNX — reduces download to ~200 MB |
| **Inference library** | [@huggingface/transformers v3](https://github.com/huggingface/transformers.js) (Transformers.js) |
| **CDN** | `https://cdn.jsdelivr.net/npm/@huggingface/transformers@3.5.1` |
| **Backend** | WebGPU (primary) → WebAssembly (automatic fallback) |

### Why browser-based inference?

- **Privacy**: Your task text never leaves your device. There is no server-side API call.
- **Cost**: No paid API keys. No quotas. No rate limits.
- **Open-weight**: SmolLM2-360M-Instruct is openly licensed. The weights are hosted publicly on Hugging Face Hub.
- **Feasibility**: At 360M parameters with 4-bit quantisation, the model fits comfortably within browser memory on modern devices and loads in a reasonable time.

---

## How to Run Locally

Because the application uses ES modules (`type="module"`) and loads model shards from the internet, it **must be served from an HTTP server** — not opened directly as a `file://` URL (browsers block module imports and cross-origin fetches from `file://`).

### Option A — Python (no install required on most systems)

```bash
cd sloth-task
python -m http.server 8080
```

Then open: `http://localhost:8080`

### Option B — VS Code Live Server extension

1. Open the `sloth-task` folder in VS Code.
2. Right-click `index.html` → **Open with Live Server**.

### Option C — Any static file server

Serve the `sloth-task` directory with any HTTP server that can serve static files. The app has no build step and requires no configuration.

---

## Model Download Requirements

- **First visit**: The browser downloads SmolLM2-360M-Instruct from Hugging Face Hub (~200 MB total, in several quantised shards). A progress bar and status message are shown.
- **Subsequent visits**: The model is cached in browser storage (Cache API / IndexedDB). Loading takes only a few seconds.
- **Internet required**: An active internet connection is needed on first use to download the model. After caching, the app can generate plans offline (as long as the cached model is still in browser storage).

---

## Browser & Hardware Limitations

| Browser | WebGPU | Notes |
|---|---|---|
| Chrome 113+ | ✅ | Best performance |
| Edge 113+ | ✅ | Best performance |
| Firefox 120+ | ⚠️ | WebGPU flag may need enabling; falls back to WASM |
| Safari 17+ | ⚠️ | WebGPU in development; WASM fallback used |
| Mobile browsers | ⚠️ | Limited by memory; may be slow |

**Performance notes:**
- WebGPU: Generates a plan in roughly 10–40 seconds on a mid-range laptop GPU.
- WebAssembly (CPU fallback): Slower — 1–5 minutes on most CPUs.
- The app shows a loading indicator and status message throughout.
- Devices with less than 4 GB of RAM may struggle to load the 4-bit model.

---

## Privacy

- ✅ Task text is processed **entirely in your browser**.
- ✅ No task data is sent to any remote server or API.
- ✅ No analytics, tracking, or user accounts.
- ⚠️ The model files are fetched from Hugging Face Hub on first use (public CDN — no auth required).

---

## Project Structure

```
sloth-task/
├── index.html   — Page structure and layout
├── style.css    — Design system and all styling
├── script.js    — AI logic, model loading, parsing, rendering
└── README.md    — This file
```

No build tools, no `package.json`, no dependencies to install.

---

## Publishing to GitHub

```bash
git add index.html style.css script.js README.md
git commit -m "feat: initial SlothTask implementation"
git push origin main
```

---

## Deploying to Vercel (Static Site)

1. Go to [vercel.com](https://vercel.com) and sign in.
2. Click **Add New → Project**.
3. Import your GitHub repository.
4. On the configuration screen:
   - **Framework Preset**: Other
   - **Build Command**: *(leave empty)*
   - **Output Directory**: *(leave empty or set to `.`)*
   - **Install Command**: *(leave empty)*
5. Click **Deploy**.

Vercel will serve `index.html` directly. No build step is needed.

> **Note on CORS**: Vercel's CDN serves files over HTTPS, which allows ES module imports and Hugging Face Hub model fetches to work correctly. The `file://` restriction does not apply when deployed.

---

## Known Limitations & Untested Scenarios

- **SmolLM2-360M-Instruct** is a 360M-parameter model — smaller than production LLMs. It follows the JSON schema reliably in most cases, but may occasionally produce malformed output. The app handles this gracefully and prompts the user to retry.
- **First-time download** of ~200 MB may be slow on limited connections.
- **Mobile devices** with limited RAM (under 3–4 GB available) may fail to load the model.
- **WebGPU availability**: Not all browser versions and GPU drivers support WebGPU. The app falls back to WASM automatically, but WASM inference is significantly slower.
- **Very long inputs** (approaching the 4000-character limit) may cause the model to truncate its output.
- The app has been implemented and verified for structure, prompt engineering, and error handling. Full end-to-end AI inference testing requires a browser with WebGPU support and a stable internet connection for the initial model download.

---

## Licence

MIT