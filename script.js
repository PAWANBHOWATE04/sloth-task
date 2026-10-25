/**
 * SlothTask — script.js
 *
 * Inference library : @huggingface/transformers v3 (via jsDelivr CDN)
 * Model             : HuggingFaceTB/SmolLM2-360M-Instruct
 * Source            : https://huggingface.co/HuggingFaceTB/SmolLM2-360M-Instruct
 *
 * The model is open-weight, instruction-tuned, and runs entirely in the
 * browser via WebGPU (if available) or WebAssembly fallback.
 * No task text is sent to any remote server.
 */

import { pipeline, env } from "https://cdn.jsdelivr.net/npm/@huggingface/transformers@3.5.1";

// ─── Library configuration ─────────────────────────────────────────────────
env.allowLocalModels  = false;   // fetch from Hugging Face Hub only
env.useBrowserCache   = true;    // reuse IndexedDB / Cache API on repeat visits

// ─── Constants ─────────────────────────────────────────────────────────────
const MODEL_ID   = "HuggingFaceTB/SmolLM2-360M-Instruct";
const MAX_INPUT  = 4000;   // chars — enforced by textarea maxlength too
const MAX_TOKENS = 512;    // max new tokens for the plan output

/** Category definitions (icon, label, CSS modifier, key in parsed plan) */
const CATEGORIES = [
  {
    key:      "start_here",
    label:    "Start Here",
    icon:     "🚀",
    modifier: "start",
  },
  {
    key:      "next_steps",
    label:    "Next Steps",
    icon:     "📋",
    modifier: "next",
  },
  {
    key:      "quick_wins",
    label:    "Quick Wins",
    icon:     "⚡",
    modifier: "quick",
  },
  {
    key:      "do_later",
    label:    "Do Later",
    icon:     "🌙",
    modifier: "later",
  },
];

// ─── State ─────────────────────────────────────────────────────────────────
let generator  = null;   // loaded pipeline, reused across requests
let isLoading  = false;  // guard against concurrent submissions

// ─── DOM references ────────────────────────────────────────────────────────
const taskInput      = document.getElementById("task-input");
const charCounter    = document.getElementById("char-counter");
const simplifyBtn    = document.getElementById("simplify-btn");
const startOverBtn   = document.getElementById("start-over-btn");
const validationMsg  = document.getElementById("validation-msg");

const statusSection  = document.getElementById("status-section");
const statusTitle    = document.getElementById("status-title");
const statusDetail   = document.getElementById("status-detail");
const progressWrap   = document.getElementById("progress-wrap");
const progressFill   = document.getElementById("progress-bar-fill");
const progressLabel  = document.getElementById("progress-label");
const progressAria   = document.getElementById("progress-bar-aria");

const errorSection   = document.getElementById("error-section");
const errorTitleEl   = document.getElementById("error-title");
const errorDetailEl  = document.getElementById("error-detail");
const retryBtn       = document.getElementById("retry-btn");

const resultsSection = document.getElementById("results-section");
const resultsGrid    = document.getElementById("results-grid");

// ─── Utility helpers ────────────────────────────────────────────────────────

function show(el)  { el.classList.remove("hidden"); }
function hide(el)  { el.classList.add("hidden"); }

function setStatus(title, detail, showProgress = false) {
  statusTitle.textContent  = title;
  statusDetail.textContent = detail;
  show(statusSection);
  if (showProgress) {
    show(progressWrap);
  } else {
    hide(progressWrap);
  }
}

function setProgress(fraction) {
  const pct = Math.round(Math.min(1, Math.max(0, fraction)) * 100);
  progressFill.style.width              = pct + "%";
  progressLabel.textContent             = pct + "%";
  progressAria.setAttribute("aria-valuenow", pct);
}

function showError(title, detail) {
  errorTitleEl.textContent  = title;
  errorDetailEl.textContent = detail;
  show(errorSection);
  hide(statusSection);
}

function clearError() {
  hide(errorSection);
}

function showValidation(msg) {
  validationMsg.textContent = msg;
  show(validationMsg);
}

function clearValidation() {
  hide(validationMsg);
}

function setUILocked(locked) {
  taskInput.disabled   = locked;
  simplifyBtn.disabled = locked;
}

// ─── Char counter ──────────────────────────────────────────────────────────

taskInput.addEventListener("input", () => {
  const len = taskInput.value.length;
  charCounter.textContent = `${len} / ${MAX_INPUT}`;
  charCounter.classList.toggle("near-limit", len > MAX_INPUT * 0.85);
  if (len > 0) clearValidation();
});

// ─── Model loading ─────────────────────────────────────────────────────────

/**
 * Loads the SmolLM2 text-generation pipeline once and caches it.
 * Reports download progress via the status card and progress bar.
 */
async function loadModel() {
  if (generator) return;   // already loaded — reuse

  setStatus(
    "Downloading AI model…",
    "SmolLM2-360M-Instruct is being fetched from Hugging Face Hub (~200 MB on first visit). This is cached in your browser for future sessions.",
    true
  );
  setProgress(0);

  /**
   * Transformers.js progress callback.
   * Called repeatedly during shard downloads; `progress` is 0–100 (per file).
   * We accumulate a running average to give an overall feel.
   */
  const progressTracker = {
    files: {},
    overall() {
      const vals = Object.values(this.files);
      if (!vals.length) return 0;
      return vals.reduce((a, b) => a + b, 0) / vals.length / 100;
    },
  };

  generator = await pipeline(
    "text-generation",
    MODEL_ID,
    {
      device: "webgpu",      // Transformers.js auto-falls back to WASM if unsupported
      dtype:  "q4f16",       // 4-bit weights + fp16 activations — optimal for WebGPU
      progress_callback: (p) => {
        if (p && p.file && typeof p.progress === "number") {
          progressTracker.files[p.file] = p.progress;
          const overall = progressTracker.overall();
          setProgress(overall);
          setStatus(
            "Downloading AI model…",
            `Fetching: ${p.file} — overall ${Math.round(overall * 100)}%`,
            true
          );
        }
      },
    }
  );

  setProgress(1);
  hide(progressWrap);
}

// ─── Prompt engineering ────────────────────────────────────────────────────

/**
 * Builds a structured chat prompt that instructs SmolLM2 to return a clean
 * JSON object with up to four task categories.  We give the model a concrete
 * output schema and several rules to follow.
 */
function buildPrompt(userInput) {
  const systemPrompt = `You are SlothTask, a calm and focused AI task organiser. Your ONLY job is to analyse the user's unorganised thoughts and return a valid JSON object that groups their tasks into a prioritised plan.

Output ONLY a raw JSON object — no markdown fences, no explanation, no preamble. The JSON must follow this exact schema:

{
  "start_here": [
    {"task": "...", "tip": "..."}
  ],
  "next_steps": [
    {"task": "..."}
  ],
  "quick_wins": [
    {"task": "..."}
  ],
  "do_later": [
    {"task": "..."}
  ]
}

Rules:
- Include only the categories that are relevant. Omit empty arrays entirely.
- "start_here" must contain exactly ONE item — the single best task to start with right now.
- "quick_wins" are tasks that take less than 10 minutes or require minimal effort.
- "do_later" are tasks with no immediate deadline or lower urgency.
- "next_steps" are everything else that matters soon.
- Do NOT invent deadlines, commitments, or tasks not mentioned by the user.
- Keep task descriptions concise (under 15 words each).
- The optional "tip" field is a brief helpful note (under 12 words). Only add one if genuinely useful.
- Preserve the user's original intent and meaning.
- Output ONLY the JSON object. Nothing else.`;

  return [
    { role: "system",    content: systemPrompt },
    { role: "user",      content: `Here are my tasks and thoughts:\n\n${userInput.trim()}` },
  ];
}

// ─── Output parsing ────────────────────────────────────────────────────────

/**
 * Attempts to extract and parse JSON from the raw model output.
 * The model is instructed to output only JSON, but we defensively
 * look for the first { ... } block in case it adds commentary.
 *
 * Returns a parsed plan object, or throws a descriptive error.
 */
function parseModelOutput(rawText) {
  if (!rawText || typeof rawText !== "string") {
    throw new Error("The model returned an empty response.");
  }

  // Strip leading/trailing whitespace
  let text = rawText.trim();

  // Remove markdown code fences if the model added them despite instructions
  text = text.replace(/^```(?:json)?\s*/i, "").replace(/```\s*$/i, "").trim();

  // Find the outermost JSON object
  const start = text.indexOf("{");
  const end   = text.lastIndexOf("}");

  if (start === -1 || end === -1 || end <= start) {
    throw new Error(
      "The model's response did not contain a recognisable task plan. " +
      "This can happen on low-memory devices or if the model is too small for the input. " +
      "Try simplifying your input or refreshing the page."
    );
  }

  const jsonString = text.slice(start, end + 1);

  let parsed;
  try {
    parsed = JSON.parse(jsonString);
  } catch {
    throw new Error(
      "The model returned a malformed response that could not be interpreted as a task plan. " +
      "Please try again — the model may produce better output on a second attempt."
    );
  }

  // Validate structure: must be an object with at least one known category key
  const validKeys = ["start_here", "next_steps", "quick_wins", "do_later"];
  const hasAny = validKeys.some(
    (k) => Array.isArray(parsed[k]) && parsed[k].length > 0
  );

  if (!hasAny) {
    throw new Error(
      "The model returned a response without any recognisable task categories. " +
      "Please try again with a clearer description of your tasks."
    );
  }

  // Sanitise each array — ensure items are objects with a 'task' string
  for (const key of validKeys) {
    if (!Array.isArray(parsed[key])) {
      delete parsed[key];
      continue;
    }
    parsed[key] = parsed[key]
      .filter((item) => item && typeof item.task === "string" && item.task.trim())
      .map((item) => ({
        task: String(item.task).trim(),
        tip:  item.tip ? String(item.tip).trim() : undefined,
      }));

    if (parsed[key].length === 0) delete parsed[key];
  }

  // Re-validate after sanitisation
  const hasAnyAfterSanitise = validKeys.some((k) => Array.isArray(parsed[k]) && parsed[k].length > 0);
  if (!hasAnyAfterSanitise) {
    throw new Error(
      "The model's output could not be turned into a usable task plan. " +
      "Please try again."
    );
  }

  return parsed;
}

// ─── Results rendering ─────────────────────────────────────────────────────

function renderResults(plan) {
  resultsGrid.innerHTML = "";

  let cardsRendered = 0;

  for (const cat of CATEGORIES) {
    const items = plan[cat.key];
    if (!Array.isArray(items) || items.length === 0) continue;

    const card = document.createElement("article");
    card.className = `result-card result-card--${cat.modifier}`;
    card.setAttribute("aria-label", cat.label);

    const header = document.createElement("div");
    header.className = "result-card-header";
    header.innerHTML = `
      <span class="result-card-icon" aria-hidden="true">${cat.icon}</span>
      <span class="result-card-label">${escapeHtml(cat.label)}</span>
    `;
    card.appendChild(header);

    const list = document.createElement("ul");
    list.className = "result-card-tasks";
    list.setAttribute("role", "list");

    for (const item of items) {
      const li = document.createElement("li");
      li.className = "result-task-item";
      li.setAttribute("role", "listitem");

      const textSpan = document.createElement("span");
      textSpan.className   = "result-task-text";
      textSpan.textContent = item.task;

      li.appendChild(textSpan);

      if (item.tip) {
        const tipSpan = document.createElement("span");
        tipSpan.className   = "result-task-tip";
        tipSpan.textContent = `💡 ${item.tip}`;
        textSpan.appendChild(tipSpan);
      }

      list.appendChild(li);
    }

    card.appendChild(list);
    resultsGrid.appendChild(card);
    cardsRendered++;
  }

  if (cardsRendered === 0) {
    throw new Error("No task categories were produced. Please try again.");
  }

  show(resultsSection);
  resultsSection.scrollIntoView({ behavior: "smooth", block: "start" });
}

function escapeHtml(str) {
  return str
    .replace(/&/g, "&amp;")
    .replace(/</g, "&lt;")
    .replace(/>/g, "&gt;")
    .replace(/"/g, "&quot;");
}

// ─── Core: simplify tasks ──────────────────────────────────────────────────

async function simplifyTasks() {
  if (isLoading) return;

  clearValidation();
  clearError();

  // ── Validate input ─────────────────────────────────────────────────────
  const rawInput = taskInput.value;

  if (!rawInput.trim()) {
    showValidation("Please enter at least one task or thought before simplifying.");
    taskInput.focus();
    return;
  }

  if (rawInput.trim().length < 5) {
    showValidation("Your input is very short — please describe your tasks in a bit more detail.");
    taskInput.focus();
    return;
  }

  if (rawInput.length > MAX_INPUT) {
    showValidation(`Your input is too long (${rawInput.length} / ${MAX_INPUT} characters). Please shorten it.`);
    taskInput.focus();
    return;
  }

  // ── Lock UI ────────────────────────────────────────────────────────────
  isLoading = true;
  setUILocked(true);
  hide(resultsSection);
  hide(errorSection);
  show(statusSection);

  try {
    // ── Step 1: Load model (no-op if already cached) ───────────────────
    if (!generator) {
      await loadModel();
    }

    // ── Step 2: Generate task plan ─────────────────────────────────────
    setStatus(
      "Organising your tasks…",
      "The AI is thinking. This may take 30–90 seconds on first run.",
      false
    );

    const messages = buildPrompt(rawInput);

    // Pass messages array directly — Transformers.js v3 applies the chat
    // template internally when the model defines one (SmolLM2 does).
    const result = await generator(messages, {
      max_new_tokens:     MAX_TOKENS,
      temperature:        0.3,      // low temperature = more deterministic / structured
      do_sample:          true,
      repetition_penalty: 1.15,
    });

    // ── Step 3: Extract generated text ────────────────────────────────
    // Transformers.js v3 chat pipeline returns generated_text as an array
    // of message objects; the last item is the assistant's response.
    // Fall back to string handling for non-chat or older behaviour.
    let generatedText = "";
    const rawOutput = result[0]?.generated_text;
    if (Array.isArray(rawOutput)) {
      // Chat format: [{role:"system",...},{role:"user",...},{role:"assistant",content:"..."}]
      const lastMsg = rawOutput[rawOutput.length - 1];
      generatedText = (lastMsg?.content ?? "").trim();
    } else if (typeof rawOutput === "string") {
      // Legacy string format — the full prompt+response is returned; we
      // cannot reliably strip the prompt, so just use the whole string.
      generatedText = rawOutput.trim();
    }

    // ── Step 4: Parse and render ───────────────────────────────────────
    const plan = parseModelOutput(generatedText);
    hide(statusSection);
    renderResults(plan);

    // Show Start Over button
    show(startOverBtn);
    simplifyBtn.querySelector(".btn-label").textContent = "Simplify Again";

  } catch (err) {
    console.error("[SlothTask] Error during task generation:", err);

    hide(statusSection);

    // Distinguish model-load errors from inference errors
    const isLoadError =
      err.message?.toLowerCase().includes("fetch") ||
      err.message?.toLowerCase().includes("load") ||
      err.message?.toLowerCase().includes("network") ||
      err.message?.toLowerCase().includes("webgpu") ||
      !generator;

    if (isLoadError && !generator) {
      // Reset so user can retry loading
      generator = null;
      showError(
        "Model failed to load",
        `Could not download or initialise the AI model. ${err.message || ""} ` +
        "Please check your internet connection and try again. " +
        "WebGPU is required on some browsers — try Chrome or Edge if you're on Firefox."
      );
    } else {
      showError(
        "Couldn't generate a plan",
        err.message ||
        "An unexpected error occurred during AI inference. " +
        "Please try again — smaller or clearer input sometimes helps."
      );
    }

    // Unlock input so the user can retry
    show(startOverBtn);

  } finally {
    isLoading = false;
    setUILocked(false);
  }
}

// ─── Start Over ────────────────────────────────────────────────────────────

function startOver() {
  taskInput.value         = "";
  charCounter.textContent = `0 / ${MAX_INPUT}`;
  charCounter.classList.remove("near-limit");

  clearValidation();
  clearError();
  hide(statusSection);
  hide(resultsSection);
  hide(startOverBtn);

  simplifyBtn.querySelector(".btn-label").textContent = "Simplify My Day";

  taskInput.focus();
  // Scroll back to top
  window.scrollTo({ top: 0, behavior: "smooth" });
}

// ─── Retry ─────────────────────────────────────────────────────────────────

retryBtn.addEventListener("click", () => {
  clearError();
  simplifyTasks();
});

// ─── Event listeners ────────────────────────────────────────────────────────

simplifyBtn.addEventListener("click", simplifyTasks);
startOverBtn.addEventListener("click", startOver);

// Allow Ctrl+Enter / Cmd+Enter to submit
taskInput.addEventListener("keydown", (e) => {
  if ((e.ctrlKey || e.metaKey) && e.key === "Enter") {
    e.preventDefault();
    simplifyTasks();
  }
});
