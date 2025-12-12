// Elements
const chatEl = document.getElementById("chat");
const questionInput = document.getElementById("question");
const sendBtn = document.getElementById("send");
const reindexBtn = document.getElementById("reindex");
const fileLinkEl = document.getElementById("file-link");

const attachBtn = document.getElementById("attachBtn");
const uploadPop = document.getElementById("uploadPop");
const realFileInput = document.getElementById("realFileInput");
const fileChips = document.getElementById("fileChips");
const uploadNow = document.getElementById("uploadNow");
const clearFiles = document.getElementById("clearFiles");
const uploadResult = document.getElementById("uploadResult");

let selectedFiles = []; // FileList -> array
let chat_history = [];

// helper: render message
function pushMessage(role, text, meta = null) {
  const wrapper = document.createElement("div");
  wrapper.className = "msg " + (role === "user" ? "user" : "assistant");
  const bubble = document.createElement("div");
  bubble.className = "bubble";
  bubble.innerText = text;
  wrapper.appendChild(bubble);

  if (meta) {
    const metaEl = document.createElement("div");
    metaEl.className = "meta";
    // meta is HTML created by sendQuestion (escaped where needed)
    metaEl.innerHTML = meta;
    wrapper.appendChild(metaEl);
  }

  chatEl.appendChild(wrapper);
  chatEl.scrollTop = chatEl.scrollHeight;
}

// initial assistance message
pushMessage(
  "assistant",
  "Hi, I am Charlie, Your Chromeleon AI Assistant — ask me anything about Chromeleon. Use the + to attach files for indexing."
);

// ping backend for file link (optional)
fetch("/api/query", {
  method: "POST",
  headers: { "Content-Type": "application/json" },
  body: JSON.stringify({ question: "ping file", chat_history: [] }),
})
  .then((r) => r.json())
  .then((d) => {
    fileLinkEl.innerText = d.file_url || "none";
  })
  .catch(() => (fileLinkEl.innerText = "none"));

// Attach button toggles popover
attachBtn.addEventListener("click", (e) => {
  uploadPop.classList.toggle("show");
  uploadPop.setAttribute(
    "aria-hidden",
    uploadPop.classList.contains("show") ? "false" : "true"
  );
});

// File chooser handling
realFileInput.addEventListener("change", (e) => {
  const files = Array.from(e.target.files || []);
  selectedFiles = files;
  renderChips();
});

clearFiles.addEventListener("click", () => {
  selectedFiles = [];
  realFileInput.value = "";
  renderChips();
});

function renderChips() {
  fileChips.innerHTML = "";
  if (!selectedFiles || selectedFiles.length === 0) {
    fileChips.style.display = "none";
    return;
  }
  fileChips.style.display = "flex";
  selectedFiles.forEach((f) => {
    const chip = document.createElement("div");
    chip.className = "chip";
    chip.title = f.name;
    chip.innerText = f.name.length > 20 ? f.name.slice(0, 18) + "..." : f.name;
    fileChips.appendChild(chip);
  });
}

// Upload & Index button inside popover
uploadNow.addEventListener("click", async () => {
  if (!selectedFiles || selectedFiles.length === 0) {
    alert("Choose files first.");
    return;
  }
  uploadNow.disabled = true;
  uploadNow.innerText = "Working...";
  uploadResult.style.display = "none";
  try {
    const form = new FormData();
    selectedFiles.forEach((f) => form.append("files", f));
    const resp = await fetch("/upload", { method: "POST", body: form });
    const data = await resp.json();
    if (!resp.ok) {
      uploadResult.style.display = "block";
      uploadResult.innerText =
        "Upload failed: " + JSON.stringify(data, null, 2);
    } else {
      uploadResult.style.display = "block";
      uploadResult.innerText =
        "Indexing result:\n" + JSON.stringify(data, null, 2);
      // clear selection
      selectedFiles = [];
      realFileInput.value = "";
      renderChips();
      // optionally close pop
      uploadPop.classList.remove("show");
    }
  } catch (err) {
    uploadResult.style.display = "block";
    uploadResult.innerText = "Upload error: " + String(err);
  } finally {
    uploadNow.disabled = false;
    uploadNow.innerText = "Upload & Index";
  }
});

// Helper: extract KB number digits from filename, return padded 9-digit string or null
function extractKbDigits(filename) {
  if (!filename || typeof filename !== "string") return null;
  // match KB_12345 or KB-12345 or KB12345 (case-insensitive)
  const m = filename.match(/KB[_\-]?(\d{3,})/i);
  if (!m) return null;
  const raw = m[1].replace(/^0+/, "") || m[1]; // remove leading zeros but keep if all zeros
  return raw.padStart(9, "0");
}

// Send question (updated: show deduped KB links with separate download button)
async function sendQuestion() {
  const q = questionInput.value.trim();
  if (!q) return;
  pushMessage("user", q);
  questionInput.value = "";
  questionInput.disabled = true;
  sendBtn.disabled = true;

  const payload = { question: q, chat_history: chat_history };
  try {
    const resp = await fetch("/api/query", {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify(payload),
    });
    const data = await resp.json();
    if (data.error) {
      pushMessage("assistant", "Error: " + (data.detail || data.error));
    } else {
      const answer = data.answer || "I don't know — not in the documents.";

      // Build sources display WITHOUT chunk information and dedupe names,
      // show KB page link (if available) and a separate download button (⬇️) which calls /download_kb
      let metaHtml = "";
      if (Array.isArray(data.sources) && data.sources.length > 0) {
        const seen = new Set();
        const names = [];

        data.sources.forEach((s) => {
          // prefer various metadata keys
          const rawName = (s && (s.source || s.title || s.file || s.filename)) || "";
          if (!seen.has(rawName)) {
            seen.add(rawName);
            names.push(rawName);
          }
        });

        if (names.length > 0) {
          metaHtml += "<div class='sources'><strong>Sources:</strong><br/>";
          names.forEach((nm) => {
            const padded = extractKbDigits(nm);

            if (padded) {
              // build external KB page link (open KB page)
              const externalUrl = `https://resource.digital.thermofisher.com/kb/article.aspx?n=${encodeURIComponent(padded)}`;
              const downloadUrl = `/download_kb?kb=${encodeURIComponent(padded)}`;
              const safeText = nm ? nm : "KB Article";

              // show KB name as a link, and a separate download icon/link next to it
              metaHtml += `<div class='source-item'>
                  - <a class="kb-link" href="${externalUrl}" target="_blank" rel="noopener noreferrer">${escapeHtml(safeText)}</a>
                  &nbsp;
                  <a class="kb-download" href="${downloadUrl}" target="_blank" rel="noopener noreferrer" title="Download PDF" aria-label="Download PDF">⬇️</a>
                </div>`;
            } else {
              // Not a KB file or no KB id found — just show filename
              metaHtml += `<div class='source-item'>- <em>${escapeHtml(nm || "unknown")}</em></div>`;
            }
          });
          metaHtml += "</div>";
        }
      } else {
        // optionally omit entirely if you don't want "no sources" shown
        metaHtml += "<div class='small'>No sources returned.</div>";
      }

      // Render assistant message with KB page links + separate download buttons
      pushMessage("assistant", answer, metaHtml);

      // Maintain chat history
      chat_history.push([q, answer]);
    }
  } catch (err) {
    pushMessage("assistant", "Error calling backend: " + String(err));
  } finally {
    questionInput.disabled = false;
    sendBtn.disabled = false;
    questionInput.focus();
  }
}

// small helper to escape HTML in the filename text
function escapeHtml(unsafe) {
  if (typeof unsafe !== "string") return "";
  return unsafe
    .replace(/&/g, "&amp;")
    .replace(/</g, "&lt;")
    .replace(/>/g, "&gt;")
    .replace(/"/g, "&quot;")
    .replace(/'/g, "&#039;");
}

sendBtn.addEventListener("click", sendQuestion);
questionInput.addEventListener("keydown", (e) => {
  if (e.key === "Enter") sendQuestion();
});

// Reindex button
reindexBtn.addEventListener("click", async () => {
  reindexBtn.disabled = true;
  reindexBtn.innerText = "Reindexing...";
  try {
    const resp = await fetch("/api/reindex", { method: "POST" });
    const data = await resp.json();
    if (data.status === "ok")
      pushMessage("assistant", "Reindex completed successfully.");
    else pushMessage("assistant", "Reindex response: " + JSON.stringify(data));
  } catch (err) {
    pushMessage("assistant", "Reindex failed: " + String(err));
  } finally {
    reindexBtn.disabled = false;
    reindexBtn.innerText = "Reindex";
  }
});

// close pop if click outside
document.addEventListener("click", (ev) => {
  if (!uploadPop.contains(ev.target) && !attachBtn.contains(ev.target)) {
    uploadPop.classList.remove("show");
  }
});

// small accessibility: keyboard attach toggling
attachBtn.addEventListener("keydown", (e) => {
  if (e.key === "Enter" || e.key === " ") {
    e.preventDefault();
    attachBtn.click();
  }
});
