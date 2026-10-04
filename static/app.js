/**
 * DatasheetQA — Frontend Application
 */
(function () {
    "use strict";

    // ── DOM ───────────────────────────────────────────
    const dropZone = document.getElementById("drop-zone");
    const fileInput = document.getElementById("file-input");
    const uploadStatus = document.getElementById("upload-status");
    const documentList = document.getElementById("document-list");
    const headerStatus = document.getElementById("header-status");
    const chatMessages = document.getElementById("chat-messages");
    const questionInput = document.getElementById("question-input");
    const btnSend = document.getElementById("btn-send");
    const btnClearAll = document.getElementById("btn-clear-all");
    const toastContainer = document.getElementById("toast-container");

    let isAsking = false;

    // ── Init ──────────────────────────────────────────
    function init() {
        setupDropZone();
        setupInput();
        setupClearAll();
        loadDocuments();
    }

    // ── Drop Zone ─────────────────────────────────────
    function setupDropZone() {
        dropZone.addEventListener("click", () => fileInput.click());
        dropZone.addEventListener("dragover", (e) => {
            e.preventDefault();
            dropZone.classList.add("drag-over");
        });
        dropZone.addEventListener("dragleave", () => dropZone.classList.remove("drag-over"));
        dropZone.addEventListener("drop", (e) => {
            e.preventDefault();
            dropZone.classList.remove("drag-over");
            const files = [...e.dataTransfer.files].filter(f => f.type === "application/pdf");
            if (files.length) uploadFiles(files);
            else showToast("Only PDF files are supported.", "error");
        });
        fileInput.addEventListener("change", () => {
            if (fileInput.files.length) {
                uploadFiles([...fileInput.files]);
                fileInput.value = "";
            }
        });
    }

    // ── Upload ────────────────────────────────────────
    async function uploadFiles(files) {
        uploadStatus.classList.remove("hidden", "error", "success");
        uploadStatus.textContent = `Uploading ${files.length} file(s)…`;

        const formData = new FormData();
        files.forEach(f => formData.append("files", f));

        try {
            const res = await fetch("/api/upload", { method: "POST", body: formData });
            const data = await res.json();

            if (!res.ok) {
                uploadStatus.classList.add("error");
                uploadStatus.textContent = `Error: ${data.error || "Upload failed."}`;
                showToast(data.error || "Upload failed.", "error");
                return;
            }

            const ok = data.results.filter(r => r.status === "success").length;
            uploadStatus.classList.add("success");
            uploadStatus.textContent = `✓ ${ok} file(s) processed · ${data.total_chunks} chunks indexed`;
            showToast(`Uploaded ${ok} file(s). Total: ${data.total_chunks} chunks.`, "success");
            loadDocuments();

            setTimeout(() => uploadStatus.classList.add("hidden"), 5000);
        } catch (err) {
            uploadStatus.classList.add("error");
            uploadStatus.textContent = "Upload failed. Check your connection.";
            showToast("Upload failed. Check your connection.", "error");
        }
    }

    // ── Load Documents ────────────────────────────────
    async function loadDocuments() {
        try {
            const res = await fetch("/api/documents");
            const data = await res.json();
            const docs = data.documents || [];
            const statusDot = headerStatus.querySelector(".status-dot");

            if (docs.length === 0) {
                documentList.innerHTML = `<p class="empty-state">No documents uploaded yet</p>`;
                statusDot.classList.remove("active");
                headerStatus.querySelector(".status-dot").nextSibling.textContent = " No documents loaded";
            } else {
                statusDot.classList.add("active");
                headerStatus.querySelector(".status-dot").nextSibling.textContent = ` ${docs.length} document${docs.length > 1 ? "s" : ""} · ${data.total_chunks} chunks`;
                documentList.innerHTML = docs.map(doc => `
                    <div class="doc-item">
                        <div class="doc-icon">
                            <svg viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2">
                                <path d="M14 2H6a2 2 0 0 0-2 2v16a2 2 0 0 0 2 2h12a2 2 0 0 0 2-2V8z"/>
                                <polyline points="14 2 14 8 20 8"/>
                            </svg>
                        </div>
                        <span class="doc-name" title="${esc(doc)}">${esc(doc)}</span>
                    </div>
                `).join("");
            }
        } catch (err) {
            console.error("Failed to load documents:", err);
        }
    }

    // ── Input ─────────────────────────────────────────
    function setupInput() {
        questionInput.addEventListener("input", () => {
            btnSend.disabled = questionInput.value.trim().length === 0 || isAsking;
        });
        questionInput.addEventListener("keydown", (e) => {
            if (e.key === "Enter" && !btnSend.disabled) askQuestion();
        });
        btnSend.addEventListener("click", () => {
            if (!btnSend.disabled) askQuestion();
        });
        document.querySelectorAll(".hint-chip").forEach(chip => {

            chip.addEventListener("click", () => {
                questionInput.value = chip.dataset.question;
                btnSend.disabled = false;
                questionInput.focus();
            });
        });
    }

    // ── Ask Question ──────────────────────────────────
    async function askQuestion() {
        const question = questionInput.value.trim();
        if (!question || isAsking) return;

        isAsking = true;
        btnSend.disabled = true;

        // Hide welcome
        const welcome = document.getElementById("welcome-screen");
        if (welcome) welcome.remove();

        addMessage(question, "user");
        questionInput.value = "";

        const thinkingEl = addThinking();

        try {
            const res = await fetch("/api/ask", {
                method: "POST",
                headers: { "Content-Type": "application/json" },
                body: JSON.stringify({ question }),
            });
            const data = await res.json();
            thinkingEl.remove();

            if (!res.ok) {
                addMessage(data.error || "Something went wrong.", "assistant", { isError: true });
                return;
            }

            addMessage(data.answer, "assistant", {
                confidence: data.confidence,
                citations: data.citations,
                answerable: data.answerable,
            });
        } catch (err) {
            thinkingEl.remove();
            addMessage("Failed to get a response. Please check your connection.", "assistant", { isError: true });
        } finally {
            isAsking = false;
            btnSend.disabled = questionInput.value.trim().length === 0;
        }
    }

    // ── Add Message ───────────────────────────────────
    function addMessage(text, role, meta = {}) {
        const el = document.createElement("div");
        el.className = `message ${role}`;

        let metaHtml = "";
        if (role === "assistant" && !meta.isError) {
            const conf = meta.confidence || "medium";
            metaHtml = `<div class="msg-meta"><span class="badge ${conf}">${conf} confidence</span>`;
            if (meta.citations && meta.citations.length) {
                meta.citations.forEach(c => {
                    metaHtml += `<span class="cite-tag">
                        <svg viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2">
                            <path d="M14 2H6a2 2 0 0 0-2 2v16a2 2 0 0 0 2 2h12a2 2 0 0 0 2-2V8z"/>
                            <polyline points="14 2 14 8 20 8"/>
                        </svg>
                        ${esc(c.document)} p.${c.page}
                    </span>`;
                });
            }
            metaHtml += `</div>`;
        }

        el.innerHTML = `
            <div class="msg-avatar">${role === "user" ? "You" : "AI"}</div>
            <div class="msg-body">
                <div class="msg-bubble">${formatText(text)}</div>
                ${metaHtml}
            </div>
        `;
        chatMessages.appendChild(el);
        chatMessages.scrollTop = chatMessages.scrollHeight;
    }

    // ── Thinking ──────────────────────────────────────
    function addThinking() {
        const el = document.createElement("div");
        el.className = "message assistant";
        el.innerHTML = `
            <div class="msg-avatar">AI</div>
            <div class="msg-body">
                <div class="msg-bubble"><div class="thinking"><div class="dot"></div><div class="dot"></div><div class="dot"></div></div></div>
            </div>
        `;
        chatMessages.appendChild(el);
        chatMessages.scrollTop = chatMessages.scrollHeight;
        return el;
    }

    // ── Format ────────────────────────────────────────
    function formatText(text) {
        if (!text) return "";
        let s = esc(text);
        s = s.replace(/\*\*(.*?)\*\*/g, "<strong>$1</strong>");
        s = s.replace(/^[-•]\s+(.+)$/gm, "<li>$1</li>");
        s = s.replace(/(<li>.*?<\/li>)/gs, "<ul>$1</ul>");
        s = s.replace(/\n/g, "<br>");
        return s;
    }

    // ── Clear All ─────────────────────────────────────
    function setupClearAll() {
        btnClearAll.addEventListener("click", async () => {
            if (!confirm("Clear all documents and reset the index?")) return;
            try {
                const res = await fetch("/api/documents", { method: "DELETE" });
                const data = await res.json();
                showToast(data.message || "All documents cleared.", "success");
                loadDocuments();
                // Restore welcome screen
                location.reload();
            } catch (err) {
                showToast("Failed to clear documents.", "error");
            }
        });
    }

    // ── Toast ─────────────────────────────────────────
    function showToast(msg, type = "info") {
        const t = document.createElement("div");
        t.className = `toast ${type}`;
        t.textContent = msg;
        toastContainer.appendChild(t);
        setTimeout(() => {
            t.classList.add("removing");
            t.addEventListener("animationend", () => t.remove());
        }, 5000);
    }

    // ── Escape HTML ───────────────────────────────────
    function esc(s) {
        const d = document.createElement("div");
        d.textContent = s;
        return d.innerHTML;
    }

    // ── Boot ──────────────────────────────────────────
    document.addEventListener("DOMContentLoaded", init);
})();
