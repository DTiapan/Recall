/**
 * Recall Web UI v8
 */

document.addEventListener("DOMContentLoaded", () => {
    const dropZone = document.getElementById("drop-zone");
    const fileInput = document.getElementById("file-input");
    const browseBtn = document.getElementById("browse-btn");
    const uploadStatus = document.getElementById("upload-status");
    const uploadProgress = document.getElementById("upload-progress");
    const docList = document.getElementById("doc-list");
    const docSearch = document.getElementById("doc-search");
    const indexChunkCount = document.getElementById("index-chunk-count");
    const indexDocCount = document.getElementById("index-doc-count");
    const modelSelect = document.getElementById("model-select");
    const messageList = document.getElementById("message-list");
    const queryForm = document.getElementById("query-form");
    const queryInput = document.getElementById("query-input");
    const sendBtn = document.getElementById("send-btn");
    const stopBtn = document.getElementById("stop-btn");
    const clearChatBtn = document.getElementById("clear-chat-btn");
    const exportChatBtn = document.getElementById("export-chat-btn");
    const citationModal = document.getElementById("citation-modal");
    const confirmDialog = document.getElementById("confirm-dialog");
    const confirmTitle = document.getElementById("confirm-title");
    const confirmBody = document.getElementById("confirm-body");
    const confirmCancelBtn = document.getElementById("confirm-cancel-btn");
    const confirmOkBtn = document.getElementById("confirm-ok-btn");
    const closeModalBtn = document.getElementById("close-modal-btn");
    const modalSource = document.getElementById("modal-source");
    const modalPage = document.getElementById("modal-page");
    const modalPageLabel = document.getElementById("modal-page-label");
    const modalChunkId = document.getElementById("modal-chunk-id");
    const modalSnippet = document.getElementById("modal-snippet");
    const topKSlider = document.getElementById("top-k-slider");
    const topKValue = document.getElementById("top-k-value");
    const fastModeToggle = document.getElementById("fast-mode-toggle");
    const apiKeyOverlay = document.getElementById("api-key-overlay");
    const apiKeyInput = document.getElementById("api-key-input");
    const apiKeySaveBtn = document.getElementById("api-key-save-btn");
    const sidebarToggle = document.getElementById("sidebar-toggle");
    const sidebar = document.getElementById("sidebar");

    let activeCitations = {};
    let isStreaming = false;
    let streamAbort = null;
    let indexedDocs = [];
    let uiConfig = {};
    let authRequired = false;

    const STORAGE = {
        model: "recall.model",
        topK: "recall.top_k",
        fastMode: "recall.fast_mode",
        apiKey: "recall.api_key",
    };

    const DEFAULT_STARTERS = [
        "What topics are in my documents?",
        "Summarize the key policies",
        "What are the main technical requirements?",
    ];

    browseBtn.addEventListener("click", (e) => { e.stopPropagation(); fileInput.click(); });
    dropZone.addEventListener("click", () => fileInput.click());
    dropZone.addEventListener("dragover", (e) => { e.preventDefault(); dropZone.classList.add("dragover"); });
    dropZone.addEventListener("dragleave", () => dropZone.classList.remove("dragover"));
    dropZone.addEventListener("drop", (e) => {
        e.preventDefault();
        dropZone.classList.remove("dragover");
        if (e.dataTransfer.files?.length) handleFileUpload(e.dataTransfer.files);
    });
    fileInput.addEventListener("change", () => {
        if (fileInput.files?.length) handleFileUpload(fileInput.files);
    });

    stopBtn.addEventListener("click", () => streamAbort?.abort());
    clearChatBtn.addEventListener("click", () => {
        messageList.innerHTML = "";
        renderWelcome();
    });
    exportChatBtn.addEventListener("click", exportChat);
    closeModalBtn.addEventListener("click", () => citationModal.close());
    citationModal.addEventListener("click", (e) => {
        if (e.target === citationModal) citationModal.close();
    });
    confirmCancelBtn.addEventListener("click", () => confirmDialog.close());

    queryForm.addEventListener("submit", (e) => {
        e.preventDefault();
        const q = queryInput.value.trim();
        if (!q || isStreaming) return;
        submitQuery(q);
    });

    modelSelect.addEventListener("change", () => {
        localStorage.setItem(STORAGE.model, modelSelect.value);
    });

    topKSlider.addEventListener("input", () => {
        topKValue.textContent = topKSlider.value;
        localStorage.setItem(STORAGE.topK, topKSlider.value);
    });

    fastModeToggle.addEventListener("change", () => {
        localStorage.setItem(STORAGE.fastMode, fastModeToggle.checked ? "1" : "0");
    });

    docSearch.addEventListener("input", () => renderDocList(filterDocs(indexedDocs)));

    apiKeySaveBtn.addEventListener("click", () => {
        const key = apiKeyInput.value.trim();
        if (!key) return;
        sessionStorage.setItem(STORAGE.apiKey, key);
        apiKeyOverlay.classList.add("hidden");
        fetchConfig();
        fetchDocuments();
    });

    sidebarToggle.addEventListener("click", () => {
        sidebar.classList.toggle("sidebar-open");
    });

    window.addEventListener("focus", () => fetchConfig());
    setInterval(() => fetchConfig(), 60000);

    loadSettings();
    fetchConfig().then(() => fetchDocuments());

    function loadSettings() {
        const savedTopK = localStorage.getItem(STORAGE.topK);
        if (savedTopK) {
            topKSlider.value = savedTopK;
            topKValue.textContent = savedTopK;
        }
        fastModeToggle.checked = localStorage.getItem(STORAGE.fastMode) === "1";
    }

    function apiHeaders(extra = {}) {
        const headers = { ...extra };
        const key = sessionStorage.getItem(STORAGE.apiKey);
        if (key) headers.Authorization = `Bearer ${key}`;
        return headers;
    }

    async function fetchConfig() {
        try {
            const res = await fetch("/v1/config");
            if (!res.ok) return;
            uiConfig = await res.json();
            authRequired = Boolean(uiConfig.auth_required);
            if (authRequired && !sessionStorage.getItem(STORAGE.apiKey)) {
                apiKeyOverlay.classList.remove("hidden");
            }
            populateModelSelect(uiConfig.models || [], uiConfig.default_model);
            if (uiConfig.top_k_default) {
                topKSlider.max = Math.max(10, uiConfig.top_k_default);
            }
        } catch (_) { /* ignore */ }
    }

    function populateModelSelect(models, defaultModel) {
        if (!modelSelect) return;

        let list = models;
        if (!list.length && uiConfig.model) {
            // Back-compat when server has not been restarted (old /v1/config shape)
            const slug = String(uiConfig.model).replace(/^openrouter\//, "");
            list = [{
                id: slug,
                label: uiConfig.model_display || slug,
            }];
            defaultModel = defaultModel || slug;
        }

        if (!list.length) {
            modelSelect.innerHTML = "<option value=\"\">No models — restart server</option>";
            return;
        }

        const saved = localStorage.getItem(STORAGE.model);
        modelSelect.innerHTML = "";
        list.forEach((m) => {
            const opt = document.createElement("option");
            opt.value = m.id;
            opt.textContent = m.label;
            opt.title = m.id;
            modelSelect.appendChild(opt);
        });
        const pick = saved && list.some((m) => m.id === saved) ? saved : defaultModel;
        if (pick && list.some((m) => m.id === pick)) modelSelect.value = pick;
    }

    async function fetchDocuments() {
        try {
            const res = await fetch("/v1/documents");
            if (!res.ok) return;
            const data = await res.json();
            indexedDocs = (data.documents || []).filter((d) => !isTempSource(d.source_uri));
            if (indexChunkCount) indexChunkCount.textContent = `${data.total_chunks || 0} chunks`;
            if (indexDocCount) indexDocCount.textContent = `${indexedDocs.length} docs`;
            renderDocList(filterDocs(indexedDocs));
            renderStarters(document.getElementById("starter-questions"));
        } catch (_) { /* ignore */ }
    }

    function filterDocs(docs) {
        const q = (docSearch?.value || "").trim().toLowerCase();
        if (!q) return docs;
        return docs.filter((d) => basename(d.source_uri).toLowerCase().includes(q));
    }

    function isTempSource(uri) {
        const text = uri || "";
        const lower = text.toLowerCase();
        if (lower.includes("/tmp/") || lower.includes("/var/folders/")) return true;
        const name = text.split("/").pop() || "";
        return /^tmp[a-z0-9]+(\.[a-z0-9]+)?$/i.test(name);
    }

    function renderDocList(docs) {
        docList.innerHTML = "";
        if (!docs.length) {
            const li = document.createElement("li");
            li.className = "doc-list-empty";
            li.textContent = docSearch?.value ? "No matching documents" : "Upload files to get started";
            docList.appendChild(li);
            return;
        }
        docs.forEach((doc) => {
            const li = document.createElement("li");
            li.className = "doc-list-item";

            const main = document.createElement("button");
            main.type = "button";
            main.className = "doc-list-main";
            main.innerHTML = `
                <span class="doc-name">${escapeHtml(basename(doc.source_uri))}</span>
                <span class="doc-meta">${doc.chunk_count} chunk${doc.chunk_count === 1 ? "" : "s"}${doc.file_type ? ` · ${doc.file_type}` : ""}</span>
            `;
            main.title = doc.source_uri;
            main.addEventListener("click", () => {
                queryInput.value = `What does ${basename(doc.source_uri)} cover?`;
                queryInput.focus();
            });

            const del = document.createElement("button");
            del.type = "button";
            del.className = "doc-delete-btn";
            del.setAttribute("aria-label", `Delete ${basename(doc.source_uri)}`);
            del.textContent = "✕";
            del.addEventListener("click", (e) => {
                e.stopPropagation();
                confirmDelete(doc.source_uri);
            });

            li.appendChild(main);
            li.appendChild(del);
            docList.appendChild(li);
        });
    }

    function confirmDelete(sourceUri) {
        confirmTitle.textContent = "Delete document";
        confirmBody.textContent = `Remove all chunks for "${basename(sourceUri)}" from the knowledge base?`;
        confirmDialog.showModal();
        confirmOkBtn.onclick = async () => {
            confirmDialog.close();
            try {
                const res = await fetch(
                    `/v1/documents?source_uri=${encodeURIComponent(sourceUri)}`,
                    { method: "DELETE", headers: apiHeaders() },
                );
                if (!res.ok) throw new Error((await res.json()).detail || "Delete failed");
                const data = await res.json();
                showToast(`Deleted ${data.chunks_deleted} chunk(s) from ${basename(sourceUri)}`, "success");
                await fetchDocuments();
            } catch (err) {
                showToast(err.message, "error");
            }
        };
    }

    function renderStarters(container) {
        if (!container) return;
        container.innerHTML = "";
        buildStarters().forEach((q) => {
            const btn = document.createElement("button");
            btn.type = "button";
            btn.className = "chip-btn";
            btn.textContent = q;
            btn.addEventListener("click", () => submitQuery(q));
            container.appendChild(btn);
        });
    }

    function buildStarters() {
        const ranked = [...indexedDocs].sort((a, b) => (b.chunk_count || 0) - (a.chunk_count || 0));
        const fromDocs = ranked.slice(0, 3).map((d) => `What does ${basename(d.source_uri)} cover?`);
        return [...fromDocs, ...DEFAULT_STARTERS].slice(0, 4);
    }

    function renderWelcome() {
        const hero = document.createElement("section");
        hero.className = "hero";
        hero.id = "welcome-card";
        hero.innerHTML = `
            <p class="hero-eyebrow">Enterprise RAG</p>
            <h2 class="hero-title">Ask your knowledge base</h2>
            <p class="hero-body">Hybrid search, reranking, and cited answers — grounded only in what you upload.</p>
            <div id="starter-questions" class="chips"></div>
        `;
        messageList.appendChild(hero);
        renderStarters(hero.querySelector("#starter-questions"));
    }

    async function handleFileUpload(files) {
        const fileArr = Array.from(files);
        uploadProgress.classList.remove("hidden");
        uploadProgress.innerHTML = "";
        fileArr.forEach((file) => {
            const li = document.createElement("li");
            li.className = "upload-progress-item";
            li.dataset.filename = file.name;
            li.textContent = `${file.name} — indexing…`;
            uploadProgress.appendChild(li);
        });

        for (let i = 0; i < fileArr.length; i++) {
            const file = fileArr[i];
            const row = uploadProgress.querySelector(`[data-filename="${CSS.escape(file.name)}"]`);
            const fd = new FormData();
            fd.append("file", file);
            try {
                const res = await fetch("/v1/ingest", { method: "POST", body: fd, headers: apiHeaders() });
                if (!res.ok) throw new Error((await res.json()).detail || "Upload failed");
                const data = await res.json();
                if (row) {
                    row.textContent = `${file.name} — ${data.chunks_indexed} chunk(s) indexed`;
                    row.classList.add("done");
                }
            } catch (err) {
                if (row) {
                    row.textContent = `${file.name} — failed: ${err.message}`;
                    row.classList.add("error");
                }
                showToast(err.message, "error");
                break;
            }
        }
        setTimeout(() => uploadProgress.classList.add("hidden"), 4000);
        await fetchDocuments();
        fileInput.value = "";
    }

    function showToast(msg, type) {
        uploadStatus.textContent = msg;
        uploadStatus.className = `toast ${type}`;
        uploadStatus.classList.remove("hidden");
        if (type === "success") setTimeout(() => uploadStatus.classList.add("hidden"), 3500);
    }

    function basename(uri) {
        return (uri || "Unknown").split("/").pop() || uri;
    }

    function escapeHtml(text) {
        return String(text)
            .replace(/&/g, "&amp;")
            .replace(/</g, "&lt;")
            .replace(/>/g, "&gt;")
            .replace(/"/g, "&quot;");
    }

    function renderAnswerHtml(answer, msgId) {
        let html = escapeHtml(answer);
        html = html.replace(/\*\*([^*]+)\*\*/g, "<strong>$1</strong>");
        html = html.replace(/(?<!\*)\*([^*]+)\*(?!\*)/g, "<em>$1</em>");
        html = blocksToParagraphs(html);
        html = html.replace(
            /\[Doc\s*(\d+)(?:,\s*(?:p\.?|page)\s*(\d+))?\]/gi,
            (_, idx, page) => {
                const pl = page ? `, p. ${page}` : "";
                return `<button type="button" class="cite-inline" data-msg="${msgId}" data-idx="${idx}">Doc ${idx}${pl}</button>`;
            },
        );
        return html;
    }

    function blocksToParagraphs(html) {
        return html
            .split(/\n{2,}/)
            .map((block) => `<p>${block.replace(/\n/g, "<br>")}</p>`)
            .join("");
    }

    function currentModelLabel(meta) {
        if (modelSelect?.selectedOptions?.[0]?.textContent) {
            return modelSelect.selectedOptions[0].textContent;
        }
        if (meta?.model_name) {
            return String(meta.model_name).replace(/^openrouter\//, "").split("/").pop();
        }
        return "";
    }

    function formatLatency(meta) {
        const model = currentModelLabel(meta);
        const modelPart = model ? `${model} · ` : "";
        if (!meta?.timing) {
            return meta?.latency_seconds
                ? `${modelPart}Total ${(meta.latency_seconds * 1000).toFixed(0)}ms`
                : model;
        }
        const t = meta.timing;
        return `${modelPart}Retrieve ${t.retrieve_ms.toFixed(0)}ms · Rerank ${t.rerank_ms.toFixed(0)}ms · LLM ${t.synthesis_ms.toFixed(0)}ms · Total ${t.total_ms.toFixed(0)}ms`;
    }

    function chatPayload(query) {
        const payload = {
            query,
            top_k: parseInt(topKSlider.value, 10) || 5,
            stream: true,
        };
        if (modelSelect?.value) payload.model = modelSelect.value;
        if (fastModeToggle.checked) payload.rerank = false;
        return payload;
    }

    function submitQuery(query) {
        document.getElementById("welcome-card")?.remove();
        appendMessage("user", query);
        queryInput.value = "";
        const id = appendMessage("assistant", "Searching documents…", true);
        startStream(query, id);
    }

    async function startStream(query, msgId) {
        isStreaming = true;
        sendBtn.disabled = true;
        queryInput.disabled = true;
        stopBtn.classList.remove("hidden");
        streamAbort = new AbortController();
        try {
            await streamChat(query, msgId, streamAbort.signal);
        } catch (err) {
            if (err.name !== "AbortError") finalizeMessage(msgId, `Error: ${err.message}`, [], {});
        } finally {
            isStreaming = false;
            sendBtn.disabled = false;
            queryInput.disabled = false;
            stopBtn.classList.add("hidden");
            streamAbort = null;
            queryInput.focus();
        }
    }

    async function streamChat(query, msgId, signal) {
        const res = await fetch("/v1/chat", {
            method: "POST",
            headers: apiHeaders({ "Content-Type": "application/json", Accept: "text/event-stream" }),
            body: JSON.stringify(chatPayload(query)),
            signal,
        });
        if (!res.ok) {
            const err = await res.json().catch(() => ({}));
            throw new Error(err.detail || `HTTP ${res.status}`);
        }

        const bubble = document.getElementById(msgId)?.querySelector(".msg-bubble");
        if (!bubble) return;

        const reader = res.body.getReader();
        const decoder = new TextDecoder();
        let buffer = "";
        let text = "";

        while (true) {
            const { done, value } = await reader.read();
            if (done) break;
            buffer += decoder.decode(value, { stream: true });
            const parts = buffer.split("\n\n");
            buffer = parts.pop() || "";

            for (const part of parts) {
                const line = part.trim();
                if (!line.startsWith("data: ")) continue;
                const ev = JSON.parse(line.slice(6));

                if (ev.event === "status") {
                    if (ev.phase === "retrieving") {
                        bubble.className = "msg-bubble loading";
                        bubble.textContent = "Searching documents…";
                    } else if (ev.phase === "generating") {
                        bubble.className = "msg-bubble streaming";
                        bubble.textContent = "";
                    }
                } else if (ev.event === "token") {
                    bubble.className = "msg-bubble streaming";
                    text += ev.delta;
                    bubble.textContent = text;
                } else if (ev.event === "done") {
                    finalizeMessage(msgId, ev.answer || text, ev.citations || [], ev);
                }
                messageList.parentElement.scrollTop = messageList.parentElement.scrollHeight;
            }
        }
    }

    function appendMessage(role, text, loading = false) {
        const id = `msg-${Date.now()}-${Math.random().toString(36).slice(2, 6)}`;
        const el = document.createElement("article");
        el.id = id;
        el.className = `msg ${role}`;
        el.dataset.role = role;
        el.dataset.rawText = text;

        const body = document.createElement("div");
        body.className = "msg-body";

        const label = document.createElement("p");
        label.className = "msg-label";
        label.textContent = role === "assistant" ? "Recall" : "You";
        body.appendChild(label);

        const bubble = document.createElement("div");
        bubble.className = `msg-bubble${loading ? " loading" : ""}`;
        bubble.textContent = text;
        body.appendChild(bubble);

        if (role === "assistant") {
            const actions = document.createElement("div");
            actions.className = "msg-actions";
            const copy = document.createElement("button");
            copy.type = "button";
            copy.className = "btn btn-ghost";
            copy.textContent = "Copy";
            copy.addEventListener("click", () => {
                navigator.clipboard.writeText(bubble.textContent || "").then(() => {
                    copy.textContent = "Copied";
                    setTimeout(() => { copy.textContent = "Copy"; }, 1200);
                });
            });
            actions.appendChild(copy);
            body.appendChild(actions);
        }

        el.appendChild(body);
        messageList.appendChild(el);
        messageList.parentElement.scrollTop = messageList.parentElement.scrollHeight;
        return id;
    }

    function finalizeMessage(msgId, answer, citations, meta) {
        const root = document.getElementById(msgId);
        if (!root) return;
        root.dataset.rawText = answer;
        const body = root.querySelector(".msg-body");
        body.querySelectorAll(".cite-row, .latency").forEach((n) => n.remove());

        const bubble = root.querySelector(".msg-bubble");
        bubble.className = "msg-bubble";

        citations.forEach((c) => { activeCitations[`${msgId}-${c.doc_index}`] = c; });

        bubble.innerHTML = renderAnswerHtml(answer, msgId);

        bubble.querySelectorAll(".cite-inline").forEach((btn) => {
            btn.addEventListener("click", () => {
                const c = activeCitations[`${btn.dataset.msg}-${btn.dataset.idx}`];
                if (c) openCitation(c);
            });
        });

        const copyBtn = body.querySelector(".msg-actions .btn");
        if (copyBtn) {
            copyBtn.onclick = () => {
                navigator.clipboard.writeText(answer).then(() => {
                    copyBtn.textContent = "Copied";
                    setTimeout(() => { copyBtn.textContent = "Copy"; }, 1200);
                });
            };
        }

        if (citations.length) {
            const row = document.createElement("div");
            row.className = "cite-row";
            const label = document.createElement("span");
            label.className = "cite-label";
            label.textContent = "Sources";
            row.appendChild(label);
            citations.forEach((c) => {
                const b = document.createElement("button");
                b.type = "button";
                b.className = "cite-btn";
                b.textContent = `[${c.doc_index}] ${basename(c.source_uri)}`;
                b.addEventListener("click", () => openCitation(c));
                row.appendChild(b);
            });
            body.insertBefore(row, body.querySelector(".msg-actions"));
        }

        const lat = formatLatency(meta);
        if (lat) {
            const p = document.createElement("p");
            p.className = "latency";
            p.textContent = lat;
            body.insertBefore(p, body.querySelector(".msg-actions"));
        }

        messageList.parentElement.scrollTop = messageList.parentElement.scrollHeight;
    }

    function exportChat() {
        const lines = ["# Recall chat export", ""];
        messageList.querySelectorAll(".msg").forEach((msg) => {
            const role = msg.dataset.role === "assistant" ? "Recall" : "You";
            const text = msg.dataset.rawText || msg.querySelector(".msg-bubble")?.textContent || "";
            lines.push(`## ${role}`, "", text, "");
        });
        const blob = new Blob([lines.join("\n")], { type: "text/markdown" });
        const a = document.createElement("a");
        a.href = URL.createObjectURL(blob);
        a.download = `recall-chat-${Date.now()}.md`;
        a.click();
        URL.revokeObjectURL(a.href);
    }

    function openCitation(c) {
        modalSource.textContent = basename(c.source_uri);
        modalSource.title = c.source_uri || "";
        if (c.page_number != null) {
            modalPage.textContent = String(c.page_number);
            modalPageLabel.style.display = "";
            modalPage.style.display = "";
        } else {
            modalPageLabel.style.display = "none";
            modalPage.style.display = "none";
        }
        modalChunkId.textContent = c.chunk_id;
        modalSnippet.textContent = c.snippet;
        citationModal.showModal();
    }
});
