/**
 * Recall Web UI Client
 * Handles drag-and-drop document upload, real-time chat, and interactive citation inspection.
 */

document.addEventListener("DOMContentLoaded", () => {
    // Elements
    const dropZone = document.getElementById("drop-zone");
    const fileInput = document.getElementById("file-input");
    const browseBtn = document.getElementById("browse-btn");
    const uploadStatus = document.getElementById("upload-status");
    const statChunks = document.getElementById("stat-chunks");
    const messageList = document.getElementById("message-list");
    const queryForm = document.getElementById("query-form");
    const queryInput = document.getElementById("query-input");
    const clearChatBtn = document.getElementById("clear-chat-btn");

    // Modal elements
    const citationModal = document.getElementById("citation-modal");
    const closeModalBtn = document.getElementById("close-modal-btn");
    const modalSource = document.getElementById("modal-source");
    const modalPage = document.getElementById("modal-page");
    const modalPageRow = document.getElementById("modal-page-row");
    const modalChunkId = document.getElementById("modal-chunk-id");
    const modalSnippet = document.getElementById("modal-snippet");

    // State
    let activeCitations = {};

    // Initial stats fetch
    fetchStats();

    // File Upload Setup
    browseBtn.addEventListener("click", () => fileInput.click());
    dropZone.addEventListener("click", (e) => {
        if (e.target !== browseBtn) fileInput.click();
    });

    dropZone.addEventListener("dragover", (e) => {
        e.preventDefault();
        dropZone.classList.add("dragover");
    });

    dropZone.addEventListener("dragleave", () => {
        dropZone.classList.remove("dragover");
    });

    dropZone.addEventListener("drop", (e) => {
        e.preventDefault();
        dropZone.classList.remove("dragover");
        if (e.dataTransfer.files && e.dataTransfer.files.length > 0) {
            handleFileUpload(e.dataTransfer.files);
        }
    });

    fileInput.addEventListener("change", () => {
        if (fileInput.files && fileInput.files.length > 0) {
            handleFileUpload(fileInput.files);
        }
    });

    async function handleFileUpload(files) {
        showStatus(`Uploading and indexing ${files.length} file(s)...`, "loading");

        for (const file of files) {
            const formData = new FormData();
            formData.append("file", file);

            try {
                const response = await fetch("/v1/ingest", {
                    method: "POST",
                    body: formData,
                });

                if (!response.ok) {
                    const err = await response.json();
                    throw new Error(err.detail || "Upload failed");
                }

                const data = await response.json();
                showStatus(`Indexed ${data.chunks_indexed} chunks from ${file.name}!`, "success");
                fetchStats();
            } catch (err) {
                showStatus(`Error uploading ${file.name}: ${err.message}`, "error");
                break;
            }
        }
    }

    function showStatus(msg, type) {
        uploadStatus.textContent = msg;
        uploadStatus.className = `status-indicator ${type}`;
        uploadStatus.classList.remove("hidden");
        if (type === "success") {
            setTimeout(() => {
                uploadStatus.classList.add("hidden");
            }, 4000);
        }
    }

    async function fetchStats() {
        try {
            const res = await fetch("/v1/stats");
            if (res.ok) {
                const data = await res.json();
                if (statChunks && data.total_chunks !== undefined) {
                    statChunks.textContent = `${data.total_chunks} chunks`;
                }
            }
        } catch (e) {
            console.debug("Could not fetch stats:", e);
        }
    }

    // Chat Submission
    queryForm.addEventListener("submit", async (e) => {
        e.preventDefault();
        const query = queryInput.value.trim();
        if (!query) return;

        // Clear welcome card if present
        const welcome = document.querySelector(".welcome-card");
        if (welcome) welcome.remove();

        // Append User Message
        appendMessage("user", query);
        queryInput.value = "";

        // Append Temporary Assistant Loading Message
        const loadingMsgId = appendMessage("assistant", "Searching and synthesizing verified answer...", true);

        try {
            const res = await fetch("/v1/chat", {
                method: "POST",
                headers: { "Content-Type": "application/json" },
                body: JSON.stringify({ query: query, top_k: 5 }),
            });

            if (!res.ok) {
                throw new Error(`HTTP ${res.status}`);
            }

            const data = await res.json();
            updateAssistantMessage(loadingMsgId, data.answer, data.citations);
        } catch (err) {
            updateAssistantMessage(loadingMsgId, `Error communicating with Recall engine: ${err.message}`, []);
        }
    });

    clearChatBtn.addEventListener("click", () => {
        messageList.innerHTML = `
            <div class="welcome-card">
                <div class="welcome-icon">
                    <svg width="32" height="32" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="1.8">
                        <circle cx="12" cy="12" r="10"></circle>
                        <path d="M12 16v-4"></path>
                        <path d="M12 8h.01"></path>
                    </svg>
                </div>
                <h2>Ask Recall Anything Across Your Enterprise Documents</h2>
                <p>Every answer is synthesized using concurrent hybrid search (Dense HNSW + BM25+), deep cross-encoder reranking, and citation attribution with exact document citations.</p>
            </div>
        `;
    });

    function appendMessage(role, text, isLoading = false) {
        const id = `msg-${Date.now()}-${Math.random().toString(36).substr(2, 5)}`;
        const wrapper = document.createElement("div");
        wrapper.id = id;
        wrapper.className = `chat-message ${role}`;

        const avatar = document.createElement("div");
        avatar.className = "chat-avatar";
        avatar.textContent = role === "assistant" ? "R" : "U";

        const contentWrapper = document.createElement("div");
        contentWrapper.className = "message-content-wrapper";

        const content = document.createElement("div");
        content.className = "message-content";
        if (isLoading) content.classList.add("loading-text");
        content.textContent = text;

        contentWrapper.appendChild(content);
        if (role === "assistant") {
            wrapper.appendChild(avatar);
            wrapper.appendChild(contentWrapper);
        } else {
            wrapper.appendChild(contentWrapper);
            wrapper.appendChild(avatar);
        }

        messageList.appendChild(wrapper);
        messageList.scrollTop = messageList.scrollHeight;
        return id;
    }

    function updateAssistantMessage(msgId, rawAnswer, citations = []) {
        const msgEl = document.getElementById(msgId);
        if (!msgEl) return;

        const contentEl = msgEl.querySelector(".message-content");
        contentEl.classList.remove("loading-text");
        contentEl.innerHTML = "";

        // Store citations keyed by doc_index
        const citationsByDoc = {};
        citations.forEach(c => {
            citationsByDoc[c.doc_index] = c;
            activeCitations[`${msgId}-${c.doc_index}`] = c;
        });

        // Parse [Doc X] and [Doc X, p. Y] into interactive pills
        const regex = /\[Doc\s*(\d+)(?:,\s*(?:p\.?|page)\s*(\d+))?\]/gi;
        const formatted = rawAnswer.replace(regex, (match, docIdx, page) => {
            const pageLabel = page ? `, p. ${page}` : "";
            return `<button class="citation-pill" data-msg-id="${msgId}" data-doc-idx="${docIdx}">Doc ${docIdx}${pageLabel}</button>`;
        });

        contentEl.innerHTML = formatted;

        // Attach click listeners to citations
        contentEl.querySelectorAll(".citation-pill").forEach(pill => {
            pill.addEventListener("click", () => {
                const docIdx = pill.getAttribute("data-doc-idx");
                const mId = pill.getAttribute("data-msg-id");
                const citation = activeCitations[`${mId}-${docIdx}`];
                if (citation) openCitationModal(citation);
            });
        });

        // Add Citations Footer if verified citations exist
        if (citations.length > 0) {
            const contentWrapper = msgEl.querySelector(".message-content-wrapper");
            const footer = document.createElement("div");
            footer.className = "citations-footer";

            const label = document.createElement("span");
            label.className = "citations-footer-label";
            label.textContent = "Verified Sources:";
            footer.appendChild(label);

            citations.forEach(c => {
                const btn = document.createElement("button");
                btn.className = "citation-pill";
                const shortSrc = c.source_uri ? c.source_uri.split("/").pop() : `Doc ${c.doc_index}`;
                btn.textContent = `[${c.doc_index}] ${shortSrc}`;
                btn.addEventListener("click", () => openCitationModal(c));
                footer.appendChild(btn);
            });

            contentWrapper.appendChild(footer);
        }

        messageList.scrollTop = messageList.scrollHeight;
    }

    function openCitationModal(c) {
        modalSource.textContent = c.source_uri || "Direct Input Document";
        if (c.page_number !== null && c.page_number !== undefined) {
            modalPage.textContent = `Page ${c.page_number}`;
            modalPageRow.classList.remove("hidden");
        } else {
            modalPageRow.classList.add("hidden");
        }
        modalChunkId.textContent = c.chunk_id;
        modalSnippet.textContent = c.snippet;
        citationModal.classList.remove("hidden");
    }

    closeModalBtn.addEventListener("click", () => citationModal.classList.add("hidden"));
    citationModal.addEventListener("click", (e) => {
        if (e.target === citationModal) citationModal.classList.add("hidden");
    });
});
