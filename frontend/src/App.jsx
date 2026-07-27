import { useState, useEffect, useRef, useCallback } from "react";
import axios from "axios";
import "./App.css";

const backendUrl = "http://127.0.0.1:8000";

function App() {
  const [documents, setDocuments] = useState([]);
  const [messages, setMessages] = useState([]);
  const [question, setQuestion] = useState("");
  const [uploading, setUploading] = useState(false);
  const [asking, setAsking] = useState(false);
  const [dragOver, setDragOver] = useState(false);
  const [notice, setNotice] = useState(null); // { type: "error" | "info", text }

  const fileInputRef = useRef(null);
  const scrollRef = useRef(null);

  // Load the existing library (if the backend is already running).
  useEffect(() => {
    axios
      .get(`${backendUrl}/documents`)
      .then((res) => setDocuments(res.data.documents || []))
      .catch(() => {
        /* backend may be offline at first paint — ignore */
      });
  }, []);

  // Keep the conversation scrolled to the newest message.
  useEffect(() => {
    if (scrollRef.current) {
      scrollRef.current.scrollTop = scrollRef.current.scrollHeight;
    }
  }, [messages, asking]);

  const uploadFiles = useCallback(async (fileList) => {
    const pdfs = Array.from(fileList).filter(
      (f) => f.type === "application/pdf" || f.name.toLowerCase().endsWith(".pdf")
    );

    if (pdfs.length === 0) {
      setNotice({ type: "error", text: "Those weren't PDFs. Add one or more .pdf files." });
      return;
    }

    const formData = new FormData();
    pdfs.forEach((f) => formData.append("files", f));

    try {
      setUploading(true);
      setNotice(null);
      const res = await axios.post(`${backendUrl}/upload-pdfs`, formData, {
        headers: { "Content-Type": "multipart/form-data" },
      });
      setDocuments(res.data.documents || []);

      const added = (res.data.processed || []).filter((p) => !p.skipped);
      const pages = added.reduce((n, p) => n + (p.pages || 0), 0);
      setNotice({
        type: "info",
        text: `Added ${added.length} PDF${added.length === 1 ? "" : "s"} · ${pages} pages ready to search.`,
      });
    } catch (err) {
      const detail = err.response?.data?.detail || err.message || "Upload failed.";
      setNotice({ type: "error", text: `Couldn't process that: ${detail}` });
    } finally {
      setUploading(false);
    }
  }, []);

  const onDrop = (e) => {
    e.preventDefault();
    setDragOver(false);
    if (e.dataTransfer.files?.length) uploadFiles(e.dataTransfer.files);
  };

  const removeDocument = async (name) => {
    try {
      const res = await axios.delete(`${backendUrl}/documents/${encodeURIComponent(name)}`);
      setDocuments(res.data.documents || []);
    } catch (err) {
      setNotice({ type: "error", text: `Couldn't remove ${name}.` });
    }
  };

  const ask = async () => {
    const q = question.trim();
    if (!q) return;

    if (documents.length === 0) {
      setNotice({ type: "error", text: "Add at least one PDF before asking." });
      return;
    }

    setNotice(null);
    setMessages((m) => [...m, { role: "user", text: q }]);
    setQuestion("");

    try {
      setAsking(true);
      const res = await axios.get(`${backendUrl}/ask`, { params: { query: q } });
      if (res.data.error) {
        setMessages((m) => [...m, { role: "assistant", text: res.data.error, citations: [] }]);
      } else {
        setMessages((m) => [
          ...m,
          { role: "assistant", text: res.data.answer || "", citations: res.data.citations || [] },
        ]);
      }
    } catch (err) {
      const detail = err.response?.data?.detail || err.message || "Something went wrong.";
      setMessages((m) => [
        ...m,
        { role: "assistant", text: `Couldn't reach the answer engine: ${detail}`, citations: [] },
      ]);
    } finally {
      setAsking(false);
    }
  };

  const onQuestionKeyDown = (e) => {
    if (e.key === "Enter" && !e.shiftKey) {
      e.preventDefault();
      ask();
    }
  };

  const totalPages = documents.reduce((n, d) => n + (d.pages || 0), 0);

  return (
    <div className="app">
      <header className="topbar">
        <div className="brand">
          <span className="brand-mark" aria-hidden="true" />
          <div>
            <h1 className="brand-name">Statictextbook</h1>
            <p className="brand-sub">Ask your textbooks</p>
          </div>
        </div>
        <div className="library-stat">
          {documents.length > 0
            ? `${documents.length} source${documents.length === 1 ? "" : "s"} · ${totalPages} pages`
            : "No sources yet"}
        </div>
      </header>

      <main className="layout">
        <aside className="sources">
          <div className="panel-label">Sources</div>

          <div
            className={`dropzone${dragOver ? " is-over" : ""}${uploading ? " is-busy" : ""}`}
            onDragOver={(e) => {
              e.preventDefault();
              setDragOver(true);
            }}
            onDragLeave={() => setDragOver(false)}
            onDrop={onDrop}
            onClick={() => fileInputRef.current?.click()}
            role="button"
            tabIndex={0}
            onKeyDown={(e) =>
              (e.key === "Enter" || e.key === " ") && fileInputRef.current?.click()
            }
          >
            <input
              ref={fileInputRef}
              type="file"
              accept="application/pdf"
              multiple
              className="file-input"
              onChange={(e) => {
                if (e.target.files?.length) uploadFiles(e.target.files);
                e.target.value = "";
              }}
            />
            {uploading ? (
              <span className="dropzone-text">Processing…</span>
            ) : (
              <>
                <span className="dropzone-title">Drop PDFs here</span>
                <span className="dropzone-hint">or click to browse · multiple allowed</span>
              </>
            )}
          </div>

          <ul className="source-list">
            {documents.map((d) => (
              <li className="source-item" key={d.name}>
                <div className="source-info">
                  <span className="source-name" title={d.name}>
                    {d.name}
                  </span>
                  <span className="source-meta">{d.pages} pages</span>
                </div>
                <button
                  className="source-remove"
                  aria-label={`Remove ${d.name}`}
                  onClick={() => removeDocument(d.name)}
                >
                  ×
                </button>
              </li>
            ))}
            {documents.length === 0 && !uploading && (
              <li className="source-empty">Your uploaded books will appear here.</li>
            )}
          </ul>
        </aside>

        <section className="chat">
          <div className="messages" ref={scrollRef}>
            {messages.length === 0 ? (
              <div className="empty">
                <span className="empty-mark" aria-hidden="true" />
                <h2 className="empty-title">
                  {documents.length === 0 ? "Add a book to begin" : "Ask anything from your books"}
                </h2>
                <p className="empty-text">
                  {documents.length === 0
                    ? "Upload one or more PDFs on the left. Once they're processed, ask questions in plain language and get answers pulled straight from the pages."
                    : "Questions search across every source you've added. Each answer shows the exact document and page it came from."}
                </p>
              </div>
            ) : (
              messages.map((m, i) => (
                <div className={`msg msg-${m.role}`} key={i}>
                  <div className="msg-role">{m.role === "user" ? "You" : "From your books"}</div>
                  <div className="msg-text">{m.text}</div>
                  {m.role === "assistant" && m.citations?.length > 0 && (
                    <div className="citations">
                      {m.citations.map((c, j) => (
                        <span className="chip" key={j}>
                          <span className="chip-src">{c.source}</span>
                          <span className="chip-page">p.{c.page}</span>
                        </span>
                      ))}
                    </div>
                  )}
                </div>
              ))
            )}

            {asking && (
              <div className="msg msg-assistant">
                <div className="msg-role">From your books</div>
                <div className="thinking">
                  <span></span>
                  <span></span>
                  <span></span>
                </div>
              </div>
            )}
          </div>

          {notice && (
            <div className={`notice notice-${notice.type}`} role="status">
              {notice.text}
            </div>
          )}

          <div className="composer">
            <textarea
              className="composer-input"
              value={question}
              onChange={(e) => setQuestion(e.target.value)}
              onKeyDown={onQuestionKeyDown}
              placeholder={
                documents.length === 0
                  ? "Add a PDF to start asking…"
                  : "Ask something from your books…"
              }
              rows={1}
            />
            <button className="composer-send" onClick={ask} disabled={asking || uploading}>
              {asking ? "Thinking…" : "Ask"}
            </button>
          </div>
        </section>
      </main>
    </div>
  );
}

export default App;
