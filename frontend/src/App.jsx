import { useState } from "react";
import axios from "axios";
import "./App.css";

function App() {
  const [file, setFile] = useState(null);
  const [query, setQuery] = useState("");
  const [uploadMessage, setUploadMessage] = useState("");
  const [answer, setAnswer] = useState("");
  const [citations, setCitations] = useState([]);
  const [loading, setLoading] = useState(false);

  const backendUrl = "http://127.0.0.1:8000";

  const handleFileChange = (event) => {
    setFile(event.target.files[0]);
  };

  const handleUpload = async () => {
    if (!file) {
      setUploadMessage("Please select a PDF first.");
      return;
    }

    const formData = new FormData();
    formData.append("file", file);

    try {
      setLoading(true);

      const response = await axios.post(`${backendUrl}/upload-pdf`, formData, {
        headers: {
          "Content-Type": "multipart/form-data",
        },
      });

      setUploadMessage(
        `${response.data.message} | Pages: ${response.data.total_pages} | Chunks: ${response.data.total_chunks}`
      );
      setAnswer("");
      setCitations([]);
    } catch (error) {
      console.error(error);

      const errorMessage =
        error.response?.data?.detail ||
        error.response?.data?.error ||
        error.message ||
        "Error uploading PDF.";

      setUploadMessage(`Error uploading PDF: ${errorMessage}`);
    } finally {
      setLoading(false);
    }
  };

  const handleAsk = async () => {
    if (!query.trim()) {
      return;
    }

    try {
      setLoading(true);

      const response = await axios.get(`${backendUrl}/ask`, {
        params: { query },
      });

      if (response.data.error) {
        setAnswer(response.data.error);
        setCitations([]);
        return;
      }

      setAnswer(response.data.answer || "");
      setCitations(response.data.citations || []);
    } catch (error) {
      console.error(error);

      const errorMessage =
        error.response?.data?.detail ||
        error.response?.data?.error ||
        error.message ||
        "Error getting answer.";

      setAnswer(errorMessage);
      setCitations([]);
    } finally {
      setLoading(false);
    }
  };

  return (
    <div className="app">
      <h1>Interactive Textbook using RAG</h1>

      <div className="card">
        <h2>Upload PDF</h2>

        <input
          type="file"
          accept="application/pdf"
          onChange={handleFileChange}
        />

        <button onClick={handleUpload} disabled={loading}>
          {loading ? "Processing..." : "Upload PDF"}
        </button>

        {uploadMessage && <p className="message">{uploadMessage}</p>}
      </div>

      <div className="card">
        <h2>Ask a Question</h2>

        <textarea
          value={query}
          onChange={(e) => setQuery(e.target.value)}
          placeholder="Ask something from the uploaded textbook..."
          rows="4"
        />

        <button onClick={handleAsk} disabled={loading}>
          {loading ? "Thinking..." : "Ask"}
        </button>
      </div>

      <div className="card">
        <h2>Answer</h2>

        {answer ? <p>{answer}</p> : <p>No answer yet.</p>}

        {citations.length > 0 && (
          <div>
            <h3>Citations</h3>
            <p>{citations.join(", ")}</p>
          </div>
        )}
      </div>
    </div>
  );
}

export default App;