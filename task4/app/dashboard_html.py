"""
Industrial Safety & Maintenance RAG Assistant — Interactive Dashboard HTML
"""

DASHBOARD_HTML = """<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>Industrial Safety & Maintenance RAG Assistant | AjnaLens</title>
    <link rel="preconnect" href="https://fonts.googleapis.com">
    <link rel="preconnect" href="https://fonts.gstatic.com" crossorigin>
    <link href="https://fonts.googleapis.com/css2?family=Inter:wght@300;400;500;600;700;800&family=JetBrains+Mono:wght@400;500;600&display=swap" rel="stylesheet">
    <style>
        :root {
            --bg-base: #0B0F19;
            --bg-surface: #111827;
            --bg-card: rgba(17, 24, 39, 0.85);
            --bg-card-hover: rgba(31, 41, 55, 0.7);
            --border-subtle: rgba(255, 255, 255, 0.08);
            --border-accent: rgba(245, 158, 11, 0.35);
            --text-main: #F3F4F6;
            --text-muted: #9CA3AF;
            --text-dim: #6B7280;
            --accent-amber: #F59E0B;
            --accent-amber-glow: rgba(245, 158, 11, 0.2);
            --accent-cyan: #06B6D4;
            --accent-emerald: #10B981;
            --accent-red: #EF4444;
            --font-sans: 'Inter', system-ui, -apple-system, sans-serif;
            --font-mono: 'JetBrains Mono', monospace;
        }

        * {
            box-sizing: border-box;
            margin: 0;
            padding: 0;
        }

        body {
            background-color: var(--bg-base);
            background-image: 
                radial-gradient(circle at 15% 10%, rgba(245, 158, 11, 0.07) 0%, transparent 40%),
                radial-gradient(circle at 85% 60%, rgba(6, 182, 212, 0.06) 0%, transparent 40%);
            color: var(--text-main);
            font-family: var(--font-sans);
            min-height: 100vh;
            line-height: 1.5;
            padding: 24px;
        }

        .container {
            max-width: 1200px;
            margin: 0 auto;
        }

        /* Top Header */
        header {
            display: flex;
            justify-content: space-between;
            align-items: center;
            flex-wrap: wrap;
            gap: 16px;
            padding-bottom: 24px;
            border-bottom: 1px solid var(--border-subtle);
            margin-bottom: 28px;
        }

        .brand-section {
            display: flex;
            align-items: center;
            gap: 14px;
        }

        .brand-badge {
            width: 44px;
            height: 44px;
            background: linear-gradient(135deg, #F59E0B 0%, #D97706 100%);
            border-radius: 10px;
            display: flex;
            align-items: center;
            justify-content: center;
            font-weight: 800;
            font-size: 22px;
            color: #000;
            box-shadow: 0 4px 14px rgba(245, 158, 11, 0.35);
        }

        .brand-title h1 {
            font-size: 20px;
            font-weight: 700;
            letter-spacing: -0.02em;
            color: #FFF;
        }

        .brand-title p {
            font-size: 13px;
            color: var(--text-muted);
        }

        .header-meta {
            display: flex;
            align-items: center;
            gap: 12px;
            flex-wrap: wrap;
        }

        .status-chip {
            display: inline-flex;
            align-items: center;
            gap: 6px;
            background: rgba(16, 185, 129, 0.12);
            color: var(--accent-emerald);
            border: 1px solid rgba(16, 185, 129, 0.25);
            padding: 5px 12px;
            border-radius: 9999px;
            font-size: 12px;
            font-weight: 600;
        }

        .pulse-dot {
            width: 8px;
            height: 8px;
            border-radius: 50%;
            background-color: var(--accent-emerald);
            box-shadow: 0 0 8px var(--accent-emerald);
            animation: pulse 2s infinite;
        }

        @keyframes pulse {
            0%, 100% { opacity: 1; transform: scale(1); }
            50% { opacity: 0.4; transform: scale(0.85); }
        }

        .meta-chip {
            background: var(--bg-surface);
            border: 1px solid var(--border-subtle);
            padding: 5px 12px;
            border-radius: 8px;
            font-size: 12px;
            font-family: var(--font-mono);
            color: var(--text-muted);
        }

        .swagger-link {
            background: rgba(6, 182, 212, 0.12);
            border: 1px solid rgba(6, 182, 212, 0.25);
            color: var(--accent-cyan);
            padding: 5px 12px;
            border-radius: 8px;
            font-size: 12px;
            font-weight: 600;
            text-decoration: none;
            transition: all 0.2s ease;
        }

        .swagger-link:hover {
            background: rgba(6, 182, 212, 0.25);
            color: #FFF;
        }

        /* Search & Query Bar */
        .query-card {
            background: var(--bg-card);
            backdrop-filter: blur(12px);
            border: 1px solid var(--border-subtle);
            border-radius: 14px;
            padding: 24px;
            box-shadow: 0 10px 25px -5px rgba(0, 0, 0, 0.4);
            margin-bottom: 24px;
        }

        .input-group {
            display: flex;
            gap: 12px;
            margin-bottom: 16px;
        }

        .input-wrapper {
            position: relative;
            flex: 1;
        }

        .query-input {
            width: 100%;
            background: var(--bg-base);
            border: 1px solid var(--border-subtle);
            border-radius: 10px;
            padding: 14px 18px;
            font-size: 15px;
            font-family: var(--font-sans);
            color: var(--text-main);
            outline: none;
            transition: all 0.2s ease;
        }

        .query-input:focus {
            border-color: var(--accent-amber);
            box-shadow: 0 0 0 3px var(--accent-amber-glow);
        }

        .top-k-select {
            background: var(--bg-base);
            border: 1px solid var(--border-subtle);
            border-radius: 10px;
            padding: 0 14px;
            color: var(--text-muted);
            font-size: 14px;
            outline: none;
            cursor: pointer;
        }

        .submit-btn {
            background: linear-gradient(135deg, #F59E0B 0%, #D97706 100%);
            color: #000;
            border: none;
            border-radius: 10px;
            padding: 0 24px;
            font-weight: 700;
            font-size: 14px;
            cursor: pointer;
            transition: transform 0.15s ease, box-shadow 0.15s ease;
            display: flex;
            align-items: center;
            gap: 8px;
        }

        .submit-btn:hover {
            transform: translateY(-1px);
            box-shadow: 0 4px 16px rgba(245, 158, 11, 0.4);
        }

        .submit-btn:active {
            transform: translateY(1px);
        }

        .submit-btn:disabled {
            opacity: 0.6;
            cursor: not-allowed;
        }

        /* Preset Chips */
        .preset-title {
            font-size: 12px;
            font-weight: 600;
            color: var(--text-dim);
            text-transform: uppercase;
            letter-spacing: 0.05em;
            margin-bottom: 8px;
        }

        .preset-list {
            display: flex;
            flex-wrap: wrap;
            gap: 8px;
        }

        .preset-btn {
            background: rgba(255, 255, 255, 0.04);
            border: 1px solid var(--border-subtle);
            color: var(--text-muted);
            padding: 7px 12px;
            border-radius: 8px;
            font-size: 12.5px;
            cursor: pointer;
            transition: all 0.2s ease;
            text-align: left;
        }

        .preset-btn:hover {
            background: rgba(245, 158, 11, 0.1);
            border-color: rgba(245, 158, 11, 0.3);
            color: var(--text-main);
        }

        /* Results Display */
        .results-section {
            display: none;
            animation: fadeIn 0.3s ease;
        }

        @keyframes fadeIn {
            from { opacity: 0; transform: translateY(8px); }
            to { opacity: 1; transform: translateY(0); }
        }

        .metrics-bar {
            display: flex;
            justify-content: space-between;
            align-items: center;
            flex-wrap: wrap;
            gap: 12px;
            margin-bottom: 16px;
            padding: 10px 16px;
            background: rgba(0, 0, 0, 0.2);
            border: 1px solid var(--border-subtle);
            border-radius: 10px;
        }

        .metrics-group {
            display: flex;
            gap: 16px;
            align-items: center;
        }

        .metric-item {
            font-size: 13px;
            color: var(--text-muted);
            font-family: var(--font-mono);
        }

        .metric-item strong {
            color: var(--accent-amber);
        }

        .answer-box {
            background: linear-gradient(180deg, rgba(17, 24, 39, 0.95) 0%, rgba(15, 20, 30, 0.95) 100%);
            border: 1px solid var(--border-accent);
            border-radius: 14px;
            padding: 24px;
            margin-bottom: 24px;
            box-shadow: 0 4px 20px rgba(0, 0, 0, 0.3);
        }

        .answer-header {
            display: flex;
            justify-content: space-between;
            align-items: center;
            margin-bottom: 14px;
        }

        .answer-title {
            font-size: 14px;
            font-weight: 700;
            color: var(--accent-amber);
            text-transform: uppercase;
            letter-spacing: 0.05em;
            display: flex;
            align-items: center;
            gap: 6px;
        }

        .answer-body {
            font-size: 14.5px;
            line-height: 1.7;
            color: #E5E7EB;
            white-space: pre-line;
        }

        /* Retrieved Documents Grid */
        .retrieved-heading {
            font-size: 15px;
            font-weight: 600;
            color: var(--text-muted);
            margin-bottom: 14px;
            display: flex;
            align-items: center;
            gap: 8px;
        }

        .docs-grid {
            display: grid;
            grid-template-columns: repeat(auto-fit, minmax(320px, 1fr));
            gap: 16px;
            margin-bottom: 32px;
        }

        .doc-card {
            background: var(--bg-card);
            border: 1px solid var(--border-subtle);
            border-radius: 12px;
            padding: 18px;
            transition: all 0.2s ease;
            position: relative;
            display: flex;
            flex-direction: column;
            justify-content: space-between;
        }

        .doc-card:hover {
            border-color: rgba(255, 255, 255, 0.2);
            transform: translateY(-2px);
        }

        .doc-header {
            display: flex;
            justify-content: space-between;
            align-items: flex-start;
            margin-bottom: 10px;
        }

        .doc-id {
            font-family: var(--font-mono);
            font-size: 12px;
            font-weight: 700;
            color: var(--accent-cyan);
            background: rgba(6, 182, 212, 0.1);
            padding: 2px 8px;
            border-radius: 6px;
            border: 1px solid rgba(6, 182, 212, 0.2);
        }

        .similarity-score {
            font-family: var(--font-mono);
            font-size: 12px;
            font-weight: 600;
            color: var(--accent-emerald);
        }

        .doc-title {
            font-size: 14px;
            font-weight: 600;
            color: #FFF;
            margin-bottom: 6px;
        }

        .doc-osha {
            font-size: 12px;
            color: var(--accent-amber);
            font-family: var(--font-mono);
            margin-bottom: 10px;
        }

        .doc-content {
            font-size: 13px;
            color: var(--text-muted);
            line-height: 1.55;
            margin-bottom: 12px;
            flex-grow: 1;
        }

        .doc-zones {
            display: flex;
            flex-wrap: wrap;
            gap: 6px;
            padding-top: 10px;
            border-top: 1px solid var(--border-subtle);
        }

        .zone-tag {
            background: rgba(255, 255, 255, 0.05);
            font-size: 11px;
            font-family: var(--font-mono);
            color: var(--text-dim);
            padding: 2px 6px;
            border-radius: 4px;
        }

        /* Corpus Browser */
        .corpus-toggle {
            background: none;
            border: 1px solid var(--border-subtle);
            color: var(--text-muted);
            padding: 10px 18px;
            border-radius: 8px;
            font-size: 13px;
            cursor: pointer;
            width: 100%;
            text-align: center;
            transition: all 0.2s ease;
        }

        .corpus-toggle:hover {
            border-color: var(--text-dim);
            color: #FFF;
        }

        .corpus-tray {
            display: none;
            margin-top: 16px;
            grid-template-columns: repeat(auto-fit, minmax(280px, 1fr));
            gap: 12px;
        }

        .corpus-card {
            background: var(--bg-surface);
            border: 1px solid var(--border-subtle);
            border-radius: 10px;
            padding: 14px;
            font-size: 12.5px;
        }

        /* Loading Spinner */
        .spinner {
            display: none;
            width: 18px;
            height: 18px;
            border: 2px solid rgba(0, 0, 0, 0.2);
            border-top-color: #000;
            border-radius: 50%;
            animation: spin 0.6s linear infinite;
        }

        @keyframes spin {
            to { transform: rotate(360deg); }
        }
    </style>
</head>
<body>
    <div class="container">
        <!-- Top Navigation & System Health -->
        <header>
            <div class="brand-section">
                <div class="brand-badge">AL</div>
                <div class="brand-title">
                    <h1>Industrial Safety & SOP Assistant</h1>
                    <p>Task 4 — Semantic RAG Microservice for OSHA, LOTO & Plant Policies</p>
                </div>
            </div>
            <div class="header-meta">
                <span class="status-chip"><span class="pulse-dot"></span> System Online</span>
                <span class="meta-chip">Model: all-MiniLM-L6-v2</span>
                <span class="meta-chip">Corpus: 6 SOPs</span>
                <a href="/docs" target="_blank" class="swagger-link">OpenAPI Swagger /docs ↗</a>
            </div>
        </header>

        <!-- Query Form -->
        <div class="query-card">
            <form id="ragForm" onsubmit="handleQuery(event)">
                <div class="input-group">
                    <div class="input-wrapper">
                        <input 
                            type="text" 
                            id="queryInput" 
                            class="query-input" 
                            placeholder="Enter safety inquiry (e.g., hard hat rules in zone alpha, LOTO procedures...)" 
                            required 
                            autocomplete="off"
                        />
                    </div>
                    <select id="topKSelect" class="top-k-select" title="Number of documents to retrieve">
                        <option value="2">Top 2</option>
                        <option value="3" selected>Top 3</option>
                        <option value="5">Top 5</option>
                    </select>
                    <button type="submit" id="submitBtn" class="submit-btn">
                        <span class="spinner" id="spinner"></span>
                        <span id="btnText">Query SOPs</span>
                    </button>
                </div>
            </form>

            <!-- Presets -->
            <div class="preset-title">Verified Assessment Inquiries (Click to run):</div>
            <div class="preset-list">
                <button type="button" class="preset-btn" onclick="runPreset('What are the hard hat requirements and PPE rules for workers in Zone Alpha?')">
                    🦺 Zone Alpha Hard Hat Standard
                </button>
                <button type="button" class="preset-btn" onclick="runPreset('What happens if a worker stays idle in a restricted hazard zone for more than 5 seconds?')">
                    ⏱️ 5s Idle Dwell Alarm Protocol
                </button>
                <button type="button" class="preset-btn" onclick="runPreset('What is the proper Lockout Tagout (LOTO) procedure before servicing an induction motor or pump?')">
                    🔒 LOTO Motor & Pump Isolation
                </button>
                <button type="button" class="preset-btn" onclick="runPreset('What are the safe clearance approach distances and arc flash boundaries for 11kV transformers?')">
                    ⚡ 11kV Transformer Clearances
                </button>
                <button type="button" class="preset-btn" onclick="runPreset('How is equipment identification validated when OCR detects a voltage mismatch during maintenance?')">
                    🏷️ OCR Voltage Mismatch Lockout
                </button>
            </div>
        </div>

        <!-- RAG Output Section -->
        <div id="resultsSection" class="results-section">
            <div class="metrics-bar">
                <div class="metrics-group">
                    <span class="metric-item">⚡ Latency: <strong id="latencyDisplay">-- ms</strong></span>
                    <span class="metric-item">🎯 Cosine Match: <strong id="similarityDisplay">--</strong></span>
                </div>
                <div class="metric-item" id="queryEcho" style="color: var(--text-dim); max-width: 500px; overflow: hidden; text-overflow: ellipsis; white-space: nowrap;">
                    --
                </div>
            </div>

            <!-- Grounded Answer -->
            <div class="answer-box">
                <div class="answer-header">
                    <div class="answer-title">🛡️ Grounded Safety Advisory</div>
                    <span style="font-size: 11px; font-family: var(--font-mono); color: var(--text-dim);">Source-Verified</span>
                </div>
                <div id="answerBody" class="answer-body"></div>
            </div>

            <!-- Retrieved SOP Documents -->
            <div class="retrieved-heading">
                📚 Grounding SOP Passages Retrieved
            </div>
            <div id="docsGrid" class="docs-grid"></div>
        </div>

        <!-- SOP Knowledge Base Explorer -->
        <button type="button" class="corpus-toggle" onclick="toggleCorpus()">
            📖 Browse All 6 Indexed Plant Standard Operating Procedures (SOPs) ▼
        </button>
        <div id="corpusTray" class="corpus-tray"></div>
    </div>

    <script>
        async function handleQuery(e) {
            if (e) e.preventDefault();
            const input = document.getElementById('queryInput');
            const query = input.value.trim();
            if (!query) return;

            const topK = parseInt(document.getElementById('topKSelect').value);
            const btn = document.getElementById('submitBtn');
            const btnText = document.getElementById('btnText');
            const spinner = document.getElementById('spinner');

            btn.disabled = true;
            btnText.textContent = "Retrieving...";
            spinner.style.display = "inline-block";

            try {
                const response = await fetch('/query', {
                    method: 'POST',
                    headers: { 'Content-Type': 'application/json' },
                    body: JSON.stringify({ query: query, top_k: topK })
                });

                if (!response.ok) {
                    throw new Error(`HTTP ${response.status}: ${await response.text()}`);
                }

                const data = await response.json();
                renderResults(data);
            } catch (err) {
                alert("Error querying RAG assistant: " + err.message);
            } finally {
                btn.disabled = false;
                btnText.textContent = "Query SOPs";
                spinner.style.display = "none";
            }
        }

        function runPreset(text) {
            document.getElementById('queryInput').value = text;
            handleQuery();
        }

        function renderResults(data) {
            document.getElementById('resultsSection').style.display = 'block';
            document.getElementById('latencyDisplay').textContent = `${data.latency_ms} ms`;
            document.getElementById('similarityDisplay').textContent = `${(data.top_similarity_score * 100).toFixed(1)}%`;
            document.getElementById('queryEcho').textContent = `Query: "${data.query}"`;
            document.getElementById('answerBody').textContent = data.answer;

            const docsGrid = document.getElementById('docsGrid');
            docsGrid.innerHTML = '';

            data.retrieved_documents.forEach(doc => {
                const card = document.createElement('div');
                card.className = 'doc-card';
                const zones = (doc.applicable_zones || []).map(z => `<span class="zone-tag">${z}</span>`).join('');
                
                card.innerHTML = `
                    <div>
                        <div class="doc-header">
                            <span class="doc-id">${doc.doc_id}</span>
                            <span class="similarity-score">Score: ${(doc.similarity_score * 100).toFixed(1)}%</span>
                        </div>
                        <div class="doc-title">${doc.title}</div>
                        <div class="doc-osha">${doc.osha_reference || 'Industrial SOP'}</div>
                        <div class="doc-content">${doc.content}</div>
                    </div>
                    <div class="doc-zones">${zones}</div>
                `;
                docsGrid.appendChild(card);
            });

            // Smooth scroll to output
            document.getElementById('resultsSection').scrollIntoView({ behavior: 'smooth', block: 'start' });
        }

        let corpusLoaded = false;
        async function toggleCorpus() {
            const tray = document.getElementById('corpusTray');
            if (tray.style.display === 'grid') {
                tray.style.display = 'none';
                return;
            }

            tray.style.display = 'grid';
            if (!corpusLoaded) {
                tray.innerHTML = '<div style="color: var(--text-dim);">Loading indexed SOP database...</div>';
                try {
                    const res = await fetch('/documents');
                    const data = await res.json();
                    tray.innerHTML = '';
                    data.documents.forEach(d => {
                        const item = document.createElement('div');
                        item.className = 'corpus-card';
                        item.innerHTML = `
                            <div style="font-weight: 700; color: var(--accent-cyan); font-family: var(--font-mono);">${d.doc_id}</div>
                            <div style="font-weight: 600; color: #FFF; margin: 4px 0;">${d.title}</div>
                            <div style="color: var(--accent-amber); font-size: 11px;">${d.osha_reference || 'Standard SOP'}</div>
                            <div style="color: var(--text-dim); font-size: 11px; margin-top: 6px;">Category: ${d.category}</div>
                        `;
                        tray.appendChild(item);
                    });
                    corpusLoaded = true;
                } catch (e) {
                    tray.innerHTML = '<div style="color: var(--accent-red);">Failed to load SOP directory.</div>';
                }
            }
        }
    </script>
</body>
</html>
"""
