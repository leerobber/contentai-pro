/* ContentAI Pro — Frontend Controller */
const API = '/api/content';
let currentResult = null;

// ---------- Init ----------
document.addEventListener('DOMContentLoaded', () => {
    initTabs();
    initResultTabs();
    checkHealth();
});

function initTabs() {
    document.querySelectorAll('.tab').forEach(tab => {
        tab.addEventListener('click', () => {
            document.querySelectorAll('.tab').forEach(t => t.classList.remove('active'));
            document.querySelectorAll('.panel').forEach(p => p.classList.remove('active'));
            tab.classList.add('active');
            document.getElementById(`tab-${tab.dataset.tab}`).classList.add('active');
        });
    });
}

function initResultTabs() {
    document.addEventListener('click', e => {
        if (e.target.classList.contains('result-tab')) {
            document.querySelectorAll('.result-tab').forEach(t => t.classList.remove('active'));
            e.target.classList.add('active');
            showResultView(e.target.dataset.result);
        }
    });
}

async function checkHealth() {
    try {
        const res = await fetch('/api/health');
        const data = await res.json();
        document.getElementById('statusDot').classList.add('online');
        document.getElementById('statusText').textContent = `${data.mode} mode · ${data.agents.length} agents`;
    } catch {
        document.getElementById('statusDot').classList.add('error');
        document.getElementById('statusText').textContent = 'Offline';
    }
}

// ---------- API Calls ----------
async function apiPost(endpoint, body) {
    const res = await fetch(`${API}${endpoint}`, {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify(body),
    });
    if (!res.ok) {
        const err = await res.json().catch(() => ({ detail: 'Request failed' }));
        throw new Error(err.detail || `HTTP ${res.status}`);
    }
    return res.json();
}

async function apiGet(endpoint) {
    const res = await fetch(`${API}${endpoint}`);
    return res.json();
}

// ---------- Generate Pipeline ----------
async function runPipeline() {
    const topic = document.getElementById('genTopic').value.trim();
    if (!topic) return alert('Enter a topic');

    const tracker = document.getElementById('pipelineTracker');
    const results = document.getElementById('genResults');
    tracker.style.display = 'block';
    results.style.display = 'none';
    resetStages();

    const keywords = document.getElementById('genKeywords').value.split(',').map(k => k.trim()).filter(Boolean);

    try {
        // Use streaming endpoint
        const body = {
            topic,
            content_type: document.getElementById('genType').value,
            audience: document.getElementById('genAudience').value,
            tone: document.getElementById('genTone').value,
            word_count: parseInt(document.getElementById('genWords').value),
            keywords,
            enable_debate: document.getElementById('genDebate').checked,
            enable_atomizer: document.getElementById('genAtomize').checked,
        };

        // Start SSE stream
        const res = await fetch(`${API}/generate/stream`, {
            method: 'POST',
            headers: { 'Content-Type': 'application/json' },
            body: JSON.stringify(body),
        });

        const reader = res.body.getReader();
        const decoder = new TextDecoder();
        let buffer = '';

        while (true) {
            const { done, value } = await reader.read();
            if (done) break;
            buffer += decoder.decode(value, { stream: true });

            const lines = buffer.split('\n\n');
            buffer = lines.pop();

            for (const line of lines) {
                if (line.startsWith('data: ')) {
                    try {
                        const event = JSON.parse(line.slice(6));
                        updateStage(event.stage, event.status);

                        if (event.stage === 'pipeline' && event.status === 'completed') {
                            // Fetch full result
                            const contentId = event.data.content_id;
                            const full = await apiGet(`/content/${contentId}`);
                            // Also run non-streaming for full data
                            currentResult = await apiPost('/generate', body);
                            showGenerateResults();
                        }
                    } catch (e) { /* skip parse errors */ }
                }
            }
        }

        // Fallback: if stream didn't complete cleanly, do a direct call
        if (!currentResult) {
            currentResult = await apiPost('/generate', body);
            showGenerateResults();
        }
    } catch (err) {
        // Fallback to non-streaming
        try {
            const body = {
                topic,
                content_type: document.getElementById('genType').value,
                audience: document.getElementById('genAudience').value,
                tone: document.getElementById('genTone').value,
                word_count: parseInt(document.getElementById('genWords').value),
                keywords,
                enable_debate: document.getElementById('genDebate').checked,
                enable_atomizer: document.getElementById('genAtomize').checked,
            };
            currentResult = await apiPost('/generate', body);
            completeAllStages();
            showGenerateResults();
        } catch (e2) {
            alert(`Pipeline failed: ${e2.message}`);
        }
    }
}

async function runQuick() {
    const topic = document.getElementById('genTopic').value.trim();
    if (!topic) return alert('Enter a topic');

    try {
        currentResult = await apiPost('/generate/quick', {
            topic,
            content_type: document.getElementById('genType').value,
            tone: document.getElementById('genTone').value,
            word_count: parseInt(document.getElementById('genWords').value),
        });
        document.getElementById('pipelineTracker').style.display = 'none';
        showGenerateResults();
    } catch (err) {
        alert(`Quick gen failed: ${err.message}`);
    }
}

function showGenerateResults() {
    const el = document.getElementById('genResults');
    el.style.display = 'block';
    showResultView('final');
    document.getElementById('resultMeta').textContent =
        `Content ID: ${currentResult.content_id} · Stages: ${(currentResult.stages_completed || []).join(' → ')} · ${Math.round(currentResult.latency_ms)}ms`;
}

function showResultView(view) {
    const el = document.getElementById('resultContent');
    if (!currentResult) return;

    switch (view) {
        case 'final':
            el.textContent = currentResult.final_content || 'No content generated';
            break;
        case 'research':
            el.textContent = currentResult.research || 'Research not available';
            break;
        case 'debate':
            if (currentResult.debate) {
                el.innerHTML = renderDebate(currentResult.debate);
            } else {
                el.textContent = 'Debate not run';
            }
            break;
        case 'atomized':
            if (currentResult.atomized) {
                el.innerHTML = renderAtomized(currentResult.atomized);
            } else {
                el.textContent = 'Atomizer not run';
            }
            break;
    }
}

// ---------- Debate ----------
async function runDebate() {
    const content = document.getElementById('debateContent').value.trim();
    const topic = document.getElementById('debateTopic').value.trim();
    if (!content) return alert('Paste content to debate');

    try {
        const result = await apiPost('/debate', { content, topic: topic || 'Content Review' });
        const el = document.getElementById('debateResults');
        el.style.display = 'block';
        el.innerHTML = renderDebate(result);
    } catch (err) {
        alert(`Debate failed: ${err.message}`);
    }
}

function renderDebate(data) {
    let html = `<div style="margin-bottom:12px;font-size:14px">
        Score: <span class="debate-score ${data.passed ? 'pass' : data.final_score >= 5 ? 'revise' : 'fail'}">${data.final_score}/10</span>
        · ${data.passed ? '✅ PASSED' : '❌ NEEDS REVISION'} · ${data.total_rounds} round(s)
    </div>`;

    for (const r of (data.rounds || [])) {
        html += `<div class="debate-round">
            <h4>Round ${r.round} — Score: ${r.score} (${r.verdict})</h4>
            <div class="debate-role">🛡 ADVOCATE:</div>
            <div class="debate-text">${escapeHtml(r.advocate || '').slice(0, 500)}</div>
            <div class="debate-role">⚔ CRITIC:</div>
            <div class="debate-text">${escapeHtml(r.critic || '').slice(0, 500)}</div>
        </div>`;
    }
    return html;
}

// ---------- Atomize ----------
async function runAtomize() {
    const content = document.getElementById('atomContent').value.trim();
    const topic = document.getElementById('atomTopic').value.trim();
    if (!content) return alert('Paste content to atomize');

    try {
        const result = await apiPost('/atomize', { content, topic: topic || 'Content' });
        const el = document.getElementById('atomResults');
        el.style.display = 'block';
        el.innerHTML = renderAtomized(result);
    } catch (err) {
        alert(`Atomize failed: ${err.message}`);
    }
}

function renderAtomized(data) {
    let html = `<div style="margin-bottom:12px;font-size:12px;color:var(--text-dim)">
        ${data.platforms_generated || (data.variants || []).length} platform variant(s) generated
    </div>`;
    for (const v of (data.variants || [])) {
        html += `<div class="variant-card">
            <div class="variant-header">
                <span class="variant-platform">${v.platform}</span>
                <span class="variant-chars">${v.char_count} chars · ${v.format}</span>
                <button class="variant-copy" onclick="copyText(this, '${v.platform}')">Copy</button>
            </div>
            <div class="variant-body">${escapeHtml(v.content)}</div>
        </div>`;
    }
    return html;
}

// ---------- DNA ----------
async function calibrateDNA() {
    const name = document.getElementById('dnaName').value.trim();
    const raw = document.getElementById('dnaSamples').value.trim();
    if (!name || !raw) return alert('Enter profile name and samples');

    const samples = raw.split('---').map(s => s.trim()).filter(s => s.length > 50);
    if (samples.length < 3) return alert('Need at least 3 samples (separated by ---)');

    try {
        const result = await apiPost('/dna/calibrate', { name, samples });
        const el = document.getElementById('dnaResults');
        el.style.display = 'block';
        el.innerHTML = renderDNA(result);
    } catch (err) {
        alert(`DNA calibration failed: ${err.message}`);
    }
}

function renderDNA(data) {
    let html = `<div style="margin-bottom:12px">
        <strong>Profile:</strong> ${data.name} · <strong>Samples:</strong> ${data.samples_analyzed}
        <br><em>${data.summary || ''}</em>
    </div><div class="dna-chart">`;

    for (const [dim, val] of Object.entries(data.fingerprint || {})) {
        const pct = Math.min(Math.abs(val) * 100, 100);
        html += `<div class="dna-dim">
            <span style="min-width:140px;font-size:10px;color:var(--text-dim)">${dim.replace(/_/g, ' ')}</span>
            <div class="dna-bar"><div class="dna-bar-fill" style="width:${pct}%"></div></div>
            <span class="dna-val">${val}</span>
        </div>`;
    }
    html += '</div>';
    return html;
}

// ---------- Trends ----------
async function fetchTrends() {
    const niche = document.getElementById('trendNiche').value.trim();
    try {
        const params = niche ? `?niche=${encodeURIComponent(niche)}` : '';
        const result = await apiGet(`/trends${params}`);
        const el = document.getElementById('trendResults');
        el.style.display = 'block';

        let html = `<div style="margin-bottom:10px;font-size:11px;color:var(--text-dim)">
            ${result.total_found} trends from ${(result.sources || []).join(', ')}
            ${result.cache_hit ? '(cached)' : ''} · ${Math.round(result.latency_ms)}ms
        </div>`;

        for (const t of (result.trends || [])) {
            html += `<div class="trend-card" onclick="window.open('${t.url}', '_blank')">
                <div>
                    <div class="trend-title">${escapeHtml(t.title)}</div>
                    <div class="trend-meta">
                        <span class="trend-score">▲ ${t.score}</span>
                        <span class="trend-source">${t.source}</span>
                        <span>${t.category}</span>
                    </div>
                </div>
                <button class="trend-use-btn" onclick="event.stopPropagation(); useTrend('${escapeHtml(t.title)}')">Use Topic</button>
            </div>`;
        }
        el.innerHTML = html;
    } catch (err) {
        alert(`Trend scan failed: ${err.message}`);
    }
}

function useTrend(title) {
    document.getElementById('genTopic').value = title;
    document.querySelector('.tab[data-tab="generate"]').click();
}

// ---------- Pipeline Stage Tracking ----------
function resetStages() {
    document.querySelectorAll('.stage').forEach(s => {
        s.classList.remove('active', 'completed', 'failed');
    });
}

function updateStage(stage, status) {
    const el = document.getElementById(`stage-${stage}`);
    if (!el) return;
    el.classList.remove('active', 'completed', 'failed');
    if (status === 'started') el.classList.add('active');
    else if (status === 'completed') el.classList.add('completed');
    else if (status === 'failed') el.classList.add('failed');
}

function completeAllStages() {
    document.querySelectorAll('.stage').forEach(s => {
        s.classList.remove('active');
        s.classList.add('completed');
    });
}

// ---------- Util ----------
function escapeHtml(str) {
    const div = document.createElement('div');
    div.textContent = str || '';
    return div.innerHTML;
}

function copyText(btn, platform) {
    const body = btn.closest('.variant-card').querySelector('.variant-body');
    navigator.clipboard.writeText(body.textContent).then(() => {
        btn.textContent = 'Copied!';
        setTimeout(() => btn.textContent = 'Copy', 1500);
    });
}
