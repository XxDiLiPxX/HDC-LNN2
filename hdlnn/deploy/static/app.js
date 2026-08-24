// State variables
let eventSource = null;
let scoreChart = null;
let latencyChart = null;
let threatChart = null;
let allBenchmarkRuns = [];
let threatData = { 'Normal': 0 };

// DOM Elements
const tabLiveBtn = document.getElementById('tab-live-btn');
const tabBenchBtn = document.getElementById('tab-bench-btn');
const viewLive = document.getElementById('view-live');
const viewBench = document.getElementById('view-benchmark');

const datasetSelect = document.getElementById('dataset-select');
const uploadZone = document.getElementById('upload-zone');
const fileUploader = document.getElementById('file-uploader');
const uploadFilename = document.getElementById('upload-filename');

const streamSpeedInput = document.getElementById('stream-speed');
const speedDisplay = document.getElementById('speed-display');
const flowLimitInput = document.getElementById('flow-limit');
const thresholdKInput = document.getElementById('threshold-k');
const kDisplay = document.getElementById('k-display');

const btnStartStream = document.getElementById('btn-start-stream');
const btnStopStream = document.getElementById('btn-stop-stream');
const btnClearConsole = document.getElementById('btn-clear-console');
const consoleStream = document.getElementById('console-stream');
const statusText = document.getElementById('status-text');

// KPI elements
const kpiF1 = document.getElementById('kpi-f1');
const kpiF1Sub = document.getElementById('kpi-f1-sub');
const kpiPrecRec = document.getElementById('kpi-prec-rec');
const kpiPrecRecSub = document.getElementById('kpi-prec-rec-sub');
const kpiAnomalies = document.getElementById('kpi-anomalies');
const kpiAnomalyRate = document.getElementById('kpi-anomaly-rate');
const kpiLatency = document.getElementById('kpi-latency');
const kpiThroughput = document.getElementById('kpi-throughput');

// Benchmark elements
const benchDatasetSelect = document.getElementById('bench-dataset-select');
const benchModelSelect = document.getElementById('bench-model-select');
const benchLimit = document.getElementById('bench-limit');
const btnRunBench = document.getElementById('btn-run-bench');
const evalStatusBox = document.getElementById('eval-status-box');
const benchResultCards = document.getElementById('bench-result-cards');
const matrixTableBody = document.getElementById('matrix-table-body');
const tableSearch = document.getElementById('table-search');

// ================= TAB SWITCHING =================
function switchTab(tab) {
    if (tab === 'live') {
        tabLiveBtn.classList.add('active');
        tabBenchBtn.classList.remove('active');
        viewLive.style.display = 'flex';
        viewBench.style.display = 'none';
    } else {
        tabLiveBtn.classList.remove('active');
        tabBenchBtn.classList.add('active');
        viewLive.style.display = 'none';
        viewBench.style.display = 'flex';
        loadBenchmarkTable();
    }
}

// ================= EVENT LISTENERS =================
if (streamSpeedInput) {
    streamSpeedInput.addEventListener('input', (e) => {
        speedDisplay.textContent = `${e.target.value} flows/s`;
    });
}

if (thresholdKInput) {
    thresholdKInput.addEventListener('input', (e) => {
        kDisplay.textContent = Number(e.target.value).toFixed(1);
    });
}

if (datasetSelect) {
    datasetSelect.addEventListener('change', (e) => {
        // Dropdown selection handled automatically
    });
}

const uploadStatus = document.getElementById('upload-status-indicator');

// Upload handler function
async function handleFileUpload(file) {
    if (!file) return;
    
    if (uploadStatus) {
        uploadStatus.textContent = `Uploading ${file.name}...`;
        uploadStatus.style.color = '#3b82f6';
    }
    
    const formData = new FormData();
    formData.append('file', file);
    
    try {
        const res = await fetch('/api/upload', {
            method: 'POST',
            body: formData
        });
        const data = await res.json();
        if (data.status === 'success') {
            if (uploadStatus) {
                uploadStatus.textContent = `✅ Ready: ${file.name} (${Math.round(file.size/1024)} KB)`;
                uploadStatus.style.color = '#10b981';
            }
            if (datasetSelect) {
                datasetSelect.value = 'custom';
            }
            appendConsole(`[SYSTEM] Loaded custom dataset '${file.name}' (${data.size} bytes). Click "Start Ingestion" to run telemetry.`, 'system');
        } else {
            if (uploadStatus) {
                uploadStatus.textContent = `❌ Upload Error: ${data.message}`;
                uploadStatus.style.color = '#ef4444';
            }
            appendConsole(`[ERROR] File upload failed: ${data.message}`, 'alert');
        }
    } catch (err) {
        if (uploadStatus) {
            uploadStatus.textContent = `❌ Error: ${err.message}`;
            uploadStatus.style.color = '#ef4444';
        }
    }
}

if (fileUploader) {
    fileUploader.addEventListener('change', (e) => {
        if (e.target.files && e.target.files.length > 0) {
            handleFileUpload(e.target.files[0]);
        }
    });
}

// Drag & Drop on Sidebar
const sidebarElem = document.querySelector('.sidebar');
if (sidebarElem) {
    ['dragenter', 'dragover'].forEach(name => {
        sidebarElem.addEventListener(name, (e) => {
            e.preventDefault();
            e.stopPropagation();
            sidebarElem.style.background = '#152033';
        });
    });

    ['dragleave', 'drop'].forEach(name => {
        sidebarElem.addEventListener(name, (e) => {
            e.preventDefault();
            e.stopPropagation();
            sidebarElem.style.background = 'var(--bg-surface)';
        });
    });

    sidebarElem.addEventListener('drop', (e) => {
        const dt = e.dataTransfer;
        if (dt && dt.files && dt.files.length > 0) {
            handleFileUpload(dt.files[0]);
        }
    });
}

if (btnClearConsole) {
    btnClearConsole.addEventListener('click', () => {
        consoleStream.innerHTML = '';
    });
}

// ================= CHART INITIALIZATION =================
function initCharts() {
    const ctxScore = document.getElementById('scoreChart')?.getContext('2d');
    if (ctxScore) {
        scoreChart = new Chart(ctxScore, {
            type: 'line',
            data: {
                labels: [],
                datasets: [
                    {
                        label: 'Drift Score',
                        data: [],
                        borderColor: '#3b82f6',
                        backgroundColor: 'rgba(59, 130, 246, 0.1)',
                        borderWidth: 1.8,
                        pointRadius: 0,
                        tension: 0.1,
                        fill: true
                    },
                    {
                        label: 'Threshold',
                        data: [],
                        borderColor: '#ef4444',
                        borderWidth: 1.5,
                        borderDash: [4, 4],
                        pointRadius: 0,
                        tension: 0
                    }
                ]
            },
            options: {
                responsive: true,
                maintainAspectRatio: false,
                animation: false,
                scales: {
                    x: { display: false },
                    y: {
                        grid: { color: '#1e293b' },
                        ticks: { color: '#94a3b8', font: { size: 10 } }
                    }
                },
                plugins: {
                    legend: {
                        position: 'top',
                        labels: { color: '#94a3b8', boxWidth: 10, font: { size: 10 } }
                    }
                }
            }
        });
    }

    const ctxLat = document.getElementById('latencyChart')?.getContext('2d');
    if (ctxLat) {
        latencyChart = new Chart(ctxLat, {
            type: 'line',
            data: {
                labels: [],
                datasets: [{
                    label: 'Latency (ms)',
                    data: [],
                    borderColor: '#10b981',
                    borderWidth: 1.5,
                    pointRadius: 0,
                    tension: 0.1
                }]
            },
            options: {
                responsive: true,
                maintainAspectRatio: false,
                animation: false,
                scales: {
                    x: { display: false },
                    y: {
                        grid: { color: '#1e293b' },
                        ticks: { color: '#94a3b8', font: { size: 10 } }
                    }
                },
                plugins: { legend: { display: false } }
            }
        });
    }

    const ctxThreat = document.getElementById('threatChart')?.getContext('2d');
    if (ctxThreat) {
        threatChart = new Chart(ctxThreat, {
            type: 'doughnut',
            data: {
                labels: ['Normal'],
                datasets: [{
                    data: [1],
                    backgroundColor: ['#10b981', '#f59e0b', '#ef4444', '#8b5cf6', '#06b6d4', '#ec4899'],
                    borderWidth: 0
                }]
            },
            options: {
                responsive: true,
                maintainAspectRatio: false,
                plugins: {
                    legend: {
                        position: 'bottom',
                        labels: { color: '#94a3b8', boxWidth: 8, font: { size: 10 } }
                    }
                }
            }
        });
    }
}

function appendConsole(text, type = 'info') {
    if (!consoleStream) return;
    const entry = document.createElement('div');
    entry.className = `log-${type}`;
    entry.textContent = text;
    consoleStream.appendChild(entry);
    consoleStream.scrollTop = consoleStream.scrollHeight;
}

// ================= STREAM TELEMETRY =================
function startStreamTelemetry() {
    if (eventSource) {
        eventSource.close();
    }

    const dataset = datasetSelect ? datasetSelect.value : 'UNSW_NB15_testing-set.csv';
    const speed = streamSpeedInput ? streamSpeedInput.value : 200;
    const limit = flowLimitInput ? flowLimitInput.value : 3000;
    const k = thresholdKInput ? thresholdKInput.value : 3.0;

    if (scoreChart) {
        scoreChart.data.labels = [];
        scoreChart.data.datasets[0].data = [];
        scoreChart.data.datasets[1].data = [];
        scoreChart.update();
    }

    if (latencyChart) {
        latencyChart.data.labels = [];
        latencyChart.data.datasets[0].data = [];
        latencyChart.update();
    }

    threatData = { 'Normal': 0 };
    if (threatChart) {
        threatChart.data.labels = ['Normal'];
        threatChart.data.datasets[0].data = [0];
        threatChart.update();
    }

    if (btnStartStream) btnStartStream.disabled = true;
    if (btnStopStream) btnStopStream.disabled = false;
    if (statusText) statusText.textContent = 'Streaming Telemetry...';
    const dot = document.getElementById('system-status-indicator')?.querySelector('.status-dot');
    if (dot) dot.className = 'status-dot yellow';

    appendConsole(`[SYSTEM] Starting live telemetry: dataset=${dataset}, speed=${speed} flows/s, k=${k}`, 'system');

    const url = `/api/stream?dataset_name=${encodeURIComponent(dataset)}&speed=${speed}&limit=${limit}&threshold_k=${k}`;
    eventSource = new EventSource(url);

        eventSource.onmessage = (event) => {
            const data = JSON.parse(event.data);

            if (data.type === 'system') {
                appendConsole(`[SYSTEM] ${data.message}`, 'system');
            } else if (data.type === 'alert') {
                appendConsole(`[ALERT] ${data.text}`, 'alert');
            } else if (data.type === 'progress') {
                if (kpiF1) kpiF1.textContent = Number(data.f1_score).toFixed(4);
                if (kpiF1Sub) kpiF1Sub.textContent = `Flow #${data.index} of ${limit}`;
                if (kpiPrecRec) kpiPrecRec.textContent = `${Number(data.precision).toFixed(3)} / ${Number(data.recall).toFixed(3)}`;
                if (kpiPrecRecSub) kpiPrecRecSub.textContent = `P: ${(data.precision*100).toFixed(1)}% | R: ${(data.recall*100).toFixed(1)}%`;
                if (kpiAnomalies) kpiAnomalies.textContent = data.anomalies;
                if (kpiAnomalyRate) kpiAnomalyRate.textContent = `Drift Rate: ${((data.anomalies / data.processed)*100).toFixed(1)}%`;
                if (kpiLatency) kpiLatency.textContent = `${Number(data.latency_ms).toFixed(2)} ms`;
                if (kpiThroughput) kpiThroughput.textContent = `${Math.round(data.throughput)} flows/s`;

                if (scoreChart) {
                    scoreChart.data.labels.push(data.index);
                    scoreChart.data.datasets[0].data.push(data.score);
                    scoreChart.data.datasets[1].data.push(data.threshold);
                    if (scoreChart.data.labels.length > 50) {
                        scoreChart.data.labels.shift();
                        scoreChart.data.datasets[0].data.shift();
                        scoreChart.data.datasets[1].data.shift();
                    }
                    scoreChart.update();
                }

                if (latencyChart) {
                    latencyChart.data.labels.push(data.index);
                    latencyChart.data.datasets[0].data.push(data.latency_ms);
                    if (latencyChart.data.labels.length > 50) {
                        latencyChart.data.labels.shift();
                        latencyChart.data.datasets[0].data.shift();
                    }
                    latencyChart.update();
                }

                const cat = data.threat_category || 'Normal';
                threatData[cat] = (threatData[cat] || 0) + 1;
                if (threatChart) {
                    threatChart.data.labels = Object.keys(threatData);
                    threatChart.data.datasets[0].data = Object.values(threatData);
                    threatChart.update();
                }
            } else if (data.type === 'complete') {
                appendConsole(`[COMPLETE] Finished: ${data.total_processed} flows, F1=${Number(data.f1_score).toFixed(4)}, P=${Number(data.precision).toFixed(4)}, R=${Number(data.recall).toFixed(4)}`, 'system');
                cleanupStreamUI();
            } else if (data.type === 'error') {
                appendConsole(`[ERROR] ${data.message}`, 'alert');
                cleanupStreamUI();
            }
        };

        eventSource.onerror = () => {
            appendConsole('[SYSTEM] SSE stream ended.', 'info');
            cleanupStreamUI();
        };
}

async function stopStreamTelemetry() {
    try {
        await fetch('/api/stop', { method: 'POST' });
        appendConsole('[SYSTEM] Ingestion paused by operator.', 'system');
    } catch (e) {}
    cleanupStreamUI();
}

if (btnStopStream) {
    btnStopStream.addEventListener('click', stopStreamTelemetry);
}

function cleanupStreamUI() {
    if (eventSource) {
        eventSource.close();
        eventSource = null;
    }
    if (btnStartStream) btnStartStream.disabled = false;
    if (btnStopStream) btnStopStream.disabled = true;
    if (statusText) statusText.textContent = 'System Standby';
    const dot = document.getElementById('system-status-indicator')?.querySelector('.status-dot');
    if (dot) dot.className = 'status-dot green';
}

// ================= BENCHMARK HARNESS RUNNER =================
if (btnRunBench) {
    btnRunBench.addEventListener('click', async () => {
        const dataset = benchDatasetSelect.value;
        const model = benchModelSelect.value;
        const limit = benchLimit.value;

        evalStatusBox.textContent = `Evaluating ${model} on ${dataset}...`;
        evalStatusBox.style.color = '#3b82f6';
        btnRunBench.disabled = true;

        try {
            const res = await fetch(`/api/evaluate_benchmark?dataset_name=${encodeURIComponent(dataset)}&baseline_model=${encodeURIComponent(model)}&limit=${limit}`, {
                method: 'POST'
            });
            const data = await res.json();

            if (data.status === 'success') {
                const m = data.metrics;
                evalStatusBox.textContent = `Done: Run ${data.run_id}`;
                evalStatusBox.style.color = '#10b981';

                benchResultCards.style.display = 'grid';
                document.getElementById('bench-res-f1').textContent = Number(m.f1_score || 0).toFixed(4);
                document.getElementById('bench-res-prec-rec').textContent = `${Number(m.precision || 0).toFixed(4)} / ${Number(m.recall || 0).toFixed(4)}`;
                document.getElementById('bench-res-auc').textContent = `${Number(m.auroc || 0).toFixed(4)} / ${Number(m.pr_auc || 0).toFixed(4)}`;
                document.getElementById('bench-res-fpr95').textContent = Number(m.fpr_at_95_tpr || 0).toFixed(4);

                loadBenchmarkTable();
            } else {
                evalStatusBox.textContent = `Failed: ${data.message}`;
                evalStatusBox.style.color = '#ef4444';
            }
        } catch (err) {
            evalStatusBox.textContent = `Error: ${err.message}`;
            evalStatusBox.style.color = '#ef4444';
        } finally {
            btnRunBench.disabled = false;
        }
    });
}

// ================= BENCHMARK MATRIX TABLE =================
async function loadBenchmarkTable() {
    if (!matrixTableBody) return;
    matrixTableBody.innerHTML = '<tr><td colspan="11" style="text-align: center; color: var(--text-muted); padding: 20px;">Loading empirical benchmark results...</td></tr>';
    try {
        const res = await fetch('/api/benchmarks_summary');
        const data = await res.json();
        if (data.status === 'success') {
            allBenchmarkRuns = data.runs;
            renderBenchmarkTable(allBenchmarkRuns);
        }
    } catch (err) {
        matrixTableBody.innerHTML = `<tr><td colspan="11" style="text-align: center; color: #ef4444; padding: 20px;">Failed to load benchmark table: ${err.message}</td></tr>`;
    }
}

function renderBenchmarkTable(runs) {
    if (!matrixTableBody) return;
    if (!runs || runs.length === 0) {
        matrixTableBody.innerHTML = '<tr><td colspan="11" style="text-align: center; color: var(--text-muted); padding: 20px;">No evaluated benchmarks found. Run an evaluation above.</td></tr>';
        return;
    }

    const query = (tableSearch?.value || '').toLowerCase();
    const filtered = runs.filter(r => 
        r.dataset.toLowerCase().includes(query) || 
        r.baseline.toLowerCase().includes(query) ||
        r.run_id.toLowerCase().includes(query)
    );

    matrixTableBody.innerHTML = filtered.map(r => {
        const isOurs = r.baseline.includes('hdc-lnn');
        const f1 = Number(r.f1_score).toFixed(4);
        let badgeClass = 'metric-bad';
        if (r.f1_score >= 0.85) badgeClass = 'metric-good';
        else if (r.f1_score >= 0.65) badgeClass = 'metric-avg';

        return `
            <tr class="${isOurs ? 'highlight' : ''}">
                <td><strong>${r.baseline.toUpperCase()}</strong></td>
                <td>${r.dataset}</td>
                <td><span class="metric-badge ${badgeClass}">${f1}</span></td>
                <td>${Number(r.precision).toFixed(4)}</td>
                <td>${Number(r.recall).toFixed(4)}</td>
                <td>${Number(r.auroc).toFixed(4)}</td>
                <td>${Number(r.pr_auc).toFixed(4)}</td>
                <td>${Number(r.fpr_at_95_tpr).toFixed(4)}</td>
                <td>${Number(r.latency_ms_per_flow).toFixed(2)} ms</td>
                <td>${Math.round(r.throughput_flows_sec)} /s</td>
                <td>${Math.round(r.peak_rss_mb)} MB</td>
            </tr>
        `;
    }).join('');
}

if (tableSearch) {
    tableSearch.addEventListener('input', () => {
        renderBenchmarkTable(allBenchmarkRuns);
    });
}

window.addEventListener('DOMContentLoaded', () => {
    initCharts();
});
