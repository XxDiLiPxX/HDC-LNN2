// HYDRA-LNN v2: Web Dashboard Client Javascript

// 1. Dom elements selectors
const dropZone = document.getElementById('drop-zone');
const fileInput = document.getElementById('csv-file-input');
const fileName = document.getElementById('selected-file-name');
const btnStart = document.getElementById('btn-start');
const btnStop = document.getElementById('btn-stop');
const speedLimit = document.getElementById('speed-limit');
const limitRows = document.getElementById('limit-rows');
const driftThreshold = document.getElementById('drift-threshold');
const thresholdVal = document.getElementById('threshold-val');
const btnClearTerminal = document.getElementById('btn-clear-terminal');
const alertsConsole = document.getElementById('alerts-console');
const statusIndicator = document.getElementById('system-status-indicator');
const statusText = document.getElementById('system-status-text');

// KPI elements selectors
const kpiProcessed = document.getElementById('kpi-processed');
const kpiProgress = document.getElementById('kpi-progress');
const kpiLatency = document.getElementById('kpi-latency');
const kpiJitter = document.getElementById('kpi-jitter');
const kpiAnomalies = document.getElementById('kpi-anomalies');
const kpiRate = document.getElementById('kpi-rate');
const kpiThroughput = document.getElementById('kpi-throughput');

// Stream state tracker
let activeEventSource = null;
let currentUploadedFileName = null;
let totalProcessed = 0;
let totalAnomalies = 0;
let latencies = [];
let jitterSum = 0.0;
let lastLatency = 0.0;

// Chart.js references
let scoreChart = null;
let latencyChart = null;
let threatChart = null;

// Track threat distributions
let threatCounts = {
    "Normal": 0,
    "Anomalous (Generic)": 0,
};

// 2. Slider threshold listener
driftThreshold.addEventListener('input', (e) => {
    thresholdVal.textContent = parseFloat(e.target.value).toFixed(1);
});

// 3. Drag & Drop CSV dataset upload listeners
dropZone.addEventListener('click', () => fileInput.click());

fileInput.addEventListener('change', (e) => {
    if (e.target.files.length > 0) {
        handleSelectedFile(e.target.files[0]);
    }
});

dropZone.addEventListener('dragover', (e) => {
    e.preventDefault();
    dropZone.classList.add('drag-over');
});

dropZone.addEventListener('dragleave', () => {
    dropZone.classList.remove('drag-over');
});

dropZone.addEventListener('drop', (e) => {
    e.preventDefault();
    dropZone.classList.remove('drag-over');
    if (e.dataTransfer.files.length > 0) {
        handleSelectedFile(e.dataTransfer.files[0]);
    }
});

function handleSelectedFile(file) {
    if (!file.name.endsWith('.csv')) {
        alert('Please select a valid CSV dataset file.');
        return;
    }
    fileName.textContent = file.name;
    
    // Upload file immediately via fetch
    const formData = new FormData();
    formData.append('file', file);
    
    updateStatus('orange', 'Uploading File...');
    
    fetch('/api/upload', {
        method: 'POST',
        body: formData
    })
    .then(res => res.json())
    .then(data => {
        if (data.status === 'success') {
            currentUploadedFileName = file.name;
            btnStart.disabled = false;
            updateStatus('green', 'Ready to Ingest');
            logTerminalLine(`[SYSTEM] Loaded dataset '${file.name}' successfully. Press Start to stream.`, 'system-line');
        } else {
            alert('Upload failed: ' + data.message);
            updateStatus('red', 'Upload Error');
        }
    })
    .catch(err => {
        console.error(err);
        alert('Error uploading file: ' + err.message);
        updateStatus('red', 'Server Disconnected');
    });
}

// Status updating utility
function updateStatus(dotClass, text) {
    const dot = statusIndicator.querySelector('.status-dot');
    dot.className = `status-dot ${dotClass}`;
    statusText.textContent = text;
}

// Log line printing
function logTerminalLine(text, cssClass) {
    const line = document.createElement('div');
    line.className = `console-line ${cssClass}`;
    line.textContent = text;
    alertsConsole.appendChild(line);
    alertsConsole.scrollTop = alertsConsole.scrollHeight;
    
    // Cap log lines inside browser to prevent memory bloat
    if (alertsConsole.children.length > 200) {
        alertsConsole.removeChild(alertsConsole.firstChild);
    }
}

btnClearTerminal.addEventListener('click', () => {
    alertsConsole.innerHTML = '<div class="console-line system-line">[SYSTEM] Console cleared. Waiting for events...</div>';
});

// 4. Initialize charts
function initCharts() {
    // Score Chart (Line)
    const ctxScore = document.getElementById('scoreChart').getContext('2d');
    scoreChart = new Chart(ctxScore, {
        type: 'line',
        data: {
            labels: [],
            datasets: [
                {
                    label: 'Calculated Drift Score',
                    data: [],
                    borderColor: '#ff9f43',
                    borderWidth: 2,
                    tension: 0.15,
                    pointRadius: 0
                },
                {
                    label: 'Manifold Threshold',
                    data: [],
                    borderColor: '#ff4757',
                    borderWidth: 1.5,
                    borderDash: [5, 5],
                    pointRadius: 0
                }
            ]
        },
        options: {
            responsive: true,
            maintainAspectRatio: false,
            animation: false,
            scales: {
                x: { display: false },
                y: { grid: { color: '#24314c' }, ticks: { color: '#8892b0' } }
            },
            plugins: {
                legend: { labels: { color: '#e2e8f0' } }
            }
        }
    });

    // Latency Chart (Line)
    const ctxLatency = document.getElementById('latencyChart').getContext('2d');
    latencyChart = new Chart(ctxLatency, {
        type: 'line',
        data: {
            labels: [],
            datasets: [{
                label: 'Latency (ms)',
                data: [],
                borderColor: '#00f2fe',
                borderWidth: 2,
                tension: 0.1,
                pointRadius: 0
            }]
        },
        options: {
            responsive: true,
            maintainAspectRatio: false,
            animation: false,
            scales: {
                x: { display: false },
                y: { grid: { color: '#24314c' }, ticks: { color: '#8892b0' } }
            },
            plugins: { legend: { display: false } }
        }
    });

    // Threat Category Distribution Chart (Doughnut)
    const ctxThreat = document.getElementById('threatChart').getContext('2d');
    threatChart = new Chart(ctxThreat, {
        type: 'doughnut',
        data: {
            labels: Object.keys(threatCounts),
            datasets: [{
                data: Object.values(threatCounts),
                backgroundColor: ['#2ed573', '#ff4757', '#ff9f43', '#a55eea', '#00f2fe', '#ffa502'],
                borderWidth: 0
            }]
        },
        options: {
            responsive: true,
            maintainAspectRatio: false,
            plugins: {
                legend: {
                    position: 'bottom',
                    labels: { color: '#e2e8f0', boxWidth: 12 }
                }
            }
        }
    });
}

// Helper to push update data to charts dynamically
function updateCharts(flowIndex, score, threshold, isAnomaly, latency, category) {
    // 1. Update Score Chart (cap at 50 data points)
    scoreChart.data.labels.push(flowIndex);
    scoreChart.data.datasets[0].data.push(score);
    scoreChart.data.datasets[1].data.push(threshold);
    if (scoreChart.data.labels.length > 50) {
        scoreChart.data.labels.shift();
        scoreChart.data.datasets[0].data.shift();
        scoreChart.data.datasets[1].data.shift();
    }
    scoreChart.update();

    // 2. Update Latency Chart (cap at 50 data points)
    latencyChart.data.labels.push(flowIndex);
    latencyChart.data.datasets[0].data.push(latency);
    if (latencyChart.data.labels.length > 50) {
        latencyChart.data.labels.shift();
        latencyChart.data.datasets[0].data.shift();
    }
    latencyChart.update();

    // 3. Update Threat Breakdown
    let key = isAnomaly ? (category || "Anomalous (Generic)") : "Normal";
    if (!threatCounts[key]) {
        threatCounts[key] = 0;
    }
    threatCounts[key]++;
    
    threatChart.data.labels = Object.keys(threatCounts);
    threatChart.data.datasets[0].data = Object.values(threatCounts);
    threatChart.update();
}

// 5. Start / Stop Stream Actions
btnStart.addEventListener('click', () => {
    // Clear state
    totalProcessed = 0;
    totalAnomalies = 0;
    latencies = [];
    jitterSum = 0.0;
    lastLatency = 0.0;
    threatCounts = { "Normal": 0 };
    
    // Reset KPIs
    kpiProcessed.textContent = "0";
    kpiProgress.textContent = "0% of target";
    kpiLatency.textContent = "0.00 ms";
    kpiJitter.textContent = "Jitter: 0.00 ms";
    kpiAnomalies.textContent = "0";
    kpiRate.textContent = "Drift Rate: 0.0%";
    kpiThroughput.textContent = "0.0 /s";
    
    // Clear and redraw charts
    if (scoreChart) scoreChart.destroy();
    if (latencyChart) latencyChart.destroy();
    if (threatChart) threatChart.destroy();
    initCharts();
    
    logTerminalLine("[SYSTEM] Initializing Streaming Ingest Sidecar loop...", "system-line");
    updateStatus('orange', 'Ingesting...');
    
    btnStart.disabled = true;
    btnStop.disabled = false;
    dropZone.style.pointerEvents = 'none';
    
    // Build query args
    const speed = speedLimit.value || 100;
    const limit = limitRows.value || 5000;
    const k = driftThreshold.value || 3.0;
    
    // Initialize SSE streaming EventSource
    const sseUrl = `/api/stream?speed=${speed}&limit=${limit}&k=${k}`;
    activeEventSource = new EventSource(sseUrl);
    
    activeEventSource.onmessage = (event) => {
        const data = JSON.parse(event.data);
        
        if (data.type === 'progress') {
            totalProcessed = data.processed;
            totalAnomalies = data.anomalies;
            
            // Statistics calculation
            const progressPercent = Math.min(100, Math.round((totalProcessed / limit) * 100));
            kpiProcessed.textContent = totalProcessed;
            kpiProgress.textContent = `${progressPercent}% of target`;
            kpiAnomalies.textContent = totalAnomalies;
            kpiRate.textContent = `Drift Rate: ${((totalAnomalies / totalProcessed) * 100).toFixed(1)}%`;
            kpiThroughput.textContent = `${data.throughput.toFixed(1)} /s`;
            
            // Latency statistics
            const lat = data.latency * 1000.0; // convert to ms
            latencies.push(lat);
            if (latencies.length > 200) latencies.shift();
            
            const avgLat = latencies.reduce((a, b) => a + b, 0) / latencies.length;
            kpiLatency.textContent = `${avgLat.toFixed(2)} ms`;
            
            // Jitter calculation (average difference between sequential latencies)
            if (lastLatency > 0.0) {
                const jitter = Math.abs(lat - lastLatency);
                jitterSum = 0.95 * jitterSum + 0.05 * jitter; // EWMA jitter
            }
            lastLatency = lat;
            kpiJitter.textContent = `Jitter: ${jitterSum.toFixed(2)} ms`;
            
            // Update charts
            updateCharts(data.index, data.score, data.threshold, data.is_anomaly, lat, data.attack_cat);
            
        } else if (data.type === 'alert') {
            // Print alert in console
            logTerminalLine(`🚨 [CEF ALERT] ${data.alert_text}`, 'alert-line');
            
        } else if (data.type === 'system') {
            logTerminalLine(`[SYSTEM] ${data.message}`, 'system-line');
            
        } else if (data.type === 'complete') {
            logTerminalLine(`[SYSTEM] Ingestion simulation completed successfully.`, 'system-line');
            logTerminalLine(`[SYSTEM] Stats: ${data.total_processed} flows processed. ${data.total_anomalies} anomalies alerts triggered. Avg Latency: ${data.avg_latency_ms.toFixed(2)} ms.`, 'system-line');
            stopSimulation('Standby', 'green');
        }
    };
    
    activeEventSource.onerror = (err) => {
        console.error("SSE stream error: ", err);
        logTerminalLine("[SYSTEM ERROR] SSE connection dropped unexpectedly.", "alert-line");
        stopSimulation('Disconnected', 'red');
    };
});

btnStop.addEventListener('click', () => {
    logTerminalLine("[SYSTEM] Stopping active ingestion manually...", "system-line");
    fetch('/api/stop', { method: 'POST' })
        .then(() => {
            stopSimulation('Standby', 'green');
            logTerminalLine("[SYSTEM] Ingestion loop stopped.", "system-line");
        });
});

function stopSimulation(statusLabel, statusColor) {
    if (activeEventSource) {
        activeEventSource.close();
        activeEventSource = null;
    }
    btnStart.disabled = false;
    btnStop.disabled = true;
    dropZone.style.pointerEvents = 'auto';
    updateStatus(statusColor, statusLabel);
}

// Window load init
window.addEventListener('load', () => {
    initCharts();
});
