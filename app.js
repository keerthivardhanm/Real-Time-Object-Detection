// ============================================================
// YOLOv8 Object Detector - JavaScript
// ============================================================

// ============================================================
// APPLICATION STATE
// ============================================================
const state = {
    model: null,
    modelLoaded: false,
    currentMode: 'image',
    isProcessing: false,
    webcamActive: false,
    webcamStream: null,
    currentImage: null,
    detectionHistory: [],
    classStats: {},
    totalImages: 0,
    totalObjects: 0,
    processingTimes: [],
    pieChart: null,
    barChart: null
};

// COCO-SSD Class Names
const CLASS_NAMES = [
    'person', 'bicycle', 'car', 'motorcycle', 'airplane', 'bus', 'train', 'truck', 'boat',
    'traffic light', 'fire hydrant', 'stop sign', 'parking meter', 'bench', 'bird', 'cat',
    'dog', 'horse', 'sheep', 'cow', 'elephant', 'bear', 'zebra', 'giraffe', 'backpack',
    'umbrella', 'handbag', 'tie', 'suitcase', 'frisbee', 'skis', 'snowboard', 'sports ball',
    'kite', 'baseball bat', 'baseball glove', 'skateboard', 'surfboard', 'tennis racket',
    'bottle', 'wine glass', 'cup', 'fork', 'knife', 'spoon', 'bowl', 'banana', 'apple',
    'sandwich', 'orange', 'broccoli', 'carrot', 'hot dog', 'pizza', 'donut', 'cake', 'chair',
    'couch', 'potted plant', 'bed', 'dining table', 'toilet', 'tv', 'laptop', 'mouse', 'remote',
    'keyboard', 'cell phone', 'microwave', 'oven', 'toaster', 'sink', 'refrigerator', 'book',
    'clock', 'vase', 'scissors', 'teddy bear', 'hair drier', 'toothbrush'
];

// Color palette for classes
const COLORS = [
    '#00ff88', '#ff4444', '#4488ff', '#ffaa00', '#ff44ff', '#44ffff',
    '#88ff00', '#ff8888', '#8888ff', '#ffff44', '#44ff88', '#ff44aa'
];

// ============================================================
// DOM ELEMENTS
// ============================================================
const elements = {
    modelStatusDot: document.getElementById('modelStatusDot'),
    modelStatusText: document.getElementById('modelStatusText'),
    loadingOverlay: document.getElementById('loadingOverlay'),
    detectionArea: document.getElementById('detectionArea'),
    imageMode: document.getElementById('imageMode'),
    webcamMode: document.getElementById('webcamMode'),
    batchMode: document.getElementById('batchMode'),
    uploadZone: document.getElementById('uploadZone'),
    fileInput: document.getElementById('fileInput'),
    imageCanvas: document.getElementById('imageCanvas'),
    webcamVideo: document.getElementById('webcamVideo'),
    webcamCanvas: document.getElementById('webcamCanvas'),
    webcamPlaceholder: document.getElementById('webcamPlaceholder'),
    batchUploadZone: document.getElementById('batchUploadZone'),
    batchFileInput: document.getElementById('batchFileInput'),
    batchResults: document.getElementById('batchResults'),
    runBtn: document.getElementById('runBtn'),
    resultsSection: document.getElementById('resultsSection'),
    resultObjects: document.getElementById('resultObjects'),
    resultClasses: document.getElementById('resultClasses'),
    resultTime: document.getElementById('resultTime'),
    resultFPS: document.getElementById('resultFPS'),
    detectionTags: document.getElementById('detectionTags'),
    downloadBtn: document.getElementById('downloadBtn'),
    confidence: document.getElementById('confidence'),
    confValue: document.getElementById('confValue'),
    maxDetections: document.getElementById('maxDetections'),
    maxDetValue: document.getElementById('maxDetValue'),
    showLabels: document.getElementById('showLabels'),
    showConfidence: document.getElementById('showConfidence'),
    showFPS: document.getElementById('showFPS'),
    showBoxes: document.getElementById('showBoxes'),
    totalImages: document.getElementById('totalImages'),
    totalObjects: document.getElementById('totalObjects'),
    avgTime: document.getElementById('avgTime'),
    historyTable: document.getElementById('historyTable')
};

// --- Persistent Detection History ---
function loadDetectionHistory() {
    try {
        const data = localStorage.getItem('detectionHistory');
        if (data) {
            state.detectionHistory = JSON.parse(data);
        }
    } catch (e) {
        state.detectionHistory = [];
    }
}

function saveDetectionHistory() {
    try {
        localStorage.setItem('detectionHistory', JSON.stringify(state.detectionHistory));
    } catch (e) {}
}

// Patch addToHistory to save after update
const origAddToHistory = addToHistory;
addToHistory = function(imageName, predictions, time) {
    // Only add to history if new objects detected (not duplicate)
    const newClasses = [...new Set(predictions.map(p => p.class))].sort().join(',');
    if (state.detectionHistory.length === 0 || state.detectionHistory[0].classes !== newClasses || state.detectionHistory[0].objects !== predictions.length) {
        const entry = {
            time: new Date().toLocaleTimeString(),
            image: imageName,
            objects: predictions.length,
            classes: newClasses,
            timeMs: time.toFixed(0)
        };
        state.detectionHistory.unshift(entry);
        if (state.detectionHistory.length > 10) state.detectionHistory.pop();
        saveDetectionHistory();
    }
    // Update table
    if (state.detectionHistory.length === 0) {
        elements.historyTable.innerHTML = `
            <tr>
                <td colspan="5" style="text-align: center; color: var(--text-secondary);">
                    No detections yet
                </td>
            </tr>
        `;
    } else {
        elements.historyTable.innerHTML = state.detectionHistory.map(e => `
            <tr>
                <td>${e.time}</td>
                <td>${e.image}</td>
                <td>${e.objects}</td>
                <td>${e.classes}</td>
                <td>${e.timeMs}ms</td>
            </tr>
        `).join('');
    }
}

// ============================================================
// MODEL LOADING
// ============================================================
async function loadModel() {
    try {
        console.log('Loading COCO-SSD model...');
        state.model = await cocoSsd.load({
            base: 'lite_mobilenet_v2'
        });
        state.modelLoaded = true;
        elements.modelStatusDot.className = 'status-dot ready';
        elements.modelStatusText.textContent = 'Model Ready';
        elements.loadingOverlay.style.display = 'none';
        elements.runBtn.disabled = false;
        console.log('Model loaded successfully!');
    } catch (error) {
        console.error('Failed to load model:', error);
        elements.modelStatusDot.className = 'status-dot error';
        elements.modelStatusText.textContent = 'Model Error';
        elements.loadingOverlay.querySelector('.loading-text').textContent = 
            'Failed to load model. Please refresh.';
    }
}

// ============================================================
// MODE SWITCHING
// ============================================================
document.querySelectorAll('.mode-btn').forEach(btn => {
    btn.addEventListener('click', () => {
        document.querySelectorAll('.mode-btn').forEach(b => b.classList.remove('active'));
        btn.classList.add('active');
        state.currentMode = btn.dataset.mode;
        
        elements.imageMode.style.display = 'none';
        elements.webcamMode.style.display = 'none';
        elements.batchMode.style.display = 'none';
        
        document.getElementById(`${state.currentMode}Mode`).style.display = 'block';
        
        // Update run button
        if (state.currentMode === 'webcam') {
            elements.runBtn.innerHTML = '<span>▶️</span><span>Start Webcam</span>';
        } else if (state.currentMode === 'batch') {
            elements.runBtn.innerHTML = '<span>🔍</span><span>Process All Images</span>';
        } else {
            elements.runBtn.innerHTML = '<span>🔍</span><span>Run Detection</span>';
        }
        
        // Stop webcam if switching away
        if (state.currentMode !== 'webcam' && state.webcamActive) {
            stopWebcam();
        }
    });
});

// ============================================================
// IMAGE UPLOAD
// ============================================================
elements.uploadZone.addEventListener('click', () => elements.fileInput.click());

elements.uploadZone.addEventListener('dragover', (e) => {
    e.preventDefault();
    elements.uploadZone.classList.add('dragover');
});

elements.uploadZone.addEventListener('dragleave', () => {
    elements.uploadZone.classList.remove('dragover');
});

elements.uploadZone.addEventListener('drop', (e) => {
    e.preventDefault();
    elements.uploadZone.classList.remove('dragover');
    const file = e.dataTransfer.files[0];
    if (file && file.type.startsWith('image/')) {
        handleImageUpload(file);
    }
});

elements.fileInput.addEventListener('change', (e) => {
    if (e.target.files[0]) {
        handleImageUpload(e.target.files[0]);
    }
});

function handleImageUpload(file) {
    const reader = new FileReader();
    reader.onload = (e) => {
        const img = new Image();
        img.onload = () => {
            state.currentImage = img;
            const canvas = elements.imageCanvas;
            const ctx = canvas.getContext('2d');
            canvas.width = img.width;
            canvas.height = img.height;
            ctx.drawImage(img, 0, 0);
            elements.uploadZone.style.display = 'none';
            elements.imageCanvas.style.display = 'block';
            elements.runBtn.disabled = false;
        };
        img.src = e.target.result;
    };
    reader.readAsDataURL(file);
}

// ============================================================
// BATCH UPLOAD
// ============================================================
elements.batchUploadZone.addEventListener('click', () => elements.batchFileInput.click());

elements.batchFileInput.addEventListener('change', (e) => {
    if (e.target.files.length > 0) {
        handleBatchUpload(e.target.files);
    }
});

function handleBatchUpload(files) {
    state.batchFiles = Array.from(files);
    elements.batchUploadZone.style.display = 'none';
    elements.batchResults.style.display = 'block';
    elements.batchResults.innerHTML = `
        <div class="card">
            <div class="card-title">📁 ${files.length} Images Selected</div>
            <div style="display: grid; grid-template-columns: repeat(auto-fill, minmax(150px, 1fr)); gap: 1rem; margin-top: 1rem;">
                ${Array.from(files).map((f, i) => `
                    <div style="background: var(--bg-input); padding: 0.5rem; border-radius: 8px; text-align: center;">
                        <div style="font-size: 2rem;">🖼️</div>
                        <div style="font-size: 0.8rem; color: var(--text-secondary);">${f.name.substring(0, 15)}...</div>
                    </div>
                `).join('')}
            </div>
        </div>
    `;
    elements.runBtn.disabled = false;
}

// ============================================================
// WEBCAM
// ============================================================
async function startWebcam() {
    try {
        state.webcamStream = await navigator.mediaDevices.getUserMedia({ 
            video: { facingMode: 'environment' } 
        });
        elements.webcamVideo.srcObject = state.webcamStream;
        elements.webcamVideo.style.display = 'block';
        elements.webcamPlaceholder.style.display = 'none';
        elements.webcamCanvas.style.display = 'block';
        state.webcamActive = true;
        elements.runBtn.innerHTML = '<span>⏹️</span><span>Stop Webcam</span>';
        detectWebcam();
    } catch (error) {
        alert('Could not access webcam. Please ensure camera permissions are granted.');
        console.error('Webcam error:', error);
    }
}

function stopWebcam() {
    if (state.webcamStream) {
        state.webcamStream.getTracks().forEach(track => track.stop());
        state.webcamStream = null;
    }
    state.webcamActive = false;
    elements.webcamVideo.style.display = 'none';
    elements.webcamPlaceholder.style.display = 'block';
    elements.runBtn.innerHTML = '<span>▶️</span><span>Start Webcam</span>';
}

async function detectWebcam() {
    if (!state.webcamActive || !state.modelLoaded) return;

    const video = elements.webcamVideo;
    const canvas = elements.webcamCanvas;
    const ctx = canvas.getContext('2d');
    
    canvas.width = video.videoWidth || 640;
    canvas.height = video.videoHeight || 480;
    ctx.drawImage(video, 0, 0);

    const startTime = performance.now();
    const predictions = await state.model.detect(canvas);
    const processTime = performance.now() - startTime;

    // Filter by confidence
    const confThreshold = elements.confidence.value / 100;
    let filtered = predictions.filter(p => p.score >= confThreshold);
    
    // Apply Non-Maximum Suppression for advanced scanning
    filtered = nonMaxSuppression(filtered, 0.4);
    
    // Draw predictions with advanced visualization
    drawPredictionsAdvanced(ctx, filtered, canvas.width, canvas.height);

    // Update stats
    updateResults(filtered, processTime, 1000 / processTime);

    // Add to detection history for webcam (with timestamp and object count)
    if (filtered.length > 0) {
        const webcamEntry = {
            time: new Date().toLocaleTimeString(),
            image: 'Webcam Live',
            objects: filtered.length,
            classes: [...new Set(filtered.map(p => p.class))].sort().join(', '),
            timeMs: processTime.toFixed(0)
        };
        // Only add if different from last entry
        if (state.detectionHistory.length === 0 || 
            state.detectionHistory[0].objects !== webcamEntry.objects || 
            state.detectionHistory[0].classes !== webcamEntry.classes) {
            state.detectionHistory.unshift(webcamEntry);
            if (state.detectionHistory.length > 10) state.detectionHistory.pop();
            saveDetectionHistory();
            // Update history table
            elements.historyTable.innerHTML = state.detectionHistory.map(e => `
                <tr>
                    <td>${e.time}</td>
                    <td>${e.image}</td>
                    <td>${e.objects}</td>
                    <td>${e.classes}</td>
                    <td>${e.timeMs}ms</td>
                </tr>
            `).join('');
        }
    }

    requestAnimationFrame(detectWebcam);
}

// ============================================================
// DETECTION
// ============================================================
elements.runBtn.addEventListener('click', async () => {
    if (state.currentMode === 'webcam') {
        if (state.webcamActive) {
            stopWebcam();
        } else {
            await startWebcam();
        }
        return;
    }

    if (state.isProcessing) return;
    state.isProcessing = true;
    elements.runBtn.disabled = true;
    elements.runBtn.innerHTML = '<span>⏳</span><span>Processing...</span>';

    try {
        if (state.currentMode === 'image' && state.currentImage) {
            await detectImage(state.currentImage);
        } else if (state.currentMode === 'batch' && state.batchFiles) {
            await detectBatch(state.batchFiles);
        }
    } catch (error) {
        console.error('Detection error:', error);
        alert('Error during detection: ' + error.message);
    }

    state.isProcessing = false;
    elements.runBtn.disabled = false;
    elements.runBtn.innerHTML = '<span>🔍</span><span>Run Detection</span>';
});

// Advanced Non-Maximum Suppression
function nonMaxSuppression(predictions, iouThreshold = 0.5) {
    if (predictions.length === 0) return predictions;
    
    // Sort by confidence (highest first)
    const sorted = [...predictions].sort((a, b) => b.score - a.score);
    const keep = [];
    
    while (sorted.length > 0) {
        const best = sorted.shift();
        keep.push(best);
        
        // Remove overlapping boxes
        for (let i = sorted.length - 1; i >= 0; i--) {
            const box = sorted[i];
            const iou = calculateIOU(best.bbox, box.bbox);
            if (iou > iouThreshold) {
                sorted.splice(i, 1);
            }
        }
    }
    return keep;
}

function calculateIOU(box1, box2) {
    const [x1, y1, w1, h1] = box1;
    const [x2, y2, w2, h2] = box2;
    
    const xA = Math.max(x1, x2);
    const yA = Math.max(y1, y2);
    const xB = Math.min(x1 + w1, x2 + w2);
    const yB = Math.min(y1 + h1, y2 + h2);
    
    const interArea = Math.max(0, xB - xA) * Math.max(0, yB - yA);
    const box1Area = w1 * h1;
    const box2Area = w2 * h2;
    
    return interArea / (box1Area + box2Area - interArea);
}

async function detectImage(img) {
    const canvas = elements.imageCanvas;
    const ctx = canvas.getContext('2d');
    
    const startTime = performance.now();
    const predictions = await state.model.detect(canvas);
    const processTime = performance.now() - startTime;

    // Advanced filtering: confidence threshold
    const confThreshold = elements.confidence.value / 100;
    let filtered = predictions.filter(p => p.score >= confThreshold);
    
    // Apply Non-Maximum Suppression to remove overlapping boxes
    filtered = nonMaxSuppression(filtered, 0.4);
    
    // Limit max detections
    const maxDet = parseInt(elements.maxDetections.value);
    const limited = filtered.slice(0, maxDet);

    // Draw predictions with advanced visualization
    drawPredictionsAdvanced(ctx, limited, canvas.width, canvas.height);

    // Update results
    const fps = 1000 / processTime;
    updateResults(limited, processTime, fps);
    addToHistory('Single Image', limited, processTime);
    
    // Show results section
    elements.resultsSection.style.display = 'block';
}

async function detectBatch(files) {
    const resultsContainer = document.getElementById('batchResults');
    resultsContainer.innerHTML = '<div class="card"><div class="card-title">📊 Processing...</div></div>';
    
    let totalObjects = 0;
    let totalTime = 0;

    for (let i = 0; i < files.length; i++) {
        const file = files[i];
        const img = await loadImage(file);
        const canvas = document.createElement('canvas');
        const ctx = canvas.getContext('2d');
        canvas.width = img.width;
        canvas.height = img.height;
        ctx.drawImage(img, 0, 0);

        const startTime = performance.now();
        const predictions = await state.model.detect(canvas);
        const processTime = performance.now() - startTime;

        const confThreshold = elements.confidence.value / 100;
        const filtered = predictions.filter(p => p.score >= confThreshold);
        totalObjects += filtered.length;
        totalTime += processTime;

        // Draw on canvas
        drawPredictions(ctx, filtered, canvas.width, canvas.height);
    }

    // Update stats
    state.totalImages += files.length;
    state.totalObjects += totalObjects;
    state.processingTimes.push(totalTime / files.length);
    updateSessionStats();

    resultsContainer.innerHTML = `
        <div class="card">
            <div class="card-title">✅ Batch Complete</div>
            <div class="results-panel">
                <div class="result-card">
                    <div class="result-card-value">${files.length}</div>
                    <div class="result-card-label">Images Processed</div>
                </div>
                <div class="result-card">
                    <div class="result-card-value">${totalObjects}</div>
                    <div class="result-card-label">Total Objects</div>
                </div>
                <div class="result-card">
                    <div class="result-card-value">${(totalTime / files.length).toFixed(0)}ms</div>
                    <div class="result-card-label">Avg Time/Image</div>
                </div>
            </div>
        </div>
    `;
}

function loadImage(file) {
    return new Promise((resolve) => {
        const img = new Image();
        img.onload = () => resolve(img);
        img.src = URL.createObjectURL(file);
    });
}

// ============================================================
// ADVANCED DRAW PREDICTIONS
// ============================================================
function drawPredictionsAdvanced(ctx, predictions, width, height) {
    const showBoxes = elements.showBoxes.checked;
    const showLabels = elements.showLabels.checked;
    const showConf = elements.showConfidence.checked;
    const boxLineWidth = parseInt(elements.boxLineWidth?.value || 3);
    const fontSize = parseInt(elements.fontSize?.value || 14);

    predictions.forEach((pred, i) => {
        const [x, y, w, h] = pred.bbox;
        const color = COLORS[i % COLORS.length];

        // Draw advanced bounding box with glow effect
        if (showBoxes) {
            // Outer glow
            ctx.shadowColor = color;
            ctx.shadowBlur = 15;
            ctx.strokeStyle = color;
            ctx.lineWidth = boxLineWidth;
            ctx.strokeRect(x, y, w, h);
            
            // Reset shadow for fill
            ctx.shadowBlur = 0;
            
            // Semi-transparent fill
            ctx.fillStyle = hexToRgba(color, 0.2);
            ctx.fillRect(x, y, w, h);
            
            // Inner border
            ctx.strokeStyle = hexToRgba(color, 0.8);
            ctx.lineWidth = 1;
            ctx.strokeRect(x + 2, y + 2, w - 4, h - 4);
        }

        // Draw advanced label with professional styling
        if (showLabels) {
            let label = pred.class;
            if (showConf) {
                label += ` ${(pred.score * 100).toFixed(0)}%`;
            }

            // Label background with rounded corners effect
            ctx.fillStyle = color;
            const labelWidth = ctx.measureText(label).width + 16;
            ctx.fillRect(x, y - (fontSize + 16), labelWidth, fontSize + 12);
            
            // Label text
            ctx.fillStyle = '#000';
            ctx.font = `bold ${fontSize}px Inter, sans-serif`;
            ctx.fillText(label, x + 8, y - 8);
            
            // Confidence bar
            if (showConf) {
                const barWidth = labelWidth - 16;
                const barHeight = 4;
                ctx.fillStyle = 'rgba(0,0,0,0.3)';
                ctx.fillRect(x + 8, y - 4, barWidth, barHeight);
                ctx.fillStyle = '#fff';
                ctx.fillRect(x + 8, y - 4, barWidth * pred.score, barHeight);
            }
        }
    });

    // Add timestamp with professional styling
    ctx.fillStyle = 'rgba(0, 0, 0, 0.75)';
    roundRect(ctx, 12, height - 45, 220, 32, 6);
    ctx.fill();
    ctx.fillStyle = '#fff';
    ctx.font = '12px "JetBrains Mono", monospace';
    ctx.fillText(new Date().toLocaleString(), 20, height - 22);
    
    // Object count badge
    ctx.fillStyle = 'rgba(0, 255, 136, 0.9)';
    roundRect(ctx, width - 100, 12, 88, 28, 6);
    ctx.fill();
    ctx.fillStyle = '#000';
    ctx.font = 'bold 13px Inter, sans-serif';
    ctx.fillText(`${predictions.length} objects`, width - 90, 32);
}

// Helper function to draw rounded rectangles
function roundRect(ctx, x, y, w, h, r) {
    ctx.beginPath();
    ctx.moveTo(x + r, y);
    ctx.lineTo(x + w - r, y);
    ctx.quadraticCurveTo(x + w, y, x + w, y + r);
    ctx.lineTo(x + w, y + h - r);
    ctx.quadraticCurveTo(x + w, y + h, x + w - r, y + h);
    ctx.lineTo(x + r, y + h);
    ctx.quadraticCurveTo(x, y + h, x, y + h - r);
    ctx.lineTo(x, y + r);
    ctx.quadraticCurveTo(x, y, x + r, y);
    ctx.closePath();
}

// Helper to convert hex to rgba
function hexToRgba(hex, alpha) {
    const r = parseInt(hex.slice(1, 3), 16);
    const g = parseInt(hex.slice(3, 5), 16);
    const b = parseInt(hex.slice(5, 7), 16);
    return `rgba(${r}, ${g}, ${b}, ${alpha})`;
}

// ============================================================
// UPDATE RESULTS
// ============================================================
function updateResults(predictions, time, fps) {
    // Update result cards
    elements.resultObjects.textContent = predictions.length;
    elements.resultClasses.textContent = new Set(predictions.map(p => p.class)).size;
    elements.resultTime.textContent = `${time.toFixed(0)}ms`;
    elements.resultFPS.textContent = fps.toFixed(1);

    // Update detection tags
    const classCounts = {};
    predictions.forEach(p => {
        classCounts[p.class] = (classCounts[p.class] || 0) + 1;
    });

    elements.detectionTags.innerHTML = Object.entries(classCounts)
        .sort((a, b) => b[1] - a[1])
        .map(([cls, count]) => `
            <span class="detection-tag">
                ${cls}
                <span class="detection-tag-count">${count}</span>
            </span>
        `).join('');

    // Update charts
    updateCharts(classCounts, predictions.map(p => p.score));

    // Update session stats
    state.totalImages++;
    state.totalObjects += predictions.length;
    state.processingTimes.push(time);
    updateSessionStats();
}

function updateSessionStats() {
    elements.totalImages.textContent = state.totalImages;
    elements.totalObjects.textContent = state.totalObjects;
    const avg = state.processingTimes.length > 0 
        ? state.processingTimes.reduce((a, b) => a + b, 0) / state.processingTimes.length 
        : 0;
    elements.avgTime.textContent = `${avg.toFixed(0)}ms`;
}

// ============================================================
// CHARTS
// ============================================================
function updateCharts(classCounts, confidences) {
    // Pie Chart
    const pieCtx = document.getElementById('pieChart').getContext('2d');
    if (state.pieChart) state.pieChart.destroy();
    
    state.pieChart = new Chart(pieCtx, {
        type: 'doughnut',
        data: {
            labels: Object.keys(classCounts),
            datasets: [{
                data: Object.values(classCounts),
                backgroundColor: COLORS.slice(0, Object.keys(classCounts).length),
                borderWidth: 0
            }]
        },
        options: {
            responsive: true,
            plugins: {
                legend: {
                    position: 'right',
                    labels: { color: '#a0a0b0' }
                }
            }
        }
    });

    // Bar Chart (Confidence Distribution)
    const barCtx = document.getElementById('barChart').getContext('2d');
    if (state.barChart) state.barChart.destroy();

    const bins = new Array(10).fill(0);
    confidences.forEach(c => {
        const bin = Math.min(Math.floor(c * 10), 9);
        bins[bin]++;
    });

    state.barChart = new Chart(barCtx, {
        type: 'bar',
        data: {
            labels: bins.map((_, i) => `${i * 10}-${(i + 1) * 10}%`),
            datasets: [{
                label: 'Detections',
                data: bins,
                backgroundColor: '#00ff88',
                borderRadius: 4
            }]
        },
        options: {
            responsive: true,
            scales: {
                x: { 
                    ticks: { color: '#a0a0b0' },
                    grid: { color: '#2a2a4a' }
                },
                y: { 
                    ticks: { color: '#a0a0b0' },
                    grid: { color: '#2a2a4a' }
                }
            },
            plugins: {
                legend: { display: false }
            }
        }
    });
}

// ============================================================
// HISTORY
// ============================================================
function addToHistory(imageName, predictions, time) {
    // Only add to history if new objects detected (not duplicate)
    const newClasses = [...new Set(predictions.map(p => p.class))].sort().join(',');
    if (state.detectionHistory.length === 0 || state.detectionHistory[0].classes !== newClasses || state.detectionHistory[0].objects !== predictions.length) {
        const entry = {
            time: new Date().toLocaleTimeString(),
            image: imageName,
            objects: predictions.length,
            classes: newClasses,
            timeMs: time.toFixed(0)
        };
        state.detectionHistory.unshift(entry);
        if (state.detectionHistory.length > 10) state.detectionHistory.pop();
        saveDetectionHistory();
    }
    // Update table
    if (state.detectionHistory.length === 0) {
        elements.historyTable.innerHTML = `
            <tr>
                <td colspan="5" style="text-align: center; color: var(--text-secondary);">
                    No detections yet
                </td>
            </tr>
        `;
    } else {
        elements.historyTable.innerHTML = state.detectionHistory.map(e => `
            <tr>
                <td>${e.time}</td>
                <td>${e.image}</td>
                <td>${e.objects}</td>
                <td>${e.classes}</td>
                <td>${e.timeMs}ms</td>
            </tr>
        `).join('');
    }
}

// ============================================================
// DOWNLOAD
// ============================================================
elements.downloadBtn.addEventListener('click', () => {
    let canvas;
    if (state.currentMode === 'image') {
        canvas = elements.imageCanvas;
    } else if (state.currentMode === 'webcam') {
        canvas = elements.webcamCanvas;
    } else {
        return;
    }

    const link = document.createElement('a');
    link.download = `detected_${Date.now()}.png`;
    link.href = canvas.toDataURL('image/png');
    link.click();
});

// ============================================================
// SLIDERS
// ============================================================
elements.confidence.addEventListener('input', (e) => {
    elements.confValue.textContent = `${e.target.value}%`;
});

elements.maxDetections.addEventListener('input', (e) => {
    elements.maxDetValue.textContent = e.target.value;
});

// ============================================================
// INITIALIZE
// ============================================================
window.addEventListener('load', () => {
    loadModel();
    loadDetectionHistory();
    if (state.detectionHistory.length === 0) {
        elements.historyTable.innerHTML = `
            <tr>
                <td colspan="5" style="text-align: center; color: var(--text-secondary);">
                    No detections yet
                </td>
            </tr>
        `;
    } else {
        elements.historyTable.innerHTML = state.detectionHistory.map(e => `
            <tr>
                <td>${e.time}</td>
                <td>${e.image}</td>
                <td>${e.objects}</td>
                <td>${e.classes}</td>
                <td>${e.timeMs}ms</td>
            </tr>
        `).join('');
    }
});