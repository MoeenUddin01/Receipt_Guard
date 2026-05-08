/**
 * ReceiptGuard UI - Frontend JavaScript
 */

const API_URL = 'http://localhost:8000';

// DOM Elements
const dropZone = document.getElementById('dropZone');
const fileInput = document.getElementById('fileInput');
const uploadSection = document.getElementById('uploadSection');
const loadingState = document.getElementById('loadingState');
const resultsSection = document.getElementById('resultsSection');
const errorState = document.getElementById('errorState');
const previewImage = document.getElementById('previewImage');
const toast = document.getElementById('toast');
const modelStatus = document.getElementById('modelStatus');

// Event Listeners
dropZone.addEventListener('click', () => fileInput.click());
dropZone.addEventListener('dragover', handleDragOver);
dropZone.addEventListener('dragleave', handleDragLeave);
dropZone.addEventListener('drop', handleDrop);
fileInput.addEventListener('change', handleFileSelect);

// Check model status on load
checkModelStatus();

function handleDragOver(e) {
    e.preventDefault();
    e.stopPropagation();
    dropZone.classList.add('dragover');
}

function handleDragLeave(e) {
    e.preventDefault();
    e.stopPropagation();
    dropZone.classList.remove('dragover');
}

function handleDrop(e) {
    e.preventDefault();
    e.stopPropagation();
    dropZone.classList.remove('dragover');
    
    const files = e.dataTransfer.files;
    if (files.length > 0) {
        processFile(files[0]);
    }
}

function handleFileSelect(e) {
    const file = e.target.files[0];
    if (file) {
        processFile(file);
    }
}

async function processFile(file) {
    // Validate file type
    const allowedTypes = ['image/png', 'image/jpeg', 'image/jpg'];
    if (!allowedTypes.includes(file.type)) {
        showError('Please upload a valid image file (PNG, JPG, JPEG)');
        return;
    }
    
    // Show loading state
    showLoading();
    
    try {
        // Create preview
        const reader = new FileReader();
        reader.onload = (e) => {
            previewImage.src = e.target.result;
        };
        reader.readAsDataURL(file);
        
        // Send to API
        const formData = new FormData();
        formData.append('file', file);
        
        const response = await fetch(`${API_URL}/extract`, {
            method: 'POST',
            body: formData
        });
        
        if (!response.ok) {
            throw new Error(`API Error: ${response.status}`);
        }
        
        const result = await response.json();
        
        if (result.success) {
            displayResults(result);
        } else {
            showError(result.error || 'Extraction failed');
        }
        
    } catch (error) {
        console.error('Error:', error);
        showError('Failed to process receipt. Is the backend running?');
    }
}

function displayResults(result) {
    // Update fields
    updateField('company', result.company);
    updateField('date', result.date);
    updateField('address', result.address);
    updateField('total', result.total);
    
    // Show results
    showResults();
    
    // Draw bounding boxes if available
    if (result.company?.bbox) drawBoundingBox('company', result.company.bbox);
    if (result.date?.bbox) drawBoundingBox('date', result.date.bbox);
    if (result.address?.bbox) drawBoundingBox('address', result.address.bbox);
    if (result.total?.bbox) drawBoundingBox('total', result.total.bbox);
}

function updateField(fieldName, data) {
    const valueEl = document.getElementById(`${fieldName}Value`);
    const confidenceEl = document.getElementById(`${fieldName}Confidence`);
    
    if (data && data.text) {
        valueEl.textContent = data.text;
        valueEl.classList.add('has-value');
        
        if (data.confidence) {
            const confidence = Math.round(data.confidence * 100);
            confidenceEl.textContent = `${confidence}%`;
            confidenceEl.classList.toggle('low', confidence < 80);
        }
    } else {
        valueEl.textContent = 'Not detected';
        valueEl.classList.remove('has-value');
        confidenceEl.textContent = '';
    }
}

function drawBoundingBox(type, bbox) {
    const container = document.querySelector('.image-container');
    const box = document.createElement('div');
    box.className = `bbox-overlay bbox-${type}`;
    
    // Convert bbox coordinates to percentages
    const img = previewImage;
    const x = (bbox[0] / img.naturalWidth) * 100;
    const y = (bbox[1] / img.naturalHeight) * 100;
    const w = ((bbox[2] - bbox[0]) / img.naturalWidth) * 100;
    const h = ((bbox[3] - bbox[1]) / img.naturalHeight) * 100;
    
    box.style.left = `${x}%`;
    box.style.top = `${y}%`;
    box.style.width = `${w}%`;
    box.style.height = `${h}%`;
    
    // Add label
    const label = document.createElement('span');
    label.textContent = type.charAt(0).toUpperCase() + type.slice(1);
    label.style.cssText = `
        position: absolute;
        top: -20px;
        left: 0;
        background: inherit;
        color: white;
        padding: 2px 8px;
        font-size: 12px;
        border-radius: 4px;
        text-transform: uppercase;
    `;
    box.appendChild(label);
    
    document.getElementById('boundingBoxes').appendChild(box);
}

// UI State Management
function showLoading() {
    uploadSection.style.display = 'none';
    loadingState.style.display = 'block';
    resultsSection.style.display = 'none';
    errorState.style.display = 'none';
}

function showResults() {
    uploadSection.style.display = 'none';
    loadingState.style.display = 'none';
    resultsSection.style.display = 'block';
    errorState.style.display = 'none';
}

function showError(message) {
    uploadSection.style.display = 'none';
    loadingState.style.display = 'none';
    resultsSection.style.display = 'none';
    errorState.style.display = 'block';
    document.getElementById('errorMessage').textContent = message;
}

function resetApp() {
    // Clear bounding boxes
    document.getElementById('boundingBoxes').innerHTML = '';
    
    // Reset fields
    ['company', 'date', 'address', 'total'].forEach(field => {
        document.getElementById(`${field}Value`).textContent = '-';
        document.getElementById(`${field}Confidence`).textContent = '';
    });
    
    // Reset file input
    fileInput.value = '';
    
    // Show upload
    uploadSection.style.display = 'block';
    loadingState.style.display = 'none';
    resultsSection.style.display = 'none';
    errorState.style.display = 'none';
}

// Copy functionality
function copyField(fieldName) {
    const value = document.getElementById(`${fieldName}Value`).textContent;
    if (value && value !== '-' && value !== 'Not detected') {
        navigator.clipboard.writeText(value).then(() => {
            showToast(`${fieldName.charAt(0).toUpperCase() + fieldName.slice(1)} copied!`);
        });
    }
}

function copyAll() {
    const fields = ['company', 'date', 'address', 'total'];
    const data = {};
    
    fields.forEach(field => {
        const value = document.getElementById(`${field}Value`).textContent;
        if (value && value !== '-' && value !== 'Not detected') {
            data[field] = value;
        }
    });
    
    const text = Object.entries(data)
        .map(([key, value]) => `${key}: ${value}`)
        .join('\n');
    
    navigator.clipboard.writeText(text).then(() => {
        showToast('All fields copied!');
    });
}

function showToast(message) {
    toast.textContent = message;
    toast.classList.add('show');
    
    setTimeout(() => {
        toast.classList.remove('show');
    }, 2000);
}

// Model status check
async function checkModelStatus() {
    try {
        const response = await fetch(`${API_URL}/health`);
        const data = await response.json();
        
        if (data.model_loaded) {
            modelStatus.textContent = `✅ Model ready on ${data.device}`;
            modelStatus.style.color = '#10b981';
        } else {
            modelStatus.textContent = '⚠️ Model not loaded';
            modelStatus.style.color = '#f59e0b';
        }
    } catch (error) {
        modelStatus.textContent = '❌ Backend offline';
        modelStatus.style.color = '#ef4444';
    }
}

// Keyboard shortcuts
document.addEventListener('keydown', (e) => {
    if (e.key === 'Escape' && resultsSection.style.display === 'block') {
        resetApp();
    }
});
