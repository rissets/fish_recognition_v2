/**
 * Fish Recognition API Test Application
 * Comprehensive JavaScript application for testing all API features
 */

class FishRecognitionApp {
    constructor() {
        this.apiBase = '/api/v1';
        this.wsUrl = `wss://${window.location.host}/ws/recognition/`;
        this.ws = null;
        this.videoStream = null;
        this.isProcessing = false;
        this.stats = {
            processed: 0,
            successful: 0,
            fishDetected: 0,
            totalTime: 0
        };
        
        this.currentMode = 'image';
        this.settings = {
            includeFaces: true,
            includeSegmentation: true,
            includeVisualization: true,  // Enable visualization by default
            qualityThreshold: 0.3,
            processingMode: 'accuracy',
            autoProcess: true
        };
        
        // Face filter configuration
        this.faceFilterConfig = {
            enabled: true,
            iouThreshold: 0.3
        };
        
        this.init();
    }
    
    init() {
        this.setupEventListeners();
        this.syncInitialSettings();
        this.checkApiHealth();
        this.connectWebSocket();
        this.updateUI();
    }
    
    syncInitialSettings() {
        // Sync settings with HTML checkbox states
        const includeFacesEl = document.getElementById('includeFaces');
        const includeSegmentationEl = document.getElementById('includeSegmentation');
        const includeVisualizationEl = document.getElementById('includeVisualization');
        const qualityThresholdEl = document.getElementById('qualityThreshold');
        
        if (includeFacesEl) this.settings.includeFaces = includeFacesEl.checked;
        if (includeSegmentationEl) this.settings.includeSegmentation = includeSegmentationEl.checked;
        if (includeVisualizationEl) this.settings.includeVisualization = includeVisualizationEl.checked;
        if (qualityThresholdEl) this.settings.qualityThreshold = parseFloat(qualityThresholdEl.value);
        
        // Sync face filter settings
        const faceFilterEnabledEl = document.getElementById('faceFilterEnabled');
        const faceFilterThresholdEl = document.getElementById('faceFilterThreshold');
        
        if (faceFilterEnabledEl) this.faceFilterConfig.enabled = faceFilterEnabledEl.checked;
        if (faceFilterThresholdEl) this.faceFilterConfig.iouThreshold = parseFloat(faceFilterThresholdEl.value);
        
        console.log('Initial settings synced:', this.settings);
        console.log('Initial face filter config synced:', this.faceFilterConfig);
        
        // Load current face filter configuration from server
        this.loadFaceFilterConfig();
    }
    
    setupEventListeners() {
        // Mode selection
        document.getElementById('imageMode').addEventListener('click', () => this.switchMode('image'));
        document.getElementById('cameraMode').addEventListener('click', () => this.switchMode('camera'));
        document.getElementById('batchMode').addEventListener('click', () => this.switchMode('batch'));
        
        // Settings
        document.getElementById('includeFaces').addEventListener('change', (e) => {
            this.settings.includeFaces = e.target.checked;
            this.updateWebSocketSettings();
        });
        
        document.getElementById('includeSegmentation').addEventListener('change', (e) => {
            this.settings.includeSegmentation = e.target.checked;
            this.updateWebSocketSettings();
        });
        
        document.getElementById('includeVisualization').addEventListener('change', (e) => {
            this.settings.includeVisualization = e.target.checked;
            this.updateWebSocketSettings();
        });
        
        document.getElementById('qualityThreshold').addEventListener('input', (e) => {
            this.settings.qualityThreshold = parseFloat(e.target.value);
            document.getElementById('qualityValue').textContent = e.target.value;
            this.updateWebSocketSettings();
        });
        
        // Face filter settings
        document.getElementById('faceFilterEnabled').addEventListener('change', (e) => {
            this.faceFilterConfig.enabled = e.target.checked;
        });
        
        document.getElementById('faceFilterThreshold').addEventListener('input', (e) => {
            this.faceFilterConfig.iouThreshold = parseFloat(e.target.value);
            document.getElementById('faceFilterThresholdValue').textContent = e.target.value;
        });
        
        document.getElementById('applyFaceFilterBtn').addEventListener('click', this.applyFaceFilterConfig.bind(this));
        document.getElementById('resetFaceFilterBtn').addEventListener('click', this.resetFaceFilterConfig.bind(this));
        
        // Image upload
        document.getElementById('uploadBtn').addEventListener('click', () => {
            document.getElementById('imageInput').click();
        });
        
        document.getElementById('imageInput').addEventListener('change', this.handleImageSelect.bind(this));
        document.getElementById('analyzeBtn').addEventListener('click', this.analyzeImage.bind(this));
        
        // Drag and drop
        const dropZone = document.getElementById('dropZone');
        dropZone.addEventListener('dragover', this.handleDragOver.bind(this));
        dropZone.addEventListener('drop', this.handleDrop.bind(this));
        dropZone.addEventListener('click', () => document.getElementById('imageInput').click());
        
        // Camera controls
        document.getElementById('startCameraBtn').addEventListener('click', this.startCamera.bind(this));
        document.getElementById('stopCameraBtn').addEventListener('click', this.stopCamera.bind(this));
        document.getElementById('captureBtn').addEventListener('click', this.captureFrame.bind(this));
        
        // Camera settings
        document.getElementById('processingMode').addEventListener('change', (e) => {
            this.settings.processingMode = e.target.value;
            this.updateWebSocketSettings();
        });
        
        document.getElementById('autoProcess').addEventListener('change', (e) => {
            this.settings.autoProcess = e.target.checked;
            this.updateWebSocketSettings();
        });
        
        // Batch processing
        document.getElementById('batchUploadBtn').addEventListener('click', () => {
            document.getElementById('batchInput').click();
        });
        
        document.getElementById('batchInput').addEventListener('change', this.handleBatchSelect.bind(this));
        document.getElementById('processBatchBtn').addEventListener('click', this.processBatch.bind(this));
        
        // Batch drag and drop
        const batchDropZone = document.getElementById('batchDropZone');
        batchDropZone.addEventListener('dragover', this.handleDragOver.bind(this));
        batchDropZone.addEventListener('drop', this.handleBatchDrop.bind(this));
        batchDropZone.addEventListener('click', () => document.getElementById('batchInput').click());
        
        // Status refresh
        document.getElementById('refreshStatusBtn').addEventListener('click', this.checkApiHealth.bind(this));
        
        // Help modal
        document.getElementById('helpBtn').addEventListener('click', this.showHelp.bind(this));
        document.getElementById('closeHelpBtn').addEventListener('click', this.hideHelp.bind(this));
        document.getElementById('helpModal').addEventListener('click', (e) => {
            if (e.target.id === 'helpModal') this.hideHelp();
        });
    }
    
    switchMode(mode) {
        this.currentMode = mode;
        
        // Update button styles
        document.querySelectorAll('.mode-btn').forEach(btn => {
            btn.classList.remove('bg-fish-blue', 'bg-blue-600');
            btn.classList.add('bg-gray-500');
        });
        
        document.getElementById(`${mode}Mode`).classList.remove('bg-gray-500');
        document.getElementById(`${mode}Mode`).classList.add('bg-fish-blue');
        
        // Show/hide sections
        document.getElementById('imageUploadSection').classList.toggle('hidden', mode !== 'image');
        document.getElementById('cameraSection').classList.toggle('hidden', mode !== 'camera');
        document.getElementById('batchSection').classList.toggle('hidden', mode !== 'batch');
        
        if (mode !== 'camera' && this.videoStream) {
            this.stopCamera();
        }
    }
    
    async checkApiHealth() {
        try {
            const response = await fetch(`${this.apiBase}/health/`);
            const data = await response.json();
            
            document.getElementById('healthStatus').textContent = data.status;
            document.getElementById('healthStatus').className = `font-semibold ${data.status === 'healthy' ? 'text-green-500' : 'text-red-500'}`;
            
            document.getElementById('modelsStatus').textContent = data.models_loaded ? 'Loaded' : 'Not Loaded';
            document.getElementById('modelsStatus').className = `font-semibold ${data.models_loaded ? 'text-green-500' : 'text-red-500'}`;
            
            document.getElementById('deviceStatus').textContent = data.device || 'Unknown';
            
        } catch (error) {
            console.error('Health check failed:', error);
            document.getElementById('healthStatus').textContent = 'Error';
            document.getElementById('healthStatus').className = 'font-semibold text-red-500';
        }
    }
    
    async loadFaceFilterConfig() {
        try {
            const response = await fetch(`${this.apiBase}/config/face-filter/`);
            if (response.ok) {
                const data = await response.json();
                this.faceFilterConfig.enabled = data.enabled;
                this.faceFilterConfig.iouThreshold = data.iou_threshold;
                
                // Update UI
                const enabledEl = document.getElementById('faceFilterEnabled');
                const thresholdEl = document.getElementById('faceFilterThreshold');
                const thresholdValueEl = document.getElementById('faceFilterThresholdValue');
                
                if (enabledEl) enabledEl.checked = data.enabled;
                if (thresholdEl) thresholdEl.value = data.iou_threshold;
                if (thresholdValueEl) thresholdValueEl.textContent = data.iou_threshold;
                
                this.updateFaceFilterStatus('Loaded current configuration');
                console.log('Face filter config loaded:', data);
            } else {
                this.updateFaceFilterStatus('Failed to load configuration', 'error');
            }
        } catch (error) {
            console.error('Failed to load face filter config:', error);
            this.updateFaceFilterStatus('Error loading configuration', 'error');
        }
    }
    
    async applyFaceFilterConfig() {
        try {
            const response = await fetch(`${this.apiBase}/config/face-filter/`, {
                method: 'POST',
                headers: {
                    'Content-Type': 'application/json',
                },
                body: JSON.stringify({
                    enabled: this.faceFilterConfig.enabled,
                    iou_threshold: this.faceFilterConfig.iouThreshold
                })
            });
            
            if (response.ok) {
                const data = await response.json();
                this.updateFaceFilterStatus('Configuration applied successfully', 'success');
                console.log('Face filter config applied:', data);
                
                // Update local config with server response
                if (data.config) {
                    this.faceFilterConfig.enabled = data.config.enabled;
                    this.faceFilterConfig.iouThreshold = data.config.iou_threshold;
                }
            } else {
                const errorData = await response.json();
                this.updateFaceFilterStatus(`Failed to apply: ${errorData.error || 'Unknown error'}`, 'error');
            }
        } catch (error) {
            console.error('Failed to apply face filter config:', error);
            this.updateFaceFilterStatus('Error applying configuration', 'error');
        }
    }
    
    async resetFaceFilterConfig() {
        // Reset to default values
        this.faceFilterConfig.enabled = true;
        this.faceFilterConfig.iouThreshold = 0.3;
        
        // Update UI
        document.getElementById('faceFilterEnabled').checked = true;
        document.getElementById('faceFilterThreshold').value = 0.3;
        document.getElementById('faceFilterThresholdValue').textContent = '0.3';
        
        // Apply the reset configuration
        await this.applyFaceFilterConfig();
        this.updateFaceFilterStatus('Configuration reset to defaults', 'success');
    }
    
    updateFaceFilterStatus(message, type = 'info') {
        const statusEl = document.getElementById('faceFilterStatus');
        if (statusEl) {
            statusEl.textContent = message;
            statusEl.className = `mt-2 text-sm ${
                type === 'success' ? 'text-green-600' : 
                type === 'error' ? 'text-red-600' : 
                'text-gray-600'
            }`;
            
            // Clear message after 3 seconds
            setTimeout(() => {
                statusEl.textContent = '';
            }, 3000);
        }
    }
    
    connectWebSocket() {
        try {
            this.ws = new WebSocket(this.wsUrl);
            
            this.ws.onopen = () => {
                console.log('WebSocket connected');
                this.updateConnectionStatus(true);
            };
            
            this.ws.onclose = () => {
                console.log('WebSocket disconnected');
                this.updateConnectionStatus(false);
                // Attempt to reconnect after 3 seconds
                setTimeout(() => this.connectWebSocket(), 3000);
            };
            
            this.ws.onerror = (error) => {
                console.error('WebSocket error:', error);
                this.updateConnectionStatus(false);
            };
            
            this.ws.onmessage = (event) => {
                try {
                    const message = JSON.parse(event.data);
                    this.handleWebSocketMessage(message);
                } catch (error) {
                    console.error('Failed to parse WebSocket message:', error);
                }
            };
            
        } catch (error) {
            console.error('WebSocket connection failed:', error);
            this.updateConnectionStatus(false);
        }
    }
    
    updateConnectionStatus(connected) {
        const indicator = document.getElementById('statusIndicator');
        const text = document.getElementById('statusText');
        
        if (connected) {
            indicator.className = 'w-3 h-3 bg-green-500 rounded-full pulse';
            text.textContent = 'Connected';
        } else {
            indicator.className = 'w-3 h-3 bg-red-500 rounded-full';
            text.textContent = 'Disconnected';
        }
    }
    
    handleWebSocketMessage(message) {
        console.log('WebSocket message:', message);
        
        switch (message.type) {
            case 'connection_established':
                console.log('Connection established:', message.data);
                break;
                
            case 'recognition_result':
                // Extract results from the WebSocket message structure
                const resultData = {
                    ...message.data.results,  // fish_detections, faces, visualization_image, etc.
                    frame_id: message.data.frame_id,
                    processing_time: message.data.processing_time,
                    timestamp: message.data.timestamp,
                    total_processing_time: message.data.results.total_processing_time || message.data.processing_time
                };
                this.handleRecognitionResult(resultData);
                break;
                
            case 'session_stats':
                this.updateSessionStats(message.data);
                break;
                
            case 'quality_warning':
                this.showQualityWarning(message.data);
                break;
                
            case 'frame_skipped':
                console.log('Frame skipped:', message.data.reason);
                break;
                
            case 'error':
            case 'frame_error':
                this.showError(message.data.message || message.data.error);
                break;
                
            default:
                console.log('Unknown message type:', message.type);
        }
    }
    
    updateWebSocketSettings() {
        if (this.ws && this.ws.readyState === WebSocket.OPEN) {
            console.log('Updating WebSocket settings...');
            
            // Get elements with null checks
            const includeFacesEl = document.getElementById('includeFaces');
            const includeSegmentationEl = document.getElementById('includeSegmentation');
            const includeVisualizationEl = document.getElementById('includeVisualization');
            const qualityThresholdEl = document.getElementById('qualityThreshold');
            
            if (!includeFacesEl || !includeSegmentationEl || !includeVisualizationEl || !qualityThresholdEl) {
                console.error('Some settings elements not found');
                return;
            }
            
            const settings = {
                include_faces: includeFacesEl.checked,
                include_segmentation: includeSegmentationEl.checked,
                include_visualization: includeVisualizationEl.checked,
                quality_threshold: parseFloat(qualityThresholdEl.value),
                auto_process: this.autoProcessEnabled,
                processing_mode: 'accuracy'
            };
            
            console.log('Sending settings to WebSocket:', settings);
            
            this.ws.send(JSON.stringify({
                type: 'settings_update',
                data: settings
            }));
        }
    }
    
    handleImageSelect(event) {
        const file = event.target.files[0];
        if (file) {
            this.displayImagePreview(file);
        }
    }
    
    displayImagePreview(file) {
        const reader = new FileReader();
        reader.onload = (e) => {
            const img = document.getElementById('previewImg');
            img.src = e.target.result;
            document.getElementById('imagePreview').classList.remove('hidden');
        };
        reader.readAsDataURL(file);
    }
    
    async analyzeImage() {
        const fileInput = document.getElementById('imageInput');
        const file = fileInput.files[0];
        
        if (!file) {
            this.showError('Please select an image first');
            return;
        }
        
        this.setAnalyzeLoading(true);
        
        try {
            const formData = new FormData();
            formData.append('image', file);
            formData.append('include_faces', this.settings.includeFaces ? 'true' : 'false');
            formData.append('include_segmentation', this.settings.includeSegmentation ? 'true' : 'false');
            formData.append('include_visualization', this.settings.includeVisualization ? 'true' : 'false');
            
            const response = await fetch(`${this.apiBase}/recognize/`, {
                method: 'POST',
                body: formData
            });
            
            const result = await response.json();
            
            // DEBUG: Log detailed response information
            console.log('=== API RESPONSE DEBUG ===');
            console.log('Response Status:', response.status);
            console.log('Response OK:', response.ok);
            console.log('Full Result:', result);
            console.log('Include Visualization Setting:', this.settings.includeVisualization);
            
            // Check if visualization is included
            if (result.visualization_image) {
                console.log('✅ Visualization image found in response');
                console.log('Visualization image length:', result.visualization_image.length);
                console.log('Visualization image starts with:', result.visualization_image.substring(0, 50));
            } else {
                console.log('❌ No visualization image in response');
            }
            
            // Check fish detections and segmentation
            if (result.fish_detections) {
                console.log('Fish detections count:', result.fish_detections.length);
                result.fish_detections.forEach((fish, index) => {
                    console.log(`Fish ${index + 1}:`, fish);
                    if (fish.segmentation) {
                        console.log(`  - Segmentation:`, fish.segmentation);
                        if (fish.segmentation.has_segmentation) {
                            console.log(`  - Has segmentation: TRUE`);
                            console.log(`  - Polygon data:`, fish.segmentation.polygon_data);
                        } else {
                            console.log(`  - Has segmentation: FALSE`);
                        }
                    } else {
                        console.log(`  - No segmentation data`);
                    }
                });
            }
            console.log('=== END DEBUG ===');
            
            if (response.ok) {
                this.handleRecognitionResult(result);
                this.updateStats(result);
            } else {
                this.showError(result.error || 'Recognition failed');
            }
            
        } catch (error) {
            console.error('Analysis failed:', error);
            this.showError('Network error occurred');
        } finally {
            this.setAnalyzeLoading(false);
        }
    }
    
    setAnalyzeLoading(loading) {
        const text = document.getElementById('analyzeText');
        const spinner = document.getElementById('analyzeSpinner');
        const btn = document.getElementById('analyzeBtn');
        
        if (loading) {
            text.classList.add('hidden');
            spinner.classList.remove('hidden');
            btn.disabled = true;
        } else {
            text.classList.remove('hidden');
            spinner.classList.add('hidden');
            btn.disabled = false;
        }
    }
    
    async startCamera() {
        try {
            this.videoStream = await navigator.mediaDevices.getUserMedia({
                video: { width: 640, height: 480 }
            });
            
            const video = document.getElementById('videoElement');
            video.srcObject = this.videoStream;
            
            // Ensure camera mode is set
            this.currentMode = 'camera';
            console.log('Camera started, mode set to:', this.currentMode);
            
            document.getElementById('startCameraBtn').classList.add('hidden');
            document.getElementById('stopCameraBtn').classList.remove('hidden');
            document.getElementById('captureBtn').classList.remove('hidden');
            
            // Start automatic processing if enabled
            if (this.settings.autoProcess) {
                this.startAutomaticProcessing();
            }
            
        } catch (error) {
            console.error('Camera access failed:', error);
            this.showError('Failed to access camera');
        }
    }
    
    stopCamera() {
        if (this.videoStream) {
            this.videoStream.getTracks().forEach(track => track.stop());
            this.videoStream = null;
        }
        
        document.getElementById('startCameraBtn').classList.remove('hidden');
        document.getElementById('stopCameraBtn').classList.add('hidden');
        document.getElementById('captureBtn').classList.add('hidden');
        
        this.stopAutomaticProcessing();
    }
    
    startAutomaticProcessing() {
        if (this.processingInterval) return;
        
        const interval = this.settings.processingMode === 'speed' ? 1000 : 2000; // 1s or 2s
        
        this.processingInterval = setInterval(() => {
            if (this.videoStream && !this.isProcessing) {
                this.captureAndSendFrame();
            }
        }, interval);
    }
    
    stopAutomaticProcessing() {
        if (this.processingInterval) {
            clearInterval(this.processingInterval);
            this.processingInterval = null;
        }
    }
    
    captureFrame() {
        this.captureAndSendFrame();
    }
    
    captureAndSendFrame() {
        const video = document.getElementById('videoElement');
        const canvas = document.createElement('canvas');
        canvas.width = video.videoWidth;
        canvas.height = video.videoHeight;
        
        const ctx = canvas.getContext('2d');
        ctx.drawImage(video, 0, 0);
        
        const frameData = canvas.toDataURL('image/jpeg', 0.8);
        
        if (this.ws && this.ws.readyState === WebSocket.OPEN) {
            this.isProcessing = true;
            
            const framePayload = {
                frame_data: frameData,
                frame_id: Date.now(),
                include_faces: this.settings.includeFaces,
                include_segmentation: this.settings.includeSegmentation,
                include_visualization: this.settings.includeVisualization,
                quality_threshold: this.settings.qualityThreshold
            };
            
            console.log('Sending frame with settings:', {
                include_faces: framePayload.include_faces,
                include_segmentation: framePayload.include_segmentation,
                include_visualization: framePayload.include_visualization
            });
            
            this.ws.send(JSON.stringify({
                type: 'camera_frame',
                data: framePayload
            }));
        }
    }
    
    handleBatchSelect(event) {
        const files = Array.from(event.target.files);
        if (files.length > 10) {
            this.showError('Maximum 10 images allowed');
            return;
        }
        
        this.displayBatchPreview(files);
    }
    
    handleBatchDrop(event) {
        event.preventDefault();
        const files = Array.from(event.dataTransfer.files).filter(file => file.type.startsWith('image/'));
        
        if (files.length > 10) {
            this.showError('Maximum 10 images allowed');
            return;
        }
        
        this.displayBatchPreview(files);
    }
    
    displayBatchPreview(files) {
        const container = document.getElementById('batchImages');
        container.innerHTML = '';
        
        files.forEach((file, index) => {
            const reader = new FileReader();
            reader.onload = (e) => {
                const div = document.createElement('div');
                div.className = 'relative';
                div.innerHTML = `
                    <img src="${e.target.result}" class="w-full h-24 object-cover rounded-lg">
                    <div class="absolute top-2 right-2 bg-fish-blue text-white text-xs px-2 py-1 rounded">
                        ${index + 1}
                    </div>
                `;
                container.appendChild(div);
            };
            reader.readAsDataURL(file);
        });
        
        document.getElementById('batchPreview').classList.remove('hidden');
        
        // Store files for processing
        this.batchFiles = files;
    }
    
    async processBatch() {
        if (!this.batchFiles || this.batchFiles.length === 0) {
            this.showError('Please select images first');
            return;
        }
        
        this.setBatchLoading(true);
        
        try {
            const formData = new FormData();
            
            this.batchFiles.forEach(file => {
                formData.append('images', file);
            });
            
            formData.append('include_faces', this.settings.includeFaces);
            formData.append('include_segmentation', this.settings.includeSegmentation);
            formData.append('include_visualization', this.settings.includeVisualization);
            
            const response = await fetch(`${this.apiBase}/recognize/batch/`, {
                method: 'POST',
                body: formData
            });
            
            const result = await response.json();
            
            if (response.ok) {
                this.handleBatchResults(result);
            } else {
                this.showError(result.error || 'Batch processing failed');
            }
            
        } catch (error) {
            console.error('Batch processing failed:', error);
            this.showError('Network error occurred');
        } finally {
            this.setBatchLoading(false);
        }
    }
    
    setBatchLoading(loading) {
        const text = document.getElementById('batchText');
        const spinner = document.getElementById('batchSpinner');
        const btn = document.getElementById('processBatchBtn');
        
        if (loading) {
            text.classList.add('hidden');
            spinner.classList.remove('hidden');
            btn.disabled = true;
        } else {
            text.classList.remove('hidden');
            spinner.classList.add('hidden');
            btn.disabled = false;
        }
    }
    
    handleBatchResults(result) {
        result.results.forEach((imageResult, index) => {
            this.handleRecognitionResult(imageResult, `Batch Image ${index + 1}`);
        });
        
        this.updateStats({
            total_processing_time: result.total_processing_time,
            fish_detections: result.results.reduce((sum, r) => sum + (r.fish_detections?.length || 0), 0)
        });
    }
    
    handleRecognitionResult(result, title = null) {
        this.isProcessing = false;
        
        // DEBUG: Log visualization handling
        console.log('=== HANDLE RECOGNITION RESULT DEBUG ===');
        console.log('Result object:', result);
        console.log('Has visualization_image:', !!result.visualization_image);
        if (result.visualization_image) {
            console.log('Visualization image data length:', result.visualization_image.length);
        }
        console.log('=== END HANDLE DEBUG ===');
        
        const container = document.getElementById('resultsContainer');
        
        // Remove placeholder if exists
        if (container.children.length === 1 && container.children[0].textContent.includes('No results yet')) {
            container.innerHTML = '';
        }
        
        const resultCard = this.createResultCard(result, title);
        container.insertBefore(resultCard, container.firstChild);
        
        // Keep only last 5 results
        while (container.children.length > 5) {
            container.removeChild(container.lastChild);
        }
        
        // Update overlay for camera mode
        if (this.currentMode === 'camera' && result.fish_detections) {
            console.log('=== DRAWING OVERLAY FOR CAMERA MODE ===');
            console.log('Current mode:', this.currentMode);
            console.log('Fish detections:', result.fish_detections.length);
            console.log('Fish detections data:', result.fish_detections);
            this.drawOverlay(result.fish_detections, result.faces);
            console.log('=== OVERLAY DRAWING COMPLETED ===');
        } else {
            console.log('=== NOT DRAWING OVERLAY ===');
            console.log('Current mode:', this.currentMode);
            console.log('Has fish detections:', !!result.fish_detections);
            console.log('Fish detections count:', result.fish_detections?.length || 0);
        }
    }
    
    createResultCard(result, title) {
        const div = document.createElement('div');
        div.className = 'result-card bg-gray-50 p-4 rounded-lg border';
        
        const fishCount = result.fish_detections?.length || 0;
        const faceCount = result.faces?.length || 0;
        const processingTime = result.total_processing_time || result.processing_time?.total || 0;
        
        let fishDetails = '';
        if (result.fish_detections && result.fish_detections.length > 0) {
            fishDetails = result.fish_detections.map((fish, i) => {
                const classification = fish.classification?.[0];
                if (classification) {
                    return `
                        <div class="text-xs bg-white p-2 rounded mt-2">
                            <strong>Fish ${i + 1}:</strong> ${classification.name}<br>
                            <span class="text-gray-600">Accuracy: ${(classification.accuracy * 100).toFixed(1)}%</span>
                        </div>
                    `;
                }
                return `<div class="text-xs bg-white p-2 rounded mt-2"><strong>Fish ${i + 1}:</strong> Detected</div>`;
            }).join('');
        }
        
        let visualizationSection = '';
        if (result.visualization_image) {
            console.log('Adding visualization image to result card');
            visualizationSection = `
                <div class="mt-3">
                    <div class="text-xs font-semibold text-gray-700 mb-1">Visualization with Segmentation:</div>
                    <img src="${result.visualization_image}" alt="Visualization" class="w-full rounded border" style="max-height: 200px; object-fit: contain;">
                </div>
            `;
        } else {
            console.log('No visualization image to display');
        }
        
        div.innerHTML = `
            <div class="flex justify-between items-start mb-2">
                <div class="font-semibold text-sm">${title || 'Recognition Result'}</div>
                <div class="text-xs text-gray-500">${new Date().toLocaleTimeString()}</div>
            </div>
            <div class="grid grid-cols-3 gap-2 text-xs">
                <div class="text-center">
                    <div class="font-semibold text-fish-blue">${fishCount}</div>
                    <div class="text-gray-600">Fish</div>
                </div>
                <div class="text-center">
                    <div class="font-semibold text-purple-500">${faceCount}</div>
                    <div class="text-gray-600">Faces</div>
                </div>
                <div class="text-center">
                    <div class="font-semibold text-green-500">${processingTime.toFixed(2)}s</div>
                    <div class="text-gray-600">Time</div>
                </div>
            </div>
            ${fishDetails}
            ${visualizationSection}
        `;
        
        return div;
    }
    
    drawOverlay(fishDetections, faces) {
        console.log('=== drawOverlay CALLED ===');
        const video = document.getElementById('videoElement');
        const canvas = document.getElementById('overlayCanvas');
        const ctx = canvas.getContext('2d');
        
        console.log('Video element:', video);
        console.log('Canvas element:', canvas);
        console.log('Canvas context:', ctx);
        
        // Ensure video is loaded
        if (!video.videoWidth || !video.videoHeight || video.offsetWidth === 0 || video.offsetHeight === 0) {
            console.log('Video not ready for overlay, retrying in 100ms...');
            setTimeout(() => this.drawOverlay(fishDetections, faces), 100);
            return;
        }
        
        // Set canvas size to match video element display size
        canvas.width = video.offsetWidth;
        canvas.height = video.offsetHeight;
        canvas.style.width = video.offsetWidth + 'px';
        canvas.style.height = video.offsetHeight + 'px';
        
        ctx.clearRect(0, 0, canvas.width, canvas.height);
        
        // Calculate scaling factors from video native resolution to display size
        const scaleX = video.offsetWidth / video.videoWidth;
        const scaleY = video.offsetHeight / video.videoHeight;
        
        console.log('Drawing overlay - Video dimensions:', video.videoWidth, 'x', video.videoHeight);
        console.log('Drawing overlay - Display dimensions:', video.offsetWidth, 'x', video.offsetHeight);
        console.log('Drawing overlay - Scale factors:', scaleX, 'x', scaleY);
        console.log('Drawing overlay for', fishDetections?.length || 0, 'fish detections');
        
        // Draw fish detections
        if (fishDetections && fishDetections.length > 0) {
            fishDetections.forEach((fish, i) => {
                console.log('Drawing fish', i + 1, ':', fish);
                
                const [x1, y1, x2, y2] = fish.bbox;
                console.log('Original bbox:', [x1, y1, x2, y2]);
                console.log('Scaled bbox:', [x1 * scaleX, y1 * scaleY, x2 * scaleX, y2 * scaleY]);
                
                // Draw segmentation polygon first (if available)
                const segmentation = fish.segmentation;
                if (segmentation && segmentation.has_segmentation && segmentation.polygon_data) {
                    console.log('Drawing segmentation polygon for fish', i + 1);
                    console.log('First few polygon points:', segmentation.polygon_data.slice(0, 5));
                    
                    ctx.beginPath();
                    ctx.strokeStyle = '#FBBF24'; // Yellow for segmentation
                    ctx.fillStyle = 'rgba(251, 191, 36, 0.2)'; // Semi-transparent yellow
                    ctx.lineWidth = 2;
                    
                    const polygonData = segmentation.polygon_data;
                    if (polygonData && polygonData.length > 2) {
                        const firstPoint = [polygonData[0][0] * scaleX, polygonData[0][1] * scaleY];
                        console.log('First polygon point scaled:', firstPoint);
                        ctx.moveTo(firstPoint[0], firstPoint[1]);
                        for (let j = 1; j < polygonData.length; j++) {
                            ctx.lineTo(polygonData[j][0] * scaleX, polygonData[j][1] * scaleY);
                        }
                        ctx.closePath();
                        ctx.fill();
                        ctx.stroke();
                        console.log('Polygon drawn successfully');
                    }
                }
                
                // Draw bounding box
                ctx.strokeStyle = '#10B981';
                ctx.lineWidth = 3;
                const rectX = x1 * scaleX;
                const rectY = y1 * scaleY;
                const rectWidth = (x2 - x1) * scaleX;
                const rectHeight = (y2 - y1) * scaleY;
                console.log('Drawing bounding box at:', [rectX, rectY, rectWidth, rectHeight]);
                ctx.strokeRect(rectX, rectY, rectWidth, rectHeight);
                
                // Draw label
                const classification = fish.classification?.[0];
                if (classification) {
                    let label = `${classification.name} (${(classification.accuracy * 100).toFixed(0)}%)`;
                    if (segmentation && segmentation.has_segmentation) {
                        label += ' [S]'; // Indicate segmentation
                    }
                    
                    ctx.fillStyle = '#10B981';
                    const labelWidth = ctx.measureText(label).width + 10;
                    ctx.fillRect(x1 * scaleX, (y1 - 25) * scaleY, labelWidth, 20);
                    ctx.fillStyle = 'white';
                    ctx.font = '14px Arial';
                    ctx.fillText(label, (x1 + 5) * scaleX, (y1 - 8) * scaleY);
                } else {
                    // Default label
                    let label = `Fish ${i + 1}`;
                    if (segmentation && segmentation.has_segmentation) {
                        label += ' [S]';
                    }
                    
                    ctx.fillStyle = '#10B981';
                    const labelWidth = ctx.measureText(label).width + 10;
                    ctx.fillRect(x1 * scaleX, (y1 - 25) * scaleY, labelWidth, 20);
                    ctx.fillStyle = 'white';
                    ctx.font = '14px Arial';
                    ctx.fillText(label, (x1 + 5) * scaleX, (y1 - 8) * scaleY);
                }
            });
        }
        
        // Draw face detections
        if (faces && faces.length > 0) {
            faces.forEach(face => {
                const [x1, y1, x2, y2] = face.bbox;
                
                ctx.strokeStyle = '#EF4444';
                ctx.lineWidth = 2;
                ctx.strokeRect(x1 * scaleX, y1 * scaleY, (x2 - x1) * scaleX, (y2 - y1) * scaleY);
                
                ctx.fillStyle = '#EF4444';
                ctx.fillRect(x1 * scaleX, (y1 - 20) * scaleY, 40, 15);
                ctx.fillStyle = 'white';
                ctx.font = '12px Arial';
                ctx.fillText('Face', (x1 + 2) * scaleX, (y1 - 8) * scaleY);
            });
        }
    }
    
    updateStats(result) {
        this.stats.processed++;
        if (result.success !== false) {
            this.stats.successful++;
        }
        if (result.fish_detections && Array.isArray(result.fish_detections)) {
            this.stats.fishDetected += result.fish_detections.length;
        }
        if (result.total_processing_time && !isNaN(result.total_processing_time)) {
            this.stats.totalTime += result.total_processing_time;
        }
        
        this.updateStatsDisplay();
    }
    
    updateStatsDisplay() {
        const processed = this.stats.processed || 0;
        const successful = this.stats.successful || 0;
        const totalTime = this.stats.totalTime || 0;
        const fishDetected = this.stats.fishDetected || 0;
        
        document.getElementById('processedCount').textContent = processed;
        document.getElementById('successRate').textContent = 
            processed > 0 ? `${((successful / processed) * 100).toFixed(1)}%` : '0.0%';
        document.getElementById('avgTime').textContent = 
            successful > 0 ? `${(totalTime / successful).toFixed(2)}s` : '0.00s';
        document.getElementById('fishCount').textContent = fishDetected;
    }
    
    updateSessionStats(stats) {
        console.log('Updating session stats:', stats);
        
        if (stats.frames_processed !== undefined && !isNaN(stats.frames_processed)) {
            document.getElementById('processedCount').textContent = stats.frames_processed;
        }
        if (stats.avg_processing_time !== undefined && !isNaN(stats.avg_processing_time)) {
            document.getElementById('avgTime').textContent = `${stats.avg_processing_time.toFixed(2)}s`;
        }
        if (stats.processing_rate !== undefined && !isNaN(stats.processing_rate)) {
            document.getElementById('successRate').textContent = `${(stats.processing_rate * 100).toFixed(1)}%`;
        }
        if (stats.total_fish_detected !== undefined && !isNaN(stats.total_fish_detected)) {
            document.getElementById('fishCount').textContent = stats.total_fish_detected;
        }
    }
    
    showQualityWarning(data) {
        this.showNotification(`Quality Warning: ${data.message}`, 'warning');
    }
    
    showError(message) {
        this.showNotification(`Error: ${message}`, 'error');
    }
    
    showNotification(message, type = 'info') {
        // Create notification element
        const notification = document.createElement('div');
        notification.className = `fixed top-4 right-4 p-4 rounded-lg shadow-lg z-50 max-w-sm ${
            type === 'error' ? 'bg-red-500' : 
            type === 'warning' ? 'bg-yellow-500' : 
            'bg-blue-500'
        } text-white`;
        
        notification.textContent = message;
        
        document.body.appendChild(notification);
        
        // Auto remove after 5 seconds
        setTimeout(() => {
            if (notification.parentNode) {
                notification.parentNode.removeChild(notification);
            }
        }, 5000);
    }
    
    handleDragOver(event) {
        event.preventDefault();
        event.currentTarget.classList.add('border-fish-blue');
    }
    
    handleDrop(event) {
        event.preventDefault();
        event.currentTarget.classList.remove('border-fish-blue');
        
        const files = Array.from(event.dataTransfer.files).filter(file => file.type.startsWith('image/'));
        if (files.length > 0) {
            this.displayImagePreview(files[0]);
            
            // Set file to input for processing
            const dt = new DataTransfer();
            dt.items.add(files[0]);
            document.getElementById('imageInput').files = dt.files;
        }
    }
    
    showHelp() {
        document.getElementById('helpModal').classList.remove('hidden');
    }
    
    hideHelp() {
        document.getElementById('helpModal').classList.add('hidden');
    }
    
    updateUI() {
        // Update quality threshold display
        document.getElementById('qualityValue').textContent = this.settings.qualityThreshold;
        
        // Set initial settings
        document.getElementById('includeFaces').checked = this.settings.includeFaces;
        document.getElementById('includeSegmentation').checked = this.settings.includeSegmentation;
        document.getElementById('includeVisualization').checked = this.settings.includeVisualization;
        document.getElementById('qualityThreshold').value = this.settings.qualityThreshold;
        document.getElementById('processingMode').value = this.settings.processingMode;
        document.getElementById('autoProcess').checked = this.settings.autoProcess;
    }
}

// Initialize the application when DOM is loaded
document.addEventListener('DOMContentLoaded', () => {
    window.fishApp = new FishRecognitionApp();
});