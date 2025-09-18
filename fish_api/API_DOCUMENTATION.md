# Fish Recognition API Documentation

## Overview

The Fish Recognition API is a high-performance Django REST API that combines object detection, classification, and segmentation to provide accurate fish recognition capabilities. It supports both single image processing and real-time camera feed analysis through WebSocket connections.

## Features

- **Multi-Model Fish Recognition**: Combines detection, classification, and segmentation models
- **Real-time Processing**: WebSocket support for live camera feeds
- **High Accuracy**: Optimized for accuracy over speed with adaptive processing
- **Batch Processing**: Efficient processing of multiple images
- **Performance Monitoring**: Comprehensive metrics and statistics
- **Image Quality Validation**: Automatic quality assessment and recommendations
- **Caching System**: Redis-based caching for improved performance
- **Face Detection**: Fish face detection capabilities

## API Endpoints

### Base URL
```
http://localhost:8000/api/v1/
```

### Health Check
**GET** `/health/`

Returns the current health status of the API and all loaded models.

**Response:**
```json
{
  "status": "healthy",
  "models_loaded": true,
  "device": "cpu",
  "models": {
    "classification": true,
    "detection": true,
    "segmentation": true,
    "face_detection": true
  },
  "uptime": 123.45
}
```

### Performance Statistics
**GET** `/stats/`

Returns detailed performance statistics for all models.

**Response:**
```json
{
  "face_detection": {
    "count": 50,
    "mean": 0.15,
    "median": 0.14,
    "min": 0.10,
    "max": 0.25,
    "std": 0.03
  },
  "fish_detection": { "..." },
  "classification": { "..." },
  "segmentation": { "..." }
}
```

### Single Image Recognition
**POST** `/recognize/`

Process a single image for fish recognition.

**Request Parameters:**
- `image` (file, optional): Image file upload
- `image_base64` (string, optional): Base64 encoded image
- `include_faces` (boolean, default: true): Include face detection
- `include_segmentation` (boolean, default: true): Include segmentation
- `include_visualization` (boolean, default: false): Include annotated image

**Example Request (multipart/form-data):**
```bash
curl -X POST http://localhost:8000/api/v1/recognize/ \
  -F "image=@fish_image.jpg" \
  -F "include_faces=true" \
  -F "include_segmentation=true" \
  -F "include_visualization=true"
```

**Example Request (JSON with base64):**
```bash
curl -X POST http://localhost:8000/api/v1/recognize/ \
  -H "Content-Type: application/json" \
  -d '{
    "image_base64": "data:image/jpeg;base64,/9j/4AAQSkZJRgABAQAAAQ...",
    "include_faces": true,
    "include_segmentation": true
  }'
```

**Response:**
```json
{
  "success": true,
  "image_shape": [480, 640, 3],
  "fish_detections": [
    {
      "id": 0,
      "bbox": [100, 50, 300, 200],
      "confidence": 0.89,
      "area": 30000,
      "classification": [
        {
          "name": "Ikan Mas",
          "species_id": 123,
          "accuracy": 0.92,
          "distance": 0.15
        }
      ],
      "segmentation": {
        "has_segmentation": true,
        "processing_time": 0.12
      }
    }
  ],
  "faces": [
    {
      "bbox": [120, 70, 180, 120],
      "confidence": 0.85,
      "area": 3000
    }
  ],
  "processing_time": {
    "face_detection": 0.15,
    "fish_detection": 0.23,
    "classification": 0.18,
    "segmentation": 0.12,
    "total": 0.68
  },
  "total_processing_time": 0.68,
  "quality_validation": {
    "valid": true,
    "warnings": [],
    "quality_score": 0.78
  },
  "visualization_image": "data:image/jpeg;base64,..." // if requested
}
```

### Batch Image Recognition
**POST** `/recognize/batch/`

Process multiple images in a single request.

**Request Parameters:**
- `images` (array of files, optional): Multiple image files
- `images_base64` (array of strings, optional): Multiple base64 encoded images
- `include_faces` (boolean, default: true): Include face detection
- `include_segmentation` (boolean, default: true): Include segmentation
- `include_visualization` (boolean, default: false): Include annotated images

**Example Request:**
```bash
curl -X POST http://localhost:8000/api/v1/recognize/batch/ \
  -F "images=@fish1.jpg" \
  -F "images=@fish2.jpg" \
  -F "include_faces=true"
```

**Response:**
```json
{
  "results": [
    {
      "success": true,
      "image_index": 0,
      "fish_detections": [...],
      "faces": [...],
      "processing_time": {...}
    },
    {
      "success": true,
      "image_index": 1,
      "fish_detections": [...],
      "faces": [...],
      "processing_time": {...}
    }
  ],
  "total_images": 2,
  "successful_images": 2,
  "failed_images": 0,
  "total_processing_time": 1.34
}
```

### Model Configuration
**GET** `/config/`

Get current model configuration.

**Response:**
```json
{
  "confidence_threshold": 0.5,
  "nms_threshold": 0.3,
  "segmentation_threshold": 0.5,
  "device": "cpu",
  "enable_caching": true,
  "batch_size": 1
}
```

**POST** `/config/`

Update model configuration (requires restart for changes to take effect).

**Request:**
```json
{
  "confidence_threshold": 0.7,
  "processing_mode": "accuracy"
}
```

### Face Filter Configuration
**GET** `/config/face-filter/`

Get current face filter configuration. The face filter prevents human faces from being detected as fish.

**Response:**
```json
{
  "enabled": true,
  "iou_threshold": 0.3,
  "description": "Face filter prevents human faces from being detected as fish"
}
```

**POST** `/config/face-filter/`

Update face filter configuration.

**Request:**
```json
{
  "enabled": true,
  "iou_threshold": 0.3
}
```

**Response:**
```json
{
  "message": "Face filter configuration updated successfully",
  "config": {
    "enabled": true,
    "iou_threshold": 0.3
  }
}
```

**Parameters:**
- `enabled` (boolean): Whether to enable face filtering (default: true)
- `iou_threshold` (float): IoU threshold for detecting face-fish overlap (0.0-1.0, default: 0.3)

## WebSocket API

### Connection
Connect to the WebSocket endpoint for real-time fish recognition:

```javascript
const ws = new WebSocket('ws://localhost:8000/ws/recognition/');
```

### Message Format

All WebSocket messages follow this format:
```json
{
  "type": "message_type",
  "data": { ... },
  "timestamp": "2024-01-01T12:00:00Z"
}
```

### Client to Server Messages

#### Camera Frame
Send camera frame for processing:
```json
{
  "type": "camera_frame",
  "data": {
    "frame_data": "data:image/jpeg;base64,/9j/4AAQSkZJRgABAQAAAQ...",
    "frame_id": 123,
    "include_faces": true,
    "include_segmentation": true,
    "quality_threshold": 0.3
  }
}
```

#### Settings Update
Update processing settings:
```json
{
  "type": "settings_update",
  "data": {
    "include_faces": false,
    "include_segmentation": true,
    "processing_mode": "accuracy",
    "quality_threshold": 0.4,
    "min_processing_interval": 0.5
  }
}
```

#### Get Statistics
Request current session statistics:
```json
{
  "type": "get_stats",
  "data": {}
}
```

#### Ping
Send heartbeat ping:
```json
{
  "type": "ping",
  "data": {}
}
```

### Server to Client Messages

#### Connection Established
```json
{
  "type": "connection_established",
  "data": {
    "message": "Connected to Fish Recognition WebSocket",
    "channel": "channel_name",
    "settings": { ... },
    "session_id": "1640995200"
  }
}
```

#### Recognition Result
```json
{
  "type": "recognition_result",
  "data": {
    "frame_id": 123,
    "processing_time": 0.68,
    "results": {
      "fish_detections": [...],
      "faces": [...],
      "processing_time": {...}
    }
  }
}
```

#### Quality Warning
```json
{
  "type": "quality_warning",
  "data": {
    "message": "Poor image quality detected",
    "validation": {
      "quality_score": 0.15,
      "warnings": ["Image appears to be too dark"],
      "recommendations": ["Increase lighting or brightness"]
    }
  }
}
```

#### Session Statistics
```json
{
  "type": "session_stats",
  "data": {
    "session_duration": 300.5,
    "frames_received": 150,
    "frames_processed": 120,
    "frames_skipped": 30,
    "processing_rate": 0.4,
    "avg_processing_time": 0.65
  }
}
```

#### Frame Skipped
```json
{
  "type": "frame_skipped",
  "data": {
    "reason": "Processing in progress or too frequent",
    "last_processing_time": 1640995200.123
  }
}
```

#### Error Messages
```json
{
  "type": "frame_error",
  "data": {
    "message": "Failed to process frame",
    "error": "Invalid image format"
  }
}
```

#### Heartbeat
```json
{
  "type": "heartbeat",
  "data": {
    "processing_active": false
  }
}
```

## Processing Modes

### Accuracy Mode (Default)
- Higher quality thresholds
- Longer processing intervals (0.5s minimum)
- More thorough validation
- Best for detailed analysis

### Speed Mode
- Lower quality thresholds
- Shorter processing intervals (0.1s minimum)
- Faster processing
- Better for real-time monitoring

## Image Quality Requirements

### Minimum Requirements
- **Resolution**: 224x224 pixels minimum
- **File Size**: 10MB maximum
- **Formats**: JPEG, PNG, WebP
- **Quality Score**: 0.3 minimum (0-1 scale)

### Quality Factors
- **Sharpness**: Measured using Laplacian variance
- **Contrast**: Standard deviation of pixel values
- **Brightness**: Average pixel intensity
- **Blur Detection**: Automatic blur detection

### Recommendations for Best Results
1. **Lighting**: Ensure adequate, even lighting
2. **Focus**: Keep fish in sharp focus
3. **Resolution**: Use at least 640x640 pixels
4. **Angle**: Capture fish from the side for best classification
5. **Background**: Use contrasting background colors
6. **Distance**: Keep fish filling 30-70% of the frame

## Error Handling

### HTTP Status Codes
- **200**: Success
- **400**: Bad Request (validation errors)
- **500**: Internal Server Error

### Error Response Format
```json
{
  "success": false,
  "error": "Error message",
  "validation_errors": ["Field specific errors"],
  "quality_validation": {
    "valid": false,
    "errors": ["Quality specific errors"]
  }
}
```

## Performance Optimization

### Caching
- **Redis**: Model predictions cached for 1 hour
- **Memory**: Recent results cached in memory
- **Image Hashing**: SHA256 hashing for cache keys

### Batch Processing
- **Optimal Batch Size**: 4 images
- **Parallel Processing**: Models process batches efficiently
- **Memory Management**: Automatic cleanup of large batches

### Model Optimization
- **TorchScript**: Pre-compiled models for faster inference
- **CPU Optimization**: Optimized for CPU inference
- **Memory Efficiency**: Minimal memory footprint

## Development Setup

### Requirements
```bash
pip install -r requirements.txt
```

### Environment Variables
Create a `.env` file:
```env
DEBUG=True
SECRET_KEY=your-secret-key
ALLOWED_HOSTS=localhost,127.0.0.1
```

### Database Migration
```bash
python manage.py migrate
```

### Run Development Server
```bash
# HTTP server
python manage.py runserver

# WebSocket server (if using separate process)
daphne -b 0.0.0.0 -p 8001 fish_recognition_api.asgi:application
```

### Redis Setup (for caching and WebSocket)
```bash
# Install Redis
brew install redis  # macOS
# or
sudo apt-get install redis-server  # Ubuntu

# Start Redis
redis-server
```

## Deployment

### Production Configuration
1. Set `DEBUG=False`
2. Configure proper `ALLOWED_HOSTS`
3. Use environment variables for sensitive settings
4. Set up Redis for caching and WebSocket
5. Configure proper logging

### Docker Deployment
```dockerfile
FROM python:3.9-slim

WORKDIR /app
COPY requirements.txt .
RUN pip install -r requirements.txt

COPY . .

EXPOSE 8000
CMD ["daphne", "-b", "0.0.0.0", "-p", "8000", "fish_recognition_api.asgi:application"]
```

### Scaling Considerations
- Use multiple worker processes
- Implement load balancing
- Consider GPU acceleration for production
- Monitor memory usage with large batches
- Use dedicated Redis cluster for high traffic

## Monitoring and Logging

### Logs Location
- **Application Logs**: `fish_api.log`
- **Django Logs**: Console output
- **WebSocket Logs**: Included in application logs

### Metrics Available
- Request count and success rate
- Processing times per model
- Cache hit rates
- WebSocket connection statistics
- Image quality statistics

### Health Monitoring
Regular health checks should monitor:
- Model loading status
- Memory usage
- Processing queue length
- Redis connectivity
- Average response times

## Troubleshooting

### Common Issues

1. **Models Not Loading**
   - Check model file paths in settings
   - Verify model files exist and are accessible
   - Check PyTorch version compatibility

2. **Poor Recognition Accuracy**
   - Verify image quality (lighting, focus, resolution)
   - Check confidence thresholds
   - Ensure fish is clearly visible and properly framed

3. **Slow Processing**
   - Enable caching in settings
   - Use batch processing for multiple images
   - Consider reducing image resolution
   - Monitor memory usage

4. **WebSocket Connection Issues**
   - Verify Redis is running
   - Check CORS settings
   - Ensure proper WebSocket URL
   - Monitor connection timeouts

5. **Memory Issues**
   - Reduce batch size
   - Enable garbage collection
   - Monitor memory usage patterns
   - Consider using GPU if available

### Debug Mode
Enable debug logging by setting log level to DEBUG in Django settings.

### Support
For technical support or questions, please refer to the API logs and error messages for detailed troubleshooting information.