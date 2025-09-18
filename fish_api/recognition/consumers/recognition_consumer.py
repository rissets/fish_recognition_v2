"""
WebSocket Consumer for Real-time Fish Recognition
Handles live camera feed processing with focus on accuracy over speed
"""

import json
import asyncio
import logging
import time
from datetime import datetime
from typing import Dict, Any, Optional

from channels.generic.websocket import AsyncWebsocketConsumer
from channels.db import database_sync_to_async
from django.conf import settings

from ..ml_models.fish_engine import get_fish_engine
from ..utils.image_utils import base64_to_image, ImageQualityValidator, draw_detection_results, image_to_base64
from ..serializers import CameraFrameSerializer, WebSocketMessageSerializer

logger = logging.getLogger(__name__)


class RecognitionConsumer(AsyncWebsocketConsumer):
    """
    WebSocket consumer for real-time fish recognition
    Optimized for accuracy with adaptive processing
    """
    
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.processing_active = False
        self.last_processing_time = 0
        self.frame_count = 0
        self.session_stats = {
            'frames_received': 0,
            'frames_processed': 0,
            'frames_skipped': 0,
            'total_processing_time': 0,
            'session_start': None,
            'last_recognition_time': None
        }
        
        # Adaptive processing settings
        self.min_processing_interval = 0.5  # Minimum seconds between processing
        self.quality_threshold = 0.3
        self.adaptive_quality = True
        
        # Client settings
        self.client_settings = {
            'include_faces': True,
            'include_segmentation': True,
            'include_visualization': True,  # Enable visualization by default for live stream
            'auto_process': True,
            'processing_mode': 'accuracy'  # 'accuracy' or 'speed'
        }
    
    async def connect(self):
        """Handle WebSocket connection"""
        try:
            await self.accept()
            self.session_stats['session_start'] = datetime.now()
            
            logger.info(f"WebSocket connected: {self.channel_name}")
            
            # Send welcome message
            await self.send_message('connection_established', {
                'message': 'Connected to Fish Recognition WebSocket',
                'channel': self.channel_name,
                'settings': self.client_settings,
                'session_id': str(int(time.time()))
            })
            
        except Exception as e:
            logger.error(f"WebSocket connection failed: {str(e)}")
            await self.close()
    
    async def disconnect(self, close_code):
        """Handle WebSocket disconnection"""
        logger.info(f"WebSocket disconnected: {self.channel_name}, code: {close_code}")
        
        # Send final stats if possible
        try:
            await self.send_session_stats()
        except:
            pass
    
    async def receive(self, text_data):
        """Handle incoming WebSocket messages"""
        try:
            data = json.loads(text_data)
            message_type = data.get('type', 'unknown')
            
            logger.debug(f"Received message type: {message_type}")
            
            if message_type == 'camera_frame':
                await self.handle_camera_frame(data.get('data', {}))
            
            elif message_type == 'settings_update':
                await self.handle_settings_update(data.get('data', {}))
            
            elif message_type == 'get_stats':
                await self.send_session_stats()
            
            elif message_type == 'ping':
                await self.send_message('pong', {'timestamp': datetime.now().isoformat()})
            
            else:
                await self.send_message('error', {
                    'message': f'Unknown message type: {message_type}',
                    'supported_types': ['camera_frame', 'settings_update', 'get_stats', 'ping']
                })
                
        except json.JSONDecodeError as e:
            await self.send_message('error', {
                'message': 'Invalid JSON format',
                'error': str(e)
            })
        except Exception as e:
            logger.error(f"Error handling WebSocket message: {str(e)}")
            await self.send_message('error', {
                'message': 'Internal server error',
                'error': str(e)
            })
    
    async def handle_camera_frame(self, frame_data: Dict[str, Any]):
        """Process incoming camera frame"""
        try:
            self.session_stats['frames_received'] += 1
            
            # Validate frame data
            serializer = CameraFrameSerializer(data=frame_data)
            if not serializer.is_valid():
                await self.send_message('frame_error', {
                    'message': 'Invalid frame data',
                    'errors': serializer.errors
                })
                return
            
            validated_data = serializer.validated_data
            
            # Check if we should process this frame (adaptive processing)
            if not await self.should_process_frame():
                self.session_stats['frames_skipped'] += 1
                await self.send_message('frame_skipped', {
                    'reason': 'Processing in progress or too frequent',
                    'last_processing_time': self.last_processing_time
                })
                return
            
            # Set processing flag
            self.processing_active = True
            frame_start_time = time.time()
            
            try:
                # Process the frame
                result = await self.process_frame(validated_data)
                
                if result:
                    self.session_stats['frames_processed'] += 1
                    processing_time = time.time() - frame_start_time
                    self.session_stats['total_processing_time'] += processing_time
                    self.session_stats['last_recognition_time'] = datetime.now()
                    
                    # Send results
                    await self.send_message('recognition_result', {
                        'frame_id': validated_data.get('frame_id', self.frame_count),
                        'processing_time': processing_time,
                        'timestamp': datetime.now().isoformat(),
                        'results': result
                    })
                    
                    # Send stats periodically
                    if self.session_stats['frames_processed'] % 10 == 0:
                        await self.send_session_stats()
                
            finally:
                self.processing_active = False
                self.last_processing_time = time.time()
                self.frame_count += 1
                
        except Exception as e:
            logger.error(f"Error processing camera frame: {str(e)}")
            await self.send_message('frame_error', {
                'message': 'Failed to process frame',
                'error': str(e)
            })
            self.processing_active = False
    
    async def process_frame(self, frame_data: Dict[str, Any]) -> Optional[Dict[str, Any]]:
        """Process a single camera frame"""
        try:
            # Convert base64 to image
            image_bgr = await database_sync_to_async(base64_to_image)(frame_data['frame_data'])
            
            # Convert to bytes for engine processing
            import cv2
            _, buffer = cv2.imencode('.jpg', image_bgr)
            image_bytes = buffer.tobytes()
            
            # Validate image quality if enabled
            if self.adaptive_quality:
                validator = ImageQualityValidator(
                    min_quality_score=frame_data.get('quality_threshold', self.quality_threshold)
                )
                validation_result = await database_sync_to_async(validator.validate)(image_bytes)
                
                if not validation_result['valid']:
                    await self.send_message('quality_warning', {
                        'message': 'Poor image quality detected',
                        'validation': validation_result
                    })
                    
                    # Skip processing if quality is too low
                    if validation_result.get('quality_score', 0) < self.quality_threshold * 0.5:
                        return None
            
            # Get fish recognition engine
            engine = await database_sync_to_async(get_fish_engine)()
            
            # Process image with engine
            recognition_results = await database_sync_to_async(engine.process_image)(
                image_data=image_bytes,
                include_faces=frame_data.get('include_faces', self.client_settings['include_faces']),
                include_segmentation=frame_data.get('include_segmentation', self.client_settings['include_segmentation'])
            )
            
            # Generate visualization if requested
            include_visualization = frame_data.get('include_visualization', self.client_settings.get('include_visualization', False))
            logger.info(f"include_visualization setting: {include_visualization}")
            logger.info(f"client_settings: {self.client_settings}")
            
            if include_visualization:
                try:
                    logger.info("Generating visualization for WebSocket frame")
                    visualization = await database_sync_to_async(draw_detection_results)(image_bgr, recognition_results)
                    recognition_results['visualization_image'] = await database_sync_to_async(image_to_base64)(visualization)
                    logger.info("Visualization generated successfully")
                except Exception as e:
                    logger.warning(f"Failed to generate visualization: {str(e)}")
                    recognition_results['visualization_image'] = None
            else:
                logger.info("Visualization not requested, skipping generation")
            
            return recognition_results
            
        except Exception as e:
            logger.error(f"Frame processing failed: {str(e)}")
            raise
    
    async def should_process_frame(self) -> bool:
        """Determine if current frame should be processed"""
        current_time = time.time()
        
        # Don't process if already processing
        if self.processing_active:
            return False
        
        # Respect minimum processing interval
        if current_time - self.last_processing_time < self.min_processing_interval:
            return False
        
        # Auto-processing check
        if not self.client_settings.get('auto_process', True):
            return False
        
        return True
    
    async def handle_settings_update(self, settings_data: Dict[str, Any]):
        """Update client settings"""
        try:
            # Update settings
            for key, value in settings_data.items():
                if key in self.client_settings:
                    self.client_settings[key] = value
            
            # Update adaptive settings
            if 'quality_threshold' in settings_data:
                self.quality_threshold = float(settings_data['quality_threshold'])
            
            if 'min_processing_interval' in settings_data:
                self.min_processing_interval = float(settings_data['min_processing_interval'])
            
            if 'processing_mode' in settings_data:
                mode = settings_data['processing_mode']
                if mode == 'speed':
                    self.min_processing_interval = 0.1
                    self.quality_threshold = 0.2
                elif mode == 'accuracy':
                    self.min_processing_interval = 0.5
                    self.quality_threshold = 0.3
            
            await self.send_message('settings_updated', {
                'message': 'Settings updated successfully',
                'current_settings': self.client_settings,
                'adaptive_settings': {
                    'quality_threshold': self.quality_threshold,
                    'min_processing_interval': self.min_processing_interval
                }
            })
            
        except Exception as e:
            await self.send_message('settings_error', {
                'message': 'Failed to update settings',
                'error': str(e)
            })
    
    async def send_session_stats(self):
        """Send current session statistics"""
        try:
            current_time = datetime.now()
            session_duration = (current_time - self.session_stats['session_start']).total_seconds()
            
            avg_processing_time = (
                self.session_stats['total_processing_time'] / self.session_stats['frames_processed']
                if self.session_stats['frames_processed'] > 0 else 0
            )
            
            processing_rate = (
                self.session_stats['frames_processed'] / session_duration
                if session_duration > 0 else 0
            )
            
            stats = {
                'session_duration': session_duration,
                'frames_received': self.session_stats['frames_received'],
                'frames_processed': self.session_stats['frames_processed'],
                'frames_skipped': self.session_stats['frames_skipped'],
                'processing_rate': processing_rate,
                'avg_processing_time': avg_processing_time,
                'last_recognition': (
                    self.session_stats['last_recognition_time'].isoformat()
                    if self.session_stats['last_recognition_time'] else None
                )
            }
            
            await self.send_message('session_stats', stats)
            
        except Exception as e:
            logger.error(f"Failed to send session stats: {str(e)}")
    
    async def send_message(self, message_type: str, data: Dict[str, Any]):
        """Send a formatted message to the client"""
        try:
            message = {
                'type': message_type,
                'data': data,
                'timestamp': datetime.now().isoformat()
            }
            
            await self.send(text_data=json.dumps(message))
            
        except Exception as e:
            logger.error(f"Failed to send message: {str(e)}")
    
    async def send_heartbeat(self):
        """Send periodic heartbeat to keep connection alive"""
        while True:
            try:
                if self.channel_layer:
                    await self.send_message('heartbeat', {
                        'timestamp': datetime.now().isoformat(),
                        'processing_active': self.processing_active
                    })
                await asyncio.sleep(30)  # Send heartbeat every 30 seconds
            except:
                break