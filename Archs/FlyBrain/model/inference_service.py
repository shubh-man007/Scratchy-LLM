import os
import json
import logging
from typing import Dict, Any
import boto3
import psycopg2
from kafka import KafkaConsumer
from dotenv import load_dotenv
import torch
import cv2
import numpy as np
from PIL import Image
import io
import tempfile

from utils import FlyDepth, load_and_preprocess_image, apply_guided_filter, enhance_sharpness

load_dotenv()

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

class InferenceService:
    def __init__(self):
        self.s3_client = boto3.client(
            's3',
            aws_access_key_id=os.getenv('AWS_ACCESS_KEY_ID'),
            aws_secret_access_key=os.getenv('AWS_SECRET_ACCESS_KEY'),
            region_name=os.getenv('AWS_REGION')
        )
        
        self.consumer = KafkaConsumer(
            'frames-to-process',
            bootstrap_servers=['localhost:9092'],
            value_deserializer=lambda m: json.loads(m.decode('utf-8')),
            auto_offset_reset='earliest',
            group_id='inference-group'
        )
        
        self.db_conn = psycopg2.connect(os.getenv('DATABASE_URL'))
        
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.model = self.load_model()
        
        self.img_height = 480
        self.img_width = 640
        
        logger.info(f"Initialized inference service on device: {self.device}")
    
    def load_model(self):
        try:
            model = FlyDepth(cnn_out_channels=64, gnn_hidden_dim=64, num_gnn_layers=2, fly_embed_dim=64)
            
            checkpoint_path = "cnn2gnn_flyprior.pth"
            checkpoint = torch.load(checkpoint_path, map_location=self.device)
            
            model.load_state_dict(checkpoint['model_state_dict'])
            model.to(self.device)
            model.eval()
            
            logger.info("Successfully loaded FlyDepth model")
            return model
            
        except Exception as e:
            logger.error(f"Failed to load model: {e}")
            raise
    
    def download_frame_from_s3(self, s3_path: str) -> str:
        bucket = s3_path.split('/')[2]
        key = '/'.join(s3_path.split('/')[3:])
        
        try:
            temp_file = tempfile.NamedTemporaryFile(delete=False, suffix='.jpg')
            temp_path = temp_file.name
            temp_file.close()
            
            self.s3_client.download_file(bucket, key, temp_path)
            logger.info(f"Downloaded frame from S3: {s3_path}")
            return temp_path
            
        except Exception as e:
            logger.error(f"Failed to download frame from S3: {e}")
            raise
    
    def run_inference(self, image_path: str) -> np.ndarray:
        try:
            img_tensor, rgb_img = load_and_preprocess_image(image_path, self.img_height, self.img_width)
            img_tensor = img_tensor.to(self.device)
            
            with torch.no_grad():
                pred_depth = self.model(img_tensor)
            
            pred_depth = torch.nn.functional.interpolate(
                pred_depth, 
                size=(self.img_height, self.img_width), 
                mode='bilinear', 
                align_corners=True
            )
            
            pred_depth_np = pred_depth.squeeze().cpu().numpy()
            
            pred_depth_norm = cv2.normalize(pred_depth_np, None, 0, 255, cv2.NORM_MINMAX)
            pred_depth_norm = np.uint8(pred_depth_norm)
            
            refined_depth = apply_guided_filter(rgb_img, pred_depth_norm)
            refined_depth_sharp = enhance_sharpness(refined_depth)
            
            logger.info("Successfully ran inference on frame")
            return refined_depth_sharp
            
        except Exception as e:
            logger.error(f"Failed to run inference: {e}")
            raise
    
    def upload_heatmap_to_s3(self, heatmap: np.ndarray, video_id: str, frame_number: int) -> str:
        bucket = os.getenv('S3_BUCKET')
        key = f"heatmaps/{video_id}/frame_{frame_number:04d}_heatmap.jpg"
        
        try:
            if heatmap.dtype != np.uint8:
                heatmap = heatmap.astype(np.uint8)
            
            heatmap_image = Image.fromarray(heatmap)
            img_buffer = io.BytesIO()
            heatmap_image.save(img_buffer, format='JPEG')
            img_buffer.seek(0)
            
            self.s3_client.put_object(
                Bucket=bucket,
                Key=key,
                Body=img_buffer.getvalue(),
                ContentType='image/jpeg'
            )
            
            s3_path = f"s3://{bucket}/{key}"
            logger.info(f"Uploaded heatmap to {s3_path}")
            return s3_path
            
        except Exception as e:
            logger.error(f"Failed to upload heatmap to S3: {e}")
            raise
    
    def update_frame_status(self, frame_id: str, status: str, heatmap_s3_path: str = None):
        try:
            cursor = self.db_conn.cursor()
            if heatmap_s3_path:
                cursor.execute(
                    "UPDATE frames SET status = %s, heatmap_s3_path = %s WHERE frame_id = %s",
                    (status, heatmap_s3_path, frame_id)
                )
            else:
                cursor.execute(
                    "UPDATE frames SET status = %s WHERE frame_id = %s",
                    (status, frame_id)
                )
            self.db_conn.commit()
            cursor.close()
            logger.info(f"Updated frame {frame_id} status to {status}")
        except Exception as e:
            logger.error(f"Failed to update frame status: {e}")
            self.db_conn.rollback()
            raise
    
    def process_frame(self, message: Dict[str, Any]):
        temp_file_path = None
        try:
            video_id = message['video_id']
            frame_id = message['frame_id']
            frame_number = message['frame_number']
            s3_path = message['s3_path']
            
            logger.info(f"Processing frame {frame_number} for video {video_id}")
            
            self.update_frame_status(frame_id, 'processing')
            
            temp_file_path = self.download_frame_from_s3(s3_path)
            
            heatmap = self.run_inference(temp_file_path)
            
            heatmap_s3_path = self.upload_heatmap_to_s3(heatmap, video_id, frame_number)
            
            self.update_frame_status(frame_id, 'done', heatmap_s3_path)
            
            logger.info(f"Successfully processed frame {frame_number}")
            
        except Exception as e:
            logger.error(f"Failed to process frame {frame_number}: {e}")
            self.update_frame_status(frame_id, 'failed')
        finally:
            if temp_file_path and os.path.exists(temp_file_path):
                os.unlink(temp_file_path)
    
    def run(self):
        logger.info("Starting inference service...")
        logger.info("Waiting for Kafka messages...")
        
        try:
            for message in self.consumer:
                logger.info(f"Received message: {message.value}")
                self.process_frame(message.value)
                
        except KeyboardInterrupt:
            logger.info("Shutting down inference service...")
        finally:
            self.consumer.close()
            self.db_conn.close()
            logger.info("Inference service stopped")

if __name__ == "__main__":
    service = InferenceService()
    service.run() 
