import cv2
import numpy as np
from deepface import DeepFace
import threading
import time

class FacialExpressionAnalyzer:
    def __init__(self):
        print("Initializing Facial Expression Analyzer with DeepFace...")
        
        # Emotion colors for visualization
        self.emotion_colors = {
            'happy': (0, 255, 0),      # Green
            'sad': (255, 0, 0),        # Blue
            'angry': (0, 0, 255),      # Red
            'fear': (128, 0, 128),     # Purple
            'surprise': (0, 255, 255), # Yellow
            'disgust': (0, 128, 0),    # Dark Green
            'neutral': (128, 128, 128) # Gray
        }
        
        # Load face cascade for backup detection
        self.face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')
        
        print("Analyzer ready!")
    
    def analyze_frame_emotions(self, frame):
        """Analyze emotions in a frame using DeepFace"""
        try:
            # DeepFace.analyze returns emotion analysis
            result = DeepFace.analyze(frame, actions=['emotion'], enforce_detection=False)
            
            # Handle both single face and multiple faces
            if isinstance(result, list):
                return result
            else:
                return [result]
                
        except Exception as e:
            print(f"Analysis error: {e}")
            return []
    
    def draw_results(self, frame, analysis_results):
        """Draw emotion results on the frame"""
        for result in analysis_results:
            try:
                # Get face region
                face_region = result.get('region', {})
                x = face_region.get('x', 0)
                y = face_region.get('y', 0)
                w = face_region.get('w', 100)
                h = face_region.get('h', 100)
                
                # Get emotions
                emotions = result.get('emotion', {})
                
                if emotions:
                    # Get dominant emotion
                    dominant_emotion = result.get('dominant_emotion', 'neutral').lower()
                    confidence = emotions.get(dominant_emotion, 0)
                    
                    # Get color
                    color = self.emotion_colors.get(dominant_emotion, (255, 255, 255))
                    
                    # Draw bounding box
                    cv2.rectangle(frame, (x, y), (x+w, y+h), color, 2)
                    
                    # Draw emotion label
                    label = f"{dominant_emotion.upper()}: {confidence:.1f}%"
                    
                    # Background for text
                    (text_width, text_height), _ = cv2.getTextSize(
                        label, cv2.FONT_HERSHEY_SIMPLEX, 0.7, 2)
                    
                    cv2.rectangle(frame, (x, y - text_height - 10), 
                                 (x + text_width + 10, y), color, -1)
                    
                    # Text
                    cv2.putText(frame, label, (x + 5, y - 5), 
                               cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
                    
                    # Draw emotion bars
                    self.draw_emotion_bars(frame, emotions, x, y + h + 20, w)
                    
            except Exception as e:
                print(f"Drawing error: {e}")
        
        return frame
    
    def draw_emotion_bars(self, frame, emotions, x, y, width):
        """Draw emotion confidence bars"""
        bar_height = 12
        bar_spacing = 15
        
        sorted_emotions = sorted(emotions.items(), key=lambda item: item[1], reverse=True)
        
        for i, (emotion, confidence) in enumerate(sorted_emotions[:5]):  # Show top 5
            bar_y = y + i * bar_spacing
            bar_width = int((width * confidence) / 100)
            
            # Background bar
            cv2.rectangle(frame, (x, bar_y), (x + width, bar_y + bar_height), 
                         (200, 200, 200), 1)
            
            # Filled bar
            if bar_width > 0:
                color = self.emotion_colors.get(emotion.lower(), (255, 255, 255))
                cv2.rectangle(frame, (x + 1, bar_y + 1), 
                             (x + bar_width - 1, bar_y + bar_height - 1), color, -1)
            
            # Emotion text
            cv2.putText(frame, f"{emotion}: {confidence:.1f}%", 
                       (x + width + 10, bar_y + 10), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.4, (0, 0, 0), 1)
    
    def analyze_webcam(self):
        """Real-time webcam emotion analysis"""
        cap = cv2.VideoCapture(0)
        
        if not cap.isOpened():
            print("Error: Could not open webcam")
            return
        
        print("Starting webcam analysis...")
        print("Note: First analysis may take a few seconds while model loads")
        
        frame_count = 0
        analysis_interval = 3  # Analyze every 10 frames for better performance
        current_results = []
        
        while True:
            ret, frame = cap.read()
            if not ret:
                break
            
            # Flip for mirror effect
            frame = cv2.flip(frame, 1)
            
            # Analyze emotions every few frames (for performance)
            if frame_count % analysis_interval == 0:
                try:
                    current_results = self.analyze_frame_emotions(frame)
                except Exception as e:
                    print(f"Analysis error: {e}")
                    current_results = []
            
            # Draw results from last analysis
            if current_results:
                frame = self.draw_results(frame, current_results)
            
            # Add instructions
            cv2.putText(frame, "Press 'q' to quit, 's' to save", (10, 30), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
            
            # Display frame
            cv2.imshow('Facial Expression Analysis - DeepFace', frame)
            
            # Handle key presses
            key = cv2.waitKey(1) & 0xFF
            if key == ord('q'):
                break
            elif key == ord('s'):
                cv2.imwrite('facial_emotion_result.jpg', frame)
                print("Screenshot saved!")
            
            frame_count += 1
        
        cap.release()
        cv2.destroyAllWindows()
    
    def analyze_image(self, image_path):
        """Analyze emotions from static image"""
        try:
            # Load image
            frame = cv2.imread(image_path)
            if frame is None:
                print(f"Error: Could not load image '{image_path}'")
                return
            
            print(f"Analyzing emotions in: {image_path}")
            
            # Analyze emotions
            results = self.analyze_frame_emotions(frame)
            
            if not results:
                print("No faces detected in the image")
                cv2.imshow('No Faces Detected', frame)
                cv2.waitKey(0)
                cv2.destroyAllWindows()
                return
            
            # Draw results
            frame = self.draw_results(frame, results)
            
            # Print detailed results
            print(f"\nEmotion Analysis Results:")
            print("=" * 40)
            
            for i, result in enumerate(results, 1):
                emotions = result.get('emotion', {})
                dominant = result.get('dominant_emotion', 'neutral')
                
                print(f"\nFace {i}:")
                print(f"Dominant Emotion: {dominant.upper()} ({emotions.get(dominant, 0):.1f}%)")
                print("All Emotions:")
                for emotion, confidence in sorted(emotions.items(), key=lambda x: x[1], reverse=True):
                    print(f"  {emotion.capitalize()}: {confidence:.1f}%")
            
            # Display result
            cv2.imshow('Facial Expression Analysis Results', frame)
            print("\nPress any key to close...")
            cv2.waitKey(0)
            cv2.destroyAllWindows()
            
        except Exception as e:
            print(f"Error: {e}")
    
    def run(self):
        """Main program loop"""
        print("="*60)
        print("FACIAL EXPRESSION SENTIMENT ANALYSIS")
        print("="*60)
        print("Using DeepFace pre-trained models for high accuracy")
        print("\nDetectable Emotions: Happy, Sad, Angry, Fear, Surprise, Disgust, Neutral")
        
        while True:
            print("\n" + "="*40)
            print("Select Analysis Mode:")
            print("1. Real-time Webcam Analysis")
            print("2. Analyze Image File")
            print("3. Exit")
            
            choice = input("\nEnter choice (1-3): ").strip()
            
            if choice == '1':
                self.analyze_webcam()
            elif choice == '2':
                image_path = input("Enter image path: ").strip()
                if image_path:
                    self.analyze_image(image_path)
                else:
                    print("Please provide a valid image path")
            elif choice == '3':
                print("Goodbye!")
                break
            else:
                print("Invalid choice. Please enter 1, 2, or 3.")

# Example usage and testing
def test_with_sample():
    """Quick test function"""
    analyzer = FacialExpressionAnalyzer()
    
    # You can test with webcam or provide image path
    print("Testing facial expression analysis...")
    
    # For quick webcam test
    analyzer.analyze_webcam()

if __name__ == "__main__":
    test_with_sample()