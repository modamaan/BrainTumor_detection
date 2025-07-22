import gradio as gr
from ultralytics import YOLO
import cv2
import numpy as np
from PIL import Image
import os

# Load the model
model = YOLO('runs/detect/train/weights/best.pt')

def predict_brain_tumor(image):
    """
    Predict brain tumor from uploaded image
    """
    try:
        # Convert PIL image to numpy array if needed
        if isinstance(image, Image.Image):
            image_array = np.array(image)
        else:
            image_array = image
        
        # Run prediction
        results = model.predict(source=image_array, imgsz=640, conf=0.25)
        
        # Get annotated image
        annotated_image = results[0].plot()
        
        # Convert BGR to RGB for proper display
        rgb_image = annotated_image[:, :, ::-1]
        
        # Get detection information
        detections = results[0].boxes
        
        if detections is not None and len(detections) > 0:
            # Extract detection details
            num_detections = len(detections)
            confidences = detections.conf.cpu().numpy()
            classes = detections.cls.cpu().numpy()
            
            # Get class names
            class_names = model.names
            
            # Create detection summary
            detection_info = f"**Detections Found: {num_detections}**\n\n"
            
            for i, (conf, cls) in enumerate(zip(confidences, classes)):
                class_name = class_names[int(cls)]
                detection_info += f"• Detection {i+1}: {class_name} (Confidence: {conf:.2%})\n"
            
            status = "⚠️ Brain tumor detected!"
        else:
            detection_info = "No brain tumors detected in the image."
            status = "✅ No tumors detected"
        
        return rgb_image, detection_info, status
        
    except Exception as e:
        error_msg = f"Error processing image: {str(e)}"
        return None, error_msg, "❌ Processing failed"

# Create Gradio interface
with gr.Blocks(title="Brain Tumor Detection", theme=gr.themes.Soft()) as demo:
    gr.Markdown(
        """
        # 🧠 Brain Tumor Detection System
        
        Upload an MRI brain scan image to detect potential tumors using AI-powered analysis.
        
        **Instructions:**
        1. Upload a brain MRI image (JPG, PNG, etc.)
        2. Click "Analyze Image" to run detection
        3. View the results with annotated regions and confidence scores
        """
    )
    
    with gr.Row():
        with gr.Column(scale=1):
            # Input section
            gr.Markdown("### 📤 Upload Image")
            input_image = gr.Image(
                label="Upload Brain MRI Scan",
                type="pil",
                height=400
            )
            
            analyze_btn = gr.Button(
                "🔍 Analyze Image", 
                variant="primary",
                size="lg"
            )
            
            # Status display
            status_display = gr.Markdown(
                "**Status:** Ready for analysis",
                elem_id="status"
            )
        
        with gr.Column(scale=1):
            # Output section
            gr.Markdown("### 📊 Analysis Results")
            output_image = gr.Image(
                label="Annotated Results",
                height=400
            )
            
            detection_info = gr.Markdown(
                "Upload an image and click 'Analyze Image' to see results.",
                label="Detection Details"
            )
    
    # Example images section
    gr.Markdown("### 📋 Example Images")
    gr.Markdown("Click on any example below to load it for testing:")
    
    # Try to find example images in the dataset
    example_paths = []
    possible_dirs = [
        "MRI-3/valid/images",
        "MRI-3/test/images", 
        "dataset/valid/images",
        "dataset/test/images"
    ]
    
    for dir_path in possible_dirs:
        if os.path.exists(dir_path):
            files = [f for f in os.listdir(dir_path) if f.lower().endswith(('.jpg', '.jpeg', '.png'))]
            example_paths = [os.path.join(dir_path, f) for f in files[:3]]  # Take first 3 images
            break
    
    if example_paths:
        gr.Examples(
            examples=[[path] for path in example_paths],
            inputs=input_image,
            label="Sample MRI Images"
        )
    
    # Event handlers
    analyze_btn.click(
        fn=predict_brain_tumor,
        inputs=[input_image],
        outputs=[output_image, detection_info, status_display]
    )
    
    # Footer
    gr.Markdown(
        """
        ---
        **Note:** This is an AI-powered detection system for educational/research purposes. 
        Always consult with medical professionals for actual diagnosis.
        """
    )

if __name__ == "__main__":
    # Launch the interface
    demo.launch(
        share=True,  # Creates a public link
        server_name="0.0.0.0",  # Allow external connections
        server_port=7860,
        show_error=True
    )