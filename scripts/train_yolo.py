from ultralytics import YOLO
# Load a pretrained YOLO model (recommended for training)
model = YOLO('yolov8n.pt')

# Train the model using the 'coco128.yaml' dataset for 3 epochs
#results = model.train(data='training_data.yaml', epochs=100, batch=32, workers=0)

# Evaluate the model's performance on the validation set
#results = model.val()

# Perform object detection on an image using the model
#results = model('https://ultralytics.com/images/zidane.jpg', visualize=True)

# Export the model to ONNX format
#success = model.export(format='onnx')