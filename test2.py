import cv2
import numpy as np
import ncnn

# Initialize NCNN network
net = ncnn.Net()
param_path = "int8.param"  # Adjust to your local path
bin_path = "int8.bin"     # Adjust to your local path
if net.load_param(param_path) != 0 or net.load_model(bin_path) != 0:
    print(f"Error: Failed to load NCNN model from {param_path} or {bin_path}")
    exit()

# Initialize camera (0 for default webcam)
cap = cv2.VideoCapture(0)
if not cap.isOpened():
    print("Error: Cannot open camera")
    exit()

while True:
    ret, frame = cap.read()
    if not ret:
        print("Error: Empty frame")
        break

    # Preprocess: Resize and convert to NCNN Mat
    resized = cv2.resize(frame, (224, 224))  # Match model input size
    img = np.ascontiguousarray(resized, dtype=np.float32)
    # Workaround for PIXEL_RGB issue: Manually create and populate Mat
    mat_in = ncnn.Mat(224, 224, 3)  # 224x224x3 matrix
    for i in range(224):
        for j in range(224):
            mat_in[i, j, 0] = img[i, j, 0]  # B
            mat_in[i, j, 1] = img[i, j, 1]  # G
            mat_in[i, j, 2] = img[i, j, 2]  # R

    # Run inference
    ex = net.create_extractor()
    ex.set_num_threads(4)  # Use available cores
    ex.input("input", mat_in)  # Replace with your model's input layer name
    mat_out = ncnn.Mat()
    ex.extract("output", mat_out)  # Replace with your model's output layer name

    # Post-process: Example for classification (adjust for YOLO)
    max_score = float('-inf')
    max_idx = 0
    for i in range(mat_out.h):
        if mat_out[i] > max_score:
            max_score = mat_out[i]
            max_idx = i

    # Display result on frame
    cv2.putText(frame, f"Class: {max_idx} ({max_score:.2f})", (10, 30),
                cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
    cv2.imshow("Camera Feed", frame)
    if cv2.waitKey(1) == 27:  # ESC to exit
        break

cap.release()
cv2.destroyAllWindows()