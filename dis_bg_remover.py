import onnxruntime as ort
import numpy as np
import cv2
import traceback

def normalize(image, mean, std):
    """Normalize a numpy image with mean and standard deviation."""
    return (image / 255.0 - mean) / std

def remove_background(model_path,image_path):
    if model_path == None or image_path==None:
        return None,None
    
    input_size = (1024, 1024)

    try:
        # Load the ONNX model
        session = ort.InferenceSession(model_path)
        im = cv2.imread(image_path, cv2.IMREAD_COLOR)
        im = cv2.cvtColor(im, cv2.COLOR_BGR2RGB)  # Convert from BGR to RGB if using OpenCV

            
        # If image is grayscale, convert to RGB
        if len(im.shape) == 2:
            im = cv2.cvtColor(im, cv2.COLOR_GRAY2RGB)
        
        # Normalize the image using NumPy
        im = im.astype(np.float32)  # Convert to float
        im_normalized = normalize(im, mean=[0.5, 0.5, 0.5], std=[1.0, 1.0, 1.0])
            
        # Resize the image
        im_resized = cv2.resize(im_normalized, input_size, interpolation=cv2.INTER_LINEAR)
        im_resized = np.transpose(im_resized, (2, 0, 1))  # CHW format
        im_resized = np.expand_dims(im_resized, axis=0)  # Add batch dimension

        # Run inference
        im_resized = im_resized.astype(np.float32)  
        ort_inputs = {session.get_inputs()[0].name: im_resized}
        ort_outs = session.run(None, ort_inputs)
            
        # Process the model output
        result = ort_outs[0][0]  # Assuming single output and single batch
        result = np.clip(result, 0, 1)  # Assuming you want to clip the result to [0, 1]
        result = (result * 255).astype(np.uint8)  # Rescale to [0, 255]
        result = np.transpose(result, (1, 2, 0))  # HWC format
        result_bgr = cv2.cvtColor(result, cv2.COLOR_RGB2BGR)
        # Resize to original shape
        original_shape = im.shape[:2]
        result = cv2.resize(result, (original_shape[1], original_shape[0]), interpolation=cv2.INTER_LINEAR)

        # Ensure 'result' is 2D (H x W) and add an axis to make it (H x W x 1)
        alpha_channel = result[:, :, np.newaxis]

        # Concatenate the RGB channels of 'im' with the alpha channel
        im_rgba = np.concatenate((im, alpha_channel), axis=2)
        im_bgra = cv2.cvtColor(im_rgba, cv2.COLOR_RGBA2BGRA)

        return im_bgra, result
    except Exception as e:
        print("An error occurred:")
        traceback.print_exc()
        return None,None