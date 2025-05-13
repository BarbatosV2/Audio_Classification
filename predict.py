import torch
import sounddevice as sd
import soundfile as sf
import keyboard 
import queue
import threading
import argparse
import os
import numpy as np # Added for concatenating audio chunks
from datetime import datetime
from model import AudioClassifier # LABELS will be loaded from the model file
from utils import get_device # Removed LABELS import from utils
import torchaudio # Added for loading audio
import torchaudio.transforms as T # Added for resampling

TARGET_SAMPLE_RATE = 16000
FIXED_AUDIO_LENGTH = 16000

# Global LABELS list, to be populated from the loaded model checkpoint
# This allows predict_file and predict_mic to access it.
LABELS = []

def predict_file(path, model, device, target_sr=TARGET_SAMPLE_RATE, fixed_length=FIXED_AUDIO_LENGTH):
    """
    Function to predict the label of an audio file.
    Args:
    - path (str): Path to the audio file.
    - model (nn.Module): The trained model.
    - device (torch.device): The device to run the model on (CPU or GPU).
    """
    try:
        waveform, sr = torchaudio.load(path, normalize=True)

        # Resample if needed
        if sr != target_sr:
            resampler = T.Resample(sr, target_sr)
            waveform = resampler(waveform)

        # Convert to mono
        if waveform.shape[0] > 1:
            waveform = torch.mean(waveform, dim=0, keepdim=True)

        # Pad or trim to fixed length
        current_len = waveform.shape[1]
        if current_len < fixed_length:
            padding = torch.zeros((1, fixed_length - current_len))
            waveform = torch.cat([waveform, padding], dim=1)
        else:
            waveform = waveform[:, :fixed_length]

        # Add batch dimension and channel dimension: [1, 1, fixed_length]
        input_tensor = waveform.unsqueeze(0).to(device) # Model expects [B, 1, N_samples]
        with torch.no_grad():
            output = model(input_tensor)  # Forward pass
            prediction = torch.argmax(output, dim=1).item()  # Get the predicted class
            print(f"Predicted: {LABELS[prediction]}")

            # Log the prediction to a text file
            log_file = os.path.join("recorded", "wav_predict.txt") 
            with open(log_file, "a") as f:
                f.write(f"{os.path.basename(path)}, {LABELS[prediction]}\n")
            print(f"Prediction logged to {log_file}")


    except Exception as e:
        print(f"Error processing file {path}: {e}")

def get_next_numbered_filename(log_dir, prefix, extension):
    """
    Determines the next available numbered filename (e.g., prefix_1.ext, prefix_2.ext).
    Args:
        log_dir (str): The directory where log files are stored.
        prefix (str): The prefix for the filename.
        extension (str): The file extension (e.g., ".txt", ".wav").
    Returns:
        str: The full path to the next log file.
    """
    os.makedirs(log_dir, exist_ok=True) # Ensure the directory exists
    i = 1
    while True:
        filename = os.path.join(log_dir, f"{prefix}_{i}{extension}")
        if not os.path.exists(filename):
            return filename
        i += 1

def format_timedelta_to_hms(td):
    """Formats a timedelta object to HH:MM:SS string."""
    total_seconds = int(td.total_seconds())
    hours, remainder = divmod(total_seconds, 3600)
    minutes, seconds = divmod(remainder, 60)
    return f"{hours:02}:{minutes:02}:{seconds:02}"

def log_detection(log_file_path, timestamp, session_time_str, label):
    """
    Appends a detection event to the specified log file.
    """
    with open(log_file_path, 'a') as f:
        f.write(f"{timestamp} (Session: {session_time_str}) - Detected: {label}\n")

def predict_mic(model, device, frame_duration=1.0, target_sr=TARGET_SAMPLE_RATE, fixed_length=FIXED_AUDIO_LENGTH, confidence_threshold=0.7, null_label="null"):
    """
    Real-time microphone prediction until Ctrl+C or 'q' is pressed.
    Args:
        model: Trained audio classification model.
        device: torch.device (CPU/GPU).
        frame_duration: Duration of each prediction window in seconds.
    """
    print(f"\n🎙️ Starting real-time audio classification (Confidence threshold: {confidence_threshold:.2f})...")
    log_dir = "recorded"
    text_log_file_path = get_next_numbered_filename(log_dir, "detect", ".txt")
    session_start_time = datetime.now() # Record session start time
    print("Press 'q' or Ctrl+C to stop.\n")

    assert int(target_sr * frame_duration) == fixed_length, "frame_duration and fixed_length are inconsistent with target_sr"
    audio_queue = queue.Queue()
    all_audio_data_chunks = [] # List to store all incoming audio chunks

    def callback(indata, frames, time, status):
        if status:
            print(f"[Warning] {status}")
        audio_queue.put(indata.copy())

    stop_flag = threading.Event()

    def keyboard_listener():
        keyboard.wait('q')
        stop_flag.set()

    # Start keyboard listener thread
    listener_thread = threading.Thread(target=keyboard_listener)
    listener_thread.daemon = True
    listener_thread.start()

    try:
        with sd.InputStream(samplerate=target_sr, channels=1, callback=callback, blocksize=fixed_length):
            while not stop_flag.is_set():
                audio_data = audio_queue.get()
                all_audio_data_chunks.append(audio_data.copy()) # Store the chunk

                # Convert numpy array to tensor, ensure mono, and correct shape
                # audio_data is (fixed_length, 1) for mono
                waveform = torch.tensor(audio_data, dtype=torch.float32).T # Transpose to [1, fixed_length]

                # Pad or trim (should be unnecessary if blocksize == fixed_length and no data loss)
                current_len = waveform.shape[1]
                if current_len < fixed_length:
                    padding = torch.zeros((1, fixed_length - current_len))
                    waveform = torch.cat([waveform, padding], dim=1)
                elif current_len > fixed_length:
                    waveform = waveform[:, :fixed_length]

                try:
                    # Add batch dimension: [1, 1, fixed_length]
                    input_tensor = waveform.unsqueeze(0).to(device)
 
                    with torch.no_grad():
                        output = model(input_tensor)
                        # --- DIAGNOSTIC PRINT ---
                        # print(f"Raw logits: {output.cpu().numpy()}") # Temporarily uncomment to see raw scores
                        # ------------------------
                        # Apply softmax to get probabilities
                        probabilities = torch.softmax(output, dim=1)
                        # Get the highest probability and its class index
                        max_probability, predicted_idx = torch.max(probabilities, dim=1)

                        if max_probability.item() > confidence_threshold:
                            label = LABELS[predicted_idx.item()]
                            # Only log and print if it's not the null_label
                            if label.lower() != null_label.lower(): # Case-insensitive check
                                timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S.%f")[:-3] # Milliseconds
                                current_session_time = datetime.now() - session_start_time
                                session_time_str = format_timedelta_to_hms(current_session_time)
                                
                                print(f"[{timestamp}] (Session: {session_time_str}) Detected: {label} (Accuracy: {max_probability.item()*100:.0f}%)")
                                log_detection(text_log_file_path, timestamp, session_time_str, label)
                        else:
                            # This block handles cases where max_probability.item() <= confidence_threshold
                            timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S.%f")[:-3]
                            current_session_time = datetime.now() - session_start_time
                            session_time_str = format_timedelta_to_hms(current_session_time)
                            print(f"[{timestamp}] (Session: {session_time_str}) Detected: null (Accuracy: {max_probability.item()*100:.0f}%)")
                except Exception as e:
                    print(f"[Error] Failed to process frame: {e}")
    except KeyboardInterrupt:
        print("\n[INFO] Interrupted by user.")
    finally:
        print("[INFO] Stopped real-time prediction.")
        if all_audio_data_chunks:
            # Concatenate all audio chunks
            continuous_audio = np.concatenate(all_audio_data_chunks, axis=0)
            # Save the continuous recording
            continuous_audio_save_path = get_next_numbered_filename(log_dir, "recording", ".wav")
            sf.write(continuous_audio_save_path, continuous_audio, target_sr)
            print(f"[INFO] Continuous audio saved to {continuous_audio_save_path}")
        
if __name__ == '__main__':
    # Parse the arguments from the command line
    parser = argparse.ArgumentParser()
    parser.add_argument('--source', type=str, default='0', help="Source for prediction (0 for microphone or file path for audio file)")
    parser.add_argument('--device', type=int, default=0, help="Device to run the model on (default: 0 for GPU if available, else CPU)")
    parser.add_argument('--confidence_threshold', type=float, default=0.7, help="Minimum confidence for a detection to be considered valid.")
    parser.add_argument('--null_label', type=str, default="null", help="Label to consider as 'null' or background.")
    parser.add_argument('model_path', type=str, help="Path to the trained model file (e.g., 'sound_model.pth')")
    args = parser.parse_args()

    # Get the device (GPU/CPU)
    device = get_device(args.device)

    try:
        checkpoint = torch.load(args.model_path, map_location=device)
        
        num_classes_from_checkpoint = checkpoint['num_classes']
        label_to_idx_from_checkpoint = checkpoint['label_to_idx']
        
        # Populate global LABELS list
        LABELS = [""] * num_classes_from_checkpoint # Initialize with placeholders
        for label, idx in label_to_idx_from_checkpoint.items():
            if idx < num_classes_from_checkpoint:
                LABELS[idx] = label
            else:
                raise ValueError(f"Index {idx} from label_to_idx is out of bounds for num_classes {num_classes_from_checkpoint}")

        if any(s == "" for s in LABELS): # Basic check
            raise ValueError("Failed to reconstruct all labels from model file. Some indices might be missing in label_to_idx.")

        print(f"Loading model trained with {num_classes_from_checkpoint} classes.")
        print(f"Labels from model: {LABELS}")

        model = AudioClassifier(num_classes=num_classes_from_checkpoint).to(device)
        model.load_state_dict(checkpoint['model_state_dict'])
        model.eval()  # Set the model to evaluation mode

    except FileNotFoundError:
        print(f"Error: Model file '{args.model_path}' not found.")
        exit(1)
    except RuntimeError as e:
        print(f"Error loading model state_dict: {e}")
        exit(1)
    except KeyError as e:
        print(f"Error: Model file '{args.model_path}' is missing required key: {e}. It might be an old model format or not contain label info.")
        exit(1)
    except ValueError as e: # For the label reconstruction check
        print(f"Error processing model file: {e}")
        exit(1)

    # Make predictions based on input source (microphone or file)
    if args.source == '0':  # Microphone input
        predict_mic(model, device, confidence_threshold=args.confidence_threshold, null_label=args.null_label)
    else:  # File input
        predict_file(args.source, model, device)
