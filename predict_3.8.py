import torch
import sounddevice as sd
import soundfile as sf
import keyboard 
import queue
import threading
import argparse
import os
import requests # For HTTP requests (streaming and download)
import tempfile # For downloading files before prediction
import asyncio # For WebSocket client
import websockets # For WebSocket client
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

def download_audio_from_url_and_predict(url, model, device, target_sr=TARGET_SAMPLE_RATE, fixed_length=FIXED_AUDIO_LENGTH):
    """
    Downloads a complete audio file from a URL, saves it to a temporary file,
    and then predicts its label using predict_file.
    Args:
    - url (str): The URL to fetch the audio file from.
    - model (nn.Module): The trained model.
    - device (torch.device): The device to run the model on (CPU or GPU).
    - target_sr (int): Target sample rate for audio processing.
    - fixed_length (int): Fixed length to pad/trim audio to.
    """
    temp_file_path = None
    try:
        print(f"Attempting to download audio from: {url}")
        # Use a timeout for the request (10s connect, 30s read)
        response = requests.get(url, stream=True, timeout=(10, 30))
        response.raise_for_status()  # Raise an exception for HTTP errors (4xx or 5xx)

        with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as tmp_file: # Suffix helps torchaudio
            for chunk in response.iter_content(chunk_size=8192): # Read in chunks
                tmp_file.write(chunk)
            temp_file_path = tmp_file.name
        
        print(f"Audio downloaded and saved to temporary file: {temp_file_path}")
        predict_file(temp_file_path, model, device, target_sr, fixed_length) # Process the downloaded file

    except requests.exceptions.RequestException as e:
        print(f"Error downloading audio from {url}: {e}")
    except Exception as e:
        print(f"An unexpected error occurred while processing URL {url}: {e}")
    finally:
        if temp_file_path and os.path.exists(temp_file_path):
            os.remove(temp_file_path)
            print(f"Temporary file {temp_file_path} deleted.")

def predict_mic(model, device, frame_duration=1.0, target_sr=TARGET_SAMPLE_RATE, fixed_length=FIXED_AUDIO_LENGTH, confidence_threshold=0.7, null_label="null"):
    """
    Real-time microphone prediction until Ctrl+C or 'q' is pressed.
    Args:
        model: Trained audio classification model.
        device: torch.device (CPU/GPU).
        frame_duration: Duration of each prediction window in seconds.
    """
    print(f"\n Starting real-time audio classification (Confidence threshold: {confidence_threshold:.2f})...")
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

def predict_http_stream(url, model, device, target_sr=TARGET_SAMPLE_RATE, fixed_length=FIXED_AUDIO_LENGTH, confidence_threshold=0.7, null_label="null"):
    """
    Real-time prediction from an HTTP audio stream.
    IMPORTANT ASSUMPTIONS:
    - The stream provides raw PCM mono audio.
    - The audio is 16-bit signed integers (np.int16).
    - The sample rate of the stream matches `target_sr`.
    """
    print(f"\n Connecting to audio stream: {url} (Confidence threshold: {confidence_threshold:.2f})...")
    log_dir = "recorded"
    text_log_file_path = get_next_numbered_filename(log_dir, "stream_detect", ".txt")
    session_start_time = datetime.now()
    print("Press Ctrl+C to stop.\n")

    bytes_per_sample = 2  # For 16-bit PCM (np.int16)
    # Number of samples in one frame * bytes per sample
    frame_size_bytes = fixed_length * bytes_per_sample
    internal_buffer = bytearray() # Buffer to hold incoming bytes from the stream
    
    response = None # Initialize for the finally block

    try:
        # Connect timeout: 10s, Read timeout (for each block from iter_content): 30s
        response = requests.get(url, stream=True, timeout=(10, 30))
        response.raise_for_status()  # Check for HTTP errors (4xx or 5xx)
        print(f"Successfully connected to stream: {url}")

        # iter_content chunk_size is how much data requests reads from the socket at a time.
        # This is not necessarily our audio frame size.
        # Reading a moderate chunk size (e.g., 4096 bytes)
        for chunk_bytes in response.iter_content(chunk_size=4096): # Read raw bytes
            if not chunk_bytes: # Handle keep-alive or empty chunks if any
                continue
            
            internal_buffer.extend(chunk_bytes)

            # Process as many full frames as we have in the buffer
            while len(internal_buffer) >= frame_size_bytes:
                frame_bytes = internal_buffer[:frame_size_bytes]
                internal_buffer = internal_buffer[frame_size_bytes:] # Consume the frame from buffer

                # Convert bytes to numpy array (assuming 16-bit PCM, mono)
                waveform_np = np.frombuffer(frame_bytes, dtype=np.int16)
                
                # Normalize to float32 between -1 and 1 (standard for many models)
                waveform_np = waveform_np.astype(np.float32) / 32768.0 # Max value of int16

                # Convert to PyTorch tensor: [fixed_length]
                waveform = torch.from_numpy(waveform_np).float()
                
                # Add channel dimension (model expects mono): [1, fixed_length]
                waveform = waveform.unsqueeze(0)

                # Add batch dimension (model expects batch): [1, 1, fixed_length]
                input_tensor = waveform.unsqueeze(0).to(device)

                try:
                    with torch.no_grad():
                        output = model(input_tensor)
                        probabilities = torch.softmax(output, dim=1)
                        max_probability, predicted_idx = torch.max(probabilities, dim=1)

                        if max_probability.item() > confidence_threshold:
                            label = LABELS[predicted_idx.item()]
                            if label.lower() != null_label.lower(): # Case-insensitive
                                timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S.%f")[:-3]
                                current_session_time = datetime.now() - session_start_time
                                session_time_str = format_timedelta_to_hms(current_session_time)
                                print(f"[{timestamp}] (Session: {session_time_str}) Detected: {label} (Accuracy: {max_probability.item()*100:.0f}%) from stream")
                                log_detection(text_log_file_path, timestamp, session_time_str, label)
                        else: # Low confidence, treat as null
                            timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S.%f")[:-3]
                            current_session_time = datetime.now() - session_start_time
                            session_time_str = format_timedelta_to_hms(current_session_time)
                            print(f"[{timestamp}] (Session: {session_time_str}) Detected: null (Accuracy: {max_probability.item()*100:.0f}%) from stream")
                
                except Exception as e:
                    print(f"[Error] Failed to process stream frame: {e}")

    except requests.exceptions.RequestException as e:
        print(f"Stream connection error for {url}: {e}")
    except KeyboardInterrupt:
        print("\n[INFO] Stream interrupted by user.")
    except Exception as e:
        print(f"An unexpected error occurred with the stream: {e}")
    finally:
        print("[INFO] Stopped stream prediction.")
        if response: # Ensure response object exists before trying to close
            response.close() # Important to close the connection
        
async def predict_websocket_stream(url, model, device, target_sr=TARGET_SAMPLE_RATE, fixed_length=FIXED_AUDIO_LENGTH, confidence_threshold=0.7, null_label="null"):
    """
    Real-time prediction from a WebSocket audio stream.
    Assumes the stream provides raw PCM mono audio, 16-bit signed integers,
    at the target_sr.
    """
    print(f"\n Connecting to WebSocket audio stream: {url} (Confidence threshold: {confidence_threshold:.2f})...")
    log_dir = "recorded"
    text_log_file_path = get_next_numbered_filename(log_dir, "websocket_detect", ".txt")
    session_start_time = datetime.now()
    print("Press Ctrl+C to stop.\n")

    bytes_per_sample = 2  # For 16-bit PCM (np.int16)
    # Number of samples in one frame * bytes per sample
    frame_size_bytes = fixed_length * bytes_per_sample
    internal_buffer = bytearray() # Buffer to hold incoming bytes from the stream
    all_audio_data_chunks = [] # List to store all incoming audio chunks for saving

    try:
        # Adjust connect_timeout and ping_interval/timeout as needed
        async with websockets.connect(url, open_timeout=10, ping_interval=20, ping_timeout=20) as websocket:
            print(f"Successfully connected to WebSocket stream: {url}")
            async for message in websocket:
                if not isinstance(message, bytes):
                    print(f"[Warning] Received non-bytes message from WebSocket: {type(message)}. Skipping.")
                    continue

                internal_buffer.extend(message)

                # Process as many full frames as we have in the buffer
                while len(internal_buffer) >= frame_size_bytes:
                    frame_bytes = internal_buffer[:frame_size_bytes]
                    internal_buffer = internal_buffer[frame_size_bytes:] # Consume the frame

                    # Convert bytes to numpy array (assuming 16-bit PCM, mono)
                    waveform_np = np.frombuffer(frame_bytes, dtype=np.int16)
                    all_audio_data_chunks.append(waveform_np.copy()) # Store the chunk for saving
                    
                    # Normalize to float32 between -1 and 1
                    waveform_np = waveform_np.astype(np.float32) / 32768.0

                    # Convert to PyTorch tensor: [fixed_length]
                    waveform = torch.from_numpy(waveform_np).float()
                    
                    # Add channel dimension (model expects mono): [1, fixed_length]
                    waveform = waveform.unsqueeze(0)

                    # Add batch dimension (model expects batch): [1, 1, fixed_length]
                    input_tensor = waveform.unsqueeze(0).to(device)

                    try:
                        with torch.no_grad():
                            output = model(input_tensor)
                            probabilities = torch.softmax(output, dim=1)
                            max_probability, predicted_idx = torch.max(probabilities, dim=1)

                            if max_probability.item() > confidence_threshold:
                                label = LABELS[predicted_idx.item()]
                                if label.lower() != null_label.lower(): # Case-insensitive
                                    timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S.%f")[:-3]
                                    current_session_time = datetime.now() - session_start_time
                                    session_time_str = format_timedelta_to_hms(current_session_time)
                                    print(f"[{timestamp}] (Session: {session_time_str}) Detected: {label} (Accuracy: {max_probability.item()*100:.0f}%) from WebSocket")
                                    log_detection(text_log_file_path, timestamp, session_time_str, label)
                            else: # Low confidence, treat as null
                                timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S.%f")[:-3]
                                current_session_time = datetime.now() - session_start_time
                                session_time_str = format_timedelta_to_hms(current_session_time)
                                print(f"[{timestamp}] (Session: {session_time_str}) Detected: null (Accuracy: {max_probability.item()*100:.0f}%) from WebSocket")
                    except Exception as e:
                        print(f"[Error] Failed to process WebSocket frame: {e}")
    except (websockets.exceptions.ConnectionClosedError, websockets.exceptions.ConnectionClosedOK) as e:
        print(f"WebSocket connection closed: {e}")
    except websockets.exceptions.InvalidURI:
        print(f"Invalid WebSocket URI: {url}")
    except ConnectionRefusedError:
        print(f"WebSocket connection refused for {url}")
    except asyncio.TimeoutError: # Covers connection timeout from websockets.connect
        print(f"WebSocket connection to {url} timed out.")
    except KeyboardInterrupt:
        print("\n[INFO] WebSocket stream interrupted by user.")
    except Exception as e:
        print(f"An unexpected error occurred with the WebSocket stream: {e}")
    finally:
        print("\n[INFO] Stopped WebSocket stream prediction.")
        if all_audio_data_chunks:
            # Concatenate all audio chunks (waveform_np was 1D array of int16 samples)
            continuous_audio = np.concatenate(all_audio_data_chunks, axis=0)
            # Save the continuous recording
            continuous_audio_save_path = get_next_numbered_filename(log_dir, "websocket_recording", ".wav")
            sf.write(continuous_audio_save_path, continuous_audio, target_sr)
            print(f"[INFO] Continuous WebSocket audio saved to {continuous_audio_save_path}")

if __name__ == '__main__':
    # Parse the arguments from the command line
    parser = argparse.ArgumentParser()
    parser.add_argument('--source', type=str, default='0', help="Source: '0' (mic), file path, 'http(s)://...' (download file), 'stream:http(s)://...' (live stream)")
    parser.add_argument('--device', type=int, default=0, help="Device to run the model on (default: 0 for GPU if available, else CPU)")
    parser.add_argument('--confidence_threshold', type=float, default=0.8, help="Minimum confidence for a detection to be considered valid.")
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
    if args.source.startswith('stream:ws://') or args.source.startswith('stream:wss://'):
        stream_url = args.source.split('stream:', 1)[1] # Extract the URL part
        asyncio.run(predict_websocket_stream(stream_url, model, device,
                                           target_sr=TARGET_SAMPLE_RATE, fixed_length=FIXED_AUDIO_LENGTH,
                                           confidence_threshold=args.confidence_threshold,
                                           null_label=args.null_label))
    elif args.source.startswith('stream:http://') or args.source.startswith('stream:https://'):
        stream_url = args.source.split('stream:', 1)[1] # Extract the URL part
        predict_http_stream(stream_url, model, device,
                            target_sr=TARGET_SAMPLE_RATE, fixed_length=FIXED_AUDIO_LENGTH,
                            confidence_threshold=args.confidence_threshold, null_label=args.null_label)
    elif args.source.startswith('http://') or args.source.startswith('https://'):
        # This is for downloading a complete file from a URL
        download_audio_from_url_and_predict(args.source, model, device,
                                            target_sr=TARGET_SAMPLE_RATE, fixed_length=FIXED_AUDIO_LENGTH)
    elif args.source == '0':  # Microphone input
        predict_mic(model, device, confidence_threshold=args.confidence_threshold, null_label=args.null_label)
    else:  # File input
        predict_file(args.source, model, device, target_sr=TARGET_SAMPLE_RATE, fixed_length=FIXED_AUDIO_LENGTH)
