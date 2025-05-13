import soundfile as sf
import os
import argparse

def get_hz(wav_file_path):
    """
    Extracts the sample rate (Hz) from a WAV file.

    Args:
        wav_file_path (str): The path to the WAV file.

    Returns:
        int or None: The sample rate in Hz, or None if an error occurs. It now
                     returns an int instead of a float
    """
    try:
        with sf.SoundFile(wav_file_path, 'r') as sf_file:
            return int(sf_file.samplerate)
    except sf.LibsndfileError as e:
        print(f"Error: Could not open or read WAV file '{wav_file_path}'. {e}") # Replaced Wave.Error for SoundFile
        return None

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Get the sample rate of a WAV file.")
    parser.add_argument("wav_file", help="Path to the WAV file")
    args = parser.parse_args()

    wav_file = args.wav_file
    hz = get_hz(wav_file)

    if hz is not None:
        with open("hz.txt", "a") as f:
            f.write(f"{os.path.basename(wav_file)}, {hz} Hz\n")
        print(f"Sample rate for '{wav_file}' is: {hz} Hz.  Logged to hz.txt")
    else:
        print("Failed to determine the sample rate.")
