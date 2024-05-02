from transformers import pipeline

model_path = "April01524/cmomay"  # Path to your trained model directory
pipe = pipeline(model=model_path)

# Example audio file path
audio_file = r"data\batch_6.mp3"

# Perform inference
result = pipe(audio_file)
transcribed_text = result["text"]

print("Transcribed Text:", transcribed_text)
