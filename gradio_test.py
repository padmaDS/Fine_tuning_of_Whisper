from transformers import pipeline
import gradio as gr

model_path = "whisper-small-te" # Path to your trained model directory
pipe = pipeline(model=model_path)

def transcribe(audio):
    text = pipe(audio)["text"]
    return text

iface = gr.Interface(
    fn=transcribe, 
    inputs=gr.Audio(source="microphone", type="filepath"), 
    outputs="text",
    title="Whisper Small Telugu",
    description="Realtime demo for Telugu speech recognition using a fine-tuned Whisper small model.",
)

iface.launch(share=True, debug=True)
