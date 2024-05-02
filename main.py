from dotenv import load_dotenv
import os
from datasets import Audio
from datasets import load_dataset, DatasetDict
from transformers import WhisperFeatureExtractor
from transformers import WhisperProcessor
from transformers import AutoTokenizer




# Load environment variables from .env file
load_dotenv()

# Get environment variables
# hf_home = os.getenv('HF_HOME')     

# Set HF_HOME environment variable

hf_token = os.environ['HF_TOKEN']

common_voice = DatasetDict()

# Load Dataset
common_voice["train"] = load_dataset("google/fleurs", "te_in", split="train+validation", use_auth_token=True)
common_voice["test"] = load_dataset("google/fleurs", "te_in", split="test", use_auth_token=True)

# google/fleurs

print(common_voice)

common_voice = common_voice.remove_columns(['id', 'num_samples', 'path',  'raw_transcription', 'gender', 'lang_id', 'language', 'lang_group_id'])
print(common_voice)

### Prepare Feature Extractor, Tokenizer and Data
## Load WhisperFeatureExtractor

feature_extractor = WhisperFeatureExtractor.from_pretrained("openai/whisper-small")



processor = WhisperProcessor.from_pretrained("openai/whisper-small", language="Telugu", task="transcribe")

#### Prepare Data

# print(common_voice["train"][0])

common_voice = common_voice.cast_column("audio", Audio(sampling_rate=16000))

print(common_voice["train"][0])


# Instantiate the tokenizer
from transformers import WhisperTokenizer

tokenizer = WhisperTokenizer.from_pretrained("openai/whisper-small", language="Telugu", task="transcribe")


def prepare_dataset(batch):
    # load and resample audio data from 48 to 16kHz
    audio = batch["audio"]

    # compute log-Mel input features from input audio array 
    batch["input_features"] = feature_extractor(audio["array"], sampling_rate=audio["sampling_rate"]).input_features[0]

    # encode target text to label ids 
    batch["labels"] = tokenizer(batch["transcription"]).input_ids
    # batch["labels"] = tokenizer(batch["sentence"]).input_ids
    return batch

common_voice = common_voice.map(prepare_dataset, remove_columns=common_voice.column_names["train"], num_proc=1)

### Training and Evaluation
## Define a Data Collator

import torch

from dataclasses import dataclass
from typing import Any, Dict, List, Union

@dataclass
class DataCollatorSpeechSeq2SeqWithPadding:
    processor: Any

    def __call__(self, features: List[Dict[str, Union[List[int], torch.Tensor]]]) -> Dict[str, torch.Tensor]:
        # split inputs and labels since they have to be of different lengths and need different padding methods
        # first treat the audio inputs by simply returning torch tensors
        input_features = [{"input_features": feature["input_features"]} for feature in features]
        batch = self.processor.feature_extractor.pad(input_features, return_tensors="pt")

        # get the tokenized label sequences
        label_features = [{"input_ids": feature["labels"]} for feature in features]
        # pad the labels to max length
        labels_batch = self.processor.tokenizer.pad(label_features, return_tensors="pt")

        # replace padding with -100 to ignore loss correctly
        labels = labels_batch["input_ids"].masked_fill(labels_batch.attention_mask.ne(1), -100)

        # if bos token is appended in previous tokenization step,
        # cut bos token here as it's append later anyways
        if (labels[:, 0] == self.processor.tokenizer.bos_token_id).all().cpu().item():
            labels = labels[:, 1:]

        batch["labels"] = labels

        return batch
    
data_collator = DataCollatorSpeechSeq2SeqWithPadding(processor=processor)

### Evaluation Metrics

import evaluate

metric = evaluate.load("wer")

def compute_metrics(pred):
    pred_ids = pred.predictions
    label_ids = pred.label_ids

    # replace -100 with the pad_token_id
    label_ids[label_ids == -100] = tokenizer.pad_token_id

    # we do not want to group tokens when computing the metrics
    pred_str = tokenizer.batch_decode(pred_ids, skip_special_tokens=True)
    label_str = tokenizer.batch_decode(label_ids, skip_special_tokens=True)

    wer = 100 * metric.compute(predictions=pred_str, references=label_str)

    return {"wer": wer}

### Load a Pre-Trained Checkpoint

from transformers import WhisperForConditionalGeneration

model = WhisperForConditionalGeneration.from_pretrained("openai/whisper-small")

model.config.forced_decoder_ids = None
model.config.suppress_tokens = []

#### Define the Training Configuration

from transformers import Seq2SeqTrainingArguments

training_args = Seq2SeqTrainingArguments(
    output_dir="./whisper-small-te",  # change to a repo name of your choice
    per_device_train_batch_size=4, # reducing this to 4 from 16 to avoid OOM errors
    gradient_accumulation_steps=1,  # increase by 2x for every 2x decrease in batch size
    learning_rate=1e-5,
    warmup_steps=50,
    max_steps=100, # reducing this by 10% from the original
    gradient_checkpointing=True,
    fp16=True,
    evaluation_strategy="steps",
    per_device_eval_batch_size=2, # reducing this to 2 from 8 to avoid OOM errors
    predict_with_generate=True,
    generation_max_length=225,
    save_steps=50,
    eval_steps=50,
    logging_steps=25,
    report_to=["tensorboard"],
    load_best_model_at_end=True,
    metric_for_best_model="wer",
    greater_is_better=False,
    push_to_hub=True,
)

from transformers import Seq2SeqTrainer

trainer = Seq2SeqTrainer(
    args=training_args,
    model=model,
    train_dataset=common_voice["train"],
    eval_dataset=common_voice["test"],
    data_collator=data_collator,
    compute_metrics=compute_metrics,
    tokenizer=processor.feature_extractor,
)

processor.save_pretrained(training_args.output_dir)

#### Training

# trainer.train()

kwargs = {
    "dataset_tags": "google/fleurs",
    "dataset": "Google Fleurs",  # a 'pretty' name for the training dataset
    "dataset_args": "config: te, split: test",
    "language": "te",
    "model_name": "Whisper Small te-arthink",  # a 'pretty' name for our model
    "finetuned_from": "openai/whisper-small",
    "tasks": "automatic-speech-recognition",
    "tags": "hf-asr-leaderboard",
}


trainer.push_to_hub(**kwargs)