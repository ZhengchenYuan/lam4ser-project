import audb
import torch
import numpy as np
from torch.utils.data import Dataset, DataLoader
from transformers import Wav2Vec2Processor, Wav2Vec2Model
import audiofile

print("Loading EMODB...")
db = audb.load(
    'emodb',
    version='2.0.0',
    sampling_rate=16000,
    mixdown=True,
    format='wav',
)

# Get the emotion labels table
df = db['emotion'].get()
print(f"Dataset loaded: {len(df)} samples")
print(f"Emotion classes: {df['emotion'].unique().tolist()}")

# Label encoding
emotion_classes = sorted(df['emotion'].unique().tolist())
label2idx = {label: idx for idx, label in enumerate(emotion_classes)}
idx2label = {idx: label for label, idx in label2idx.items()}

print(f"Label mapping: {label2idx}")

# Load Wav2Vec2
print("Loading wav2vec2...")
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

processor = Wav2Vec2Processor.from_pretrained("facebook/wav2vec2-base-960h")
encoder = Wav2Vec2Model.from_pretrained("facebook/wav2vec2-base-960h")
encoder = encoder.to(device)
encoder.eval()

print(f"Using device: {device}")

# Dataset class
MAX_DURATION_SEC = 8.0
SAMPLING_RATE = 16000
MAX_SAMPLES = int(MAX_DURATION_SEC * SAMPLING_RATE)

class EmoDBDataset(Dataset):
    def __init__(self, df, label2idx, processor, encoder, device, max_samples):
        self.samples = []
        self.label2idx = label2idx
                
        print(f"Pre-extracting embeddings for {len(df)} clips...")
        
        for i, (idx, row) in enumerate(df.iterrows()):
            file_path = idx if isinstance(idx, str) else idx[0]
            label_str = row['emotion']
            label_int = label2idx[label_str]
            
            # Load audio
            signal, _ = audiofile.read(file_path, always_2d=False)
            # signal shape: (num_samples,)
            
            # Pad or truncate to MAX_SAMPLES
            if len(signal) > max_samples:
                signal = signal[:max_samples]
            else:
                pad_len = max_samples - len(signal)
                signal = np.pad(signal, (0, pad_len), mode='constant')
            
            # Run through wav2vec2 feature extractor + encoder
            inputs = processor(
                signal,
                sampling_rate=SAMPLING_RATE,
                return_tensors="pt",
                padding=False,
            )
            # [1, max_samples]
            input_values = inputs.input_values.to(device)
            
            with torch.no_grad():
                outputs = encoder(input_values)
            
            # embedding: [1, T_audio, 768]
            embedding = outputs.last_hidden_state.squeeze(0).cpu()
            
            self.samples.append((embedding, label_int, file_path))
            
            if (i + 1) % 100 == 0:
                print(f"Processed {i+1}/{len(df)} clips")
        
        print("Pre-extraction complete.")
    
    def __len__(self):
        return len(self.samples)
    
    def __getitem__(self, idx):
        embedding, label, _ = self.samples[idx]
        return embedding, torch.tensor(label, dtype=torch.long)


# Build dataset and dataloader
dataset = EmoDBDataset(
    df=df,
    label2idx=label2idx,
    processor=processor,
    encoder=encoder,
    device=device,
    max_samples=MAX_SAMPLES,
)

# All embeddings have the same T_audio
dataloader = DataLoader(dataset, batch_size=8, shuffle=True)

# Sanity check
embeddings_batch, labels_batch = next(iter(dataloader))
print(f"\nSanity check:")
# [8, T_audio, 768]
print(f"Embedding batch shape: {embeddings_batch.shape}")
# [8]
print(f"Labels batch shape: {labels_batch.shape}")
print(f"Label example: {[idx2label[l.item()] for l in labels_batch]}")

# Save to disk
SAVE_PATH = "embeddings/emodb_embeddings.pt"

print(f"\nSaving embeddings to {SAVE_PATH}...")
torch.save({
    # list of [T_audio, 768]
    'embeddings': [s[0] for s in dataset.samples],
    # list of ints
    'labels': [s[1] for s in dataset.samples],
    # list of str — used for speaker-independent splitting
    'file_paths': [s[2] for s in dataset.samples],
    'label2idx': label2idx,
    'idx2label': idx2label,
    'T_audio': dataset.samples[0][0].shape[0],
    'hidden_dim': 768,
}, SAVE_PATH)

print(f"Done!\nSummary:")
print(f"Num samples: {len(dataset)}")
print(f"T_audio: {dataset.samples[0][0].shape[0]}")
print(f"Hidden dim: 768")
print(f"Num classes: {len(label2idx)}  ->  {label2idx}")