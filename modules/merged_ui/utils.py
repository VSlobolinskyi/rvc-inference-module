import os
import re
import threading
import logging
from queue import PriorityQueue, Queue
import torch

from merged_ui.buffer_queue import OrderedAudioBufferQueue
from rvc_ui.initialization import vc

# Initialize the Spark TTS model (moved outside function to avoid reinitializing)
distributed_sentence_queue = Queue()
model_dir = "spark/pretrained_models/Spark-TTS-0.5B"
device = 0

def initialize_temp_dirs():
    os.makedirs("./TEMP/spark", exist_ok=True)
    os.makedirs("./TEMP/rvc", exist_ok=True)

def prepare_audio_buffer(buffer_time=1.0):
    """
    Create and return an OrderedAudioBufferQueue for managing audio output order.
    """
    return OrderedAudioBufferQueue(buffer_time)

def split_text_and_validate(text):
    sentences = split_into_sentences(text)
    if not sentences:
        raise ValueError("No valid text to process.")
    return sentences

def get_base_fragment_num(sentences):
    base_fragment_num = 1
    while any(
        os.path.exists(f"./TEMP/spark/fragment_{base_fragment_num + i}.wav") or 
        os.path.exists(f"./TEMP/rvc/fragment_{base_fragment_num + i}.wav")
        for i in range(len(sentences))
    ):
        base_fragment_num += 1
    return base_fragment_num

def prepare_prompt(prompt_wav_upload, prompt_wav_record, prompt_text):
    prompt_speech = prompt_wav_upload if prompt_wav_upload else prompt_wav_record
    prompt_text_clean = None if not prompt_text or len(prompt_text) < 2 else prompt_text
    return prompt_speech, prompt_text_clean

def initialize_cuda_streams(num_tts_workers, num_rvc_workers):
    use_cuda = torch.cuda.is_available()
    if use_cuda:
        tts_streams = [torch.cuda.Stream() for _ in range(num_tts_workers)]
        rvc_streams = [torch.cuda.Stream() for _ in range(num_rvc_workers)]
        logging.info(f"Using {num_tts_workers} CUDA streams for Spark TTS and {num_rvc_workers} for RVC")
    else:
        tts_streams = [None] * num_tts_workers
        rvc_streams = [None] * num_rvc_workers
        logging.info("CUDA not available, parallel processing will be limited")
    return tts_streams, rvc_streams

def create_sentence_batches(sentences, num_tts_workers):
    sentence_batches = []
    batch_size = len(sentences) // num_tts_workers
    remainder = len(sentences) % num_tts_workers
    start_idx = 0
    for i in range(num_tts_workers):
        current_batch_size = batch_size + (1 if i < remainder else 0)
        end_idx = start_idx + current_batch_size
        batch = sentences[start_idx:end_idx]
        batch_indices = list(range(start_idx, end_idx))
        sentence_batches.append((batch, batch_indices))
        start_idx = end_idx
    return sentence_batches

def split_into_sentences(text):
    """
    Split text into sentences using regular expressions.
    
    Args:
        text (str): The input text to split
        
    Returns:
        list: A list of sentences
    """
    # Split on period, exclamation mark, or question mark followed by space or end of string
    sentences = re.split(r'(?<=[.!?])\s+|(?<=[.!?])$', text)
    # Remove any empty sentences
    sentences = [s.strip() for s in sentences if s.strip()]
    return sentences


def modified_get_vc(sid0_value, protect0_value, file_index2_component):
    """
    Modified function to get voice conversion parameters
    """
    protect1_value = protect0_value
    outputs = vc.get_vc(sid0_value, protect0_value, protect1_value)
    
    if isinstance(outputs, tuple) and len(outputs) >= 3:
        return outputs[0], outputs[1], outputs[3]
    
    return 0, protect0_value, file_index2_component.choices[0] if file_index2_component.choices else ""