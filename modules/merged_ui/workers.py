import os
import shutil
import platform
import logging
from contextlib import nullcontext
from queue import Empty, PriorityQueue, Queue
import threading
import torch
import soundfile as sf

from spark.cli.SparkTTS import SparkTTS
from rvc_ui.initialization import vc

# Global shared sentence queue
distributed_sentence_queue = PriorityQueue()  # Changed to PriorityQueue to ensure order

def create_queues_and_events(num_tts_workers, num_rvc_workers):
    """
    Create queues and events for inter-thread communication.
    Using PriorityQueue for tts_to_rvc_queue to prioritize by fragment number.
    """
    tts_to_rvc_queue = PriorityQueue()  # Priority by fragment number
    rvc_results_queue = PriorityQueue()  # Changed to PriorityQueue to maintain order in results
    tts_complete_events = [threading.Event() for _ in range(num_tts_workers)]
    rvc_complete_events = [threading.Event() for _ in range(num_rvc_workers)]
    processing_complete = threading.Event()
    return tts_to_rvc_queue, rvc_results_queue, tts_complete_events, rvc_complete_events, processing_complete

def create_sentence_batches(sentences, num_tts_workers):
    """
    Create batches that distribute work across TTS workers.
    Modified to use a shared priority queue to strictly maintain order.
    """
    global distributed_sentence_queue
    
    # Clear any existing items
    while not distributed_sentence_queue.empty():
        try:
            distributed_sentence_queue.get_nowait()
        except Empty:
            break
    
    # Add all sentences to the queue with their original index as priority
    for idx, sentence in enumerate(sentences):
        distributed_sentence_queue.put((idx, (sentence, idx)))  # Use index as priority
    
    # Return empty batches for all workers
    # They will pull from the shared queue
    batches = [([], []) for _ in range(num_tts_workers)]
    
    return batches

def tts_worker(worker_id, sentences_batch, global_indices, cuda_stream, base_fragment_num,
               prompt_speech, prompt_text_clean, tts_to_rvc_queue, tts_complete_events,
               num_rvc_workers, model_dir, device):
    """
    TTS worker thread that processes sentences and adds results to the RVC queue.
    Modified to pull from a shared priority queue and maintain strict order.
    """
    global distributed_sentence_queue
    
    logging.info(f"TTS Worker {worker_id}: Initializing Spark TTS model")
    # Determine proper device based on platform and availability
    if platform.system() == "Darwin":
        model_device = torch.device(f"mps:{device}")
    elif torch.cuda.is_available():
        model_device = torch.device(f"cuda:{device}")
    else:
        model_device = torch.device("cpu")
    
    tts_model = SparkTTS(model_dir, model_device)
    
    # Check if we should use the distributed queue (empty batch)
    if not sentences_batch and not global_indices:
        # Pull from the shared queue
        processed_count = 0
        while True:
            try:
                # Get next sentence from the queue - will get lowest index first
                priority_data = distributed_sentence_queue.get(block=False)
                if priority_data is None:
                    break
                
                # Extract the sentence and global_idx from the priority tuple
                _, (sentence, global_idx) = priority_data
                processed_count += 1
                
                # Process sentence
                fragment_num = base_fragment_num + global_idx
                tts_filename = f"fragment_{fragment_num}.wav"
                save_path = os.path.join("./TEMP/spark", tts_filename)
                
                try:
                    logging.info(f"TTS Worker {worker_id}: Processing text {global_idx+1}: {sentence[:30]}...")
                    stream_ctx = torch.cuda.stream(cuda_stream) if cuda_stream and torch.cuda.is_available() else nullcontext()
                    with stream_ctx:
                        with torch.no_grad():
                            wav = tts_model.inference(
                                sentence,
                                prompt_speech,
                                prompt_text_clean,
                                None,  # gender
                                None,  # pitch
                                None,  # speed
                            )
                    sf.write(save_path, wav, samplerate=16000)
                    logging.info(f"TTS Worker {worker_id}: Audio saved at: {save_path}")
                    
                    # Put in priority queue with fragment number as priority
                    tts_to_rvc_queue.put((global_idx, (global_idx, fragment_num, sentence, save_path)))
                    
                except Exception as e:
                    logging.error(f"TTS Worker {worker_id} error for sentence {global_idx+1}: {str(e)}")
                    tts_to_rvc_queue.put((global_idx, (global_idx, fragment_num, sentence, None, str(e))))
                
            except Empty:
                break
        
        logging.info(f"TTS Worker {worker_id}: Completed processing {processed_count} sentences from queue")
    else:
        # Process pre-assigned batch (fallback to original behavior)
        for local_idx, (sentence, global_idx) in enumerate(zip(sentences_batch, global_indices)):
            fragment_num = base_fragment_num + global_idx
            tts_filename = f"fragment_{fragment_num}.wav"
            save_path = os.path.join("./TEMP/spark", tts_filename)
            
            try:
                logging.info(f"TTS Worker {worker_id}: Processing text {global_idx+1}: {sentence[:30]}...")
                stream_ctx = torch.cuda.stream(cuda_stream) if cuda_stream and torch.cuda.is_available() else nullcontext()
                with stream_ctx:
                    with torch.no_grad():
                        wav = tts_model.inference(
                            sentence,
                            prompt_speech,
                            prompt_text_clean,
                            None,  # gender
                            None,  # pitch
                            None,  # speed
                        )
                sf.write(save_path, wav, samplerate=16000)
                logging.info(f"TTS Worker {worker_id}: Audio saved at: {save_path}")
                
                # Put in priority queue with original index as priority to maintain strict order
                tts_to_rvc_queue.put((global_idx, (global_idx, fragment_num, sentence, save_path)))
                
            except Exception as e:
                logging.error(f"TTS Worker {worker_id} error for sentence {global_idx+1}: {str(e)}")
                tts_to_rvc_queue.put((global_idx, (global_idx, fragment_num, sentence, None, str(e))))
    
    logging.info(f"TTS Worker {worker_id}: Completed processing sentences")
    tts_complete_events[worker_id].set()
    
    # If all TTS workers are done, add sentinel values for each RVC worker.
    if all(event.is_set() for event in tts_complete_events):
        for _ in range(num_rvc_workers):
            tts_to_rvc_queue.put((float('inf'), None))  # Use highest priority for sentinel values

def rvc_worker(worker_id, cuda_stream, tts_to_rvc_queue, rvc_results_queue,
               rvc_complete_events, tts_complete_events, spk_item, vc_transform, f0method,
               file_index1, file_index2, index_rate, filter_radius,
               resample_sr, rms_mix_rate, protect, processing_complete):
    """
    RVC worker thread that processes TTS outputs.
    Modified to handle priority queue items and maintain original sentence order.
    """
    logging.info(f"RVC Worker {worker_id}: Starting")
    
    while True:
        try:
            # Get item from priority queue (will prioritize lower indices)
            priority_item = tts_to_rvc_queue.get(timeout=0.5)
            
            # Extract actual item from priority tuple
            # If it's None, it's a sentinel value
            if priority_item[1] is None:
                break
                
            original_idx, item = priority_item
            
            # Check if the item indicates an error from a TTS worker (length==5)
            if len(item) == 5:
                i, fragment_num, sentence, _, error = item
                # Put exactly 5 elements expected by the main loop, with priority as the queue key
                rvc_results_queue.put((i, (i, None, None, False, f"TTS error for sentence {i+1}: {error}")))
                continue
            
            i, fragment_num, sentence, tts_path = item
            if not tts_path or not os.path.exists(tts_path):
                rvc_results_queue.put((i, (i, None, None, False, f"No TTS output for sentence {i+1}")))
                continue
            
            # Determine output file path
            rvc_path = os.path.join("./TEMP/rvc", f"fragment_{fragment_num}.wav")
            
            try:
                logging.info(f"RVC Worker {worker_id}: Processing fragment {fragment_num} (sentence {i+1})")
                
                # Merged process_with_rvc logic
                with (torch.cuda.stream(cuda_stream) if cuda_stream and torch.cuda.is_available() else nullcontext()):
                    # f0_file is not used here
                    f0_file = None
                    output_info, output_audio = vc.vc_single(
                        spk_item, tts_path, vc_transform, f0_file, f0method,
                        file_index1, file_index2, index_rate, filter_radius,
                        resample_sr, rms_mix_rate, protect
                    )
                
                # Save RVC output (CPU operation)
                rvc_saved = False
                try:
                    if isinstance(output_audio, str) and os.path.exists(output_audio):
                        # Case 1: output_audio is a file path string
                        shutil.copy2(output_audio, rvc_path)
                        rvc_saved = True
                    elif isinstance(output_audio, tuple) and len(output_audio) >= 2:
                        # Case 2: output_audio is a (sample_rate, audio_data) tuple
                        sf.write(rvc_path, output_audio[1], output_audio[0])
                        rvc_saved = True
                    elif hasattr(output_audio, 'name') and os.path.exists(output_audio.name):
                        # Case 3: output_audio is a file-like object
                        shutil.copy2(output_audio.name, rvc_path)
                        rvc_saved = True
                except Exception as e:
                    output_info += f"\nError saving RVC output: {str(e)}"
                
                logging.info(f"RVC inference completed for {tts_path}")
                
                info_message = f"Sentence {i+1}: {sentence[:30]}{'...' if len(sentence) > 30 else ''}\n"
                info_message += f"  - Spark output: {tts_path}\n"
                if rvc_saved:
                    info_message += f"  - RVC output (Worker {worker_id}): {rvc_path}"
                else:
                    info_message += f"  - Could not save RVC output to {rvc_path}"
                
                # Add to results queue with original index as priority
                # Return a tuple containing exactly 5 elements as the original code expects
                rvc_results_queue.put((i, (i, tts_path, rvc_path if rvc_saved else None, rvc_saved, info_message)))
            except Exception as e:
                logging.error(f"RVC Worker {worker_id} error for sentence {i+1}: {str(e)}")
                info_message = f"Sentence {i+1}: {sentence[:30]}{'...' if len(sentence) > 30 else ''}\n"
                info_message += f"  - Spark output: {tts_path}\n"
                info_message += f"  - RVC processing error (Worker {worker_id}): {str(e)}"
                rvc_results_queue.put((i, (i, tts_path, None, False, info_message)))
        
        except Empty:
            if all(event.is_set() for event in tts_complete_events):
                break
            continue
    
    logging.info(f"RVC Worker {worker_id}: Completed")
    rvc_complete_events[worker_id].set()
    
    if all(event.is_set() for event in rvc_complete_events):
        processing_complete.set()

# Helper function to get results in the correct order
def get_ordered_results(rvc_results_queue, num_sentences):
    """
    Retrieves results from the results queue in the correct order.
    The results queue contains tuples of (priority, result_tuple).
    Each result_tuple contains exactly 5 elements expected by the main loop.
    """
    results = []
    
    # Get all results from the queue
    while not rvc_results_queue.empty():
        try:
            # The queue contains (priority, result_tuple)
            priority, result_tuple = rvc_results_queue.get_nowait()
            results.append(result_tuple)  # Just add the result_tuple, not the priority
        except Empty:
            break
    
    # Sort results by sentence index (first element of each tuple)
    results.sort(key=lambda x: x[0])
    return results