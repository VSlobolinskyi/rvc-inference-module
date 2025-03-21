# Add to merged_ui/utils.py or create a new file

import os
import re
import time
import threading
import queue
import logging
from datetime import datetime
import torch
from pydub import AudioSegment
import soundfile as sf
import shutil

from spark.cli.SparkTTS import SparkTTS
from rvc_ui.initialization import vc

# Global models - initialize once and reuse
spark_models = []
model_dir = "spark/pretrained_models/Spark-TTS-0.5B"
device = 0

def modified_get_vc(sid0_value, protect0_value, file_index2_component):
    """
    Modified function to get voice conversion parameters
    """
    protect1_value = protect0_value
    outputs = vc.get_vc(sid0_value, protect0_value, protect1_value)
    
    if isinstance(outputs, tuple) and len(outputs) >= 3:
        return outputs[0], outputs[1], outputs[3]
    
    return 0, protect0_value, file_index2_component.choices[0] if file_index2_component.choices else ""

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

def initialize_spark_models(num_models=2):
    """Initialize multiple Spark models and keep them in GPU memory"""
    global spark_models
    
    
    # Only initialize if not already initialized
    if len(spark_models) == num_models:
        return spark_models
        
    # Clear any existing models
    spark_models = []
    
    # Determine device
    if torch.cuda.is_available():
        device_obj = torch.device(f"cuda:{device}")
        logging.info(f"Using CUDA device: {device_obj}")
    elif hasattr(torch, 'mps') and torch.backends.mps.is_available():
        device_obj = torch.device("mps")
        logging.info(f"Using MPS device: {device_obj}")
    else:
        device_obj = torch.device("cpu")
        logging.info("Using CPU (no GPU acceleration available)")
    
    # Initialize models
    for i in range(num_models):
        logging.info(f"Loading Spark model {i+1}/{num_models}...")
        model = SparkTTS(model_dir, device_obj)
        spark_models.append(model)
        logging.info(f"Spark model {i+1} loaded")
    
    return spark_models

def run_tts_with_model(
    model,  # Pre-initialized Spark model
    text,
    prompt_text=None,
    prompt_speech=None,
    gender=None,
    pitch=None,
    speed=None,
    save_dir="TEMP/spark", 
    save_filename=None,
):
    """Perform TTS inference with a pre-initialized model and save the generated audio."""
    logging.info(f"Running TTS on text: {text[:30]}...")

    if prompt_text is not None:
        prompt_text = None if len(prompt_text) <= 1 else prompt_text

    # Ensure the save directory exists
    os.makedirs(save_dir, exist_ok=True)

    # Determine the save path
    if save_filename:
        save_path = os.path.join(save_dir, save_filename)
    else:
        timestamp = datetime.now().strftime("%Y%m%d%H%M%S")
        save_path = os.path.join(save_dir, f"{timestamp}.wav")

    # Perform inference and save the output audio
    with torch.no_grad():
        wav = model.inference(
            text,
            prompt_speech,
            prompt_text,
            gender,
            pitch,
            speed,
        )
        sf.write(save_path, wav, samplerate=16000)

    logging.info(f"TTS audio saved at: {save_path}")
    return save_path

def _async_generate_and_process_with_rvc(
    text, prompt_text, prompt_wav_upload, prompt_wav_record,
    spk_item, vc_transform, f0method, 
    file_index1, file_index2, index_rate, filter_radius,
    resample_sr, rms_mix_rate, protect,
    num_spark_models=2
):
    """
    Asynchronous implementation using multiple Spark models and one RVC model.
    
    Args:
        text (str): Input text to process
        [all other parameters same as original]
        num_spark_models (int): Number of Spark models to use in parallel
        
    Yields:
        tuple: (info_message, audio_path) - The info message and path to the latest processed audio
    """
    
    # Ensure TEMP directories exist
    os.makedirs("./TEMP/spark", exist_ok=True)
    os.makedirs("./TEMP/rvc", exist_ok=True)
    
    sentences = split_into_sentences(text)
    if not sentences:
        yield "No valid text to process.", None
        return
    
    # Get next base fragment number to avoid filename collisions
    base_fragment_num = 1
    while any(os.path.exists(f"./TEMP/spark/fragment_{base_fragment_num + i}.wav") or 
              os.path.exists(f"./TEMP/rvc/fragment_{base_fragment_num + i}.wav") 
              for i in range(len(sentences))):
        base_fragment_num += 1
    
    # Process reference speech
    prompt_speech = prompt_wav_upload if prompt_wav_upload else prompt_wav_record
    prompt_text_clean = None if not prompt_text or len(prompt_text) < 2 else prompt_text
    
    info_messages = [f"Processing {len(sentences)} sentences using {num_spark_models} parallel Spark models..."]
    
    # Yield initial message with no audio yet
    yield "\n".join(info_messages), None
    
    # Initialize Spark models if needed
    if len(spark_models) != num_spark_models:
        initialize_spark_models(num_spark_models)
    
    # Create queues for communication between threads
    sentence_queue = queue.Queue()  # Sentences waiting to be processed by Spark
    tts_queue = queue.Queue()       # TTS outputs waiting to be processed by RVC
    result_queue = queue.Queue()    # Final processed outputs with metadata
    
    # Add all sentences to the queue with their original indices
    for i, sentence in enumerate(sentences):
        sentence_queue.put((i, sentence))
    
    # Event to signal when threads should stop
    stop_event = threading.Event()
    
    # Function for Spark worker thread
    def spark_worker(worker_id):
        """Worker function for Spark TTS processing"""
        model = spark_models[worker_id]
        logging.info(f"Spark worker {worker_id} started with model {id(model)}")
        
        while not stop_event.is_set():
            try:
                # Get a sentence with timeout
                i, sentence = sentence_queue.get(timeout=0.5)
                
                # Use unique fragment number based on original index
                fragment_num = base_fragment_num + i
                
                # Generate TTS audio with pre-initialized model
                try:
                    filename = f"fragment_{fragment_num}_spark{worker_id}.wav"
                    tts_path = run_tts_with_model(
                        model=model,
                        text=sentence,
                        prompt_text=prompt_text_clean,
                        prompt_speech=prompt_speech,
                        save_dir="./TEMP/spark",
                        save_filename=filename
                    )
                    
                    # Put the result in the TTS queue if successful
                    if tts_path and os.path.exists(tts_path):
                        tts_queue.put((i, sentence, tts_path))
                        logging.info(f"Spark worker {worker_id} processed sentence {i+1}")
                    else:
                        logging.error(f"Spark worker {worker_id} failed to process sentence {i+1}")
                except Exception as e:
                    logging.error(f"Error in Spark worker {worker_id} processing: {str(e)}")
                
                sentence_queue.task_done()
            except queue.Empty:
                # Check if all sentences are processed
                if sentence_queue.empty() and sentence_queue.unfinished_tasks == 0:
                    logging.info(f"Spark worker {worker_id} finished - no more sentences")
                    break
            except Exception as e:
                logging.error(f"Error in Spark worker {worker_id}: {str(e)}")
                try:
                    sentence_queue.task_done()
                except:
                    pass
    
    # Function for RVC worker thread
    def rvc_worker():
        """Worker function for RVC processing"""
        logging.info("RVC worker started")
        
        while not stop_event.is_set():
            try:
                # Get a TTS output with timeout
                i, sentence, tts_path = tts_queue.get(timeout=0.5)
                
                fragment_num = base_fragment_num + i
                
                # Process TTS output with RVC
                try:
                    f0_file = None  # Not using an F0 curve file
                    output_info, output_audio = vc.vc_single(
                        spk_item, tts_path, vc_transform, f0_file, f0method,
                        file_index1, file_index2, index_rate, filter_radius,
                        resample_sr, rms_mix_rate, protect
                    )
                    
                    # Save RVC output
                    rvc_output_path = f"./TEMP/rvc/fragment_{fragment_num}.wav"
                    rvc_saved = False
                    
                    # Handle different output formats from RVC
                    try:
                        if isinstance(output_audio, str) and os.path.exists(output_audio):
                            # Case 1: output_audio is a file path
                            shutil.copy2(output_audio, rvc_output_path)
                            rvc_saved = True
                        elif isinstance(output_audio, tuple) and len(output_audio) >= 2:
                            # Case 2: output_audio is (sample_rate, audio_data)
                            sf.write(rvc_output_path, output_audio[1], output_audio[0])
                            rvc_saved = True
                        elif hasattr(output_audio, 'name') and os.path.exists(output_audio.name):
                            # Case 3: output_audio is a file-like object
                            shutil.copy2(output_audio.name, rvc_output_path)
                            rvc_saved = True
                    except Exception as e:
                        output_info += f"\nError saving RVC output: {str(e)}"
                        logging.error(f"Error saving RVC output: {str(e)}")
                    
                    # Prepare info message
                    info_message = f"Sentence {i+1}: {sentence[:30]}{'...' if len(sentence) > 30 else ''}\n"
                    info_message += f"  - Spark output: {tts_path}\n"
                    
                    if rvc_saved:
                        info_message += f"  - RVC output: {rvc_output_path}"
                        
                        # Calculate audio duration for playback timing
                        try:
                            audio_seg = AudioSegment.from_file(rvc_output_path)
                            duration = audio_seg.duration_seconds
                        except Exception as e:
                            logging.warning(f"Could not determine audio duration: {str(e)}")
                            duration = 0
                        
                        # Put the result in the result queue with index for proper ordering
                        result_queue.put((i, rvc_output_path, info_message, duration))
                        logging.info(f"RVC processed sentence {i+1}")
                    else:
                        logging.error(f"RVC failed to save output for sentence {i+1}")
                except Exception as e:
                    logging.error(f"Error in RVC processing: {str(e)}")
                
                tts_queue.task_done()
            except queue.Empty:
                # Check if all TTS outputs are processed
                if tts_queue.empty() and tts_queue.unfinished_tasks == 0:
                    # Check if all Spark processing is also done
                    if sentence_queue.empty() and sentence_queue.unfinished_tasks == 0:
                        logging.info("RVC worker finished - no more TTS outputs")
                        break
            except Exception as e:
                logging.error(f"Error in RVC worker: {str(e)}")
                try:
                    tts_queue.task_done()
                except:
                    pass
    
    # Start worker threads
    spark_threads = []
    for i in range(min(num_spark_models, len(spark_models))):
        thread = threading.Thread(target=spark_worker, args=(i,))
        thread.daemon = True
        thread.start()
        spark_threads.append(thread)
    
    rvc_thread = threading.Thread(target=rvc_worker)
    rvc_thread.daemon = True
    rvc_thread.start()
    
    # Process results in order of original sentences
    completed_indices = set()
    next_index_to_yield = 0
    processed_results = {}
    
    # Set up timing for playback simulation
    next_available_time = time.time()
    
    # Continue until all sentences are processed
    try:
        while len(completed_indices) < len(sentences):
            # Try to get a result (non-blocking)
            try:
                i, rvc_path, info, duration = result_queue.get(block=False)
                # Store the result by index for ordered yielding
                processed_results[i] = (rvc_path, info, duration)
                completed_indices.add(i)
                result_queue.task_done()
            except queue.Empty:
                # Check if all threads are done and queues are empty
                all_threads_done = (all(not t.is_alive() for t in spark_threads) and 
                                   not rvc_thread.is_alive())
                all_queues_empty = (sentence_queue.empty() and 
                                   tts_queue.empty() and
                                   sentence_queue.unfinished_tasks == 0 and
                                   tts_queue.unfinished_tasks == 0)
                
                if all_threads_done or all_queues_empty:
                    # If we've processed all possible sentences, break
                    if len(completed_indices) < len(sentences):
                        missing = [i for i in range(len(sentences)) if i not in completed_indices]
                        info_messages.append(f"Warning: Could not process all sentences. Missing indices: {missing}")
                        yield "\n".join(info_messages), None
                    break
            
            # Check if we can yield the next result
            while next_index_to_yield in processed_results:
                rvc_path, info, duration = processed_results[next_index_to_yield]
                info_messages.append(info)
                
                # Wait until previous audio should be finished
                current_time = time.time()
                if current_time < next_available_time:
                    time.sleep(next_available_time - current_time)
                
                # Yield the result
                yield "\n".join(info_messages), rvc_path
                
                # Update the next available time for audio playback simulation
                next_available_time = time.time() + duration
                
                # Remove from our cache and increment counter
                del processed_results[next_index_to_yield]
                next_index_to_yield += 1
            
            # Small sleep to prevent CPU spinning
            time.sleep(0.01)
            
    except Exception as e:
        logging.error(f"Error in main processing loop: {str(e)}")
        info_messages.append(f"Error during processing: {str(e)}")
        yield "\n".join(info_messages), None
    finally:
        # Signal threads to stop
        stop_event.set()
        
        # Wait for threads to finish (with timeout)
        for thread in spark_threads:
            thread.join(timeout=2.0)
        rvc_thread.join(timeout=2.0)
    
    # Final yield with all info messages if we have a path
    if next_index_to_yield > 0 and len(sentences) > 0:
        final_path = None
        for i in range(len(sentences) - 1, -1, -1):
            if i in processed_results:
                final_path = processed_results[i][0]
                break
        
        if final_path:
            yield "\n".join(info_messages), final_path


# This is the wrapper function that maintains the same interface as your original function
def generate_and_process_with_rvc(
    text, prompt_text, prompt_wav_upload, prompt_wav_record,
    spk_item, vc_transform, f0method, 
    file_index1, file_index2, index_rate, filter_radius,
    resample_sr, rms_mix_rate, protect
):
    """
    Wrapper around the asynchronous implementation that maintains the same interface.
    This allows you to drop in the new implementation without changing your existing code.
    """
    # Call the async implementation with 2 Spark models
    return _async_generate_and_process_with_rvc(
        text, prompt_text, prompt_wav_upload, prompt_wav_record,
        spk_item, vc_transform, f0method, 
        file_index1, file_index2, index_rate, filter_radius,
        resample_sr, rms_mix_rate, protect,
        num_spark_models=2  # Use 2 Spark models as requested
    )