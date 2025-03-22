import time
import logging
import os
import threading
from merged_ui.utils import create_sentence_batches, get_base_fragment_num, initialize_cuda_streams, initialize_temp_dirs, prepare_audio_buffer, prepare_prompt, split_text_and_validate
from merged_ui.workers import create_queues_and_events, rvc_worker, tts_worker

def process_results(sentences, rvc_results_queue, buffer, processing_complete):
    """
    Process results from the RVC queue and feed them to the ordered buffer queue.
    Files will be added to the buffer as they complete (in any order) but will
    be output in the correct sequence.
    """
    completed_sentences = {}
    info_messages = []
    
    # Continue while processing is active or results are still pending
    while (not processing_complete.is_set() or 
           not rvc_results_queue.empty() or 
           not buffer.is_empty() or 
           (completed_sentences and len(completed_sentences) < len(sentences))):
        try:
            # Gather all available results
            while not rvc_results_queue.empty():
                i, tts_path, rvc_path, success, info = rvc_results_queue.get_nowait()
                completed_sentences[i] = True
                info_messages.append(info)
                
                if success and rvc_path and os.path.exists(rvc_path):
                    # Add directly to the ordered buffer with its position
                    buffer.add_with_position(rvc_path, i)
                    logging.info(f"Sentence {i+1} added to ordered buffer")
                else:
                    logging.warning(f"Sentence {i+1} had no valid output file")
                
            # Get next file from the buffer (this respects timing and order)
            next_file = buffer.get_next()
            if next_file:
                yield "\n".join(info_messages), next_file
            else:
                time.sleep(0.1)
                
        except Exception as e:
            logging.error(f"Error in main processing loop: {str(e)}")
            info_messages.append(f"Error in processing: {str(e)}")
            yield "\n".join(info_messages), None
            break


def generate_and_process_with_rvc_parallel(
    text, prompt_text, prompt_wav_upload, prompt_wav_record,
    spk_item, vc_transform, f0method, 
    file_index1, file_index2, index_rate, filter_radius,
    resample_sr, rms_mix_rate, protect,
    num_tts_workers=2, num_rvc_workers=1,
    model_dir="spark/pretrained_models/Spark-TTS-0.5B", device="0"  # example extra parameters
):
    """
    Orchestrates combined TTS and RVC processing using multiple worker threads.
    """
    # Step 1: Initialize environment and inputs.
    initialize_temp_dirs()
    buffer = prepare_audio_buffer()
    sentences = split_text_and_validate(text)
    base_fragment_num = get_base_fragment_num(sentences)
    prompt_speech, prompt_text_clean = prepare_prompt(prompt_wav_upload, prompt_wav_record, prompt_text)
    
    # Step 2: Set up CUDA streams (if available) and inter-thread communication.
    tts_streams, rvc_streams = initialize_cuda_streams(num_tts_workers, num_rvc_workers)
    (tts_to_rvc_queue, rvc_results_queue, 
     tts_complete_events, rvc_complete_events, 
     processing_complete) = create_queues_and_events(num_tts_workers, num_rvc_workers)
    
    sentence_batches = create_sentence_batches(sentences, num_tts_workers)
    
    info_messages = [f"Processing {len(sentences)} sentences using {num_tts_workers} TTS workers and {num_rvc_workers} RVC workers..."]
    yield "\n".join(info_messages), None  # initial status message
    
    # Step 3: Start TTS worker threads.
    tts_threads = []
    for i in range(num_tts_workers):
        if sentence_batches[i][0]:  # Only start if there are sentences for this worker.
            thread = threading.Thread(
                target=tts_worker,
                args=(
                    i,
                    sentence_batches[i][0],
                    sentence_batches[i][1],
                    tts_streams[i],
                    base_fragment_num,
                    prompt_speech,
                    prompt_text_clean,
                    tts_to_rvc_queue,
                    tts_complete_events,
                    num_rvc_workers,
                    model_dir,
                    device
                )
            )
            tts_threads.append(thread)
            thread.start()
        else:
            tts_complete_events[i].set()
    
    # Step 4: Start RVC worker threads.
    rvc_threads = []
    for i in range(num_rvc_workers):
        thread = threading.Thread(
            target=rvc_worker,
            args=(
                i,
                rvc_streams[i],
                tts_to_rvc_queue,
                rvc_results_queue,
                rvc_complete_events,
                tts_complete_events,
                spk_item,
                vc_transform,
                f0method,
                file_index1,
                file_index2,
                index_rate,
                filter_radius,
                resample_sr,
                rms_mix_rate,
                protect,
                processing_complete
            )
        )
        rvc_threads.append(thread)
        thread.start()
    
    # Step 5: Process results and yield output as they become available.
    for output in process_results(sentences, rvc_results_queue, buffer, processing_complete):
        yield output
    
    # Step 6: Wait for all worker threads to finish.
    for thread in tts_threads:
        thread.join()
    for thread in rvc_threads:
        thread.join()
    
    logging.info(f"Processing complete. Processed {len(sentences)} sentences.")