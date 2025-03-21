import gradio as gr

# Import modules from your packages
from merged_ui.utils import async_generate_and_process_with_rvc, modified_get_vc
from rvc_ui.initialization import config
from rvc_ui.main import names, index_paths

def build_merged_ui():
    """
    Build the combined TTS-RVC UI interface using Gradio
    """
    # Create the UI
    with gr.Blocks(title="Unified TTS-RVC Pipeline") as app:
        gr.Markdown("## Voice Generation and Conversion Pipeline")
        gr.Markdown("Generate speech with Spark TTS and process it through RVC for voice conversion")
        
        with gr.Tabs():
            with gr.TabItem("TTS-to-RVC Pipeline"):
                gr.Markdown("### Generate speech with Spark TTS and convert with RVC")
                gr.Markdown("*Note: For multi-sentence text, each sentence will be processed separately and streamed as it’s ready.*")
                
                # TTS Generation Section
                with gr.Row():
                    prompt_wav_upload = gr.Audio(
                        sources="upload",
                        type="filepath",
                        label="Reference voice (upload)",
                    )
                    prompt_wav_record = gr.Audio(
                        sources="microphone",
                        type="filepath",
                        label="Reference voice (record)",
                    )

                with gr.Row():
                    tts_text_input = gr.Textbox(
                        label="Text to synthesize", 
                        lines=3, 
                        placeholder="Enter text for TTS. Multiple sentences will be processed individually."
                    )
                    prompt_text_input = gr.Textbox(
                        label="Text of prompt speech (Optional)",
                        lines=3,
                        placeholder="Enter text of the reference audio",
                    )
                
                # RVC Settings
                with gr.Row():
                    with gr.Column():
                        sid0 = gr.Dropdown(
                            label="Target voice model:", choices=sorted(names)
                        )
                        vc_transform0 = gr.Number(
                            label="Transpose (semitones):",
                            value=0,
                        )
                        f0method0 = gr.Radio(
                            label="Pitch extraction algorithm:",
                            choices=(
                                ["pm", "harvest", "crepe", "rmvpe"]
                                if config.dml == False
                                else ["pm", "harvest", "rmvpe"]
                            ),
                            value="rmvpe",
                            interactive=True,
                        )
                        file_index1 = gr.Textbox(
                            label="Path to feature index file (leave blank for auto):",
                            placeholder="Leave blank to use dropdown selection",
                            interactive=True,
                        )
                        file_index2 = gr.Dropdown(
                            label="Select feature index:",
                            choices=sorted(index_paths),
                            interactive=True,
                        )
                        
                    with gr.Column():
                        index_rate1 = gr.Slider(
                            minimum=0,
                            maximum=1,
                            label="Feature search ratio (accent strength):",
                            value=0.75,
                            interactive=True,
                        )
                        filter_radius0 = gr.Slider(
                            minimum=0,
                            maximum=7,
                            label="Median filter radius (3+ reduces breathiness):",
                            value=3,
                            step=1,
                            interactive=True,
                        )
                        rms_mix_rate0 = gr.Slider(
                            minimum=0,
                            maximum=1,
                            label="Volume envelope scaling (0=original, 1=constant):",
                            value=0.25,
                            interactive=True,
                        )
                        protect0 = gr.Slider(
                            minimum=0,
                            maximum=0.5,
                            label="Consonant protection (0=max, 0.5=disable):",
                            value=0.33,
                            step=0.01,
                            interactive=True,
                        )
                        resample_sr0 = gr.Slider(
                            minimum=0,
                            maximum=48000,
                            label="Output sample rate (0=no resampling):",
                            value=0,
                            step=1,
                            interactive=True,
                        )
                
                # Speaker ID (hidden)
                spk_item = gr.Slider(
                    minimum=0,
                    maximum=2333,
                    step=1,
                    label="Speaker ID:",
                    value=0,
                    visible=False,
                    interactive=True,
                )
                
                # Combined process button and outputs
                generate_with_rvc_button = gr.Button("Generate with RVC", variant="primary")
                
                with gr.Row():
                    vc_output1 = gr.Textbox(label="Output information", lines=10)
                    vc_output2 = gr.Audio(label="Streaming concatenated audio", autoplay=True)
                
                # Connect generate function to button with streaming enabled
                generate_with_rvc_button.click(
                    async_generate_and_process_with_rvc,
                    inputs=[
                        tts_text_input,
                        prompt_text_input,
                        prompt_wav_upload,
                        prompt_wav_record,
                        spk_item,
                        vc_transform0,
                        f0method0,
                        file_index1,
                        file_index2,
                        index_rate1,
                        filter_radius0,
                        resample_sr0,
                        rms_mix_rate0,
                        protect0,
                    ],
                    outputs=[vc_output1, vc_output2]
                )
                
                # Connect modified_get_vc function for dropdown change
                sid0.change(
                    fn=lambda sid0_val, protect0_val: modified_get_vc(sid0_val, protect0_val, file_index2),
                    inputs=[sid0, protect0],
                    outputs=[spk_item, protect0, file_index2],
                )
                
    return app

if __name__ == "__main__":
    build_merged_ui()