import random
import gradio as gr

def latency_predict(file):
    if file is None:
        return "Please provide a file!", None
    with open(file.name, "rb") as f:
        latency = "111"
        topology_image_path = "./1.jpg"
    return latency, topology_image_path

def select_right_model(files, limit):
    if files is None:
        return "Please provide a file!"
    files_count = len(files)
    a = [random.random() for _ in range(files_count)]
    meet_ret, unmeet_ret = "", ""
    for i in range(files_count):
        if a[i] > limit:
            unmeet_ret += f"the latency of {files[i].name.split('/')[-1]} is {a[i]}\n"
        else:
            meet_ret += f"the latency of {files[i].name.split('/')[-1]} is {a[i]}\n"
    return meet_ret, unmeet_ret

def clear_fields():
    return None, None, None

def clear_fields2():
    return None, 0, None, None

if __name__ == '__main__':
    with gr.Blocks() as iface:
        with gr.Tabs():
            with gr.Tab("Latency Predictor"):
                gr.Markdown(
                    """
                    <div style="text-align: center;">
                        <h1>Latency Predictor</h1>
                    </div>
                    """
                )
                gr.Markdown("Please upload a file to predict its latency")

                with gr.Row(elem_id="centered-row"):
                    with gr.Column():
                        file_input = gr.File(label="Upload File")
                        latency_output = gr.Text(label="Latency")
                        with gr.Row():
                            cancel_button = gr.Button("Cancel")
                            submit_button = gr.Button("Submit")
                    topology_image = gr.Image(label="Model Topology")

            with gr.Tab("Select Right Model"):
                gr.Markdown(
                    """
                    <div style="text-align: center;">
                        <h1>Select Right Model</h1>
                    </div>
                    """
                )
                gr.Markdown("Please upload some files and set a latency limit.")
                with gr.Row(elem_id="centered-row"):
                    with gr.Column():
                        files_input = gr.File(label="Select Files", file_count="multiple")
                        latency_limit = gr.Slider(minimum=0.0, maximum=10.0, step=0.01, label="Latency Limit (ms)")
                        with gr.Row():
                            cancel_button2 = gr.Button("Cancel")
                            submit_button2 = gr.Button("Submit")
                    with gr.Column():
                        meet_files_output = gr.Text(label="Meet Files and Latency")
                        unmeet_files_output = gr.Text(label="Unmeet Files and Latency")

        # Button functionality
        submit_button.click(
            latency_predict,
            inputs=file_input,
            outputs=[latency_output, topology_image]
        )
        cancel_button.click(
            clear_fields,
            outputs=[file_input, latency_output, topology_image]
        )
        submit_button2.click(
            select_right_model,
            inputs=[files_input, latency_limit],
            outputs=[meet_files_output, unmeet_files_output]
        )
        cancel_button2.click(
            clear_fields2,
            outputs=[files_input, latency_limit, meet_files_output, unmeet_files_output]
        )

        # Custom CSS to set high-contrast based on system theme
        iface.css = """
            /* Light mode: darken content background for contrast */
            @media (prefers-color-scheme: light) {
                .gr-blocks, .gr-tab, .gr-row, .gr-column, .gr-markdown, .gr-button, .gr-slider, .gr-text, .gr-file, .gr-image {
                    background-color: #000 !important;
                    color: #fff !important;
                    border: 1px solid #fff !important;
                }
                .gr-button {
                    background-color: #222 !important;
                    color: #fff !important;
                    border-color: #fff !important;
                }
                .gr-slider input[type="range"] {
                    background-color: #444 !important;
                }
            }

            /* Dark mode: lighten content background for contrast */
            @media (prefers-color-scheme: dark) {
                .gr-blocks, .gr-tab, .gr-row, .gr-column, .gr-markdown, .gr-button, .gr-slider, .gr-text, .gr-file, .gr-image {
                    background-color: #fff !important;
                    color: #000 !important;
                    border: 1px solid #000 !important;
                }
                .gr-button {
                    background-color: #ddd !important;
                    color: #000 !important;
                    border-color: #000 !important;
                }
                .gr-slider input[type="range"] {
                    background-color: #bbb !important;
                }
            }
        """

    iface.launch()