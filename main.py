import random
import gradio as gr

CPU_arr = [
    "Intel Core i9-13900K",
    "Intel Core i7-13700K",
    "Intel Core i5-13600K",
    "AMD Ryzen 9 7950X",
    "AMD Ryzen 7 7800X3D",
    "AMD Ryzen 5 7600X",
    "Apple M2 Pro",
    "Apple M2 Max",
    "Intel Xeon W-2400",
    "AMD EPYC 9654"
]

GPU_arr = [
    "NVIDIA GeForce RTX 2080 Ti",
    "NVIDIA GeForce RTX 2070 Super",
    "NVIDIA GeForce RTX 3060",
    "NVIDIA GeForce RTX 3060 Ti",
    "NVIDIA GeForce RTX 3080",
    "NVIDIA GeForce RTX 3090",
    "NVIDIA GeForce RTX 4080",
    "NVIDIA GeForce RTX 4090",
    "NVIDIA Tesla T100",
    "NVIDIA Tesla A100"
]

CPU_dict = {
    "Intel Core i9-13900K": "24核心32线程，主频3.0GHz，睿频5.8GHz，125W TDP",
    "Intel Core i7-13700K": "16核心24线程，主频3.4GHz，睿频5.4GHz，125W TDP",
    "Intel Core i5-13600K": "14核心20线程，主频3.5GHz，睿频5.1GHz，125W TDP",
    "AMD Ryzen 9 7950X": "16核心32线程，主频4.5GHz，睿频5.7GHz，170W TDP",
    "AMD Ryzen 7 7800X3D": "8核心16线程，主频4.2GHz，睿频5.0GHz，120W TDP",
    "AMD Ryzen 5 7600X": "6核心12线程，主频4.7GHz，睿频5.3GHz，105W TDP",
    "Apple M2 Pro": "10核心，主频高达3.5GHz，集成GPU 16核",
    "Apple M2 Max": "12核心，主频高达3.7GHz，集成GPU 38核",
    "Intel Xeon W-2400": "24核心48线程，主频3.1GHz，睿频4.8GHz，225W TDP",
    "AMD EPYC 9654": "96核心192线程，主频2.4GHz，睿频3.7GHz，360W TDP"
}

GPU_dict = {
    "NVIDIA GeForce RTX 2080 Ti": "4352 CUDA核心，11GB GDDR6显存，显存带宽616GB/s，核心频率1350MHz（加速频率1635MHz）",
    "NVIDIA GeForce RTX 2070 Super": "2560 CUDA核心，8GB GDDR6显存，显存带宽448GB/s，核心频率1605MHz（加速频率1770MHz）",
    "NVIDIA GeForce RTX 3060": "3584 CUDA核心，12GB GDDR6显存，显存带宽360GB/s，核心频率1320MHz（加速频率1777MHz）",
    "NVIDIA GeForce RTX 3060 Ti": "4864 CUDA核心，8GB GDDR6显存，显存带宽448GB/s，核心频率1410MHz（加速频率1665MHz）",
    "NVIDIA GeForce RTX 3080": "8704 CUDA核心，10GB GDDR6X显存，显存带宽760GB/s，核心频率1440MHz（加速频率1710MHz）",
    "NVIDIA GeForce RTX 3090": "10496 CUDA核心，24GB GDDR6X显存，显存带宽936GB/s，核心频率1395MHz（加速频率1695MHz）",
    "NVIDIA GeForce RTX 4080": "9728 CUDA核心，16GB GDDR6X显存，显存带宽716.8GB/s，核心频率2205MHz（加速频率2505MHz）",
    "NVIDIA GeForce RTX 4090": "16384 CUDA核心，24GB GDDR6X显存，显存带宽1008GB/s，核心频率2235MHz（加速频率2520MHz）",
    "NVIDIA Tesla T100": "2560 CUDA核心，16GB GDDR6显存，显存带宽448GB/s，核心频率1410MHz（加速频率1590MHz）",
    "NVIDIA Tesla A100": "6912 CUDA核心，40GB HBM2显存，显存带宽1555GB/s，核心频率1410MHz（加速频率1530MHz）"
}


def update_dropdowns(table_column_1, table_column_2):
    if table_column_1 == "CPU":
        options = CPU_arr
    elif table_column_1 == "GPU":
        options = GPU_arr
    else:
        options = []
    table_column_2 = gr.Dropdown.update(choices=options)
    return table_column_2


def gpt_outputs(dtype, device):
    if dtype == "CPU":
        outputs = CPU_dict[device]
    elif dtype == "GPU":
        outputs = GPU_dict[device]
    else:
        outputs = []
    return outputs


def latency_predict(file):
    if file is None:
        return "Please upload a file!", None
    if file.name.split('.')[-1] != "py":
        return "Please upload a model file!", None
    latency = random.uniform(0, 1)
    # latency = ZYX's Model(file.name)
    topology_image_path = "./model_onnx.png"
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

                with gr.Row(elem_id="centered-row"):
                    with gr.Column():
                        with gr.Row():
                            Type_Dropdown = gr.Dropdown(["CPU", "GPU"], label="Choose your device model", value="CPU")
                            Device_Dropdown = gr.Dropdown(CPU_arr, label="Choose your device")
                            Device_button = gr.Button("Submit", elem_id="device-button")
                        with gr.Row():
                            with gr.Column(scale=1, min_width=50):
                                ai_avatar = gr.Image(value="GPT-copy.svg", interactive=False, elem_id="avatar",show_label=False)
                            with gr.Column(scale=10):
                                GPT_outputs = gr.Text(show_label=False)
                        file_input = gr.File(label="Upload File",elem_id="file-input")
                        latency_output = gr.Text(label="Latency")
                        with gr.Row():
                            cancel_button = gr.Button("Cancel")
                            submit_button = gr.Button("Submit")
                    topology_image = gr.Image(label="Model Topology",elem_id="image-output")

            with gr.Tab("Select Right Model"):
                gr.Markdown(
                    """
                    <div style="text-align: center;">
                        <h1>Select Right Model</h1>
                    </div>
                    """
                )
                with gr.Row(elem_id="centered-row"):
                    with gr.Column():
                        with gr.Row():
                            Type_Dropdown2 = gr.Dropdown(["CPU", "GPU"], label="Choose your device model", value="CPU")
                            Device_Dropdown2 = gr.Dropdown(CPU_arr, label="Choose your device")
                            Device_button2 = gr.Button("Submit", elem_id="device-button")
                        with gr.Row():
                            with gr.Column(scale=1, min_width=50):
                                ai_avatar2 = gr.Image(value="GPT-copy.svg", interactive=False, elem_id="avatar",show_label=False)
                            with gr.Column(scale=10):
                                GPT_outputs2 = gr.Text(show_label=False)
                        files_input = gr.File(label="Select Files", file_count="multiple", elem_id="file-input")
                        latency_limit = gr.Slider(minimum=0.0, maximum=1.0, step=0.01, label="Latency Limit (ms)")
                        with gr.Row():
                            cancel_button2 = gr.Button("Cancel")
                            submit_button2 = gr.Button("Submit")
                    with gr.Column(elem_id="second-output"):
                        meet_files_output = gr.Text(label="Meet Files and Latency", elem_id="meet")
                        unmeet_files_output = gr.Text(label="Unmeet Files and Latency", elem_id="unmeet")

        # Button functionality
        Device_button.click(
            gpt_outputs,
            inputs=[Type_Dropdown, Device_Dropdown],
            outputs=GPT_outputs
        )
        submit_button.click(
            latency_predict,
            inputs=file_input,
            outputs=[latency_output, topology_image]
        )
        cancel_button.click(
            clear_fields,
            outputs=[file_input, latency_output, topology_image]
        )
        Device_button2.click(
            gpt_outputs,
            inputs=[Type_Dropdown2, Device_Dropdown2],
            outputs=GPT_outputs2
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

        Type_Dropdown.change(
            update_dropdowns,
            inputs=[Type_Dropdown, Device_Dropdown],
            outputs=[Device_Dropdown]
        )
        Type_Dropdown2.change(
            update_dropdowns,
            inputs=[Type_Dropdown2, Device_Dropdown2],
            outputs=[Device_Dropdown2]
        )
        # Custom CSS to set high-contrast based on system theme
        iface.css = """
            /* Light mode: darken content background for contrast */
             #device-button {
                padding: 4px 8px;
                font-size: 12px;
                min-width: 60px;
            }
            #file-input {
                height: 240px;
            }
            #image-output {
                max-width: 450px;
                max-height: 600px;
            }
            #second-output {
                max-width: 450px;
            }
            /* 设置头像的样式 */
            #avatar {
                border-radius: 50%;  /* 圆形头像 */
                width: 40px;         /* 调整头像大小 */
                height: 40px;
                object-fit: cover;   /* 保持图像比例 */
                top: 50%;
                margin-top: -20px;
                left: 50%;
                margin-left: -20px;
            }
                    """

    iface.launch()
