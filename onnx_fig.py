# coding : utf-8
# Author : yuxiang Zeng
import torch
import torch.nn as nn
import torch.onnx
import onnx
from onnx.tools.net_drawer import GetPydotGraph
import os

from models.birnn import BIRNN
from models.gcn import GCN, GraphSage
from models.gru import GRU
from models.lstm import LSTM
from models.mlp import MLP


def export_and_visualize_model(model, dummy_input, export_path="models.onnx", output_image_path="model_visualization.png"):
    """
    将 PyTorch 模型导出为 ONNX 格式并生成其可视化图像

    Args:
        model (nn.Module): PyTorch 模型
        dummy_input (torch.Tensor): 模型的示例输入，用于定义输入形状
        export_path (str): 导出 ONNX 文件的路径
        output_image_path (str): 导出的网络结构图像的路径

    Returns:
        None
    """
    # 设置模型为评估模式
    model.eval()

    # 导出为 ONNX 格式
    torch.onnx.export(
        model,
        dummy_input,
        export_path,
        export_params=True,  # 保存权重
        opset_version=11,    # 使用的 ONNX opset 版本
        input_names=["input"],  # 输入节点的名称
        output_names=["output"],  # 输出节点的名称
        dynamic_axes={"input": {0: "batch_size"}, "output": {0: "batch_size"}},  # 支持动态批大小
    )
    print(f"Model has been exported to {export_path}")

    # 加载 ONNX 模型
    onnx_model = onnx.load(export_path)

    # 生成网络图
    pydot_graph = GetPydotGraph(
        onnx_model.graph,  # 注意参数传递的是 `onnx_model.graph`
        name=onnx_model.graph.name,
        rankdir="TB",  # 图形方向："TB" 表示从上到下
    )

    # 保存为 PNG 图像
    with open(output_image_path, "wb") as f:
        f.write(pydot_graph.create_png())
    print(f"ONNX models visualization saved as image at {output_image_path}")


def load_model_from_file(file_path):
    import importlib.util
    """
    动态加载模型类并实例化模型

    Args:
        file_path (str): 模型文件的路径

    Returns:
        model_instance (object): 模型实例
    """
    try:
        # 从文件名中提取模块名
        module_name = os.path.splitext(os.path.basename(file_path))[0]

        # 动态加载模块
        spec = importlib.util.spec_from_file_location(module_name, file_path)
        model_module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(model_module)

        # 根据文件名生成类名（假设类名为文件名的大写形式）
        model_class_name = module_name.upper()

        # 检查模块中是否有对应的类
        if hasattr(model_module, model_class_name):
            model_class = getattr(model_module, model_class_name)
            model_instance = model_class()  # 实例化模型
            return model_instance
        else:
            raise AttributeError(f"Module '{module_name}' does not contain a class named '{model_class_name}'")
    except Exception as e:
        raise RuntimeError(f"Error loading model from file: {e}")


def generate_the_fig(file_path):
    """
    主函数：加载模型，生成示例输入，导出 ONNX 文件并生成可视化图像

    Args:
        file_path (str): 模型文件的路径

    Returns:
        None
    """
    # 加载模型
    model = load_model_from_file(file_path)

    # 获取示例输入
    try:
        example_input = model.get_sample()  # 确保模型实现了 get_sample 方法
    except AttributeError:
        raise AttributeError(f"The model does not implement 'get_sample' method.")

    # 导出模型并生成可视化图像
    model_name = os.path.splitext(os.path.basename(file_path))[0]
    export_and_visualize_model(
        model=model,
        dummy_input=example_input,
        export_path=f"{model_name}_onnx.onnx",
        output_image_path=f"model_onnx.png",
    )
    print(f"Model '{model_name}' has been successfully exported and visualized.")
    print(model.latency)
    return model.latency


if __name__ == "__main__":
    # 假设用户选择的文件路径
    selected_file = "models/gcn.py"  # 替换为您实际选择的文件路径
    generate_the_fig(selected_file)
