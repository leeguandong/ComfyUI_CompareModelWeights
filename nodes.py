import os
import random
import torch
import numpy as np
import matplotlib.pyplot as plt

from diffusers import StableDiffusionPipeline
from io import BytesIO
from PIL import Image

import folder_paths


# def convert_preview_image(images):
#     # 转换图像为 torch.Tensor，并调整维度顺序为 NHWC
#     images_tensors = []
#     for img in images:
#         # 将 PIL.Image 转换为 numpy.ndarray
#         img_array = np.array(img)
#         # 转换 numpy.ndarray 为 torch.Tensor
#         img_tensor = torch.from_numpy(img_array).float() / 255.
#         # 转换图像格式为 CHW (如果需要)
#         if img_tensor.ndim == 3 and img_tensor.shape[-1] == 3:
#             img_tensor = img_tensor.permute(2, 0, 1)
#         # 添加批次维度并转换为 NHWC
#         img_tensor = img_tensor.unsqueeze(0).permute(0, 2, 3, 1)
#         images_tensors.append(img_tensor)
#
#     if len(images_tensors) > 1:
#         output_image = torch.cat(images_tensors, dim=0)
#     else:
#         output_image = images_tensors[0]
#     return output_image


# 提取UNet权重
def get_unet_weights(model):
    return {name: param.data for name, param in model.unet.named_parameters()}


# 计算L2范数差距
def calculate_difference(base_weights, compare_weights):
    differences = {}
    for name in base_weights.keys():
        if name in compare_weights:
            diff = torch.norm(base_weights[name] - compare_weights[name]).item()
            differences[name] = diff
    return differences


# 计算归一化差距
def calculate_normalized_difference(base_weights, compare_weights):
    normalized_differences = {}
    base_norms = {name: torch.norm(param).item() for name, param in base_weights.items()}
    for name in base_weights.keys():
        if name in compare_weights:
            diff = torch.norm(base_weights[name] - compare_weights[name]).item()
            normalized_diff = diff / base_norms[name] if base_norms[name] != 0 else 0
            normalized_differences[name] = normalized_diff
    return normalized_differences


# 筛选以特定前缀开头的权重
def filter_weights(weights, prefix):
    return {name: value for name, value in weights.items() if name.startswith(prefix)}


def plot_differences(weight_names, values, model_names, title, ylabel):
    x = range(len(weight_names))
    plt.figure(figsize=(48, 24))

    for i, model_name in enumerate(model_names):
        plt.bar([p + i * 0.4 for p in x], values[i], width=0.4, label=model_name, align='center')

        # 在条形图上标注差距，调整标识位置
        for j, value in enumerate(values[i]):
            plt.text(j + i * 0.4, value + 0.01, f'{value:.2f}', ha='center', fontsize=4)

    plt.ylabel(ylabel)
    plt.title(title)
    plt.xticks([p + 0.2 for p in x], weight_names, rotation=45, fontsize=4)
    plt.legend()
    plt.tight_layout()

    # 保存图像到内存
    buf = BytesIO()
    plt.savefig(buf, format='png', dpi=300)
    plt.close()  # 关闭当前图形，释放内存
    buf.seek(0)  # 重置指针到开始位置

    # 返回PIL图像
    return Image.open(buf)
    # 返回numpy数组
    # img = Image.open(buf)
    # return np.array(img)


class CheckPointLoader_Compare:
    @classmethod
    def INPUT_TYPES(s):
        return {"required": {"ckpt_name": (folder_paths.get_filename_list("checkpoints"),), }}

    RETURN_TYPES = ("MODEL", "STRING")
    RETURN_NAMES = ("model", "name")
    FUNCTION = "load_checkpoint_compare"
    CATEGORY = "CompareModelWieghts"

    def load_checkpoint_compare(self, ckpt_name):
        ckpt_path = folder_paths.get_full_path("checkpoints", ckpt_name)
        model = StableDiffusionPipeline.from_single_file(ckpt_path)
        unet_weights = get_unet_weights(model)
        return (unet_weights, ckpt_name)


class CompareModelWeightsDiff:
    @classmethod
    def INPUT_TYPES(s):
        return {"required": {
            "base_model": ("MODEL",),
            "compare_model1": ("MODEL",),

            "compare_model_name1": ("STRING", {"forceInput": True, "default": ""}),
        },
            "optional": {
                "compare_model2": ("MODEL",),
                "compare_model3": ("MODEL",),
                "compare_model4": ("MODEL",),

                "compare_model_name2": ("STRING", {"forceInput": True, "default": ""}),
                "compare_model_name3": ("STRING", {"forceInput": True, "default": ""}),
                "compare_model_name4": ("STRING", {"forceInput": True, "default": ""}),
            }}

    RETURN_TYPES = ("IMAGE",)
    RETURN_NAMES = ("image",)
    FUNCTION = "compare_model_weights_diff"
    CATEGORY = "CompareModelWieghts"

    def compare_model_weights_diff(self,
                                   base_model,
                                   compare_model1,
                                   compare_model_name1,
                                   compare_model2=None,
                                   compare_model3=None,
                                   compare_model4=None,
                                   compare_model_name2=None,
                                   compare_model_name3=None,
                                   compare_model_name4=None):
        compare_model = [model for model in
                         [compare_model1, compare_model2, compare_model3, compare_model4] if model is not None]
        model_names = [name for name in
                       [compare_model_name1, compare_model_name2, compare_model_name3, compare_model_name4] if
                       name is not None]
        prefixes = ['down', 'up', 'mid']

        difference_images = []
        for prefix in prefixes:
            filtered_base_weights = filter_weights(base_model, prefix)
            filtered_model_weights = [filter_weights(model_weights, prefix) for model_weights in compare_model]

            differences = [calculate_difference(filtered_base_weights, model_weights) for model_weights in
                           filtered_model_weights]
            weight_names = list(differences[0].keys())

            # 准备数据
            values = [[differences[i][name] for name in weight_names] for i in range(len(differences))]

            # 绘制原始差距图并返回PIL图像
            difference_image = plot_differences(weight_names, values, model_names,
                                                f'UNet Parameter Differences for {prefix} (Base Model = 1)',
                                                'Relative Parameter Difference')
            difference_images.append(difference_image)
        # output_images = convert_preview_image(difference_images)
        # images = np.concatenate(difference_images, axis=1)
        return (difference_images,)


class CompareModelWeightsDiffNormalized:
    @classmethod
    def INPUT_TYPES(s):
        return {"required": {
            "base_model": ("MODEL",),
            "compare_model1": ("MODEL",),

            "compare_model_name1": ("STRING", {"forceInput": True, "default": ""}),
        },
            "optional": {
                "compare_model2": ("MODEL",),
                "compare_model3": ("MODEL",),
                "compare_model4": ("MODEL",),

                "compare_model_name2": ("STRING", {"forceInput": True, "default": ""}),
                "compare_model_name3": ("STRING", {"forceInput": True, "default": ""}),
                "compare_model_name4": ("STRING", {"forceInput": True, "default": ""}),
            }}

    RETURN_TYPES = ("IMAGE",)
    RETURN_NAMES = ("image",)
    FUNCTION = "compare_model_weights_diff"
    CATEGORY = "CompareModelWieghts"

    def compare_model_weights_diff(self,
                                   base_model,
                                   compare_model1,
                                   compare_model_name1,
                                   compare_model2=None,
                                   compare_model3=None,
                                   compare_model4=None,
                                   compare_model_name2=None,
                                   compare_model_name3=None,
                                   compare_model_name4=None):
        compare_model = [model for model in
                         [compare_model1, compare_model2, compare_model3, compare_model4] if model is not None]
        model_names = [name for name in
                       [compare_model_name1, compare_model_name2, compare_model_name3, compare_model_name4] if
                       name is not None]
        prefixes = ['down', 'up', 'mid']

        normalized_images = []
        for prefix in prefixes:
            filtered_base_weights = filter_weights(base_model, prefix)
            filtered_model_weights = [filter_weights(model_weights, prefix) for model_weights in compare_model]

            normalized_differences = [calculate_normalized_difference(filtered_base_weights, model_weights) for
                                      model_weights in filtered_model_weights]
            weight_names = list(normalized_differences[0].keys())

            # 准备数据
            normalized_values = [[normalized_differences[i][name] for name in weight_names] for i in
                                 range(len(normalized_differences))]

            # 绘制归一化差距图并返回PIL图像
            normalized_image = plot_differences(weight_names, normalized_values, model_names,
                                                f'Normalized UNet Parameter Differences for {prefix}',
                                                'Normalized Parameter Difference')
            normalized_images.append(normalized_image)
        # output_images = convert_preview_image(normalized_images)
        # images = np.concatenate(normalized_images, axis=1)
        return (normalized_images,)


class PreviewImageCompareModelWeights:
    def __init__(self):
        self.output_dir = folder_paths.get_temp_directory()
        self.type = "temp"
        self.prefix_append = "_temp_" + ''.join(random.choice("abcdefghijklmnopqrstupvxyz") for x in range(5))
        self.compress_level = 1

    @classmethod
    def INPUT_TYPES(s):
        return {"required":
                    {"images": ("IMAGE",), },
                "hidden": {"prompt": "PROMPT", "extra_pnginfo": "EXTRA_PNGINFO"},
                }

    RETURN_TYPES = ()
    OUTPUT_NODE = True
    FUNCTION = "preview_images"
    CATEGORY = "CompareModelWieghts"

    def preview_images(self, images, filename_prefix="ComfyUI", prompt=None, extra_pnginfo=None):
        filename_prefix += self.prefix_append
        full_output_folder, filename, counter, subfolder, filename_prefix = folder_paths.get_save_image_path(
            filename_prefix, self.output_dir, images[0].size[0], images[0].size[1])
        results = list()
        for (batch_number, image) in enumerate(images):
            img = image

            filename_with_batch_num = filename.replace("%batch_num%", str(batch_number))
            file = f"{filename_with_batch_num}_{counter:05}_.png"
            img.save(os.path.join(full_output_folder, file), compress_level=self.compress_level)
            results.append({
                "filename": file,
                "subfolder": subfolder,
                "type": self.type
            })
            counter += 1

        return {"ui": {"images": results}}


NODE_CLASS_MAPPINGS = {
    "CheckPointLoader_Compare": CheckPointLoader_Compare,
    "CompareModelWeightsDiff": CompareModelWeightsDiff,
    "CompareModelWeightsDiffNormalized": CompareModelWeightsDiffNormalized,
    "PreviewImageCompareModelWeights": PreviewImageCompareModelWeights
}

NODE_DISPLAY_NAME_MAPPINGS = {
    "CheckPointLoader_Compare": "Compare CheckPointer Loader",
    "CompareModelWeightsDiff": "Compare ModelWeightsDiff",
    "CompareModelWeightsDiffNormalized": "Compare ModelWeightsDiff Normalized",
    "PreviewImageCompareModelWeights": "Compare Preview Image"
}
