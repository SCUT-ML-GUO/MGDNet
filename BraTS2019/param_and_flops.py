# 安装：pip install torchscan
from torchscan import summary
from networks.MGDNet import MGDNet

def calculate_flops_torchscan(model, input_size=(3, 224, 224)):
    model.eval()

    # 打印详细摘要
    summary(model, input_size)

    # 也可以获取具体数值
    from torchscan import crawl_module
    details = crawl_module(model, (input_size))

    total_flops = details['flops']
    total_params = details['params']

    print(f"总FLOPs: {total_flops:,}")
    print(f"总参数量: {total_params:,}")

    return total_flops, total_params

model = MGDNet(in_channels=2, num_classes=4)
flops, params = calculate_flops_torchscan(model, input_size=[(2, 128, 128, 128), (2, 128, 128, 128)])
