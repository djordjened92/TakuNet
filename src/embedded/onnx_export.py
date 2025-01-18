import torch
from torch import nn

import os
import argparse
from utils.net_utils import extract_net_params, extract_optim_params
from utils.net_utils import select_arch

def onnx_export(args: argparse.Namespace) -> None:
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    net_kwargs = extract_net_params(args)
    optim_kwargs = extract_optim_params(args)

    net_kwargs['classes'] = [str(i) for i in range(args.num_classes)]
    optim_kwargs['dataset_length'] = 0

    loss = nn.CrossEntropyLoss(reduction="mean", 
                            label_smoothing=args.label_smoothing, 
                            weight=torch.tensor(args.class_weights).to(device) if args.class_weights is not None else None
                            )
    
    ckpts_file = os.path.join(args.ckpts_path, os.listdir(args.ckpts_path)[0]) if not args.ckpts_path.endswith(".ckpt") else args.ckpts_path
    model = select_arch(net_kwargs, loss, optim_kwargs, ckpts_file).to(device)

    os.makedirs('exports', exist_ok=True)
    export_path = os.path.join('exports', f'{args.network}_fp32_opset{args.onnx_opset_version}.onnx')

    example_input = torch.randn(1, 3, args.img_height, args.img_width).to(device)
    try:
        torch.onnx.export(
            model,
            example_input,
            export_path,
            export_params=True,
            opset_version=args.onnx_opset_version,
            input_names=['input'],
            output_names=['output'],
            dynamic_axes={
                'input': {0: 'batch_size'},
                'output': {0: 'batch_size'}
            }
        )
        print(f"Model saved correctly in {export_path}")
    except Exception as e:
        print(f"Error during export: {e}")

    if args.network.lower() == 'takunet':
        model = model.half()
        example_input = torch.randn(1, 3, args.img_height, args.img_width).half().to(device)
        export_path = os.path.join('exports', f'{args.network}_fp16_opset{args.onnx_opset_version}.onnx')
        
        try:
            torch.onnx.export(
                model,
                example_input,
                export_path,
                export_params=True,
                opset_version=args.onnx_opset_version,
                input_names=['input'],
                output_names=['output'],
                dynamic_axes={
                    'input': {0: 'batch_size'},
                    'output': {0: 'batch_size'}
                }
            )
            print(f"Model saved correctly in {export_path}")
        except Exception as e:
            print(f"Error during export: {e}")