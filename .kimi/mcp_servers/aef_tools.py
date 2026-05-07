#!/usr/bin/env python3
"""AEF_qwen 专用 MCP 工具服务器."""
from __future__ import annotations

import json
import os
import sys
import traceback

from mcp.server import Server
from mcp.server.stdio import stdio_server
from mcp.types import TextContent, Tool

app = Server("aef-tools")


@app.list_tools()
async def list_tools() -> list[Tool]:
    return [
        Tool(
            name="execute_python",
            description="Execute a Python code snippet and return stdout/result. Useful for quick math checks, tensor shape verification, or numpy operations.",
            inputSchema={
                "type": "object",
                "properties": {
                    "code": {"type": "string", "description": "Python code to execute"},
                },
                "required": ["code"],
            },
        ),
        Tool(
            name="check_environment",
            description="Check PyTorch/CUDA environment and installed package versions.",
            inputSchema={
                "type": "object",
                "properties": {},
            },
        ),
        Tool(
            name="compute_array_stats",
            description="Compute statistics (mean, std, min, max, shape, NaN count) for a pickled numpy array or tensor file.",
            inputSchema={
                "type": "object",
                "properties": {
                    "file_path": {"type": "string", "description": "Absolute path to .npy or .pt file"},
                },
                "required": ["file_path"],
            },
        ),
    ]


@app.call_tool()
async def call_tool(name: str, arguments: dict) -> list[TextContent]:
    if name == "execute_python":
        code = arguments.get("code", "")
        old_stdout = sys.stdout
        old_stderr = sys.stderr
        from io import StringIO
        stdout_capture = StringIO()
        stderr_capture = StringIO()
        sys.stdout = stdout_capture
        sys.stderr = stderr_capture
        result_val = None
        try:
            # Execute with access to common libs
            globals_dict = {
                "__builtins__": __builtins__,
                "os": os, "sys": sys, "json": json, "math": __import__("math"),
                "np": __import__("numpy"), "torch": __import__("torch"),
                "F": __import__("torch.nn.functional", fromlist=["F"]),
            }
            exec(code, globals_dict)
            result_val = globals_dict.get("_result", None)
        except Exception:
            result_val = traceback.format_exc()
        finally:
            sys.stdout = old_stdout
            sys.stderr = old_stderr

        out_text = stdout_capture.getvalue()
        err_text = stderr_capture.getvalue()
        outputs = []
        if out_text:
            outputs.append(f"[stdout]\n{out_text}")
        if err_text:
            outputs.append(f"[stderr]\n{err_text}")
        if result_val is not None:
            outputs.append(f"[result]\n{result_val}")
        return [TextContent(type="text", text="\n".join(outputs) if outputs else "No output.")]

    elif name == "check_environment":
        lines = []
        lines.append(f"Python: {sys.version}")
        try:
            import torch
            lines.append(f"PyTorch: {torch.__version__}")
            lines.append(f"CUDA available: {torch.npu.is_available()}")
            if torch.npu.is_available():
                lines.append(f"CUDA version: {torch.version.cuda}")
                lines.append(f"GPU count: {torch.cuda.device_count()}")
                for i in range(torch.cuda.device_count()):
                    lines.append(f"  GPU {i}: {torch.cuda.get_device_name(i)}")
        except Exception as e:
            lines.append(f"PyTorch check error: {e}")
        try:
            import rasterio
            lines.append(f"rasterio: {rasterio.__version__}")
        except Exception as e:
            lines.append(f"rasterio: not available ({e})")
        try:
            import geopandas
            lines.append(f"geopandas: {geopandas.__version__}")
        except Exception as e:
            lines.append(f"geopandas: not available ({e})")
        return [TextContent(type="text", text="\n".join(lines))]

    elif name == "compute_array_stats":
        path = arguments.get("file_path", "")
        try:
            if path.endswith(".npy"):
                import numpy as np
                arr = np.load(path)
            elif path.endswith(".pt") or path.endswith(".pth"):
                import torch
                obj = torch.load(path, map_location="cpu", weights_only=False)
                if isinstance(obj, torch.Tensor):
                    arr = obj.numpy()
                else:
                    return [TextContent(type="text", text=f"Loaded object is {type(obj)}, not a tensor.")]
            else:
                return [TextContent(type="text", text="Unsupported file extension. Use .npy or .pt/.pth")]

            import numpy as np
            stats = {
                "shape": arr.shape,
                "dtype": str(arr.dtype),
                "mean": float(np.mean(arr)),
                "std": float(np.std(arr)),
                "min": float(np.min(arr)),
                "max": float(np.max(arr)),
                "nan_count": int(np.isnan(arr).sum()),
                "inf_count": int(np.isinf(arr).sum()),
            }
            return [TextContent(type="text", text=json.dumps(stats, indent=2))]
        except Exception as e:
            return [TextContent(type="text", text=f"Error: {e}\n{traceback.format_exc()}")]

    return [TextContent(type="text", text=f"Unknown tool: {name}")]


async def main():
    async with stdio_server() as (read_stream, write_stream):
        await app.run(read_stream, write_stream, app.create_initialization_options())


if __name__ == "__main__":
    import asyncio
    asyncio.run(main())
