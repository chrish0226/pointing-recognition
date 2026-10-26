"""TensorRT 10.x FP16 engine building and execution on the current NVIDIA GPU."""
import os
from pathlib import Path
import json
import time
import hashlib

import numpy as np
import torch

# Import torch first so its CUDA 12 DLLs are available on Windows.
_dll_handles=[]
if os.name=='nt':
    _dll_handles.append(os.add_dll_directory(str(Path(torch.__file__).parent/'lib')))
import tensorrt as trt
LOGGER=trt.Logger(trt.Logger.WARNING)

ROOT=Path(__file__).resolve().parent
TRT_DIR=ROOT/'models/rtmw3d_l_tensorrt_fp16'


def build_engine(onnx_path,engine_path):
    onnx_path=Path(onnx_path);engine_path=Path(engine_path)
    engine_path.parent.mkdir(parents=True,exist_ok=True)
    logger=LOGGER
    builder=trt.Builder(logger)
    network=builder.create_network(1<<int(trt.NetworkDefinitionCreationFlag.EXPLICIT_BATCH))
    parser=trt.OnnxParser(network,logger)
    if not parser.parse(onnx_path.read_bytes()):
        raise RuntimeError('\n'.join(str(parser.get_error(i)) for i in range(parser.num_errors)))
    config=builder.create_builder_config()
    config.set_flag(trt.BuilderFlag.FP16)
    config.clear_flag(trt.BuilderFlag.TF32)
    config.set_flag(trt.BuilderFlag.PREFER_PRECISION_CONSTRAINTS)
    config.set_memory_pool_limit(trt.MemoryPoolType.WORKSPACE,1024*1024*1024)
    config.profiling_verbosity=trt.ProfilingVerbosity.DETAILED
    protected=[]
    for i in range(network.num_layers):
        layer=network.get_layer(i)
        if layer.type in (trt.LayerType.REDUCE,trt.LayerType.SOFTMAX) or any(s in layer.name.lower() for s in ('norm/','/pow','/sqrt')):
            # Keep accumulation and normalization in FP32 to avoid overflow.
            layer.precision=trt.float32
            for j in range(layer.num_outputs):
                if layer.get_output(j).dtype in (trt.float32,trt.float16):layer.set_output_type(j,trt.float32)
            protected.append(layer.name)
    cache_path=engine_path.with_suffix('.timing_cache')
    cache=config.create_timing_cache(cache_path.read_bytes() if cache_path.exists() else b'')
    config.set_timing_cache(cache,False)
    print(f'BUILD {onnx_path.name}: {network.num_layers} layers, FP16, FP32-sensitive={len(protected)}',flush=True)
    start=time.perf_counter();data=builder.build_serialized_network(network,config)
    if data is None:raise RuntimeError(f'TensorRT build failed: {onnx_path}')
    engine_path.write_bytes(bytes(data));cache_path.write_bytes(bytes(config.get_timing_cache().serialize()))
    runtime=trt.Runtime(logger);engine=runtime.deserialize_cuda_engine(data)
    inspector=engine.create_engine_inspector()
    info=inspector.get_engine_information(trt.LayerInformationFormat.JSON)
    engine_path.with_suffix('.layers.json').write_text(info,encoding='utf8')
    meta={'onnx':str(onnx_path),'onnx_sha256':hashlib.sha256(onnx_path.read_bytes()).hexdigest(),
          'engine':str(engine_path),'engine_sha256':hashlib.sha256(bytes(data)).hexdigest(),
          'tensorrt':trt.__version__,'torch':torch.__version__,'gpu':torch.cuda.get_device_name(0),
          'compute_capability':torch.cuda.get_device_capability(0),'precision':'FP16 with FP32-sensitive reductions',
          'workspace_bytes':1024**3,'build_s':time.perf_counter()-start,'protected_fp32_layers':protected,
          'half_format_mentions':info.lower().count('half'),'float_format_mentions':info.lower().count('float')}
    engine_path.with_suffix('.build.json').write_text(json.dumps(meta,indent=2),encoding='utf8')
    if not meta['half_format_mentions']:raise RuntimeError('No FP16 execution formats found in engine inspection')
    print(json.dumps({k:v for k,v in meta.items() if k!='protected_fp32_layers'}),flush=True)
    return meta


class Engine:
    def __init__(self,path):
        self.logger=LOGGER;self.runtime=trt.Runtime(self.logger)
        self.engine=self.runtime.deserialize_cuda_engine(Path(path).read_bytes())
        if self.engine is None:raise RuntimeError(f'Cannot deserialize {path}')
        self.context=self.engine.create_execution_context()
        self.stream=torch.cuda.Stream();self.buffers={};self.host={};self.inputs=[];self.outputs=[]
        dtype_map={trt.float32:torch.float32,trt.float16:torch.float16,trt.int32:torch.int32,trt.int64:torch.int64}
        for i in range(self.engine.num_io_tensors):
            name=self.engine.get_tensor_name(i);shape=tuple(self.engine.get_tensor_shape(name))
            if any(x<0 for x in shape):raise ValueError(f'Dynamic output not supported here: {name} {shape}')
            self.buffers[name]=torch.empty(shape,dtype=dtype_map[self.engine.get_tensor_dtype(name)],device='cuda')
            self.context.set_tensor_address(name,self.buffers[name].data_ptr())
            if self.engine.get_tensor_mode(name)==trt.TensorIOMode.INPUT:self.inputs.append(name)
            else:
                self.outputs.append(name)
                self.host[name]=torch.empty(shape,dtype=self.buffers[name].dtype,pin_memory=True)
        if len(self.inputs)!=1:raise ValueError('Expected one input')
    def __call__(self,array):
        src=torch.from_numpy(np.ascontiguousarray(array))
        with torch.cuda.stream(self.stream):
            self.buffers[self.inputs[0]].copy_(src,non_blocking=True)
            if not self.context.execute_async_v3(self.stream.cuda_stream):raise RuntimeError('TensorRT execution failed')
            for name in self.outputs:self.host[name].copy_(self.buffers[name],non_blocking=True)
        self.stream.synchronize()
        outputs=[self.host[name].numpy().copy() for name in self.outputs]
        if not all(np.isfinite(a).all() for a in outputs):raise FloatingPointError('Nonfinite TensorRT output')
        return outputs


def export_detector_before_nms():
    """Keep official decoded YOLOX boxes/scores; NMS runs explicitly on CPU."""
    import onnx
    from onnx import numpy_helper
    source=ROOT/'models/detector_yolox_s/end2end.onnx'
    target=TRT_DIR/'yolox_s_pre_nms.onnx';TRT_DIR.mkdir(exist_ok=True)
    model=onnx.load(str(source))
    nms=next(n for n in model.graph.node if n.op_type=='NonMaxSuppression')
    onnx.utils.extract_model(str(source),str(target),['input'],list(nms.input[:2]),check_model=True)
    graph=onnx.load(str(target))
    # The source's TopK=5000 exceeds TensorRT's maximum K=3840. Keep the
    # strongest 1000 proposals; verify this against real-frame detections.
    for tensor in graph.graph.initializer:
        if tensor.name=='1207':tensor.CopyFrom(numpy_helper.from_array(np.array([1000],np.int64),'1207'))
    for value in list(graph.graph.value_info)+list(graph.graph.output):
        for dim in value.type.tensor_type.shape.dim:
            if dim.dim_value==5000:dim.dim_value=1000
    onnx.checker.check_model(graph);onnx.save(graph,str(target))
    print('Detector pre-NMS outputs:',[(x.name,[d.dim_value for d in x.type.tensor_type.shape.dim]) for x in graph.graph.output],flush=True)
    return target


if __name__=='__main__':
    import argparse
    ap=argparse.ArgumentParser();ap.add_argument('--which',choices=['pose','detector','both'],default='both')
    args=ap.parse_args()
    if args.which in ('pose','both'):
        build_engine(ROOT/'models/rtmw3d_l_384x288/rtmw3d_l.onnx',TRT_DIR/'rtmw3d_l_fp16.engine')
    if args.which in ('detector','both'):
        build_engine(export_detector_before_nms(),TRT_DIR/'yolox_s_fp16.engine')
