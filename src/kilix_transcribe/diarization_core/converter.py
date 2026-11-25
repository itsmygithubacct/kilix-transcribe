"""Restricted offline converter requiring a trusted managed export context.
The receipt-gated product tool verifies its pinned environment before loading.
Nothing is imported from an ambient ML environment.
Apache-2.0. Architecture notices are in ../../notices/.
"""
from abc import ABC, abstractmethod
from collections import OrderedDict
import fcntl
import io
import os

from .fd import SEALS, F_ADD_SEALS, memfd, Refusal, digest_fd, read_sealed, require

CHECKPOINT_BYTES = 29357703
CHECKPOINT_SHA256 = '5b1a88b6f8d85826fabef804779c3372b42f3af21457fa48bd5c097c0686b2de'
ARCHITECTURE_COMMIT = '065629c313eaf1a01c65c640c46d77e61e9607b4'
METADATA = {'framework': '3d-speaker', 'output_dim': '512', 'sample_rate': '16000',
            'normalize_samples': '1', 'language': 'en', 'feature_normalize_type': 'global-mean'}
MAX_STATE_BYTES = 64 << 20
MAX_GRAPH_BYTES = 128 << 20

class ManagedExportContext(ABC):
    """Trusted F100/License/OwnedExecution seam, never constructed from JSON.
    The existing owner must verify the complete environment/ELF/module hashes,
    dependency licences, CPU-only network-denied cgroup, deadline and input
    receipt before returning these already loaded managed modules.
    """
    @abstractmethod
    def verify_exact_input_and_environment(self, fd, sha256, size): ...
    @abstractmethod
    def modules_and_architecture(self): ...
    @abstractmethod
    def prove_teardown_on_error(self): ...

def validate_state(torch, model, state):
    require(type(state) in (dict, OrderedDict), 'PLAIN_STATE_DICT_REQUIRED')
    expected = model.state_dict()  # trusted exact CAMPPlus(80,512), not input metadata
    require(0 < len(expected) <= 8192 and set(state) == set(expected), 'STATE_KEY_MISMATCH')
    total = 0
    for key in sorted(expected):
        require(type(key) is str and len(key) <= 256, 'STATE_KEY_INVALID')
        actual, reference = state[key], expected[key]
        require(type(actual) is torch.Tensor and type(reference) is torch.Tensor,
                'EXACT_TENSOR_TYPE_REQUIRED')
        require(actual.device.type == 'cpu' and actual.layout == torch.strided,
                'CPU_DENSE_TENSOR_REQUIRED')
        require(tuple(actual.shape) == tuple(reference.shape) and actual.dtype == reference.dtype,
                'STATE_SHAPE_OR_DTYPE_MISMATCH')
        if key.endswith('.num_batches_tracked'):
            require(actual.dtype == torch.int64 and tuple(actual.shape) == (), 'COUNTER_DTYPE_MISMATCH')
        else:
            require(actual.dtype == torch.float32, 'FLOAT32_WEIGHTS_REQUIRED')
        size = actual.numel() * actual.element_size()
        require(0 <= size <= MAX_STATE_BYTES-total, 'STATE_ALLOCATION_LIMIT')
        total += size
        require(bool(torch.isfinite(actual).all().item()), 'NONFINITE_STATE')
    return state

def restricted_load(torch, model, fd):
    """Internal loader; caller first verifies the immutable exact input.
    This has exactly one load attempt. No safe-global additions, unrestricted
    pickle retry, SDK, URLs, module rewriting or dtype/key coercion.
    """
    with os.fdopen(os.dup(fd), 'rb') as source:
        source.seek(0)
        state = torch.load(source, map_location='cpu', weights_only=True, mmap=False)
    validate_state(torch, model, state)
    model.load_state_dict(state, strict=True)
    model.cpu().eval()
    return model

def validate_and_bind_graph(onnx, fd):
    """Concrete single-graph inspection using the managed, pinned ONNX module."""
    size = os.fstat(fd).st_size
    require(0 < size <= MAX_GRAPH_BYTES, 'GRAPH_SIZE_LIMIT')
    data = os.pread(fd, size+1, 0)
    require(len(data) == size, 'GRAPH_TRUNCATED')
    graph_model = onnx.load_model_from_string(data)
    graph = graph_model.graph
    # The exact fixed-batch architecture preserves batch 1. The legacy 2.7
    # exporter leaves only this output dimension symbolic.
    require(len(graph.input) == 1 and len(graph.output) == 1, 'GRAPH_IO_MISMATCH')
    input_shape = graph.input[0].type.tensor_type.shape.dim
    output_shape = graph.output[0].type.tensor_type.shape.dim
    require([d.dim_param if d.HasField('dim_param') else d.dim_value for d in input_shape]
            == [1,'T',80] and len(output_shape) == 2 and output_shape[1].dim_value == 512,
            'GRAPH_SHAPE_MISMATCH')
    if output_shape[0].dim_param == 'BatchNormalizationembs_dim_0':
        output_shape[0].ClearField('dim_param')
        output_shape[0].dim_value = 1
    # Torch excludes the extra implicit ceil-padding from AveragePool's
    # divisor. Correct only the 52 exact frozen CAM++ pool configurations.
    pools = [node for node in graph.node if node.op_type == 'AveragePool']
    require(len(pools) == 52, 'CAMPP_POOL_POPULATION_MISMATCH')
    for node in pools:
        attributes = {a.name:a for a in node.attribute}
        require(len(attributes) == len(node.attribute)
                and {'ceil_mode','pads','kernel_shape','strides','count_include_pad'} <= set(attributes)
                and attributes['ceil_mode'].i == 1 and list(attributes['pads'].ints) == [0,0]
                and list(attributes['kernel_shape'].ints) == list(attributes['strides'].ints) == [100]
                and attributes['count_include_pad'].i in (0,1), 'CAMPP_POOL_CONFIGURATION_MISMATCH')
        attributes['count_include_pad'].i = 0
    onnx.checker.check_model(graph_model, full_check=True)
    require(not graph_model.functions, 'LOCAL_FUNCTIONS_FORBIDDEN')
    require([(x.domain, x.version) for x in graph_model.opset_import] == [('',11)],
            'EXACT_OPSET_REQUIRED')
    graph = graph_model.graph
    require(0 < len(graph.node) <= 16384 and len(graph.initializer) <= 8192 and
            not graph.sparse_initializer, 'GRAPH_POPULATION_LIMIT')
    require(len(graph.input) == 1 and len(graph.output) == 1 and
            graph.input[0].name == 'feats' and graph.output[0].name == 'embs', 'GRAPH_IO_MISMATCH')
    def shape(value):
        tensor = value.type.tensor_type
        require(tensor.elem_type == onnx.TensorProto.FLOAT, 'FLOAT32_GRAPH_REQUIRED')
        return [d.dim_param if d.HasField('dim_param') else d.dim_value for d in tensor.shape.dim]
    require(shape(graph.input[0]) == [1,'T',80] and shape(graph.output[0]) == [1,512],
            'GRAPH_SHAPE_MISMATCH')
    def tensor_check(tensor):
        require(tensor.data_location != onnx.TensorProto.EXTERNAL and not tensor.external_data,
                'EXTERNAL_GRAPH_DATA_FORBIDDEN')
        require(len(tensor.dims) <= 8 and all(0 <= x <= 1<<20 for x in tensor.dims),
                'GRAPH_TENSOR_SHAPE_LIMIT')
    for tensor in graph.initializer:
        tensor_check(tensor)
    for node in graph.node:
        require(node.domain in ('','ai.onnx'), 'CUSTOM_GRAPH_DOMAIN_FORBIDDEN')
        for attribute in node.attribute:
            require(attribute.type not in (onnx.AttributeProto.GRAPH, onnx.AttributeProto.GRAPHS,
                    onnx.AttributeProto.SPARSE_TENSOR, onnx.AttributeProto.SPARSE_TENSORS),
                    'NESTED_OR_SPARSE_GRAPH_FORBIDDEN')
            if attribute.type == onnx.AttributeProto.TENSOR:
                tensor_check(attribute.t)
            for tensor in attribute.tensors:
                tensor_check(tensor)
    require(len({p.key for p in graph_model.metadata_props}) == len(graph_model.metadata_props),
            'DUPLICATE_GRAPH_METADATA')
    props = {p.key:p.value for p in graph_model.metadata_props}
    require(all(k not in props or props[k] == v for k,v in METADATA.items()), 'CONFLICTING_GRAPH_METADATA')
    props.update(METADATA)
    onnx.helper.set_model_props(graph_model, props)
    onnx.checker.check_model(graph_model, full_check=True)
    data = graph_model.SerializeToString(deterministic=True)
    require(0 < len(data) <= MAX_GRAPH_BYTES, 'GRAPH_SIZE_LIMIT')
    os.ftruncate(fd, 0)
    offset = 0
    while offset < len(data):
        written = os.pwrite(fd, data[offset:offset+65536], offset)
        require(written > 0, 'GRAPH_WRITE_FAILED')
        offset += written

def convert_exact_english(context, checkpoint_fd):
    """Receipt-gated offline conversion; never used by runtime inference.
    Output is an anonymous graph FD, never a runtime pickle. Two clean invocations
    and real waveform/feature/embedding parity remain admission requirements.
    """
    require(isinstance(context, ManagedExportContext), 'TRUSTED_EXPORT_CONTEXT_REQUIRED')
    digest_fd(checkpoint_fd, CHECKPOINT_BYTES, CHECKPOINT_BYTES, CHECKPOINT_SHA256)
    context.verify_exact_input_and_environment(checkpoint_fd, CHECKPOINT_SHA256, CHECKPOINT_BYTES)
    graph_fd = None
    try:
        torch, onnx, campplus_factory = context.modules_and_architecture()
        model = campplus_factory(feat_dim=80, embedding_size=512)
        require(type(model).__name__ == 'CAMPPlus', 'EXACT_ARCHITECTURE_REQUIRED')
        model = restricted_load(torch, model, checkpoint_fd)
        torch.set_num_threads(4)
        torch.set_num_interop_threads(1)
        torch.use_deterministic_algorithms(True)
        graph_fd = memfd('F104-OFFLINE-GRAPH')
        with torch.inference_mode():
            sample = torch.zeros((1,200,80), dtype=torch.float32, device='cpu')
            with os.fdopen(os.dup(graph_fd), 'w+b') as output:
                torch.onnx.export(model, sample, output, input_names=['feats'],
                    output_names=['embs'], dynamic_axes={'feats': {1:'T'}},
                    export_params=True, opset_version=11, dynamo=False, external_data=False)
        validate_and_bind_graph(onnx, graph_fd)
        require(0 < os.fstat(graph_fd).st_size <= MAX_GRAPH_BYTES, 'GRAPH_SIZE_LIMIT')
        fcntl.fcntl(graph_fd, F_ADD_SEALS, SEALS)
        sha256 = digest_fd(graph_fd, MAX_GRAPH_BYTES)
        return {'graph_fd':graph_fd, 'sha256':sha256, 'bytes':os.fstat(graph_fd).st_size,
                'qualified':False, 'waveform_parity':'PENDING', 'reproducibility':'PENDING'}
    except BaseException:
        if graph_fd is not None:
            os.close(graph_fd)
        context.prove_teardown_on_error()
        raise

if __name__ == '__main__':
    raise Refusal('NO_ADMITTED_EXPORT_CONTEXT; SOURCE PREPARATION ONLY')
