"""One restricted, network-denied export in the bound managed environment."""
import fcntl
import hashlib
import json
import os
from pathlib import Path
import sys

ROOT=Path(__file__).resolve().parents[1]
sys.path[:0]=[str(ROOT/'lib'),str(ROOT/'export-packages'),str(ROOT/'architecture')]
from kilix_transcribe.diarization_core.converter import ManagedExportContext,convert_exact_english
from kilix_transcribe.diarization_core.fd import digest_fd

class Context(ManagedExportContext):
    def verify_exact_input_and_environment(self,fd,sha,size):
        if sys.version_info[:3]!=(3,12,8):raise ValueError('selected Python unavailable')
        digest_fd(fd,size,size,sha)
        interfaces=[x.split(':',1)[0].strip() for x in Path('/proc/net/dev').read_text().splitlines()[2:]]
        if interfaces!=['lo']:raise ValueError('export network namespace unavailable')
        cg=Path('/sys/fs/cgroup')/Path('/proc/self/cgroup').read_text().strip().split('::',1)[1].lstrip('/')
        quota,period=(cg/'cpu.max').read_text().split()
        if (int((cg/'memory.max').read_text())>2147483648 or int((cg/'memory.swap.max').read_text())!=0
                or quota=='max' or int(quota)>4*int(period) or Path('/dev/nvidia0').exists()):
            raise ValueError('export resource policy unavailable')
        for name,expected in json.loads((ROOT/'tool.json').read_text())['files'].items():
            with (ROOT/name).open('rb') as f:
                if hashlib.file_digest(f,'sha256').hexdigest()!=expected:raise ValueError('export environment changed')
    def modules_and_architecture(self):
        import torch,onnx
        from speakerlab.models.campplus.DTDNN import CAMPPlus
        if torch.__version__!='2.7.0+cpu' or onnx.__version__!='1.18.0':raise ValueError('export version mismatch')
        return torch,onnx,CAMPPlus
    def prove_teardown_on_error(self):
        if Path(f'/proc/self/task/{os.getpid()}/children').read_text().strip():raise ValueError('unexpected export child')

result=convert_exact_english(Context(),int(sys.argv[1]))
try:
    if result['sha256']!='7cfea971b2231ed7d7dd6324de1de33f8f9dbe65bbb30cf64b4b0b203b566292' or result['bytes']!=29539501:
        raise ValueError('export does not reproduce the selected graph')
    target=int(sys.argv[2]);offset=0
    while block:=os.pread(result['graph_fd'],65536,offset):
        view=memoryview(block)
        while view:view=view[os.write(target,view):]
        offset+=len(block)
finally:os.close(result['graph_fd'])
