#!/usr/bin/python3 -I
"""Receipt-gated local conversion for the packaged diarization asset."""
import argparse
import ctypes
import fcntl
import hashlib
import json
import os
from pathlib import Path
import signal
import stat
import subprocess
import sys
import tempfile
import time
import uuid

ROOT=Path(__file__).resolve().parents[1]
def sha(path):
    with path.open('rb') as f:return hashlib.file_digest(f,'sha256').hexdigest()
def sealed(path,digest,size):
    source=os.open(path,os.O_RDONLY|os.O_NOFOLLOW|os.O_CLOEXEC)
    libc=ctypes.CDLL(None,use_errno=True);libc.memfd_create.argtypes=(ctypes.c_char_p,ctypes.c_uint);libc.memfd_create.restype=ctypes.c_int
    fd=libc.memfd_create(b'kilix-approved-conversion-input',3)
    if fd<0:os.close(source);raise OSError(ctypes.get_errno(),'memfd')
    h=hashlib.sha256();total=0
    try:
        info=os.fstat(source)
        if not stat.S_ISREG(info.st_mode) or info.st_size!=size:raise ValueError('input size mismatch')
        while block:=os.read(source,65536):
            total+=len(block)
            if total>size:raise ValueError('input exceeds bound')
            h.update(block);view=memoryview(block)
            while view:view=view[os.write(fd,view):]
        if total!=size or h.hexdigest()!=digest:raise ValueError('input digest mismatch')
        fcntl.fcntl(fd,1033,15);return fd
    except BaseException:os.close(fd);raise
    finally:os.close(source)
def death_signal():
    libc=ctypes.CDLL(None,use_errno=True)
    if libc.prctl(1,signal.SIGKILL,0,0,0):raise OSError(ctypes.get_errno(),'parent death signal')
def main():
    p=argparse.ArgumentParser(description=__doc__)
    for name in ('input','output','receipt-store'):p.add_argument('--'+name,type=Path,required=True)
    p.add_argument('--manifest-digest',required=True);args=p.parse_args()
    payload=(ROOT/'tool.json').read_bytes()
    if hashlib.sha256(payload).hexdigest()!='@TOOL_SHA256@':raise ValueError('conversion tool binding changed')
    value=json.loads(payload)
    for directory in (*reversed(ROOT.parents),ROOT):
        info=directory.lstat()
        if (not stat.S_ISDIR(info.st_mode) or info.st_uid not in (0,os.geteuid())
                or info.st_mode&0o022 and not (info.st_uid==0 and info.st_mode&stat.S_ISVTX)):
            raise ValueError('unsafe conversion tool root')
    for name,expected in value['files'].items():
        path=ROOT/name;info=path.lstat()
        if not stat.S_ISREG(info.st_mode) or info.st_mode&0o022 or sha(path)!=expected:
            raise ValueError('conversion tool population changed')
    sys.path.insert(0,str(ROOT/'lib'))
    import kilix_content as content,kilix_license as license
    spec=content.verified_packaged_catalog().require_asset('kilix-diarization-cpu')
    if spec.manifest_digest!=args.manifest_digest or spec.version!=value['revision']:
        raise ValueError('conversion manifest differs from packaged selection')
    reference=license.AssetRef(id=spec.asset_id,record_digest=spec.licenses[0].record_digest,manifest_digest=spec.manifest_digest)
    license.require(reference,records=license.RecordIndex(license.load_determined_records()),store=license.ReceiptStore(args.receipt_store))
    if sorted(x.name for x in args.input.iterdir())!=['campplus_voxceleb.bin','pyannote-segmentation.onnx']:
        raise ValueError('conversion input population differs')
    checkpoint=sealed(args.input/'campplus_voxceleb.bin','5b1a88b6f8d85826fabef804779c3372b42f3af21457fa48bd5c097c0686b2de',29357703)
    segmentation=sealed(args.input/'pyannote-segmentation.onnx','220ad67ca923bef2fa91f2390c786097bf305bceb5e261d4af67b38e938e1079',5992913)
    with tempfile.TemporaryDirectory(prefix='.kilix-diarization-conversion-',dir=args.output.parent) as temporary:
        temp=Path(temporary);graph=temp/'embedding.onnx'
        graph_fd=os.open(graph,os.O_RDWR|os.O_CREAT|os.O_EXCL|os.O_CLOEXEC,0o600)
        try:
            unit='kilix-diarization-export-'+uuid.uuid4().hex
            # A scope retains the launcher's process ancestry/group. Kernel
            # parent-death plus bwrap's namespace owner remove export work if
            # the Content installer is canceled or killed; no detached service.
            cmd=['systemd-run','--user','--scope','--quiet','--unit='+unit,
                '-p','MemoryMax=2147483648','-p','MemorySwapMax=0','-p','CPUQuota=400%',
                '/usr/bin/bwrap','--unshare-net','--unshare-pid','--die-with-parent','--ro-bind','/','/',
                '--proc','/proc','--dev','/dev','--tmpfs','/tmp','--clearenv','--setenv','PATH','/usr/bin:/bin',
                '--setenv','HOME','/tmp','--setenv','OMP_NUM_THREADS','4',
                '--preserve-fds',str(max(checkpoint,graph_fd)-2),
                str(ROOT/'python/bin/python3.12'),'-I','-S','-B',str(ROOT/'bin/export.py'),str(checkpoint),str(graph_fd)]
            parent=os.getppid();child=subprocess.Popen(cmd,pass_fds=(checkpoint,graph_fd),preexec_fn=death_signal)
            expires=time.monotonic()+600
            try:
                while child.poll() is None:
                    if os.getppid()!=parent or time.monotonic()>=expires:raise ValueError('export owner or deadline lost')
                    time.sleep(.05)
                if child.returncode:raise ValueError('restricted export failed')
            finally:
                if child.poll() is None:
                    child.kill();child.wait()
            if sha(graph)!='7cfea971b2231ed7d7dd6324de1de33f8f9dbe65bbb30cf64b4b0b203b566292':raise ValueError('derived graph differs')
            seg=temp/'segmentation.onnx'
            with seg.open('xb') as f:f.write(os.pread(segmentation,5992914,0))
            # This path runs the same committed stager in the bound stdlib.
            dest=temp/'runtime';command=[str(ROOT/'python/bin/python3.12'),'-I','-S','-B','-c',
                'import sys,runpy;sys.path.insert(0,'+repr(str(ROOT/'lib'))+');runpy.run_path('+repr(str(ROOT/'bin/stage_runtime.py'))+',run_name="__main__")',
                '--native',str(ROOT/'native'),'--segmentation',str(seg),'--embedding',str(graph),
                '--embedding-sha256','7cfea971b2231ed7d7dd6324de1de33f8f9dbe65bbb30cf64b4b0b203b566292',
                '--revision',value['revision'],'--destination',str(dest)]
            subprocess.run(command,check=True,timeout=60)
            if any(args.output.iterdir()):raise ValueError('conversion output must be empty')
            for path in dest.iterdir():os.rename(path,args.output/path.name)
        finally:
            for fd in (checkpoint,segmentation,graph_fd):os.close(fd)
if __name__=='__main__':
    try:main()
    except (OSError,ValueError,RuntimeError,subprocess.SubprocessError) as error:
        print('kilix diarization conversion: '+str(error),file=sys.stderr);raise SystemExit(1)
