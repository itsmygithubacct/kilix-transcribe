#!/usr/bin/env python3
"""Stage a relocatable optional conversion tool from exact local software.

No model is copied. The installed CPU runtime does not include this exporter.
"""
import argparse
import hashlib
import json
from pathlib import Path
import shutil
import subprocess
import sys
import tarfile
import io

ROOT=Path(__file__).resolve().parents[1]
def sha(path):
    with path.open('rb') as f:return hashlib.file_digest(f,'sha256').hexdigest()
def population(root):
    digest=hashlib.sha256()
    for path in sorted(root.rglob('*')):
        if '__pycache__' in path.parts or path.suffix=='.pyc' or not path.is_file():continue
        digest.update(str(path.relative_to(root)).encode()+b'\0'+bytes.fromhex(sha(path)))
    return digest.hexdigest()
def main():
    p=argparse.ArgumentParser(description=__doc__)
    for name in ('native','python','environment','content','destination'):
        p.add_argument('--'+name,type=Path,required=True)
    p.add_argument('--eigen-source',type=Path,required=True,
                   help='exact native-build Eigen source archive conveyed with its MPL notice')
    p.add_argument('--revision',required=True)
    args=p.parse_args()
    inputs=json.loads((ROOT/'native/diarization/export-inputs.json').read_text())
    python_root=args.python.resolve().parent.parent
    site=args.environment/'lib/python3.12/site-packages'
    if (population(python_root)!=inputs['python_population_sha256']
            or population(site)!=inputs['export_packages_population_sha256']):
        p.error('software differs from the exact admitted Python/export population')
    if subprocess.check_output(['git','-C',str(ROOT),'rev-parse','HEAD']).decode().strip()!=args.revision:
        p.error('revision must select this committed product source')
    if subprocess.check_output(['git','-C',str(ROOT),'status','--porcelain']):
        p.error('product source must be clean')
    eigen=json.loads((ROOT/'native/diarization/inputs.json').read_text())['dependencies']['eigen.tar.gz']
    if args.eigen_source.stat().st_size!=eigen['bytes'] or sha(args.eigen_source)!=eigen['sha256']:
        p.error('Eigen source differs from the admitted native-build archive')
    out=args.destination.absolute();out.mkdir(mode=0o700)
    (out/'bin').mkdir();(out/'lib').mkdir()
    # Archive committed packages, not mutable Python import paths.
    def package(repo,revision,prefix,target):
        raw=subprocess.check_output(['git','-C',str(repo),'archive',revision,prefix])
        with tarfile.open(fileobj=io.BytesIO(raw)) as t:
            for member in t:
                if not member.isfile():continue
                name=Path(member.name).relative_to(prefix)
                if '..' in name.parts:raise ValueError('unsafe package member')
                dest=target/name;dest.parent.mkdir(parents=True,exist_ok=True)
                dest.write_bytes(t.extractfile(member).read())
    package(ROOT,args.revision,'src/kilix_transcribe',out/'lib/kilix_transcribe')
    content_ref=subprocess.check_output(['git','-C',str(args.content),'rev-parse','HEAD']).decode().strip()
    package(args.content,content_ref,'src/kilix_content',out/'lib/kilix_content')
    package(args.content,content_ref,'third_party/kilix-license/src/kilix_license',out/'lib/kilix_license')
    shutil.copytree(python_root,out/'python',ignore=shutil.ignore_patterns('__pycache__','*.pyc'),symlinks=False)
    shutil.copytree(site,out/'export-packages',ignore=shutil.ignore_patterns('__pycache__','*.pyc'),symlinks=False)
    shutil.copytree(args.native,out/'native')
    (out/'source').mkdir()
    shutil.copyfile(args.eigen_source,out/'source/eigen.tar.gz')
    (out/'source/README.txt').write_text(
        'The exact Eigen source used by the native runner is provided in eigen.tar.gz.\n'
        'Its original notices and Mozilla Public License 2.0 are in native/notices/eigen/.\n'
        'The archive is unmodified; its SHA-256 is '+eigen['sha256']+'.\n')
    source=ROOT/'native/diarization'
    arch=out/'architecture/speakerlab/models/campplus';arch.mkdir(parents=True)
    for name in ('DTDNN.py','layers.py'):shutil.copyfile(source/name,arch/name)
    shutil.copyfile(source/'export-inputs.json',out/'export-inputs.json')
    shutil.copyfile(source/'export-wheels.json',out/'export-wheels.json')
    shutil.copyfile(ROOT/'tools/stage_diarization_runtime.py',out/'bin/stage_runtime.py')
    shutil.copyfile(ROOT/'tools/diarization_export.py',out/'bin/export.py')
    # Convey every actual wheel/component notice, including bundled libraries.
    notices=out/'notices/export';notices.mkdir(parents=True)
    for path in site.rglob('*'):
        if path.is_file() and any(x in path.name.lower() for x in ('license','notice','copying')):
            target=notices/path.relative_to(site);target.parent.mkdir(parents=True,exist_ok=True)
            shutil.copyfile(path,target)
    value={'schema':'kilix.transcribe.diarization-tool/v1',
        'revision':'032b8131a7ad812f87061955ca974c99060c5a03','provider_ref':args.revision,
        'content_ref':content_ref,'files':{str(x.relative_to(out)):sha(x) for x in sorted(out.rglob('*')) if x.is_file()}}
    data=(json.dumps(value,sort_keys=True,indent=2)+'\n').encode()
    (out/'tool.json').write_bytes(data)
    launcher=(ROOT/'tools/diarization_convert.py').read_text().replace('@TOOL_SHA256@',hashlib.sha256(data).hexdigest())
    (out/'bin/kilix-diarization-convert').write_text(launcher)
    (out/'bin/kilix-diarization-convert').chmod(0o755)
    # This is public software for an image carrier, with no weights, receipts
    # or user state. Installed root ownership must still allow desktop use.
    for path in out.rglob('*'):
        if path.is_dir():path.chmod(0o755)
        elif path.is_file():path.chmod(0o755 if path.stat().st_mode&0o111 else 0o644)
    out.chmod(0o755)
    print('Staged software only; models require the packaged first-use agreement.')
if __name__=='__main__':main()
