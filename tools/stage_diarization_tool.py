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
    p.add_argument('--python-full-archive',type=Path,required=True,
                   help='official matching standalone archive containing dependency notices')
    p.add_argument('--patchelf',type=Path,required=True,
                   help='pinned build tool for selecting the distro OpenMP runtime')
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
    if (args.python_full_archive.stat().st_size!=93784813
            or sha(args.python_full_archive)!='80a3625b14e1bd1caf9cbe7fb24a7adcce27ee2150630e666ba1653a925915a9'):
        p.error('Python dependency notices require the exact official standalone archive')
    if sha(args.patchelf)!='f2a22996ad184a5329cdc98a2bf7df5badda683894ab9a64c969ac951a05d772':
        p.error('patchelf differs from the selected 0.17.2.4 wheel build tool')
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
    # uv's relocatable installation rewrites build-prefix metadata to its
    # cache location. Preserve the configuration values while deriving that
    # prefix from this relocated interpreter; never convey the builder home.
    for path in (out/'python').rglob('_sysconfigdata*.py'):
        text=path.read_text()
        if str(python_root) in text:
            text=text.replace(str(python_root),'@KILIX_PYTHON_PREFIX@')
            text+=('\nimport os as _kilix_os\n'
                '_kilix_prefix = _kilix_os.path.dirname(_kilix_os.path.dirname(_kilix_os.path.dirname(__file__)))\n'
                'for _kilix_key, _kilix_value in list(build_time_vars.items()):\n'
                '    if isinstance(_kilix_value, str):\n'
                '        build_time_vars[_kilix_key] = _kilix_value.replace("@KILIX_PYTHON_PREFIX@", _kilix_prefix)\n')
            path.write_text(text)
    # Installed-wheel acquisition metadata names the builder's local cache.
    # Runtime/version/licence metadata remains; public provenance is conveyed
    # by the exact locked wheel URLs and digests in export-wheels.json.
    shutil.copytree(site,out/'export-packages',
        ignore=shutil.ignore_patterns('__pycache__','*.pyc','direct_url.json'),symlinks=False)
    # Do not redistribute an opaque GCC runtime without its corresponding
    # source. Select Debian's libgomp1 by SONAME, changing no executable code.
    gomp='libgomp-25c89faf.so.1'
    torch_lib=out/'export-packages/torch/lib'
    for name in ('libshm.so','libtorch_cpu.so','libtorch_global_deps.so'):
        path=torch_lib/name
        needed=subprocess.check_output([args.patchelf,'--print-needed',path]).decode().splitlines()
        if needed.count(gomp)!=1:p.error('unexpected Torch OpenMP dependency')
        subprocess.run([args.patchelf,'--replace-needed',gomp,'libgomp.so.1',path],check=True)
    (torch_lib/gomp).unlink()
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
    python_notices=out/'notices/python';python_notices.mkdir()
    child=subprocess.Popen(['zstd','-dc',args.python_full_archive],stdout=subprocess.PIPE)
    try:
        with tarfile.open(fileobj=child.stdout,mode='r|') as archive:
            for member in archive:
                if not member.isfile():continue
                if member.name=='python/PYTHON.json':
                    if member.size>4*1024**2:raise ValueError('Python metadata exceeds bound')
                    (python_notices/'UPSTREAM-PYTHON.json').write_bytes(archive.extractfile(member).read())
                elif member.name.startswith('python/licenses/'):
                    if member.size>4*1024**2:raise ValueError('Python notice exceeds bound')
                    (python_notices/Path(member.name).name).write_bytes(archive.extractfile(member).read())
    finally:
        child.stdout.close()
        if child.wait():raise ValueError('Python notice archive failed')
    (out/'notices/OPENMP-DEPENDENCY.txt').write_text(
        'Torch 2.7.0+cpu comes from the exact wheels in export-wheels.json.\n'
        'Only DT_NEEDED references in libshm.so, libtorch_cpu.so and libtorch_global_deps.so\n'
        'are changed from the bundled libgomp-25c89faf.so.1 to system libgomp.so.1.\n'
        'The bundled GCC runtime is omitted. Install Debian libgomp1; no compiler/runtime\n'
        'source from that package is redistributed by this software archive.\n')
    (out/'notices/SOURCE-AVAILABILITY.txt').write_text(
        'Converter source and changes: https://github.com/itsmygithubacct/kilix-transcribe/tree/'+args.revision+'\n'
        'CPython 3.12.8 source: https://www.python.org/ftp/python/3.12.8/Python-3.12.8.tar.xz\n'
        'SHA256: c909157bb25ec114e5869124cc2a9c4a4d4c1e957ca4ff553f1edc692101154e\n'
        'Standalone build scripts/patches and complete dependency source URLs/checksums:\n'
        'https://github.com/astral-sh/python-build-standalone/tree/20241206\n'
        'https://github.com/astral-sh/python-build-standalone/blob/20241206/pythonbuild/downloads.py\n'
        'Berkeley DB 6.0.19 source (Sleepycat; see LICENSE.bdb.txt):\n'
        'https://ftp.osuosl.org/pub/blfs/conglomeration/db/db-6.0.19.tar.gz\n'
        'SHA256: 2917c28f60903908c2ca4587ded1363b812c4e830a5326aaa77c9879d13ae18e\n'
        'Python dependency identities and original notices: notices/python/.\n'
        'Exact Eigen source is included in source/eigen.tar.gz.\n')
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
