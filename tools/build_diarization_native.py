#!/usr/bin/env python3
"""Build the pinned CPU diarization runner from exact local software archives."""
from __future__ import annotations
import argparse
import hashlib
import json
from pathlib import Path
import shutil
import stat
import subprocess
import tarfile
import zipfile

ROOT = Path(__file__).resolve().parents[1]
INPUTS = ROOT / 'native/diarization'

def digest(path):
    h = hashlib.sha256()
    with path.open('rb') as f:
        for block in iter(lambda: f.read(1048576), b''):
            h.update(block)
    return h.hexdigest()

def unpack(archive, destination, *, source=False):
    destination.mkdir()
    if archive.suffix == '.zip':
        with zipfile.ZipFile(archive) as z:
            if sum(x.file_size for x in z.infolist()) >= 512*1024**2:
                raise ValueError('dependency archive exceeds extraction bound')
            links = []
            for info in z.infolist():
                name = Path(info.filename)
                if name.is_absolute() or '..' in name.parts:
                    raise ValueError('unsafe archive member')
                target = destination / name
                if stat.S_ISLNK(info.external_attr >> 16):
                    links.append((target, z.read(info).decode()))
                elif info.is_dir():
                    target.mkdir(parents=True, exist_ok=True)
                else:
                    target.parent.mkdir(parents=True, exist_ok=True)
                    with z.open(info) as source, target.open('xb') as output:
                        shutil.copyfileobj(source, output)
            for target, link in links:
                if Path(link).is_absolute() or not (target.parent / link).resolve().is_relative_to(destination.resolve()):
                    raise ValueError('unsafe archive link')
                target.symlink_to(link)
    else:
        with tarfile.open(archive) as t:
            members = t.getmembers()
            if source:
                # The sealed full upstream snapshot includes example links to
                # absolute paths. They are unrelated to this CPU build. Extract
                # only the build carrier, native sources, CMake modules and licence.
                roots = {'build-native', 'sherpa-onnx', 'cmake', 'LICENSE', 'CMakeLists.txt'}
                members = [x for x in members if Path(x.name).parts[0] in roots]
            if sum(x.size for x in members) >= 512*1024**2:
                raise ValueError('archive exceeds extraction bound')
            t.extractall(destination, members=members, filter='data')
    children = list(destination.iterdir())
    return children[0] if len(children) == 1 and children[0].is_dir() else destination

def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--source-archive', type=Path, required=True)
    p.add_argument('--dependencies', type=Path, required=True)
    p.add_argument('--work', type=Path, required=True)
    p.add_argument('--output', type=Path, required=True)
    p.add_argument('--jobs', type=int, choices=(1, 2), default=2)
    args = p.parse_args()
    inputs = json.loads((INPUTS / 'inputs.json').read_text())
    if digest(args.source_archive) != inputs['source_archive_sha256']:
        raise ValueError('source archive digest mismatch')
    for name, record in inputs['dependencies'].items():
        path = args.dependencies / name
        if path.stat().st_size != record['bytes'] or digest(path) != record['sha256']:
            raise ValueError('dependency archive identity mismatch: ' + name)
    work = args.work.resolve(); work.mkdir(mode=0o700)
    source = unpack(args.source_archive, work / 'source', source=True)
    deps = work / 'dependencies'; deps.mkdir()
    roots = {}
    for name in inputs['dependencies']:
        key = name.split('.')[0]
        roots[key] = unpack(args.dependencies / name, deps / key)
    for path in (INPUTS / 'overlay').rglob('*'):
        if path.is_file():
            destination = source / path.relative_to(INPUTS / 'overlay')
            destination.parent.mkdir(parents=True, exist_ok=True)
            shutil.copyfile(path, destination)
    carrier = source / 'build-native'
    shutil.copyfile(INPUTS / 'runner.cc', carrier / 'runner.cc')
    with (carrier / 'CMakeLists.txt').open('a') as output:
        output.write('\n' + (INPUTS / 'runner.cmake').read_text())
    build = work / 'build'
    configure = ['cmake', '-S', str(carrier), '-B', str(build), '-G', 'Ninja',
        '-DBUILD_SHARED_LIBS=ON', '-DBUILD_TESTING=OFF', '-DCMAKE_BUILD_TYPE=MinSizeRel',
        '-DCMAKE_C_FLAGS_MINSIZEREL=-O1 -DNDEBUG', '-DCMAKE_CXX_FLAGS_MINSIZEREL=-O1 -DNDEBUG',
        '-DCMAKE_CXX_STANDARD=17', '-DCMAKE_EXPORT_COMPILE_COMMANDS=ON',
        '-DCMAKE_BUILD_WITH_INSTALL_RPATH=ON', '-DCMAKE_INTERPROCEDURAL_OPTIMIZATION=OFF',
        '-DCMAKE_FIND_USE_PACKAGE_REGISTRY=OFF', '-DCMAKE_FIND_USE_SYSTEM_PACKAGE_REGISTRY=OFF',
        '-DFETCHCONTENT_FULLY_DISCONNECTED=ON', '-DFETCHCONTENT_UPDATES_DISCONNECTED=ON',
        '-DSHERPA_ONNX_USE_PRE_INSTALLED_ONNXRUNTIME_IF_AVAILABLE=OFF',
        '-DSHERPA_ONNX_ENABLE_SPEAKER_DIARIZATION=ON', '-DSHERPA_ONNX_ENABLE_C_API=ON',
        '-DSHERPA_ONNX_LINK_LIBSTDCPP_STATICALLY=OFF', '-DOPENFST_USE_ABSL=OFF',
        '-DJSON_BuildTests=OFF', '-DJSON_Install=OFF', '-DEIGEN_BUILD_DOC=OFF', '-DEIGEN_BUILD_BTL=OFF',
        '-DKISSFFT_OPENMP=OFF', '-DKISSFFT_TEST=OFF', '-DKISSFFT_TOOLS=OFF']
    for key in ('BINARY', 'CHECK', 'GPU', 'PYTHON', 'JNI', 'PORTAUDIO', 'WEBSOCKET', 'WASM',
                'TTS', 'TESTS', 'ASCEND_NPU', 'AXCL', 'AXERA', 'DIRECTML', 'QNN', 'RKNN',
                'SPACEMIT', 'SANITIZER'):
        configure.append('-DSHERPA_ONNX_ENABLE_' + key + '=OFF')
    configure += ['-DSHERPA_ONNX_BUILD_C_API_EXAMPLES=OFF', '-DSHERPA_ONNX_HAS_ALSA=OFF']
    mapping = {'eigen':'EIGEN', 'hclust-cpp':'HCLUST_CPP', 'nlohmann-json':'JSON',
        'kaldifst':'KALDIFST', 'kaldi-decoder':'KALDI_DECODER', 'kaldi-native-fbank':'KALDI_NATIVE_FBANK',
        'kissfft':'KISSFFT', 'onnxruntime':'ONNXRUNTIME', 'openfst':'OPENFST',
        'simple-sentencepiece':'SIMPLE-SENTENCEPIECE'}
    for name, cmake_name in mapping.items():
        configure.append('-DFETCHCONTENT_SOURCE_DIR_' + cmake_name + '=' + str(roots[name]))
    for name, flag in (('kaldi-decoder','DECODER'), ('kaldifst','KALDIFST'), ('kaldi-native-fbank','FBANK')):
        configure.append('-DF104_' + flag + '_MODULES=' + str(roots[name] / 'cmake'))
    subprocess.run(configure, check=True)
    subprocess.run(['cmake', '--build', str(build), '--target', 'kilix-diarize-runner',
                    '--parallel', str(args.jobs)], check=True)
    output = args.output.resolve(); output.mkdir(mode=0o700)
    shutil.copyfile(build / 'kilix-diarize-runner', output / 'diarize-runner')
    (output / 'diarize-runner').chmod(0o700)
    lib = output / 'lib'; lib.mkdir(mode=0o700)
    for path in (roots['onnxruntime'] / 'lib').glob('*.so*'):
        if path.is_file():
            shutil.copyfile(path, lib / path.name)
            (lib / path.name).chmod(0o600)
    notices = output / 'notices'; notices.mkdir(mode=0o700)
    shutil.copytree(INPUTS / 'notices', notices / 'source')
    for name, root in roots.items():
        for path in root.rglob('*'):
            if path.is_file() and any(x in path.name.lower() for x in ('license', 'notice', 'copying')):
                target = notices / name / path.relative_to(root)
                target.parent.mkdir(mode=0o700, parents=True, exist_ok=True)
                shutil.copyfile(path, target)
    files = {str(x.relative_to(output)): digest(x) for x in sorted(output.rglob('*')) if x.is_file()}
    report = {'schema':'kilix.transcribe.diarization-native/v1', 'inputs':inputs,
              'files':files, 'qualified':False}
    (output / 'native.json').write_text(json.dumps(report, indent=2) + '\n')
    print('Built the pinned CPU runner; model and installed qualification remain separate.')

if __name__ == '__main__':
    main()
