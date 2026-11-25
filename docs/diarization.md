# CPU speaker turns (diarization)

The CPU speaker-turn route uses pinned pyannote 3.0 segmentation and a locally
exported English CAM++ graph. Select the receipt-covered `kilix-diarization-cpu`
asset with `serve --diarization-runtime PATH --diarization-asset
kilix-diarization-cpu`, alongside the installed Tiny selection. Each request
must use `file INPUT --task diarize --diarization-session --format json`.
Speaker turns form a separate annotation bound to the unchanged recognized
words/timestamps. Anonymous labels are new for every request; no embedding or
speaker identity is retained. Unidentifiable activity remains explicit.
Streaming diarization is refused; transcript streaming remains available for
ordinary transcription/translation. These checks do not claim a qualified
speaker-accuracy or sustained-load profile.

`tools/build_diarization_native.py` builds the CPU runner from exact local
software archives without fetching. `tools/stage_diarization_tool.py` creates
the optional receipt-gated conversion tool from committed product source and
the pinned Python/export environment, and conveys the exact native-build Eigen
source archive with its MPL notice (`--eigen-source`). Conversion reads only the approved
English checkpoint with `weights_only=True` inside a network-denied, bounded
CPU scope and requires the reproducible selected ONNX digest. Installed
runtime inference loads ONNX only. `tools/stage_diarization_runtime.py` also
supports explicit local development staging; it never creates a model receipt.
The software bundle includes the matching Python dependency notices and source
information. Its Torch and NumPy libraries use Debian's `libgomp1`, `libgfortran5`
and `libquadmath0` packages; the bundle omits those GCC runtime libraries.
