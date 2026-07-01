# Miles-Diffusion Docker

> ⚠️ This image is still **experimental**. A stable version is on the way — stay tuned.

## Build locally (no push)

```bash
just build-local
```

Builds `radixark/miles-diffusion:<version>-local` locally without pushing.
(`<version>` is read from `docker/version.txt`.)

## Experimental CUDA 13 (cu130)

> ⚠️ **Preliminary.** The CU13 image builds and boots the full RL runtime
> (SGLang diffusion engine, model load, FSDP + LoRA weight sync, NCCL). It has
> only been **partially validated at runtime** — a colocate Qwen-Image OCR GRPO
> run reaches training steps end-to-end — so treat it as experimental until a
> longer soak.

Same `docker/Dockerfile`; the only change is the sglang base image, swapped to a
cu130 build via `--build-arg SGLANG_IMAGE_TAG` (default stays `v0.5.12-cu129`, so
`release-primary` is unaffected). Base: `lmsysorg/sglang:v0.5.14-cu130`. Version
string lives in `docker/version-cu13.txt`.

```bash
just build-local-cu13   # local build, no push  -> radixark/miles-diffusion:<cu13-version>-local
just debug-cu13         # build + push to the -test namespace (experimental, shareable)
```

**Running the cu130 image** (2-GPU Qwen-Image OCR GRPO) needs, in addition to the
cu129 requirements (`--cap-add=SYS_PTRACE --security-opt seccomp=unconfined` for
colocate CUDA-IPC weight sync), a larger `/dev/shm` — the cu130 base ships NCCL
2.28.9, whose shared-memory transport exhausts docker's default 64 MB:

```bash
docker run --rm --shm-size=32g --gpus '"device=<A>,<B>"' \
  --cap-add=SYS_PTRACE --security-opt seccomp=unconfined \
  -e CUDA_VISIBLE_DEVICES=0,1 \
  radixark/miles-diffusion:<cu13-version>-local \
  bash -lc 'cd /root/miles_diffusion && bash scripts/run-diffusion-grpo-ocr-2gpu-flowgrpo-aligned.sh'
```

## Release rule

_TBD._
