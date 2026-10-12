# AuroraIMG-6M, ONNX export

This branch builds [AuroraIMG-6M](https://huggingface.co/AuroraAI-Research/AuroraIMG-6M) by
AuroraAI-Research (Apache-2.0) as ONNX for AIDungeonMaster's on-device image test, and publishes
the files as the release `aurora-img-6m-onnx-v1`. It shares no history with the rest of this
repository.

The workflow downloads the model's weights at a pinned commit, exports them with `export.py` (the
model's own classes, in `aurora_model.py`), checks the ONNX graphs against PyTorch over the whole
50-step pipeline, requires the files to match `SHA256SUMS` (the checksums the app verifies), and
then publishes them. Weights are unmodified; opset 17, fp32.

| File | Inputs → outputs |
|---|---|
| `text_encoder.onnx` | `tokens`, `mask` int32 [b,64] → `context` [b,64,128] |
| `unet.onnx` | `latents` [b,8,16,16], `t` [b] (0..1000), `context`, `mask` → `velocity` [b,8,16,16] |
| `vae_decoder.onnx` | normalized `latents` [b,8,16,16] → `image` [b,3,64,64] in 0..1 (latent stats folded in) |
| `tokenizer.json` | the model's tokenizer, unchanged |
