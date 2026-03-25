import sys
import gradio as gr
import tempfile
import torchaudio
import soundfile as sf

from f5_tts.infer.utils_infer import (
    infer_process, load_model, load_vocoder,
    preprocess_ref_audio_text, remove_silence_for_generated_wav,
)
from f5_tts.model import DiT
from f5_tts.cleantext.th_normalize import normalize_text

# ─── Config ───────────────────────────────────────────────────────────────────
DEFAULT_REF_AUDIO = "/content/F5-TTS-THAI-E3/src/f5_tts/infer/default_ref_audio/audio_1057.wav"
DEFAULT_REF_TEXT  = "ถ้าเฮาบ่ได้ลงทุนนำกันเฮาเฮาบ่มีความสัมพันธ์กับคนรอบข้าง"

# ─── Load model ───────────────────────────────────────────────────────────────
vocoder = load_vocoder()

def load_f5tts():
    cfg = dict(dim=1024, depth=22, heads=16, ff_mult=2, text_dim=512,
               text_mask_padding=False, conv_layers=4, pe_attn_head=1)
    ckpt  = "/content/F5-TTS-THAI-E3/pretrained_model/model_400.pt"
    vocab = "/content/F5-TTS-THAI-E3/vocab/v1/vocab.txt"
    return load_model(DiT, cfg, ckpt, vocab_file=vocab, use_ema=True)

model = load_f5tts()

# ─── TTS function ─────────────────────────────────────────────────────────────
def tts(gen_text, speed, nfe_step, cfg_strength):
    if not gen_text.strip():
        return None

    ref_audio, ref_text = preprocess_ref_audio_text(DEFAULT_REF_AUDIO, DEFAULT_REF_TEXT)
    gen_text_cleaned = normalize_text(gen_text)

    final_wave, final_sample_rate, _ = infer_process(
        ref_audio, ref_text, gen_text_cleaned,
        model, vocoder,
        cross_fade_duration=0.15,
        nfe_step=int(nfe_step),
        speed=speed,
        cfg_strength=cfg_strength,
    )

    with tempfile.NamedTemporaryFile(delete=False, suffix=".wav") as f:
        sf.write(f.name, final_wave, final_sample_rate)
        remove_silence_for_generated_wav(f.name)
        final_wave, _ = torchaudio.load(f.name)
    final_wave = final_wave.squeeze().cpu().numpy()

    return (final_sample_rate, final_wave)

# ─── UI ───────────────────────────────────────────────────────────────────────
with gr.Blocks(title="F5-TTS ภาษาอีสาน") as app:
    gr.Markdown("## F5-TTS ภาษาอีสาน 🎙️")

    text_input = gr.Textbox(label="ข้อความที่จะสร้าง", lines=4,
                            placeholder="พิมพ์ข้อความภาษาอีสานที่นี่...")

    with gr.Row():
        speed_slider = gr.Slider(label="ความเร็ว", minimum=0.5, maximum=1.5,
                                 value=0.8, step=0.05)
        nfe_slider   = gr.Slider(label="NFE Steps (คุณภาพ)", minimum=8, maximum=64,
                                 value=32, step=4)
        cfg_slider   = gr.Slider(label="CFG Strength (ความชัด)", minimum=1.0, maximum=4.0,
                                 value=2.0, step=0.1)

    btn       = gr.Button("🚀 สร้างเสียง", variant="primary")
    audio_out = gr.Audio(label="เสียงที่สร้าง")

    btn.click(tts,
              inputs=[text_input, speed_slider, nfe_slider, cfg_slider],
              outputs=audio_out)

app.launch(share=True)