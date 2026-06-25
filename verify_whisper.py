from faster_whisper import WhisperModel

model = WhisperModel(
    "small",
    device="cpu",
    compute_type="int8"
)

segments, info = model.transcribe(
    "movie2.mp4",
    language="hi",
    beam_size=1,
    vad_filter=False
)

segments = list(segments)

print("Duration:", info.duration)
print("Segments:", len(segments))

for s in segments:
    print(s.start, s.end, s.text)
# =================================





