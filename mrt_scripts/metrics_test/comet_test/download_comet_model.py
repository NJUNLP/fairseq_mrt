import sys
sys.path.append("COMET")
from comet import download_model, load_from_checkpoint

model_path = download_model("wmt20-comet-da")   # /home/tiger/.cache/torch/unbabel_comet/wmt20-comet-da/checkpoints/model.ckpt
#model_path = '/home/tiger/.cache/torch/unbabel_comet/wmt20-comet-da/checkpoints/model.ckpt'
model = load_from_checkpoint(model_path)
data = [
    {
        "src": "Dem Feuer konnte Einhalt geboten werden",
        "mt": "The fire could be stopped",
        "ref": "They were able to control the fire."
    },
    {
        "src": "Schulen und Kindergärten wurden eröffnet.",
        "mt": "Schools and kindergartens were open",
        "ref": "Schools and kindergartens opened"
    },
]

seg_scores = model.predict_mello(data, batch_size=8, device=1)

print(seg_scores)

print("Download comet successfully!")


# python3 mrt_scripts/metrics_test/comet_test/comet_test.py