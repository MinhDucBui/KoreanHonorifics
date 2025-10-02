from comet import download_model, load_from_checkpoint

model = load_from_checkpoint("/p/project/westai0073/Models/XCOMET-XXL/checkpoints/model.ckpt")
data = [
    {
        "src": "Boris Johnson teeters on edge of favour with Tory MPs", 
        "mt": "Boris Johnson ist bei Tory-Abgeordneten völlig in der Gunst", 
        "ref": "Boris Johnsons Beliebtheit bei Tory-MPs steht auf der Kippe"
    }
]
model_output = model.predict(data, batch_size=8, gpus=1)
# Segment-level scores
print (model_output.scores)

# System-level score
print (model_output.system_score)

# Score explanation (error spans)
print (model_output.metadata.error_spans)
