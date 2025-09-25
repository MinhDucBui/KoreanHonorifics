import pandas as pd



def load_data(model_name):
    if "Hunyuan-MT-7B" in model_name:
        template_explicit = """Translate the following segment into Korean, without additional explanation\n\n'I am addressing {addresse} and saying: {sentence}'"""
        template_implicit = """Translate the following segment into Korean, without additional explanation\n\n'{sentence}'""" 
    elif "Seed-X-PPO-7B" in model_name:
        template_explicit = """Translate the following English sentence into Korean:\nI am addressing {addresse} and saying: {sentence} <ko>"""
        template_implicit = """Translate the following English sentence into Korean:\n{sentence} <ko>""" 
    elif "nllb" in model_name:
        template_explicit = """I am addressing {addresse} and saying: {sentence}"""
        template_implicit = """{sentence}""" 
    else:
        template_explicit = """Translate the following English sentence into Korean: 'I am addressing {addresse} and saying: {sentence}'\n\nProvide only the Korean translation, without any additional text or explanation."""
        template_implicit = """Translate the following English sentence into Korean: {sentence}\n\nProvide only the Korean translation, without any additional text or explanation."""

    df = pd.read_excel("data/results.xlsx", header=None)

    df.columns = ["ID", "addresse", "sentence", "Ex_Ann1", "Ex_Ann2", "Ex_Ann3", "Ex_Ann4", "Ex_Ann5", "Ex_Ann6", "Ex_Ann7", "Ex_Ann8", "Im_Ann1", "Im_Ann2", "Im_Ann3", "Im_Ann4", "Im_Ann5", "Im_Ann6"]
    df["addresse"] = df["addresse"].str.replace("Addressee: ", "").str.lower()
    df["sentence"] = df["sentence"].str.replace('"', "")

    # Split columns into Ex and Im groups
    ex_cols = [c for c in df.columns if c.startswith("Ex_")]
    im_cols = [c for c in df.columns if c.startswith("Im_")]

    # Build two frames, one for Ex and one for Im
    df_ex = df[["ID", "addresse", "sentence"] + ex_cols].copy()
    df_ex["type"] = "Ex"
    df_ex = df_ex.rename(columns={c: c.replace("Ex_", "") for c in ex_cols})

    df_im = df[["ID", "addresse", "sentence"] + im_cols].copy()
    df_im["type"] = "Im"
    df_im = df_im.rename(columns={c: c.replace("Im_", "") for c in im_cols})

    # Align column order
    final_cols = ["ID", "addresse", "sentence", "type"] + list(set([c.replace("Ex_", "").replace("Im_", "") for c in ex_cols+im_cols]))
    df_final = pd.concat([df_ex, df_im], ignore_index=True)[final_cols]


    df_final["raw_prompts"] = df_final.apply(
        lambda row: (
            template_explicit.format(sentence=row["sentence"], addresse=row["addresse"])
            if row["type"] == "Ex"
            else template_implicit.format(sentence=row["sentence"])
        ),
        axis=1
    )

    return df_final