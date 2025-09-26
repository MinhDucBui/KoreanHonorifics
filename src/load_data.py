import pandas as pd


def create_source_sentence(df):

    source_implicit = "{sentence}"
    source_explicit = "I am addressing {addresse} and saying: {sentence}"

    df["source_sentence"] = df.apply(
        lambda row: (
            source_explicit.format(sentence=row["sentence"], addresse=row["addresse"])
            if row["type"] == "Ex"
            else source_implicit.format(sentence=row["sentence"])
        ),
        axis=1
    )

    return df


def split_response(df, file_path):

    if "Hunyuan-MT-7B" in file_path:
        splitting_words = [r"\n\n.*'"]
    else:
        splitting_words = ["assistant\\n\\n", "assistant\\n", "<\\|CHATBOT_TOKEN\\|>", " <ko> "]

    pattern = "|".join(splitting_words)
    df["extracted_response"] = df["response"].str.split(pattern).str[-1].str.strip()

    return df


def get_template(model_name, src_lang="English", tgt_lang="Korean"):
    if "Hunyuan-MT-7B" in model_name:
        template = f"""Translate the following {src_lang} segment into {tgt_lang}, without additional explanation\n\n'{{source_sentence}}'"""
    elif "Seed-X-PPO-7B" in model_name:
        if tgt_lang == "Korean":
            tag = "<ko>"
        elif tgt_lang == "English":
            tag = "<en>"
        template = f"""Translate the following {src_lang} sentence into {tgt_lang}:\n{{source_sentence}} {tag}"""
    elif "nllb" in model_name or "google" in model_name:
        template = """{source_sentence}"""
    else:
        template = f"""Translate the following {src_lang} segment into {tgt_lang}: '{{source_sentence}}'\n\nProvide only the {tgt_lang} translation, without any additional text or explanation."""

    return template

def load_data(model_name, mode="", file_path=""):

    if mode == "":
        return load_trans_data(model_name)
    elif mode == "backtrans":
        return load_backtrans_data(model_name, file_path, mode)


def process_results_file():

    df = pd.read_excel("data/results.xlsx", header=None)

    df.columns = ["ID", "addresse", "sentence", "Ex_Ann1", "Ex_Ann2", "Ex_Ann3", "Ex_Ann4", "Ex_Ann5", "Ex_Ann6", "Ex_Ann7", "Ex_Ann8", "Im_Ann1", "Im_Ann2", "Im_Ann3", "Im_Ann4", "Im_Ann5", "Im_Ann6"]
    df["addresse"] = df["addresse"].str.replace("Addressee: ", "").str.lower()
    df["sentence"] = df["sentence"].str.replace('"', "")
    df["sentence"] = df["sentence"].str.replace('“', "")
    df["sentence"] = df["sentence"].str.replace('”', "")

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

    df_final = create_source_sentence(df_final)
    return df_final

def load_trans_data(model_name):
    template = get_template(model_name)

    df_final = process_results_file()

    df_final["raw_prompts"] = df_final.apply(
        lambda row: (
            template.format(source_sentence=row["source_sentence"])
        ),
        axis=1
    )
    return df_final

def load_backtrans_data(model_name, file_path, mode):
    template = get_template(model_name, src_lang="Korean", tgt_lang="English")
    df = pd.read_csv(file_path)

    df = split_response(df, file_path)

    df[f"raw_prompts{('_' + mode) if mode else ''}"] = df.apply(
        lambda row: (
            template.format(source_sentence=row["extracted_response"])
        ),
        axis=1
    )

    return df
