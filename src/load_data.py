import pandas as pd


def create_source_sentence(df):

    source_implicit = 'I said: {sentence}'
    source_explicit = 'I was talking to {addresse}, and I said: {sentence}'

    df["source_sentence"] = df.apply(
        lambda row: (
            source_explicit.format(sentence=row["sentence"], addresse=row["addresse"])
            if row["type"] == "Ex"
            else source_implicit.format(sentence=row["sentence"])
        ),
        axis=1
    )

    return df


def reformat_addresse_sentence(df):

    # Sentences
    df["sentence"] = df["sentence"].str.replace('"', "")
    df["sentence"] = df["sentence"].str.replace('“', "")
    df["sentence"] = df["sentence"].str.replace('”', "")

    # Addresse
    mapping = {
        "Addressee: One's Professor, at university": "my professor at university",
        "Addressee: A Stranger, on the street": "a stranger on the street",
        "Addressee: A Clerk, in a store": "a clerk in a store",
        "Addressee: One's Boss, at work": "my boss at work",
        "Addressee: One's In Laws, the first meeting": "my in-laws at our first meeting",
        "Addressee: A Police Officer, on the street": "a police officer on the street",
        "Addressee: A Government Official, at a government institution": "a government official at a government institution",
        "Addressee: A group of students, giving a presentation in front of class": "a group of students while giving a presentation in front of class",
        "Addressee: A Job Interviewer, during a job interview": "a job interviewer during a job interview",
        "Addressee: A Customer, at one's company": "a customer at my company",
        "Addressee: A Waiter, in a restaurant": "a waiter in a restaurant",
        "Addressee: One's Teacher, at school": "my teacher at school",
        "Addressee: A Classmate, in school – unknown age": "a classmate at school",
        "Addressee: A Taxi Driver, in a cab": "a taxi driver in a cab",
        "Addressee: A nurse, in a clinic": "a nurse in a clinic",
        "Addressee: One's Mother, at home": "my mother at home",
        "Addressee: One's In Laws, already acquainted": "my in-laws I am already acquainted with",
        "Addressee: A Church Member, at church": "a church member at church",
        "Addressee: A Co-Worker, at work – lower rank": "a lower-ranking co-worker at work",
        "Addressee: An Online forum, a blog post": "an online forum in a blog post",
        "Addressee: One's Younger Sibling, at home": "my younger sibling at home",
        "Addressee: A Younger Cousin, holidays with the family": "my younger cousin during the holidays with the family",
        "Addressee: One's Best Friend, outside": "my best friend outside",
        "Addressee: One's Roommate, at home": "my roommate at home",
        "Addressee: A Classmate, outside – well acquainted": "a well-acquainted classmate outside",
        "Addressee: One's Romantic Partner, at home": "my romantic partner at home",
        "Addressee: A Strange Child, younger child - outside": "a strange younger child outside",
        "Addressee: One's Pet, at home": "my pet at home",
        "Addressee: Chatting with Chat-GPT, at home": "ChatGPT at home",
        "Addressee: Talking with oneself, on the street": "myself on the street",
    }

    df["addresse"] = df["addresse"].map(mapping)

    return df




def split_response(df, file_path):

    #if "Hunyuan-MT-7B" in file_path:
    #    splitting_words = [r"\n\n.*'"]
    #else:
    splitting_words = ["assistant\\n\\n", "assistant\\n", "<\\|CHATBOT_TOKEN\\|>", " <ko> ", "\nKorean:", "model\n", "<answer>\n"]

    pattern = "|".join(splitting_words)
    df["extracted_response"] = df["response"].str.split(pattern).str[-1].str.strip()
    df["extracted_response"] = df["extracted_response"].str.replace("</answer>", "").str.strip()
    return df


def get_template(model_name, src_lang="English", tgt_lang="Korean"):
    if "Hunyuan-MT-7B" in model_name:
        template = f"""Translate the following {src_lang} segment into {tgt_lang}, without additional explanation\n\n{{source_sentence}}"""
    elif "Seed-X-PPO-7B" in model_name:
        if tgt_lang == "Korean":
            tag = "<ko>"
        elif tgt_lang == "English":
            tag = "<en>"
        template = f"""Translate the following {src_lang} sentence into {tgt_lang}:\n{{source_sentence}} {tag}"""
    elif "nllb" in model_name or "google" in model_name or "opus" in model_name:
        template = """{source_sentence}"""
    elif "Tower" in model_name:
        template = f"""Translate the following {src_lang} source text to {tgt_lang}:\n{src_lang}: {{source_sentence}}\n{tgt_lang}: """
    elif "gemmax2" in model_name.lower():
        template = f"Translate this from {src_lang} to {tgt_lang}:\n{src_lang}: {{source_sentence}}\n{tgt_lang}:"
    elif "LLaMAX3-8B-Alpaca" in model_name:
        instruction = f'Translate the following sentences from {src_lang} to {tgt_lang}.'
        template = (
            'Below is an instruction that describes a task, paired with an input that provides further context. '
            'Write a response that appropriately completes the request.\n'
            f'### Instruction:\n{instruction}\n'
            f'### Input:\n{{source_sentence}}\n### Response:'
        )
    else:
        template = f"""Translate the following {src_lang} source segment into {tgt_lang}. Return only the translation, without any additional explanations or commentary.\n{src_lang}: '{{source_sentence}}'\n{tgt_lang}:"""

        if "Hunyuan-7B-Instruct" in model_name:
            template = "Be very short with your thinking. " + template

    return template

def load_data(model_name, mode="", file_path=""):

    if mode == "":
        return load_trans_data(model_name)
    elif mode == "backtrans":
        return load_backtrans_data(model_name, file_path, mode)


def process_results_file():

    df = pd.read_csv("data/processed.csv", header=None)

    df.columns = ["ID", "addresse", "sentence", "Ex_Ann1", "Ex_Ann2", "Ex_Ann3", "Ex_Ann4", "Ex_Ann5", "Ex_Ann6", "Ex_Ann7", "Ex_Ann8", "Im_Ann1", "Im_Ann2", "Im_Ann3", "Im_Ann4", "Im_Ann5"]

    df = reformat_addresse_sentence(df)

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

    # Take smaller set
    df_final = df_final.groupby(["addresse", "type"], group_keys=False).head(25)
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
