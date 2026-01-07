import streamlit as st
import json
import requests
from generate import GenerateEmail

# --- CONFIG + DATA LOAD-IN ---
st.set_page_config(page_title="AI Email Editor", page_icon="📧", layout="wide")

def load_jsonl(path):
    out = {}
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            data = json.loads(line)
            out[data['id']] = data
    return out

emails_lengthen = load_jsonl("datasets/lengthen.jsonl")
emails_shorten = load_jsonl("datasets/shorten.jsonl")
emails_tone = load_jsonl("datasets/tone.jsonl")
emails_shorten_synthetic = load_jsonl("datasets/shorten_synthetic.jsonl")
emails_lengthen_synthetic = load_jsonl("datasets/lengthen_synthetic.jsonl")
emails_tone_synthetic = load_jsonl("datasets/tone_synthetic.jsonl")

# --- UI HEADER ---
tab1, tab2 = st.tabs(["Generate", "Evaluate"])
with tab1:
    st.title("📧 AI Email Editing Tool")
    st.write("Select an email record by ID and use AI to refine it.")

    # --- ID NAVIGATION BAR ---
    task_ids = ["Lengthen", "Shorten", "Change Tone", "Lengthen (synthetic)", "Shorten (synthetic)", "Change Tone (synthetic)"]
    selected_task = st.sidebar.selectbox("✏️ Select Editing Task", options=task_ids, index=0)

    # EMAIL TYPE SELECTION
    emails = {}
    if selected_task == "Lengthen":
        emails = emails_lengthen
    elif selected_task == "Shorten":
        emails = emails_shorten
    elif selected_task == "Change Tone":
        emails = emails_tone
    elif selected_task == "Shorten (synthetic)":
        emails = emails_shorten_synthetic
    elif selected_task == "Lengthen (synthetic)":
        emails = emails_lengthen_synthetic
    elif selected_task == "Change Tone (synthetic)":
        emails = emails_tone_synthetic

    email_ids = emails.keys()
    selected_id = st.sidebar.selectbox("📂 Select Email ID", options=email_ids, index=0)

    if not emails:
        st.warning("No emails found in your JSONL file.")
        st.stop()

    # Find the selected email
    if selected_id not in email_ids:
        st.error(f"Invalid email ID.")
        st.stop()
    else:
        selected_email = emails[selected_id]

    if not selected_email:
        st.error(f"No email found with ID {selected_id}.")
        st.stop()

    # --- DISPLAY SELECTED EMAIL ---
    st.markdown(f"### ✉️ Email ID: `{selected_id}`")
    st.markdown(f"**From:** {selected_email.get('sender', '(unknown)')}")
    st.markdown(f"**Subject:** {selected_email.get('subject', '(no subject)')}")

    email_text = st.text_area(
        "Email Content",
        value=selected_email.get("content", ""),
        height=250,
        key=f"email_text_{selected_id}",
    )

    if "Change Tone" in selected_task:
        tone_choice = st.segmented_control("Tone", ["Friendly", "Sympathetic", "Professional"], default="Friendly")

    # --- PROMPT AI ---
    action = ""
    if "Lengthen" in selected_task:
        action = "lengthen"
    elif "Shorten" in selected_task:
        action = "shorten"
    elif "Change Tone" in selected_task:
        action = tone_choice.lower()

    st.markdown("### 🤖 Edited Email")
    col1, col2 = st.columns([0.18,0.82], vertical_alignment="bottom")
    with col1:
        model_choice = st.segmented_control("Model", ["gpt-4o-mini", "gpt-4.1"], default="gpt-4o-mini")
    with col2:
        generate_clicked = st.button("Generate!")
        
    if generate_clicked:
        with st.spinner("Generating..."):
            generator = GenerateEmail(model_choice)
            new_text = generator.generate(action, selected_email)

            task = ""
            if action == "lengthen":
                task = "lengthening the original text"
            elif action == "shorten":
                task = "shortening the original text"
            elif action == "friendly":
                task = "changing the tone of the original text to be friendly"
            elif action == "sympathetic":
                task = "changing the tone of the original text to be sympathetic"
            elif action == "professional":
                task = "changing the tone of the original text to be professional"

            completeness_judge_out = generator.generate_judge("completeness", task, selected_email, new_text)
            faithfulness_judge_out = generator.generate_judge("faithfulness", task, selected_email, new_text)

        new_email_text = st.text_area(
            "Generated Email Content",
            value = new_text,
            height=250,
            key = f"new_email_text_{selected_id}"
        )
        col1, col2 = st.columns(2)
        with col1:
            st.markdown("**Completeness Judge**")
            judge_text = st.json(
                body=completeness_judge_out,
                expanded=2
            )
        with col2:
            st.markdown("**Faithfulness Judge**")
            judge_text = st.json(
                body=faithfulness_judge_out,
                expanded=2
            )

with tab2:
    st.markdown("# Evaluate Model on All Shortening Data")
    model_choice = st.segmented_control("Eval Model", ["gpt-4o-mini", "gpt-4.1"], default="gpt-4o-mini")
    generate_all = st.button("Evaluate on Whole Dataset")
    if generate_all:
        outputs = []
        avg_c = 0
        avg_f = 0
        n = 0
        with st.spinner("Evaluating on whole dataset..."):
            generator = GenerateEmail(model_choice)

            for email_id in emails_shorten:
                email = emails_shorten[email_id]
                new_text = generator.generate("shorten", email)
                completeness_judge_out = json.loads(generator.generate_judge("completeness", "shorten", email, new_text))
                faithfulness_judge_out = json.loads(generator.generate_judge("faithfulness", "shorten", email, new_text))
                outputs.append({"id":email_id, "original":email["content"], "text":new_text, "c_score":completeness_judge_out, "f_score":faithfulness_judge_out})

                avg_c += completeness_judge_out['rating']
                avg_f += faithfulness_judge_out['rating']
                n += 1
             
            for email_id in emails_shorten_synthetic:
                email = emails_shorten_synthetic[email_id]
                new_text = generator.generate("shorten", email)
                completeness_judge_out = json.loads(generator.generate_judge("completeness", "shorten", email, new_text))
                faithfulness_judge_out = json.loads(generator.generate_judge("faithfulness", "shorten", email, new_text))
                outputs.append({"id":f"synthetic_{email_id}", "original":email["content"], "text":new_text, "c_score":completeness_judge_out, "f_score":faithfulness_judge_out})

                avg_c += completeness_judge_out['rating']
                avg_f += faithfulness_judge_out['rating']
                n += 1

        avg_c /= n
        avg_f /= n  
        st.markdown(f"**Average Completeness Score: {avg_c}/3**")
        st.markdown(f"**Average Faithfullness Score: {avg_f}/3**")


        for data in outputs:
            with st.expander(f"Email {data['id']}"):
                orig_email_text = st.text_area(
                    "Original Email Content",
                    value = data["original"]
                )

                new_email_text = st.text_area(
                    "Generated Email Content",
                    value = data["text"],
                    key = f"new_email_text_{data['id']}"
                )
                col1, col2 = st.columns(2)
                with col1:
                    st.markdown("**Completeness Judge**")
                    judge_text = st.json(
                        body=data["c_score"],
                        expanded=2
                    )
                with col2:
                    st.markdown("**Faithfulness Judge**")
                    judge_text = st.json(
                        body=data["f_score"],
                        expanded=2
                    )