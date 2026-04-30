import os
import gradio as gr
from pii_intent_classifier import PIIIntentClassifier

# Get token from environment variable (required for gated models in Spaces)
hf_token = os.getenv("HF_TOKEN")

print("Loading PII Intent Classifier model...")
classifier = PIIIntentClassifier(token=hf_token)
print("Model loaded successfully.")

def classify_text(text):
    if not text.strip():
        return {"privacy_asking_for_pii": 0.0, "privacy_giving_pii": 0.0}, [["Status", "SAFE"], ["Categories", "None"], ["Combined Score", "0.0000"]]
    
    result = classifier.classify(text)
    
    scores = {
        "privacy_asking_for_pii": result.asking_score,
        "privacy_giving_pii": result.giving_score
    }
    
    flagged_status = "FLAGGED" if result.is_flagged else "SAFE"
    categories = ", ".join(result.flagged_category) if result.flagged_category else "None"
    
    table_data = [
        ["Status", flagged_status],
        ["Categories", categories],
        ["Combined Score", f"{result.combined_score:.4f}"]
    ]
    
    return scores, table_data

with gr.Blocks(title="PII Intent Classifier Demo") as demo:
    gr.Markdown("# PII Intent Classifier Demo")
    gr.Markdown("Detect attempts to solicit or share Personally Identifiable Information (PII) in text. This uses the Roblox XLM-RoBERTa-Large multi-label text classifier.")
    
    with gr.Row():
        with gr.Column(scale=1):
            input_text = gr.Textbox(
                lines=8,
                label="Input Text",
                placeholder="Enter a message or conversation to analyze..."
            )
            with gr.Row():
                clear_btn = gr.ClearButton(value="Clear")
                submit_btn = gr.Button("Analyze Intent", variant="primary")
            
        with gr.Column(scale=1):
            scores_label = gr.Label(label="Intent Scores")
            details_table = gr.Dataframe(
                headers=["Metric", "Value"],
                datatype=["str", "str"],
                label="Classification Details",
                interactive=False
            )
            
    gr.Examples(
        examples=[
            ["Hey, what is your discord or phone number?"],
            ["my number is 555-0199 call me"],
            ["where do you live? just tell me the city man"],
            ["I love this game, want to play again tomorrow?"],
            ["If you need help my email is admin@example.com, what's yours?"]
        ],
        inputs=input_text
    )

    clear_btn.add([input_text, scores_label, details_table])
    
    submit_btn.click(
        fn=classify_text,
        inputs=input_text,
        outputs=[scores_label, details_table]
    )

if __name__ == "__main__":
    demo.launch()
