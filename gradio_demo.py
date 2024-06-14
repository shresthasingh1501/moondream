import gradio as gr
from PIL import Image
from moondream import Moondream, detect_device, LATEST_REVISION
from transformers import AutoTokenizer
import torch

# Model Initialization
model_id = "vikhyatk/moondream2"
tokenizer = AutoTokenizer.from_pretrained(model_id, revision=LATEST_REVISION)

def analyze_project(image, project_name, project_description, project_status, use_gpu):
    try:
        # Set device and dtype
        if use_gpu:
            device, dtype = detect_device()
            if device == torch.device("cpu"):
                raise ValueError("GPU not found, defaulting to CPU.")
        else:
            device = torch.device("cpu")
            dtype = torch.float32

        moondream = Moondream.from_pretrained(
            model_id, torch_dtype=dtype, revision=LATEST_REVISION
        ).to(device=device)
        moondream.eval()

        image_embeds = moondream.encode_image(image)

        # Preliminary check: Verify if the image is an outdoor scene, field, or building
        check_prompt = """
        is this picture of a a general scene or a object or human as centerpiece , 
        yes if its a scene , no if its a object or a human in focus
        """
        check_result = moondream.answer_question(image_embeds, check_prompt, tokenizer).strip()
        if check_result.lower() != 'yes':
            return {"Error": "The uploaded image is not an appropriate scene for construction verification."}

        # Instructions map for project status
        instructions_map = {
            0: "An empty plot with no major construction signs, but possibly minor groundwork or preliminary site preparations visible.",
            5: "An empty plot with no major construction signs, but possibly minor groundwork or preliminary site preparations visible.",
            10: "Early construction stage with visible foundations or footings; initial raised structures are evident on the site.",
            20: "Foundations raised to plinth level; completed lintel or beam structure indicating progress in the building's construction.",
            40: "Plinth level raised; completed lintel or beam structure with initial roof structure visible, showing significant construction progress.",
            70: "Beam structure or roof in place; interior flooring and plastering visible, indicating advanced stages of construction.",
            90: "Roof structure present; interior flooring and plastering visible; nearing completion with external finishes and electrical installations.",
            100: "Fully completed structure with all necessary finishes, fixtures, and fittings in place, ready for occupancy."
        }

        # Prompt 1: Verify construction progress
        prompt1 = f""" does this image have {instructions_map.get(project_status, "")} and only that in the picture. Answer with yes or no. """
        verification = moondream.answer_question(image_embeds, prompt1, tokenizer).strip()

        # Prompt 2: Reasoning
        prompt2 = f""" Describe this uploaded by a contractor of a construction crew claiming that {instructions_map.get(project_status, "")} is visible and {project_name} is {project_status}% complete and also gave a description '{project_description}'. Is he right or wrong? Explain why and be confident. """
        reasoning = moondream.answer_question(image_embeds, prompt2, tokenizer).strip()

        return {"Check": verification, "Reasoning": reasoning}
    
    except Exception as e:
        return {"Error": str(e)}

def verify_project(image, project_name, project_description, project_status, use_gpu):
    analysis = analyze_project(image, project_name, project_description, project_status, use_gpu)
    if 'Error' in analysis:
        return f"Error: {analysis['Error']}", "", ""
    else:
        return "Verification Complete!", f"AI Check: {analysis['Check']}", f"AI Reasoning: {analysis['Reasoning']}"

options = {
    0: "empty plot of land with no signs of construction.",
    5: "visible foundations or footings.",
    10: "raised foundations to the plinth level.",
    20: "completed lintel or beam structure.",
    40: "roof structure in place.",
    70: "visible flooring and plastering inside the structure.",
    90: "all external finishes and electrical installations.",
    100: "Completed building ready for handover"
}

gr.Interface(
    fn=verify_project,
    inputs=[
        gr.Image(type="pil"),
        gr.Textbox(label="Enter Project Name"),
        gr.Textbox(label="Enter Project Description"),
        gr.Dropdown(list(options.keys()), label="Project Status (%)", value=0, type="index"),
        gr.Checkbox(label="Use GPU", value=False)
    ],
    outputs=[
        gr.Textbox(label="Verification Status"),
        gr.Textbox(label="AI Check"),
        gr.Textbox(label="AI Reasoning")
    ],
    title="Construction Project Verification"
).launch(share=True)
