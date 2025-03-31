from flask import Flask, render_template, request
import model

app = Flask(__name__)
taskDict = {
            "summarization": "Summarize the text given by the user as meaningfully as possible, without losing important information!",
            "qa": "Extract the answer to the user's question from the text and provide it. If the answer to the question is not found in the text, state that you do not know the answer."
        }
llmModel = model.LLMmodel(max_new_tokens=256,
                              do_sample=True,
                              temperature = 0.4,
                              top_p = 0.9,
                              repetition_penalty=1.1)
    
llmModel.load_model("./Turkish-Llama-8b-Instruct-v0.1")


@app.route("/", methods=["GET", "POST"])
def index():
    if request.method == "POST":
        content = request.form["user_input"]
        task = request.form["option"]
        
        llmModel.system_message = f"""
            You are an AI assistant. The user will give you a task. Your goal is to complete the task as faithfully as possible.
            While performing the task, think step by step and justify your steps. The task the user wants you to complete is:
            {taskDict.get(task)}
        """
        
        if task == "qa":
            question = request.form.get("user_question", "")
            llmModel.user_message = f"""
                Text: {content},
                Question: {question}
                Answer:
            """

        elif task == "summarization":
            llmModel.user_message = f"""
                Text: {content},
                Answer:
            """
        
        result = llmModel.generate_answer()
        
        return render_template("index.html", result=result)
    
    return render_template("index.html", result=None)

if __name__ == "__main__":
    app.run(debug=False)
