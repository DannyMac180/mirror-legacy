import dspy

lm = dspy.OpenAI(base_url="http://localhost:1234/v1", api_key="not-needed")
dspy.configure(lm=lm)

class GenerateAnswer(dspy.Signature):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

    def run(self, inputs):
        return inputs
    
print(lm("Hello, how are you?"))

