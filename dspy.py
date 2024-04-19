import dspy

class GenerateAnswer(dspy.Signature):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

    def run(self, inputs):
        return inputs