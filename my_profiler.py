import typer
import upmem_llm_framework as upmem_layers
import torch
import torch.nn as nn

app = typer.Typer(callback=upmem_layers.initialize_profiling_options)

@app.command()
def profile(my_input: str):
    upmem_layers.profiler_init()

    # 모델 정의
    class SimpleModel(nn.Module):
        def __init__(self):
            super(SimpleModel, self).__init__()
            self.linear = nn.Linear(10, 5)

        def forward(self, x):
            return self.linear(x)

    model = SimpleModel()

    # 입력 텐서 정의
    myTensor = torch.randn(1, 10)

    # 프로파일링 실행
    upmem_layers.profiler_start()
    prediction = model.forward(myTensor)
    upmem_layers.profiler_end()

    print("Prediction:", prediction)

if __name__ == "__main__":
    app()
