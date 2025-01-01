import typer
import upmem_llm_framework as upmem_layers
import torch
import torch.nn as nn

app = typer.Typer(callback=upmem_layers.initialize_profiling_options)

# 간단한 PyTorch 모델
class SimpleModel(nn.Module):
    def __init__(self):
        super(SimpleModel, self).__init__()
        self.linear = nn.Linear(10, 5)

    def forward(self, x):
        return self.linear(x)

@app.command()
def profile(my_input: str):
    upmem_layers.profiler_init()

    # PyTorch 모델 생성
    model = SimpleModel()
    myTensor = torch.randn(1, 10)

    # 프로파일링 시작
    upmem_layers.profiler_start()

    # 모델 실행
    prediction = model.forward(myTensor)

    # 프로파일링 종료
    upmem_layers.profiler_end()

    # 예측 결과 출력
    print("PyTorch Model Prediction:", prediction)

if __name__ == "__main__":
    app()
