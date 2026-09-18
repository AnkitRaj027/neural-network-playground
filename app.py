import gradio as gr
import torch
import torch.nn as nn
from model import NeuralNetwork
from train import train_model
from data import X_train, y_train, X_test, y_test
def train_from_ui(hidden_layers,neurons,activation,learning_rate,epochs):
    activation_functions={
        "ReLU":nn.ReLU,
        "Sigmoid":nn.Sigmoid,
        "Tanh":nn.Tanh
    }
    activation_function=activation_functions[activation]
    hidden_sizes=[
        int(neurons)
        for _ in range(int(hidden_layers))
    ]
    model=NeuralNetwork(
        input_size=2,
        hidden_sizes=hidden_sizes,
        output_size=2,
        activation=activation_function
    )
    model,train_losses,test_accuracies=train_model(
        model,
        X_train,
        y_train,
        X_test,
        y_test,
        learning_rate,
        int(epochs)
    )
    final_accuracy=test_accuracies[-1]*100

    architecture= ("2->"+"->".join(str(size) for size in hidden_sizes)+"->2")
    return (
        f"Model trained successfully!\n"
        f"Architecture: {architecture}\n"
        f"activation: {activation}\n"
        f"learning_rate: {learning_rate}\n"
        f"epochs: {epochs} \n"
        f"Final Accuracy: {final_accuracy:.2f}%\n"
    )

with gr.Blocks() as app:
    gr.Markdown("Neural Network PlayGround")
    with gr.Row():
        with gr.Column():
            gr.Markdown("Model Configuration")
            hidden_layers=gr.Slider(
                minimum=1,
                maximum=5,
                value=2,
                step=1,
                label="No of hidden layers"
            )
            neurons=gr.Slider(
                minimum=2,
                maximum=64,
                value=16,
                step=2,
                label="Nurons per layer"
            )
            activation=gr.Dropdown(
                choices=["ReLU","Sigmoid","Tanh"],
                value="ReLU",
                label="Activation Function"
            )


        with gr.Column():
            gr.Markdown("Training Configuration")
            learning_rate=gr.Number(
                value=0.01,
                label="Learning Rate"
            )
            epochs=gr.Slider(
                minimum=10,
                maximum=500,
                value=100,
                step=10,
                label="No of Epochs"
            )
            train_button=gr.Button("Train Model")
            status=gr.Textbox(
                label="Status",
            )
            train_button.click(
                fn=train_from_ui,
                inputs=[hidden_layers,neurons,activation,learning_rate,epochs],
                outputs=status
            )
app.launch()