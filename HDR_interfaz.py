import tkinter as tk
from tkinter import Canvas, Button
from PIL import Image, ImageOps
import torch
from torchvision.transforms import ToTensor, Normalize
import io
import torch.nn as nn
import torch.nn.functional as F

# Definición de la red LeNet-5
class LeNet5(nn.Module):
    def __init__(self):
        super(LeNet5, self).__init__()
        self.conv1 = nn.Conv2d(1, 6, kernel_size=5, stride=1, padding=2)  # Padding agregado
        self.conv2 = nn.Conv2d(6, 16, kernel_size=5, stride=1, padding=0)
        self.conv3 = nn.Conv2d(16, 120, kernel_size=5, stride=1, padding=0)
        self.fc1 = nn.Linear(120, 84)
        self.fc2 = nn.Linear(84, 10)

    def forward(self, x):
        x = F.relu(self.conv1(x))
        x = F.avg_pool2d(x, kernel_size=2, stride=2)
        x = F.relu(self.conv2(x))
        x = F.avg_pool2d(x, kernel_size=2, stride=2)
        x = F.relu(self.conv3(x))
        x = x.view(-1, 120)
        x = F.relu(self.fc1(x))
        x = self.fc2(x)
        return x

# Función para predecir el dígito dibujado
def predict_digit():
    global canvas
    # Convertir la imagen del lienzo a un tensor
    ps = canvas.postscript(colormode='color')
    img = Image.open(io.BytesIO(ps.encode('utf-8'))).convert('L')
    img = ImageOps.invert(img)  # Invertir los colores
    img = img.resize((28, 28))
    img = ToTensor()(img)
    img = Normalize((0.1307,), (0.3081,))(img)
    img = img.unsqueeze(0)

    # Pasar la imagen a través del modelo
    with torch.no_grad():
        model.eval()
        output = model(img)
        pred = output.argmax(dim=1, keepdim=True)
        print(f'Predicción: {pred.item()}')
        prediction_label.config(text=f'Predicción: {pred.item()}')

# Función para borrar el lienzo
def clear_canvas():
    global canvas
    canvas.delete("all")
    prediction_label.config(text="Predicción: ")

# Crear la ventana de la interfaz
root = tk.Tk()
root.title("Digit Recognizer")

# Crear el lienzo para dibujar (fondo blanco)
canvas = Canvas(root, width=200, height=200, bg='white')  # Fondo blanco
canvas.grid(row=0, column=0, columnspan=2)

# Botón para predecir
predict_button = Button(root, text="Predecir", command=predict_digit)
predict_button.grid(row=1, column=0)

# Botón para limpiar el lienzo
clear_button = Button(root, text="Limpiar", command=clear_canvas)
clear_button.grid(row=1, column=1)

# Etiqueta para mostrar la predicción
prediction_label = tk.Label(root, text="Predicción: ")
prediction_label.grid(row=2, column=0, columnspan=2)

# Cargar el modelo previamente entrenado
model = LeNet5()
model.load_state_dict(torch.load('mnist_cnn.pt', map_location=torch.device('cpu')))  # Asegurarse de cargar en CPU
model.eval()

# Función para dibujar en el lienzo
def paint(event):
    if canvas.prev_x is not None and canvas.prev_y is not None:
        # Dibuja una línea desde las coordenadas anteriores hasta las actuales
        canvas.create_line(canvas.prev_x, canvas.prev_y, event.x, event.y, fill="black", width=15)  # Dibuja en negro
    # Actualiza las coordenadas previas
    canvas.prev_x = event.x
    canvas.prev_y = event.y

def start_drawing(event):
    # Inicializa las coordenadas previas solo cuando se hace clic en el lienzo por primera vez
    canvas.prev_x = event.x
    canvas.prev_y = event.y

canvas.prev_x = None
canvas.prev_y = None

canvas.bind("<Button-1>", start_drawing)
canvas.bind("<B1-Motion>", paint)

root.mainloop()
