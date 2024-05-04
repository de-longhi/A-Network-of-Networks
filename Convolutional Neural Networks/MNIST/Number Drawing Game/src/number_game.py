import tkinter as tk
from tkinter import Canvas, Button, messagebox
from PIL import Image, ImageDraw
from pathlib import Path
import torch
from torch import nn

class CNN(nn.Module):
    
    def __init__(self):
        super(CNN, self).__init__()
        self.device = "cuda" if torch.cuda.is_available else "cpu"
        self.features = nn.Sequential(
            nn.Conv2d(in_channels=1, out_channels=16, kernel_size=(3,3), stride=1, padding=1),
            nn.ReLU(),
            nn.BatchNorm2d(16),
            nn.MaxPool2d(2),
            nn.Conv2d(in_channels=16, out_channels=32, kernel_size=(3, 3),stride = 1, padding = 1),
            nn.ReLU(),
            nn.BatchNorm2d(32),
            nn.MaxPool2d(2)
            
        )
        self.classify = nn.Sequential(
            nn.Linear(in_features=7*7*32, out_features=256),
            nn.ReLU(),
            nn.BatchNorm1d(256),
            nn.Linear(in_features=256, out_features=10)         
        )
        

    def forward(self, x):
        x = self.features(x)
        x = x.view(-1, 32*7*7)
        x = self.classify(x)
        return x

class GUI:
    def __init__(self, master):
        self.master = master
        self.master.title("Draw a number!")

        self.canvas = Canvas(master, width=280, height=280, bg="white")
        self.canvas.grid(row=0, column=0, columnspan=2)

        self.label = tk.Label(master, text="Draw a number with your mouse")
        self.label.grid(row=1, column=0, columnspan=2)

        self.save_button = Button(master, text="Classify!", command=self.classify)
        self.save_button.grid(row=2, column=0)

        self.clear_button = Button(master, text="Clear Drawing", command=self.clear_drawing)
        self.clear_button.grid(row=2, column=1)

        self.image = Image.new("L", (280, 280), "white")
        self.draw = ImageDraw.Draw(self.image)

        self.canvas.bind("<B1-Motion>", self.paint)

    def paint(self, event):
        x1, y1 = (event.x - 10), (event.y - 10)
        x2, y2 = (event.x + 10), (event.y + 10)
        self.canvas.create_oval(x1, y1, x2, y2, fill="black")
        self.draw.ellipse([x1, y1, x2, y2], fill="black")

    def save_drawing(self, path):
        self.image = self.image.resize((28, 28))
        filename = path
        self.image.save(filename)

    def clear_drawing(self):
        self.canvas.delete("all")
        self.image = Image.new("L", (280, 280), "white")
        self.draw = ImageDraw.Draw(self.image)
    def classify(self):
        PIC_DIR = Path(__file__).resolve().parent.parent / "tmp" / "drawing.png"
        MODEL_DIR = Path(__file__).resolve().parent.parent / "models" / "CNN.pt"
        self.save_drawing(PIC_DIR)
        model = CNN().cuda()
        model = torch.load(MODEL_DIR)
        

def main():
    root = tk.Tk()
    app = GUI(root)
    root.mainloop()

if __name__ == "__main__":

    main()
