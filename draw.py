#!/usr/bin/env python3
import numpy as np
from teenygrad import Tensor
from PIL import ImageGrab
from mnist import TinyConvNet
import tkinter as tk

class DrawingApp:
  def __init__(self, master):
    self.master = master
    self.canvas = tk.Canvas(master, width=400, height=400, bg="black")
    self.canvas.pack()

    self.clear_button = tk.Button(master, text="Clear", command=self.clear_canvas)
    self.clear_button.pack()

    self.read_button = tk.Button(master, text="Read", command=self.read_canvas)
    self.read_button.pack()

    self.previous_x = None
    self.previous_y = None

    self.canvas.bind("<B1-Motion>", self.paint)
    self.canvas.bind("<ButtonRelease-1>", self.reset)

  def clear_canvas(self):
    self.canvas.delete("all")

  def paint(self, event):
    paint_color = "white"
    if self.previous_x and self.previous_y:
        self.canvas.create_line(self.previous_x, self.previous_y, event.x, event.y,
                                width=2, fill=paint_color, capstyle=tk.ROUND, smooth=tk.TRUE)
    self.previous_x = event.x
    self.previous_y = event.y

  def reset(self):
    self.previous_x = None
    self.previous_y = None

  def read_canvas_continuously(self):
    self.read_canvas()
    self.master.after(50, self.read_canvas_continuously)

  def read_canvas(self):
    # fix coordinates for my Retina display
    x = self.master.winfo_rootx() + 15
    y = self.master.winfo_rooty() + 80
    x1 = x + self.canvas.winfo_width() * 2 - 20
    y1 = y + self.canvas.winfo_height() * 2 - 20
    image = np.array(ImageGrab.grab().crop((x, y, x1, y1)).convert("L").resize((28, 28))).reshape((-1, 28*28)).astype(np.float32)
    for i in range(0, len(image[0]), 28):
      print(' '.join(f'{number.astype(int):3d}' for number in image[0][i:i+28]))
    input = Tensor(image)

    model = TinyConvNet()
    if not model.load():
      raise "model not loaded"
    output = model.forward(input)
    print(np.argmax(output.numpy(), axis=-1))

if __name__ == "__main__":
  master = tk.Tk()
  DrawingApp(master)
  master.mainloop()
