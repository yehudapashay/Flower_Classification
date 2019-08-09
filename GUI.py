#import os

#from keras import layers, optimizers
from keras.engine.saving import load_model
#from keras.initializers import RandomUniform
#from keras.preprocessing import image
#import numpy as np
#from keras import models
#from tensorflow.python.keras.preprocessing.image import ImageDataGenerator
#import matplotlib.pyplot as plt

#from keras.layers import Conv2D, MaxPooling2D, Flatten, Dense, Dropout, BatchNormalization, MaxPool2D


from tkinter import *
from tkinter import filedialog

import os
from PIL import Image, ImageTk




def make_entry(parent, caption, row, column, width=None, **options):
    Label(parent, text=caption).grid(row=row, column=column, sticky='W')
    entry = Entry(parent, **options)
    if width:
        entry.config(width=width)
    entry.grid(row=row, column=column + 1, sticky='W')
    return entry


class VerticalScrolledFrame(Frame):

    def __init__(self, parent, *args, **kw):
        Frame.__init__(self, parent, *args, **kw)

        # create a canvas object and a vertical scrollbar for scrolling it
        vscrollbar = Scrollbar(self, orient=VERTICAL)
        vscrollbar.pack(fill=Y, side=RIGHT, expand=FALSE)
        canvas = Canvas(self, bd=0, highlightthickness=0,
                        yscrollcommand=vscrollbar.set, width=800, height=500)
        canvas.pack(side=LEFT, fill=BOTH, expand=TRUE)
        vscrollbar.config(command=canvas.yview)

        # reset the view
        canvas.xview_moveto(0)
        canvas.yview_moveto(0)

        # create a frame inside the canvas which will be scrolled with it
        self.interior = interior = Frame(canvas)
        interior_id = canvas.create_window(0, 0, window=interior,
                                           anchor=NW)

        # track changes to the canvas and frame width and sync them,
        # also updating the scrollbar
        def _configure_interior(event):
            # update the scrollbars to match the size of the inner frame
            size = (interior.winfo_reqwidth(), interior.winfo_reqheight())
            canvas.config(scrollregion="0 0 %s %s" % size)
            if interior.winfo_reqwidth() != canvas.winfo_width():
                # update the canvas's width to fit the inner frame
                canvas.config(width=interior.winfo_reqwidth())

        interior.bind('<Configure>', _configure_interior)

        def _configure_canvas(event):
            if interior.winfo_reqwidth() != canvas.winfo_width():
                # update the inner frame's width to fill the canvas
                canvas.itemconfigure(interior_id, width=canvas.winfo_width())

        canvas.bind('<Configure>', _configure_canvas)


class GraphicUserInterface:

    def __init__(self, controller):
        self.controller = controller
        self.root = Tk()
        # self.root.geometry('500x800')
        self.root.protocol("WM_DELETE_WINDOW", self.root.destroy)
        self.root.title("Flower Classification")
        welcome_instruction = Label(master=self.root,text='Hello! In order to proceed, please insert Images Path and model path and then press \"Predict\":')
        welcome_instruction.grid(row=0, column=1)
        self.img_entry = make_entry(self.root, "Images Path:", 1, 0, 70)
        browse_dir = Button(master=self.root, text='Browse', width=6, command=self.browse_img_file)
        browse_dir.grid(row=1, column=2, sticky='W')
        #############################################################################
        self.model_entry = make_entry(self.root, "Model Path:", 2, 0, 70)
        model_browse_dir = Button(master=self.root, text='Browse', width=6, command=self.browse_model_file)
        model_browse_dir.grid(row=2, column=2, sticky='W')
        ####################################################################

        start_button = Button(master=self.root, text="Predict", command=self.start)
        start_button.grid(row=4, column=1)
        Label(font=("Courier", 18), text="\nResults: \n").grid(row=3, column=1)
        self.results_frame = VerticalScrolledFrame(self.root)
        labels = []
        path = os.path.dirname(os.path.abspath(__file__)) +'\\gui Examples'
        dirs_list = os.listdir(path)
        results = self.controller.predict(path)
        for i in range(0, len(results)):
            image = Image.open(path +'\\'+ dirs_list[i])
            photo = ImageTk.PhotoImage(image)
            label = Label(master=self.results_frame.interior, image=photo)
            label.image = photo  # keep a reference!
            labels.append(label)
            labels[-1].pack()
            Label(master=self.results_frame.interior, text="Example" + str(i+1) + ': ' + results[i][1]).pack()
            Label(master=self.results_frame.interior,
                  text='~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~').pack()
        self.results_frame.grid(rowspan=1, columnspan=4)
        self.root.mainloop()

    def browse_img_file(self):
        self.img_entry.delete(first=0, last=100)
        dir_path = filedialog.askdirectory()
        self.img_entry.insert(0, dir_path)


    def browse_model_file(self):
        self.model_entry.delete(first=0, last=100)
        model_dir_path = filedialog.askopenfilename()
        self.model_entry.insert(0, model_dir_path)


    def start(self):
        from Flower_Classification import ModelController
        model_path = self.model_entry.get()
        new_model = self.load_model_by_path(model_path)
        new_model_controller = ModelController(new_model)
        self.controller=new_model_controller
        self.results_frame.destroy()
        self.results_frame = VerticalScrolledFrame(self.root)
        path = self.img_entry.get()
        results = self.controller.predict(path)
        dirs_list = os.listdir(path)
        labels = []
        # self.results_frame.interior.deletecommand("all")
        for i in range(0, len(results)):
            image = Image.open(path + '\\' + dirs_list[i])
            photo = ImageTk.PhotoImage(image)
            label = Label(master=self.results_frame.interior, image=photo)
            label.image = photo  # keep a reference!
            labels.append(label)
            labels[-1].pack()
            Label(master=self.results_frame.interior, text=results[i][0] + " " + results[i][1]).pack()
            Label(master=self.results_frame.interior,
                  text='~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~').pack()
        print("finished classifying")
        self.results_frame.grid(rowspan=1, columnspan=4)

    def load_model_by_path(self,model_path):
        return load_model(model_path)

