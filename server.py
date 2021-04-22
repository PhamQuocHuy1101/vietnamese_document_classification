import prediction
from tkinter import *

# import filedialog module
from tkinter import filedialog
from tkinter import font

class App():
    def __init__(self):																								
        # Create the root window
        self.window = Tk()
        self.window.title('File Explorer')
        self.window.geometry("800x800")
        self.window.config(background = "white")


        self.label_text = Label(self.window,text = "*****", height = 3, width = 100, bg = '#DAE8FC')
        self.button_get_label = Button(self.window, text = "Predict", bg = '#DAE8FC', width = 16, command = self.get_label)
        self.button_explore = Button(self.window, text = "Browse Files", bg = '#DAE8FC', width = 16, command = self.browseFiles)
        self.button_exit = Button(self.window, text = "Exit", bg = '#DAE8FC', width = 16, command = exit)
        self.content = Text(self.window, height=50, width=100, bd = 4, wrap=WORD )

        self.content.grid(column = 1, row = 1, pady = 5, sticky = W)
        self.label_text.grid(column = 1, row = 2, pady = 5, sticky = W)
        self.button_get_label.grid(column = 1, row = 3, pady = 5, sticky = W)
        self.button_explore.grid(column = 1, row = 4, pady = 5, sticky = W)
        self.button_exit.grid(column = 1, row = 5, pady = 5, sticky = W)

    def get_label(self):
        data = self.content.get("1.0",END)
        data = data.strip()
        if not data:
            self.label_text.configure(text= "*****")
            return
        label = prediction.predict(data)
        self.label_text.configure(text= f"Label: {label[0]}")
        
    def browseFiles(self): 
        filename = filedialog.askopenfilename(initialdir = "./", title = "Select a File", filetypes = (("Text files", "*.txt*"), ("all files", "*.*"))) 
        with open(filename, 'r') as f:
            data = f.read()
            self.content.delete('1.0', END)
            self.content.insert(END, data)
            self.get_label()
    
    def click(self, btn_tag):
        if btn_tag == 'predict':
            self.get_label()
        elif btn_tag == 'browser':
            self.browseFiles()
        elif btn_tag == 'exit':
            self.window.destroy()
    
    def run(self):
        self.window.mainloop()

app = App()
app.run()