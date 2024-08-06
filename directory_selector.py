#!/usr/bin/env python
# coding: utf-8

# In[ ]:


# directory_selector.py

import tkinter as tk
from tkinter import filedialog

def get_download_directory():
    root = tk.Tk()
    root.withdraw()  # Hide the main window
    folder = filedialog.askdirectory()  # Open a file dialog for selecting a directory
    return folder

if __name__ == "__main__":
    print(get_download_directory())


# In[ ]:




