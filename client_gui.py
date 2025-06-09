import tkinter as tk
from tkinter import filedialog, ttk, messagebox
import threading
import os
import Main


def _run_pipeline(path, button):
    try:
        Main.main(path)
    except Exception as exc:
        messagebox.showerror("Error", str(exc))
    finally:
        button.config(state=tk.NORMAL)


def start_planning(stl_var, button):
    path = stl_var.get()
    if not os.path.isfile(path):
        messagebox.showerror("Invalid File", "Please select a valid STL file")
        return
    button.config(state=tk.DISABLED)
    threading.Thread(target=_run_pipeline, args=(path, button), daemon=True).start()


def create_gui():
    root = tk.Tk()
    root.title("Toolpath Planner")

    frame = ttk.Frame(root, padding=10)
    frame.pack(fill=tk.BOTH, expand=True)

    stl_var = tk.StringVar()

    ttk.Label(frame, text="STL File:").grid(row=0, column=0, sticky=tk.W)
    entry = ttk.Entry(frame, textvariable=stl_var, width=40)
    entry.grid(row=0, column=1, sticky=tk.EW)

    def browse():
        file_path = filedialog.askopenfilename(filetypes=[("STL Files", "*.stl"), ("All Files", "*.*")])
        if file_path:
            stl_var.set(file_path)

    ttk.Button(frame, text="Browse", command=browse).grid(row=0, column=2, padx=5)

    run_btn = ttk.Button(frame, text="Run", command=lambda: start_planning(stl_var, run_btn))
    run_btn.grid(row=1, column=0, columnspan=3, pady=10)

    frame.columnconfigure(1, weight=1)
    root.mainloop()


if __name__ == "__main__":
    create_gui()
