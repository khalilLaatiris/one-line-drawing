import tkinter as tk
from tkinter import ttk, filedialog, messagebox
from PIL import Image, ImageTk
import cv2
import numpy as np
from collections import deque

class SketchGeneratorApp:
    def __init__(self, root):
        self.root = root
        self.root.title("Sketch Generator/editor")

        # Main Window Configuration
        self.root.state('zoomed')      # Larger initial size
        self.root.configure(bg="#f0f0f0")  # Light gray background

        # Image Display Frames
        self.image_frame = tk.Frame(self.root, bg="#ffffff", relief=tk.RAISED, bd=2)
        self.image_frame.pack(side=tk.LEFT, padx=10, pady=10, fill=tk.BOTH, expand=True)

        # self.original_label = tk.Label(self.image_frame, text="Original Image", font=("Arial", 12))
        # self.original_label.pack(side=tk.TOP, pady=(0, 5))
        # Original Image Canvas with Scrollbars
        self.original_frame = tk.Frame(self.image_frame)
        self.original_canvas = tk.Canvas(self.original_frame, bg="gray")
        self.original_canvas.pack(side=tk.LEFT, expand=True, fill=tk.BOTH)
        self.original_frame.pack(side=tk.LEFT, padx=(0,5), fill=tk.BOTH, expand=True)
        self.original_canvas.bind("<Configure>", lambda e: self.display_images())

        # self.modified_label = tk.Label(self.image_frame, text="Modified Image", font=("Arial", 12))
        # self.modified_label.pack(side=tk.TOP, pady=(0, 5))
        # Modified Image Canvas with Scrollbars
        self.modified_frame = tk.Frame(self.image_frame)
        self.modified_canvas = tk.Canvas(self.modified_frame, bg="gray")
        self.modified_canvas.pack(side=tk.LEFT, expand=True, fill=tk.BOTH)
        self.modified_frame.pack(side=tk.LEFT, fill=tk.BOTH, expand=True)
        self.modified_canvas.bind("<Configure>", lambda e: self.display_images())


        # Toolbox Frame
        self.toolbox_frame = tk.Frame(self.root, bg="#e0e0e0", relief=tk.RAISED, bd=2, width=300)  # Fixed width
        self.toolbox_frame.pack(side=tk.RIGHT, padx=10, pady=10, fill=tk.Y)
        self.toolbox_frame.pack_propagate(False) # Important: Don't let children resize frame.

        self.toolbox_label = tk.Label(self.toolbox_frame, text="Toolbox", font=("Arial", 14, "bold"), bg="#e0e0e0")
        self.toolbox_label.pack(side=tk.TOP, pady=10)

        # Undo/Redo Buttons
        self.undo_button = ttk.Button(self.toolbox_frame, text="Undo", command=self.undo, state=tk.DISABLED)
        self.undo_button.pack(side=tk.TOP, fill=tk.X, padx=10, pady=(0, 5))
        self.redo_button = ttk.Button(self.toolbox_frame, text="Redo", command=self.redo, state=tk.DISABLED)
        self.redo_button.pack(side=tk.TOP, fill=tk.X, padx=10, pady=(0, 10))

        # Functionality Tabs
        self.notebook = ttk.Notebook(self.toolbox_frame)
        self.notebook.pack(fill=tk.BOTH, expand=True, padx=10, pady=10)

        self.create_sketch_tab()
        self.create_style_tab()
        self.create_vectorize_tab()
        self.create_canny_tab()
        self.create_xdog_tab()

        # Image Data
        self.original_image = None
        self.modified_image = None
        self.image_history = deque(maxlen=5)  # Store up to 20 states
        self.history_index = -1

    def create_sketch_tab(self):
        self.sketch_tab = ttk.Frame(self.notebook)
        self.notebook.add(self.sketch_tab, text="Sketch")

        self.blur_strength_label = tk.Label(self.sketch_tab, text="Blur Strength:", font=("Arial", 10), bg="#e0e0e0")
        self.blur_strength_label.pack(side=tk.TOP, anchor=tk.W, padx=10, pady=(5, 0))
        self.blur_strength_scale = ttk.Scale(self.sketch_tab, from_=1, to=51, value=21)
        self.blur_strength_scale.pack(side=tk.TOP, fill=tk.X, padx=10)
        self.blur_strength_value = tk.Label(self.sketch_tab, text="21", font=("Arial", 10), bg="#e0e0e0")
        self.blur_strength_value.pack(side=tk.TOP, anchor=tk.W, padx=10)

        self.dodge_intensity_label = tk.Label(self.sketch_tab, text="Dodge Intensity:", font=("Arial", 10), bg="#e0e0e0")
        self.dodge_intensity_label.pack(side=tk.TOP, anchor=tk.W, padx=10, pady=(5, 0))
        self.dodge_intensity_scale = ttk.Scale(self.sketch_tab, from_=0.1, to=5, value=2.5)
        self.dodge_intensity_scale.pack(side=tk.TOP, fill=tk.X, padx=10)
        self.dodge_intensity_value = tk.Label(self.sketch_tab, text="2.5", font=("Arial", 10), bg="#e0e0e0")
        self.dodge_intensity_value.pack(side=tk.TOP, anchor=tk.W, padx=10)
        
        self.sketch_apply_btn = ttk.Button(self.sketch_tab, text="Apply", command=self.process_image)
        self.sketch_apply_btn.pack(side=tk.TOP, fill=tk.X, padx=10, pady=5)
        self.sketch_params = {'blur_strength': 21, 'dodge_intensity': 2.5}

    def create_style_tab(self):
        self.style_tab = ttk.Frame(self.notebook)
        self.notebook.add(self.style_tab, text="Style")

        self.style_label = tk.Label(self.style_tab, text="Select Style:", font=("Arial", 10), bg="#e0e0e0")
        self.style_label.pack(side=tk.TOP, anchor=tk.W, padx=10, pady=(5, 0))
        self.style_combo = ttk.Combobox(self.style_tab, values=["None", "Manga", "Watercolor", "Blueprint"], state="readonly")
        self.style_combo.pack(side=tk.TOP, fill=tk.X, padx=10)
        self.style_combo.set("None")  # Default value
        self.style_combo.bind("<<ComboboxSelected>>", self.update_style)
        
        self.style_apply_btn = ttk.Button(self.style_tab, text="Apply", command=self.process_image)
        self.style_apply_btn.pack(side=tk.TOP, fill=tk.X, padx=10, pady=5)
        self.selected_style = "None"

    def create_vectorize_tab(self):
        self.vectorize_tab = ttk.Frame(self.notebook)
        self.notebook.add(self.vectorize_tab, text="Vectorize")

        self.draw_contours_label = tk.Label(self.vectorize_tab, text="Contour Levels:", font=("Arial", 10), bg="#e0e0e0")
        self.draw_contours_label.pack(side=tk.TOP, anchor=tk.W, padx=10, pady=(5, 0))
        self.draw_contours_scale = ttk.Scale(self.vectorize_tab, from_=1, to=10, value=2)
        self.draw_contours_scale.pack(side=tk.TOP, fill=tk.X, padx=10)
        self.draw_contours_value = tk.Label(self.vectorize_tab, text="2", font=("Arial", 10), bg="#e0e0e0")
        self.draw_contours_value.pack(side=tk.TOP, anchor=tk.W, padx=10)

        self.draw_hatch_label = tk.Label(self.vectorize_tab, text="Hatching Density:", font=("Arial", 10), bg="#e0e0e0")
        self.draw_hatch_label.pack(side=tk.TOP, anchor=tk.W, padx=10, pady=(5, 0))
        self.draw_hatch_scale = ttk.Scale(self.vectorize_tab, from_=1, to=10, value=4)
        self.draw_hatch_scale.pack(side=tk.TOP, fill=tk.X, padx=10)
        self.draw_hatch_value = tk.Label(self.vectorize_tab, text="4", font=("Arial", 10), bg="#e0e0e0")
        self.draw_hatch_value.pack(side=tk.TOP, anchor=tk.W, padx=10)
        
        self.vectorize_apply_btn = ttk.Button(self.vectorize_tab, text="Apply", command=self.process_image)
        self.vectorize_apply_btn.pack(side=tk.TOP, fill=tk.X, padx=10, pady=5)
        self.vectorize_params = {'draw_contours': 2, 'draw_hatch': 4}

    def create_canny_tab(self):
        self.canny_tab = ttk.Frame(self.notebook)
        self.notebook.add(self.canny_tab, text="Canny")
        self.canny_label = tk.Label(self.canny_tab, text="Canny Edge Detection", font=("Arial", 10), bg="#e0e0e0", wraplength=280)
        self.canny_label.pack(side=tk.TOP, padx=10, pady=10, anchor=tk.W)

        self.threshold1_label = tk.Label(self.canny_tab, text="TH1:", font=("Arial", 10), bg="#e0e0e0")
        self.threshold1_label.pack(side=tk.TOP, anchor=tk.W, padx=10, pady=(5, 0))
        self.threshold1_scale = ttk.Scale(self.canny_tab, from_=1.0, to=254, value=200)
        self.threshold1_scale.pack(side=tk.TOP, fill=tk.X, padx=10)
        self.threshold1_value = tk.Label(self.canny_tab, text="200", font=("Arial", 10), bg="#e0e0e0")
        self.threshold1_value.pack(side=tk.TOP, anchor=tk.W, padx=10)

        self.threshold2_label = tk.Label(self.canny_tab, text="TH2:", font=("Arial", 10), bg="#e0e0e0")
        self.threshold2_label.pack(side=tk.TOP, anchor=tk.W, padx=10, pady=(5, 0))
        self.threshold2_scale = ttk.Scale(self.canny_tab, from_=1.0, to=254, value=100)
        self.threshold2_scale.pack(side=tk.TOP, fill=tk.X, padx=10)
        self.threshold2_value = tk.Label(self.canny_tab, text="100", font=("Arial", 10), bg="#e0e0e0")
        self.threshold2_value.pack(side=tk.TOP, anchor=tk.W, padx=10)
        
        self.canny_apply_btn = ttk.Button(self.canny_tab, text="Apply", command=self.process_image)
        self.canny_apply_btn.pack(side=tk.TOP, fill=tk.X, padx=10, pady=5)
        self.canny_params = {'threshold1': 200, 'threshold2': 100}

    def create_xdog_tab(self):
        self.xdog_tab = ttk.Frame(self.notebook)
        self.notebook.add(self.xdog_tab, text="XDoG")

        self.sigma_label = tk.Label(self.xdog_tab, text="Sigma:", font=("Arial", 10), bg="#e0e0e0")
        self.sigma_label.pack(side=tk.TOP, anchor=tk.W, padx=10, pady=(5, 0))
        self.sigma_scale = ttk.Scale(self.xdog_tab, from_=0.1, to=3.0, value=1.0)
        self.sigma_scale.pack(side=tk.TOP, fill=tk.X, padx=10)
        self.sigma_value = tk.Label(self.xdog_tab, text="1.0", font=("Arial", 10), bg="#e0e0e0")
        self.sigma_value.pack(side=tk.TOP, anchor=tk.W, padx=10)

        self.k_label = tk.Label(self.xdog_tab, text="K:", font=("Arial", 10), bg="#e0e0e0")
        self.k_label.pack(side=tk.TOP, anchor=tk.W, padx=10, pady=(5, 0))
        self.k_scale = ttk.Scale(self.xdog_tab, from_=1.0, to=3.0, value=1.6)
        self.k_scale.pack(side=tk.TOP, fill=tk.X, padx=10)
        self.k_value = tk.Label(self.xdog_tab, text="1.6", font=("Arial", 10), bg="#e0e0e0")
        self.k_value.pack(side=tk.TOP, anchor=tk.W, padx=10)

        self.tau_label = tk.Label(self.xdog_tab, text="Tau:", font=("Arial", 10), bg="#e0e0e0")
        self.tau_label.pack(side=tk.TOP, anchor=tk.W, padx=10, pady=(5, 0))
        self.tau_scale = ttk.Scale(self.xdog_tab, from_=0.1, to=0.99, value=0.98)
        self.tau_scale.pack(side=tk.TOP, fill=tk.X, padx=10)
        self.tau_value = tk.Label(self.xdog_tab, text="0.98", font=("Arial", 10), bg="#e0e0e0")
        self.tau_value.pack(side=tk.TOP, anchor=tk.W, padx=10)
        
        self.xdog_apply_btn = ttk.Button(self.xdog_tab, text="Apply", command=self.process_image)
        self.xdog_apply_btn.pack(side=tk.TOP, fill=tk.X, padx=10, pady=5)
        self.xdog_params = {'sigma': 1.0, 'k': 1.6, 'tau': 0.98}

    def update_sketch_params(self, event):
        self.sketch_params['blur_strength'] = int(self.blur_strength_scale.get())
        self.sketch_params['dodge_intensity'] = self.dodge_intensity_scale.get()
        self.blur_strength_value.config(text=str(self.sketch_params['blur_strength']))
        self.dodge_intensity_value.config(text=f"{self.sketch_params['dodge_intensity']:.1f}")

    def update_style(self, event):
        self.selected_style = self.style_combo.get()

    def update_vectorize_params(self, event):
        self.vectorize_params['draw_contours'] = int(self.draw_contours_scale.get())
        self.vectorize_params['draw_hatch'] = int(self.draw_hatch_scale.get())
        self.draw_contours_value.config(text=str(self.vectorize_params['draw_contours']))
        self.draw_hatch_value.config(text=str(self.vectorize_params['draw_hatch']))

    def update_xdog_params(self, event):
        self.xdog_params['sigma'] = self.sigma_scale.get()
        self.xdog_params['k'] = self.k_scale.get()
        self.xdog_params['tau'] = self.tau_scale.get()
        self.sigma_value.config(text=f"{self.xdog_params['sigma']:.1f}")
        self.k_value.config(text=f"{self.xdog_params['k']:.1f}")
        self.tau_value.config(text=f"{self.xdog_params['tau']:.2f}")
    def update_canny_params(self, event):

        self.canny_params['threshold1'] = self.threshold1_scale.get()
        self.canny_params['threshold2'] = self.threshold2_scale.get()
        self.threshold1_value.config(text=f"{self.canny_params['threshold1']:.1f}")
        self.threshold2_value.config(text=f"{self.canny_params['threshold2']:.1f}")

    def load_image(self):
        file_path = filedialog.askopenfilename(
            filetypes=[("Image files", "*.png;*.jpg;*.jpeg;*.bmp")]
        )
        if file_path:
            # Read image with error handling
            try:
                self.original_image = cv2.imread(file_path)
                if self.original_image is None:
                    raise ValueError("Failed to read image file")
                
                # Initialize modified image and UI elements
                self.modified_image = self.original_image.copy()
                self.display_images()
                self.image_history.clear()
                self.image_history.append(self.modified_image.copy())
                self.history_index = 0
                self.undo_button.config(state=tk.NORMAL)
                self.redo_button.config(state=tk.DISABLED)
                # self.status_label.config(text=f"Loaded: {os.path.basename(file_path)}")
                
                # # Enable processing buttons
                # for btn in [self.sketch_btn, self.canny_btn,
                #            self.xdog_btn, self.vectorize_btn]:
                #     btn.config(state=tk.NORMAL)
                    
            except Exception as e:
                messagebox.showerror("Loading Error",
                    f"Failed to load image:\n{str(e)}")
                # self.status_label.config(text="Error loading image")

    def display_images(self):
        if self.original_image is not None:
            # Resize for display while maintaining aspect ratio
            display_img = self.resize_for_display(self.original_image)
            original_image_rgb = cv2.cvtColor(display_img, cv2.COLOR_BGR2RGB)
            original_image_pil = Image.fromarray(original_image_rgb)
            original_image_tk = ImageTk.PhotoImage(original_image_pil)
            
            # Update canvas with scrollable image
            self.original_canvas.config(scrollregion=(0, 0, original_image_tk.width(), original_image_tk.height()))
            self.original_canvas.create_image(0, 0, image=original_image_tk, anchor=tk.NW)
            self.original_canvas.image = original_image_tk

        if self.modified_image is not None:
            # Convert modified image to 3 channels if needed
            if len(self.modified_image.shape) == 2:
                display_img = cv2.cvtColor(self.modified_image, cv2.COLOR_GRAY2BGR)
            else:
                display_img = self.modified_image.copy()
                
            # Resize for display
            display_img = self.resize_for_display(display_img)
            modified_image_rgb = cv2.cvtColor(display_img, cv2.COLOR_BGR2RGB)
            modified_image_pil = Image.fromarray(modified_image_rgb)
            modified_image_tk = ImageTk.PhotoImage(modified_image_pil)
            
            # Update canvas with scrollable image
            self.modified_canvas.config(scrollregion=(0, 0, modified_image_tk.width(), modified_image_tk.height()))
            self.modified_canvas.create_image(0, 0, image=modified_image_tk, anchor=tk.NW)
            self.modified_canvas.image = modified_image_tk

    def resize_for_display(self, img):
        if not hasattr(self, 'original_canvas'):
            return img
            
        canvas_width = self.original_canvas.winfo_width()
        canvas_height = self.original_canvas.winfo_height()
        
        if canvas_width <= 1 or canvas_height <= 1:
            return img
            
        h, w = img.shape[:2]
        ratio = min(canvas_width/w, canvas_height/h)
        new_w = int(w * ratio)
        new_h = int(h * ratio)
        return cv2.resize(img, (new_w, new_h), interpolation=cv2.INTER_AREA)

    def sketch(self, img, blur_strength=21, dodge_intensity=2.5):
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        blur = cv2.GaussianBlur(gray, (blur_strength, blur_strength), 0)
        #  Dodge using true division
        dodge = np.divide(gray, 255 - blur, out=np.zeros_like(gray, dtype=np.float32), where=blur != 255)
        dodge = np.clip(dodge * 255, 0, 255).astype('uint8')
        return dodge

    def apply_style_preset(self, img, style="none"):
        # if style == 'manga':
        #     kernel_size = (3, 3)
        #     dilation_iterations = 2
        #     return cv2.dilate(img, np.ones(kernel_size, np.uint8), iterations=dilation_iterations)
        # elif style == 'watercolor':
        #     sigma_s = 150
        #     sigma_r = 0.3
        #     return cv2.bilateralFilter(img, d=15, sigmaColor=sigma_r * 255, sigmaSpace=sigma_s)
        # elif style == 'blueprint':
        #     return cv2.applyColorMap(img, cv2.COLORMAP_OCEAN)
        if style == "manga":
            gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
            kernel = np.array([[-1, -1, -1], [-1, 9, -1], [-1, -1, -1]])
            return cv2.filter2D(gray, -1, kernel)
        elif style == "watercolor":
            return cv2.bilateralFilter(img, d=9, sigmaColor=150, sigmaSpace=150)
        elif style == "blueprint":
            gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
            return cv2.applyColorMap(gray, cv2.COLORMAP_OCEAN)
        return img

    def vectorize(self, img, draw_contours=2, draw_hatch=4):
        # Very basic vectorization simulation
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        edges = cv2.Canny(gray, 50, 150)  # Use Canny for edge detection
        contours, _ = cv2.findContours(edges, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

        # Create a black background
        vectorized_image = np.zeros_like(img)

        # Draw contours
        cv2.drawContours(vectorized_image, contours, -1, (0, 0, 255), 1)  # Blue contours

        # Simulate hatching with lines
        for i in range(draw_hatch * 10):  # Control density with draw_hatch
            x1, y1 = np.random.randint(0, img.shape[1]), np.random.randint(0, img.shape[0])
            x2, y2 = np.random.randint(0, img.shape[1]), np.random.randint(0, img.shape[0])
            cv2.line(vectorized_image, (x1, y1), (x2, y2), (0, 255, 0), 1)  # Green hatching
        return vectorized_image

    def xdog(self, img, sigma=1.0, k=1.6, tau=0.98):
        sigma = self.xdog_params['sigma']
        k = self.xdog_params['k']
        tau = self.xdog_params['tau']
        print(f"XDoG parameters: sigma={sigma}, k={k}, tau={tau}")
        img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        
        gauss_small = cv2.GaussianBlur(img, (0, 0), sigma)
        gauss_large = cv2.GaussianBlur(img, (0, 0), sigma * k)
        dog = gauss_small - gauss_large
        result = 255 * (1 - np.tanh(tau * dog))
        result = np.clip(result, 0, 255).astype(np.uint8)
        return cv2.cvtColor(result, cv2.COLOR_GRAY2BGR) # Convert back to BGR for display

    def process_image(self):
        if self.original_image is None:
            return

        self.modified_image = self.original_image.copy() # Start with a fresh copy

        if self.notebook.tab(self.notebook.select(), "text") == 'Sketch':
            self.update_sketch_params(None)  # Update parameters from the UI
            self.modified_image = self.sketch(self.modified_image, **self.sketch_params)
        elif self.notebook.tab(self.notebook.select(), "text") == 'Style':
            self.modified_image = self.apply_style_preset(self.modified_image, self.selected_style)
            print(f"Applied style: {self.selected_style}")
        elif self.notebook.tab(self.notebook.select(), "text") == 'Vectorize':
            self.modified_image = self.vectorize(self.modified_image, **self.vectorize_params)
        elif self.notebook.tab(self.notebook.select(), "text") == 'Canny':
            gray = cv2.cvtColor(self.modified_image, cv2.COLOR_BGR2GRAY)
            self.update_canny_params(None)
            self.modified_image = cv2.Canny(gray, **self.canny_params)
            self.modified_image = cv2.cvtColor(self.modified_image, cv2.COLOR_GRAY2BGR)  # Convert back to BGR
        elif self.notebook.tab(self.notebook.select(), "text") == 'XDoG':
            self.update_xdog_params(None) # Update parameters from the UI
            self.modified_image = self.xdog(self.modified_image, **self.xdog_params)

        self.display_images()
        self.save_to_history()

    def save_to_history(self):
        # Only save if the current modified image is different from the last saved image
        if self.history_index == -1 or not np.array_equal(self.modified_image, self.image_history[self.history_index]):
            # If we've undone and are adding a new state, truncate the future history
            if self.history_index < len(self.image_history) - 1:
                self.image_history = self.image_history[:self.history_index + 1]
            self.image_history.append(self.modified_image.copy())
            self.history_index += 1
            self.undo_button.config(state=tk.NORMAL) # Enable undo always when a new state is saved
            self.redo_button.config(state=tk.DISABLED)  # Disable redo when a new state is added

    def undo(self):
        if self.history_index > 0:
            self.history_index -= 1
            self.modified_image = self.image_history[self.history_index].copy()
            self.display_images()
            self.redo_button.config(state=tk.NORMAL)  # Enable redo
            if self.history_index == 0:
                self.undo_button.config(state=tk.DISABLED)
        else:
            self.undo_button.config(state=tk.DISABLED)

    def redo(self):
        if self.history_index < len(self.image_history) - 1:
            self.history_index += 1
            self.modified_image = self.image_history[self.history_index].copy()
            self.display_images()
            self.undo_button.config(state=tk.NORMAL) # keep undo enabled
            if self.history_index == len(self.image_history) - 1:
                self.redo_button.config(state=tk.DISABLED)
        else:
            self.redo_button.config(state=tk.DISABLED)

    def save_image(self):
        if self.modified_image is not None:
            file_path = filedialog.asksaveasfilename(defaultextension=".png", filetypes=[("PNG files", "*.png"), ("JPEG files", "*.jpg;*.jpeg")])
            if file_path:
                try:
                    # Convert from OpenCV BGR to PIL RGB
                    modified_image_rgb = cv2.cvtColor(self.modified_image, cv2.COLOR_BGR2RGB)
                    modified_image_pil = Image.fromarray(modified_image_rgb)
                    modified_image_pil.save(file_path)
                    print(f"Image saved to {file_path}")
                except Exception as e:
                    print(f"Error saving image: {e}")

    def run(self):
        # Main Menu
        menubar = tk.Menu(self.root)
        filemenu = tk.Menu(menubar, tearoff=0)
        filemenu.add_command(label="Open", command=self.load_image)
        filemenu.add_command(label="Save", command=self.save_image)
        filemenu.add_separator()
        filemenu.add_command(label="Exit", command=self.root.quit)
        menubar.add_cascade(label="File", menu=filemenu)

        self.root.config(menu=menubar)
        # Load initial image
        self.load_image() # Removed: Load image on demand.
        self.root.mainloop()

if __name__ == "__main__":
    root = tk.Tk()
    app = SketchGeneratorApp(root)
    app.run()
