import customtkinter as ctk
from tkinter import filedialog, messagebox
from PIL import Image, ImageTk
import cv2
import os
import time

class UserDetailsApp(ctk.CTk):
    def __init__(self):
        super().__init__()

        self.title("User Details Input")
        self.geometry("800x900")
        ctk.set_appearance_mode("dark")  # Modes: system (default), light, dark
        ctk.set_default_color_theme("blue")  # Themes: blue (default), dark-blue, green

        # Initialize image paths and photos
        self.image_paths = []
        self.photos = []

        # Create widgets
        self.create_widgets()

    def create_widgets(self):
        # Title Label
        title_label = ctk.CTkLabel(self, text="User Registration Page", font=("Arial", 24))
        title_label.pack(pady=20)

        # Username input
        username_frame = ctk.CTkFrame(self)
        username_frame.pack(fill='x', padx=20, pady=10)
        ctk.CTkLabel(username_frame, text="Username:", anchor="w").pack(side='left', padx=10)
        self.entry_username = ctk.CTkEntry(username_frame)
        self.entry_username.pack(fill='x', padx=10, pady=5, expand=True)

        # User ID input
        user_id_frame = ctk.CTkFrame(self)
        user_id_frame.pack(fill='x', padx=20, pady=10)
        ctk.CTkLabel(user_id_frame, text="User ID:", anchor="w").pack(side='left', padx=10)
        self.entry_user_id = ctk.CTkEntry(user_id_frame)
        self.entry_user_id.pack(fill='x', padx=10, pady=5, expand=True)

        # Image selection buttons
        image_label = ctk.CTkLabel(self, text="Select Image:", anchor="w")
        image_label.pack(fill='x', padx=20, pady=10)

        button_frame = ctk.CTkFrame(self)
        button_frame.pack(fill='x', padx=20, pady=10)
        ctk.CTkButton(button_frame, text="Select Image from Folder", command=self.select_image).pack(side="left", padx=10, pady=5)
        ctk.CTkButton(button_frame, text="Capture Images from Webcam", command=self.capture_images).pack(side="left", padx=10, pady=5)

        self.label_image_path = ctk.CTkLabel(self, text="", anchor="w")
        self.label_image_path.pack(fill='x', padx=20, pady=10)

        self.image_display_frame = ctk.CTkFrame(self)
        self.image_display_frame.pack(pady=20)

        # Save and Redo buttons
        button_frame2 = ctk.CTkFrame(self)
        button_frame2.pack(fill='x', padx=20, pady=10)
        ctk.CTkButton(button_frame2, text="Save Details", command=self.save_details).pack(side="left", padx=10, pady=5)
        ctk.CTkButton(button_frame2, text="Redo", command=self.redo).pack(side="left", padx=10, pady=5)

    def update_image_display(self):
        # Clear previous images
        for widget in self.image_display_frame.winfo_children():
            widget.destroy()

        self.photos = []

        # Create a scrollable frame to hold the images
        canvas = ctk.CTkCanvas(self.image_display_frame, width=700, height=300)
        scrollbar = ctk.CTkScrollbar(self.image_display_frame, orientation="horizontal", command=canvas.xview)
        scrollable_frame = ctk.CTkFrame(canvas)

        scrollable_frame.bind(
            "<Configure>",
            lambda e: canvas.configure(
                scrollregion=canvas.bbox("all")
            )
        )

        canvas.create_window((0, 0), window=scrollable_frame, anchor="nw")
        canvas.configure(xscrollcommand=scrollbar.set)

        canvas.pack(side="top", fill="both", expand=True)
        scrollbar.pack(side="bottom", fill="x")

        for image_path in self.image_paths:
            img = Image.open(image_path)
            img.thumbnail((150, 150))
            photo = ImageTk.PhotoImage(img)
            self.photos.append(photo)
            img_label = ctk.CTkLabel(scrollable_frame, image=photo)
            img_label.pack(side="left", padx=10, pady=5)


    def capture_images(self):
        cam = cv2.VideoCapture(0)
        cv2.namedWindow("Webcam Capture - Follow Instructions", cv2.WINDOW_NORMAL)
        cv2.resizeWindow("Webcam Capture - Follow Instructions", 800, 600)

        instructions = [
            "Center your face"
        ]

        img_names = []
        face_cascade = cv2.CascadeClassifier("model/haarcascade_frontalface_default.xml")
        images_per_instruction = 40

        for idx, instruction in enumerate(instructions):
            captured_images = 0
            while captured_images < images_per_instruction:
                ret, frame = cam.read()
                if not ret:
                    print("Failed to grab frame")
                    break

                # Display instruction
                display_frame = frame.copy()
                cv2.putText(display_frame, instruction, (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2, cv2.LINE_AA)
                cv2.imshow("Webcam Capture - Follow Instructions", display_frame)

                # Convert frame to grayscale for face detection
                gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
                faces = face_cascade.detectMultiScale(
                    gray,
                    scaleFactor=1.2,
                    minNeighbors=10,
                    minSize=(20, 20)
                )

                # Save detected faces
                for (x, y, w, h) in faces:
                    if captured_images < images_per_instruction:
                        img_name = f"{self.entry_user_id.get()}_{idx + 1}_{captured_images + 1}.png"
                        cv2.rectangle(frame, (x, y), (x+w, y+h), (255, 0, 0), 2)
                        cv2.imwrite(img_name, gray[y:y+h, x:x+w])
                        img_names.append(img_name)
                        captured_images += 1

                        # Show the frame with bounding box
                        cv2.imshow("Webcam Capture - Follow Instructions", frame)

                        # Check if we have captured the desired number of images
                        if captured_images >= images_per_instruction:
                            break

                if cv2.waitKey(1) & 0xFF == ord('q'):
                    break

            if not ret:
                break

        cam.release()
        cv2.destroyAllWindows()

        if img_names:
            self.image_paths = img_names
            self.update_image_display()
            self.label_image_path.configure(text=", ".join(img_names))



    def select_image(self):
        # Schedule the file dialog to be opened on the main GUI thread
        self.after(0, self.open_file_dialog)

    def open_file_dialog(self):
        file_path = filedialog.askopenfilenames(filetypes=[("Image files", "*.jpg *.jpeg *.png")])
        if file_path:
            self.image_paths = list(file_path)
            self.update_image_display()
            self.label_image_path.configure(text=", ".join(file_path))

    def redo(self):
        self.entry_username.delete(0, 'end')
        self.entry_user_id.delete(0, 'end')
        self.image_paths = []
        self.label_image_path.configure(text="")
        self.update_image_display()

    def save_details(self):
        username = self.entry_username.get()
        user_id = self.entry_user_id.get()
        image_paths = self.image_paths

        if not username or not user_id or not image_paths:
            messagebox.showwarning("Input Error", "Please fill all the fields and select an image.")
            return

        # Create a directory for the user
        user_dir = f"user_data/{user_id}"
        os.makedirs(user_dir, exist_ok=True)

        # Save the details in a text file
        details_file = os.path.join(user_dir, "details.txt")
        with open(details_file, "w") as f:
            f.write(f"Username: {username}\n")
            f.write(f"User ID: {user_id}\n")
            f.write(f"Image Paths: {', '.join(image_paths)}\n")

        # Save the images in the user directory
        for image_path in image_paths:
            if os.path.exists(image_path):
                new_image_path = os.path.join(user_dir, os.path.basename(image_path))
                os.rename(image_path, new_image_path)

        messagebox.showinfo("Success", "Details saved successfully!")

if __name__ == "__main__":
    app = UserDetailsApp()
    app.mainloop()
