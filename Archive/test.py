import customtkinter as ctk
from tkinter import filedialog, messagebox
import cv2
import os

class UserDetailsApp(ctk.CTk):
    def __init__(self):
        super().__init__()

        self.title("User Details Input")
        self.geometry("600x700")

        # Create widgets
        self.create_widgets()

    def create_widgets(self):
        # Username input
        ctk.CTkLabel(self, text="Username:", anchor="w").pack(fill='x', padx=20, pady=5)
        self.entry_username = ctk.CTkEntry(self)
        self.entry_username.pack(fill='x', padx=20, pady=5)

        # User ID input
        ctk.CTkLabel(self, text="User ID:", anchor="w").pack(fill='x', padx=20, pady=5)
        self.entry_user_id = ctk.CTkEntry(self)
        self.entry_user_id.pack(fill='x', padx=20, pady=5)

        # Image selection buttons
        ctk.CTkLabel(self, text="Select Image:", anchor="w").pack(fill='x', padx=20, pady=5)

        button_frame = ctk.CTkFrame(self)
        button_frame.pack(fill='x', padx=20, pady=5)

        ctk.CTkButton(button_frame, text="Select Image from Folder", command=self.select_image).pack(side="left", padx=5, pady=5)
        ctk.CTkButton(button_frame, text="Capture Images from Webcam", command=self.capture_images).pack(side="left", padx=5, pady=5)

        self.label_image_path = ctk.CTkLabel(self, text="")
        self.label_image_path.pack(fill='x', padx=20, pady=5)

        # Save button
        ctk.CTkButton(self, text="Save Details", command=self.save_details).pack(pady=20)

    def capture_images(self):
        cam = cv2.VideoCapture(0)
        cv2.namedWindow("Webcam Capture - Follow Instructions", cv2.WINDOW_NORMAL)
        cv2.resizeWindow("Webcam Capture - Follow Instructions", 800, 600)

        instructions = [
            "Center your face",
            "Move to the left",
            "Move to the right",
            "Move up",
            "Move down",
            "Move top-left",
            "Move top-right",
            "Move bottom-left",
            "Move bottom-right",
            "Center again"
        ]

        img_names = []
        for idx, instruction in enumerate(instructions):
            ret, frame = cam.read()
            if not ret:
                print("Failed to grab frame")
                break
            cv2.putText(frame, instruction, (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2, cv2.LINE_AA)
            cv2.imshow("Webcam Capture - Follow Instructions", frame)
            cv2.waitKey(2000)  # Display each instruction for 2 seconds
            
            ret, frame = cam.read()
            img_name = f"captured_image_{idx+1}.png"
            cv2.imwrite(img_name, frame)
            img_names.append(img_name)
            cv2.waitKey(500)  # Wait 0.5 seconds before next capture

        cam.release()
        cv2.destroyAllWindows()

        self.label_image_path.configure(text=", ".join(img_names))

    def select_image(self):
        file_path = filedialog.askopenfilename(filetypes=[("Image files", "*.jpg *.jpeg *.png")])
        self.label_image_path.configure(text=file_path)

    def save_details(self):
        username = self.entry_username.get()
        user_id = self.entry_user_id.get()
        image_paths = self.label_image_path.cget("text").split(", ")

        if not username or not user_id or not image_paths:
            messagebox.showwarning("Input Error", "Please fill all the fields and select an image.")
            return

        # Create a directory for the user
        user_dir = f"user_data/{user_id}_{username}"
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
                img_ext = os.path.splitext(image_path)[1]
                new_image_path = os.path.join(user_dir, os.path.basename(image_path))
                os.rename(image_path, new_image_path)

        messagebox.showinfo("Success", "Details saved successfully!")

if __name__ == "__main__":
    app = UserDetailsApp()
    app.mainloop()
