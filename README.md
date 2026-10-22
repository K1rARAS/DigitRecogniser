# DigitRecogniser

Created by a student team for educational purposes.  
We’re just learning how to use GitHub.

This program recognizes digits in 28×28 pixel images.  
The dataset was collected manually and is included in the `nums_book.zip` file.  
Don’t expect miracles.

> ⚠️ The program was built with PyInstaller; antivirus software might flag it as suspicious.

## Getting Started

1. Put the program in a folder, for example, `test`.
2. Run the program — it will create the `ImagesForAI` folder.
3. Save a PNG image (28×28 pixels) into that folder.
4. In the terminal that opens, you must select an image from the `ImagesForAI` folder (option 1).  
   The selection is made by entering the index from the displayed list.
5. Use the menu and enter `2` to proceed—the neural network will try to recognize your image.

---

## Training & Fine‑Tuning

- Fine‑tuning functionality is not available yet (I’ll try to add it later).
- To train the neural network:
  1. Create a `train` folder.
  2. Place images in it.
  3. Run the `TnCeAI` script.
  4. Adjust the training parameters in that script as needed.
- To run from IDLE, launch `main.py`.

---

## Building the .exe

To build your own `.exe` file, install PyInstaller and run:

pyinstaller --onefile --name=DigitRecogniser --add-data "modelAIV1.pkl;." --icon "t3.ico" main.py

## Requirements
To run the program from IDLE, you need the following libraries:
pillow (for image processing).
Other libraries should be built into Python.
Learning GitHub to Enable Feedback
Python 3.13

We’re still getting comfortable with GitHub and welcome any feedback to improve the project.
