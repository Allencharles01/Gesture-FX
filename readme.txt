🖐️ Gesture FX Panel

An interactive Streamlit app that uses **MediaPipe** and **OpenCV** to trigger UI effects (balloons, snow, themes) via real-time hand gestures.

🛠 Prerequisites

Before cloning, ensure you have **Python (3.8 - 3.11)** installed. MediaPipe can be unstable on Python 3.12+.

*How to check if you have Python:*

Open your terminal (Command Prompt/PowerShell on Windows, Terminal on Mac/Linux) and type:

"python --version"

If it returns `Python 3.x.x`, you’re good. If not, download it from [python.org](https://www.python.org/).


*System-Specific Setup:*

* **Windows:** Ensure "Add Python to PATH" was checked during installation.
* **macOS:** You may need to grant "Camera Access" to your terminal in System Settings.
* **Linux (Fedora):** You might need to install OpenCV dependencies:

"sudo dnf install opencv-devel"


🚀 How to Run

1. Clone & Enter Folder:
    
    "git clone <your-repo-link>"
    "cd Gesture_FX_Project"

2. Install Libraries:   "pip install streamlit opencv-python mediapipe"

3. Launch:  "streamlit run gesturefx.py" or "streamlit run gesturefx_web.py"



⚠️ Troubleshooting

| Problem | Solution |
| --- | --- |
| **Black Camera Feed** | Change `cv2.VideoCapture(0)` to `(1)` in the code. |
| **MediaPipe Install Fail** | Ensure you aren't using Python 3.12+; downgrade to 3.10. |
| **Laggy Detection** | Improve room lighting or reduce webcam resolution. |
| **Missing DLL (Windows)** | Install the [Visual C++ Redistributable](https://learn.microsoft.com/en-us/cpp/windows/latest-supported-vc-redist). |



🤝 Connect with Me

I'm a Hyderabad-based Full-Stack Developer building AI-powered tools. Let's talk tech!

* GitHub:       [Allencharles01]    (https://github.com/Allencharles01)
* LinkedIn:     [allencharles01]    (https://www.linkedin.com/in/allencharles01/)
* Portfolio:    [allencharles.dev]  (https://allencharles.dev)

