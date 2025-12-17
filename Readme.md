# 🚀 1. Install Python 3.12.2 (Required)

The project requires **Python 3.12.2**, because MATLAB Engine is only compatible with Python versions **3.9–3.12**.

### **Step 1 — Download Python 3.12.2 (64-bit)**

Download from the official Python website:

🔗 [https://www.python.org/downloads/release/python-3122/](https://www.python.org/downloads/release/python-3122/)

### **Step 2 — During installation, make sure to:**

- ✔ Check **Add python.exe to PATH**
- ✔ Choose **Install for all users**
- ✔ Use the default installation location:

```
C:\Program Files\Python312\
```

### **Step 3 — Verify installation**

Open **Command Prompt** and type:

```cmd
python --version
```

You should see:

```
Python 3.12.2
```

---

# 🧪 2. Verify MATLAB Installation

MATLAB Engine requires MATLAB to be installed locally.

Open **Command Prompt** and type:

```cmd
matlab
```

If MATLAB opens, you're good.

---

# 🛠️ 3. Create a Virtual Environment (Python 3.12.2)

### **Step 1 — Open Command Prompt**

Press **Win + R → type `cmd` → Enter**

### **Step 2 — Navigate to your project folder**

Example:

```cmd
cd C:\Users\YourName\Documents\your-project-folder
```

### **Step 3 — Create a virtual environment named `venv`**

```cmd
py -3.12 -m venv venv
```

### **Step 4 — Activate the virtual environment**

```cmd
venv\Scripts\activate
```

You will now see:

```
(venv) C:\Users\YourName\your-project-folder>
```

---

# 📦 4. Install MATLAB Engine for Python

⚠️ **Important:** This must be done _inside_ the virtual environment.

### **Step 1 — Go to the MATLAB Engine installer folder**

```cmd
cd "C:\Program Files\MATLAB\<YOUR_VERSION>\extern\engines\python"
```

Example:

```cmd
cd "C:\Program Files\MATLAB\R2025b\extern\engines\python"
```

### **Step 2 — Install MATLAB Engine**

```cmd
py -3.12 setup.py install
```

If installation succeeds, you can now use:

```python
import matlab.engine
```

---

# 📦 5. Install Python Dependencies

Make sure your virtual environment is active (`(venv)` must be visible).

Run:

```cmd
pip install -r requirements.txt
```

This installs all required Python packages for the project.

---

# 🧪 6. Test MATLAB Engine

Inside the activated virtual environment:

```cmd
python
```

Then run:

```python
import matlab.engine
eng = matlab.engine.start_matlab()
print(eng.sqrt(25))
eng.quit()
```

Expected output:

```
5.0
```

Exit Python:

Press **Ctrl + Z**, then Enter.

---

# 🗑️ 7. Deactivate or Delete the Virtual Environment

### **Stop using the environment**

```cmd
deactivate
```

### **Delete the environment completely**

```cmd
rmdir /s /q venv
```

(or delete the folder manually)

---

# 🎉 8. Run the Project

With the virtual environment active:

```cmd
python main.py
```

Or launch Jupyter:

```cmd
pip install notebook
jupyter notebook
```

---

# 🛠️ Troubleshooting

### ❌ _Python 3.12.2 not found?_

You may be using the **Microsoft Store version of Python**, which is incompatible.
Uninstall it and install Python from python.org.

### ❌ _MATLAB Engine install fails with “Access Denied”_

Run Command Prompt **as Administrator**.

### ❌ _Cannot import matlab.engine_

You installed MATLAB Engine **outside** the virtual environment.
Delete `venv`, recreate it, reinstall MATLAB Engine.

---

# 📘 Summary (Quick Start)

```
1. Install Python 3.12.2
2. git clone <this-repo>
3. cd project-folder
4. py -3.12 -m venv venv
5. venv\Scripts\activate
6. cd "C:\Program Files\MATLAB\<VERSION>\extern\engines\python"
7. py -3.12 setup.py install
8. cd back to project
9. pip install -r requirements.txt
10. python main.py
```

---
