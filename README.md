# 🚀 [Tips Hindawi](https://www.tipshindawi.com/) Internship (August–October) 2026

> 🎓 This project was built during the [ **Tips Hindawi** ](https://www.tipshindawi.com/) **Internship (August–October) 2026**.

## 👤 Participant

| Field            | Value                                |
| ---------------- | ------------------------------------ |
| Full Name        | Shiraz Nasser                       |
| Project Name     | AI Meeting Intelligence              |
| GitHub Username  | shiraznasser                 |
| Internship Batch | August–October 2026                  |
| Training Program | Large Language Models (LLMs) Program |
| Organization     | [**Edrak for Ai**](https://edrak4ai.com/en)                         |

---

# 📖 Project Overview

**AI Meeting Intelligence** is a Streamlit web app that turns a raw meeting transcript into a clear, structured report. The user uploads a transcript (`.txt` or `.pdf`) or pastes the text, and the app returns a summary, key points, decisions, and action items (task, owner, deadline) as clean JSON. A chat tab then lets the user ask follow-up questions about the meeting.

The project combines the main techniques taught in the course in one pipeline: **transcript → chunking → embeddings → FAISS retrieval → 3 LLM chains → structured JSON**, plus a retrieval chat with memory.

---

# ✨ Features

* **Automatic summary and key points** of the whole meeting.
* **Decision extraction**: lists only what was explicitly agreed.
* **Action-item extraction** with task, owner, and deadline ("Not specified" when missing).
* **Ask the meeting**: a chat grounded in the transcript, with follow-up questions that work thanks to conversation memory.
* **Flexible input**: upload `.txt` / `.pdf`, paste text, or use the built-in sample meeting.
* **Download results** as a JSON file.

---

# 🛠️ Technologies Used

| Course technique | How it is used |
| --- | --- |
| **RAG** | Transcript is split into chunks, embedded with `all-MiniLM-L6-v2`, stored in **FAISS**, and retrieved for each task |
| **Chains** | Three prompt chains run in sequence: summary → decisions → action items |
| **Output Parser** | LangChain `StructuredOutputParser` with `ResponseSchema` returns clean JSON |
| **Memory** | The last turns of the Q&A chat are fed back into the prompt |
| **Prompt Engineering** | Grounded prompts ("use ONLY the excerpts") with explicit format instructions |

**Stack:** Python, [Streamlit](https://streamlit.io/), LangChain, Hugging Face Transformers, `mistralai/Mistral-7B-Instruct-v0.2` (4-bit quantized with bitsandbytes), Sentence-Transformers, FAISS, PyPDF, ngrok (for public access on Kaggle/Colab).

---

# ⚙️ Installation

### Option A: Kaggle / Colab (recommended)

1. Import `Meeting_Intelligence_Kaggle.ipynb` into Kaggle (or Colab).
2. Set the accelerator to **GPU** and turn **Internet** on.
3. Add your ngrok **authtoken** as a secret named `NGROK_TOKEN` (or paste it in Cell 5).
4. Run the cells in order. The notebook installs dependencies, writes the app files, and prints a public URL.

### Option B: Run locally (GPU required)

```bash
git clone https://github.com/YOUR_GITHUB_USERNAME/YOUR_REPO.git
cd YOUR_REPO
pip install -r requirements.txt
streamlit run app.py
```

To use a smaller model, set an environment variable before launching:

```bash
export MEETING_MODEL="Qwen/Qwen2.5-3B-Instruct"
```

---

# 🚀 Usage

1. Open the app URL.
2. Choose a transcript source: **Use sample**, **Upload file (.txt / .pdf)**, or **Paste text**.
3. Click **Analyze meeting**. The app indexes the transcript and runs the three chains.
4. Explore the tabs:
   * **Summary**: summary and key points.
   * **Decisions & Tasks**: agreed decisions and an action-item table.
   * **Ask the meeting**: ask questions such as *"Who is responsible for the payment sandbox?"* and follow-ups like *"And what's the deadline for that?"*
   * **JSON**: the full structured result with a download button.
5. Tune **k** (chunks retrieved) and **chunk size** from the sidebar for longer transcripts.

**Project files**

| File | Purpose |
| --- | --- |
| `app.py` | Streamlit app (pipeline and UI) |
| `requirements.txt` | Python dependencies |
| `sample_meeting.txt` | Example transcript for testing |
| `Meeting_Intelligence_Kaggle.ipynb` | Notebook that installs, writes the files, and launches the app with ngrok |

---

# 📸 Demo

<img width="3840" height="1844" alt="image" src="https://github.com/user-attachments/assets/27222b0e-6a04-49da-9c3f-86b5553b4ad5" />
<img width="3840" height="1843" alt="image" src="https://github.com/user-attachments/assets/2bf48959-80ce-45e7-a106-c7c92e64c03a" />


* <img width="3840" height="1858" alt="image" src="https://github.com/user-attachments/assets/d037d77b-eafb-4aca-aa2c-92272f80ab06" />
: Summary tab
* <img width="3840" height="1834" alt="image" src="https://github.com/user-attachments/assets/2e122e6e-a4c1-48e0-895f-34ddd0cd1ff2" />
: Decisions & Tasks tab
* <img width="3840" height="1837" alt="image" src="https://github.com/user-attachments/assets/d2aa511e-6616-4260-91bd-72ba8d11b724" />
: Ask the meeting tab
* <img width="3811" height="1800" alt="image" src="https://github.com/user-attachments/assets/28a7123a-2b97-44cf-9a14-4f0da2d68ded" />
: JSON download

---

# 📈 Results

* Converts an unstructured transcript into a structured report (summary, key points, decisions, action items) in one click.
* On the included sample meeting, the app extracts the agreed decisions (feature freeze, launch moved to May 4) and the action items with their owners and deadlines.
* Answers to follow-up questions stay grounded in the transcript, and the model says so when the answer is not in the meeting.
* Combines RAG, chains, output parsing, memory, and prompt engineering in a single deployable Streamlit app.

---

# 🔮 Future Improvements

* Support audio input with speech-to-text (e.g. Whisper) so recordings can be analyzed directly.
* Add speaker-aware analysis and per-person task views.
* Export reports to PDF/Word and sync action items to tools like Trello or Google Calendar.
* Support Arabic and multilingual meetings.
* Add automatic evaluation of extraction accuracy on a labeled set of transcripts.

---

# 📚 About the Internship

This project was developed as part of the [**Tips Hindawi**](https://www.tipshindawi.com/) **Internship (August–October) 2026**, and it will be showcased on the official [Tips Hindawi](https://www.tipshindawi.com/) website.

[Tips Hindawi](https://www.tipshindawi.com/) is the internships department of [**Edrak for Ai**](https://edrak4ai.com/en), and the internship encourages participants to build real-world projects, apply practical skills, and showcase their work through GitHub.

For more information about the internship, training programs, and upcoming batches, visit the official [Tips Hindawi](https://www.tipshindawi.com/) website.

---

# 📄 License

This project is shared for educational and portfolio purposes.
