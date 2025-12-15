AI-Powered Regulatory Compliance Checker for Contracts
🎯 Project Overview

This project automates the detection of regulatory compliance issues in contracts using AI. It extracts text from uploaded contracts, identifies key clauses, matches them against regulations, and highlights potential risks or violations. The system can also suggest amendments for non-compliant sections.

Key Features:

Upload and review contracts in PDF format.

Detects high-risk areas: missing clauses, conflicts, ambiguities.

Matches contract clauses with regulatory clauses using AI.

Generates recommended amendments or rewritten clauses.

Sends notifications via a built-in notifier module.

Agile documentation included for development transparency.

🛠️ Tech Stack

Backend: Python 3.7

Frontend: Streamlit (web interface)

AI & ML: LLM reasoning, RAG retrieval, vector similarity search

Version Control: Git & GitHub

Additional Tools: FAISS, Pandas, OpenAI API (for embeddings), Email/Notifier integration

🏗️ System Architecture

Upload Module: Accepts PDF contracts and extracts text.

Clause Detection Module: Identifies key clauses like Termination, Indemnification, Confidentiality, Data Protection, and Liability.

Compliance Matching: Uses AI to compare clauses against regulatory standards.

Risk Analysis: Highlights violations, missing clauses, and ambiguous terms.

Amendment Generator: Suggests compliant rewrites for risky clauses.

Notifier Module: Sends alerts for high-risk contracts.

(Optional: Add a diagram image here for visual clarity.)

⚡ How to Run Locally

Clone the repository:

git clone https://github.com/Sonam-5nayak/AI-Powered-Regulatory-Compliance-Checker-for-Contracts.git
cd AI-Powered-Regulatory-Compliance-Checker-for-Contracts


Install dependencies:

pip install -r requirements.txt


Run the app:

streamlit run app.py


Upload a contract and review compliance results on the web interface.

📁 Project Structure

AI-Powered-Regulatory-Compliance-Checker-for-Contracts/
│
├── app.py               # Streamlit web interface
├── main.py              # Core application logic
├── notifier.py          # Email/notification module
├── requirements.txt     # Python dependencies
├── .gitignore           # Git ignore rules
├── agile_doc.xlsx       # Agile sprint and documentation
└── README.md            # Project documentation
