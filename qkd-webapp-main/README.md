# QKD Web Application

A web-based implementation of Quantum Key Distribution protocols.

## Setup

1. **Clone the repository**
   ```bash
   git clone <repository-url>
   cd qkd_webapp
   ```

2. **Create and activate a virtual environment**
   ```bash
   python -m venv venv
   source venv/bin/activate  # On Windows: .\venv\Scripts\activate
   ```

3. **Install dependencies**
   ```bash
   pip install -r requirements.txt
   ```

4. **Set up environment variables**
   - Copy `.env.example` to `.env`:
     ```bash
     cp .env.example .env
     ```
   - Edit `.env` and add your API keys:
     ```
     SUPERSTAQ_API_KEY=your_superstaq_api_key_here
     ```

5. **Run the application**
   ```bash
   uvicorn qkd_webapp.backend.socketio_app:app --reload
   ```

6. **Open in browser**
   - Visit `http://localhost:8000`

