# QuantEdge Deployment Guide

This guide provides step-by-step instructions to deploy **QuantEdge** using free, zero-cost platforms suitable for research portfolios, recruiter demonstrations, or local production.

---

## Deployment Options Overview

| Platform | Cost | Setup Time | Best For | Link Type |
| :--- | :--- | :--- | :--- | :--- |
| **Streamlit Community Cloud** *(Recommended)* | **$0.00** (Free forever) | 2 minutes | Resumes, LinkedIn, Recruiter Demos | Public HTTPS URL (`*.streamlit.app`) |
| **Hugging Face Spaces** | **$0.00** (Free 16GB RAM) | 3 minutes | Machine Learning Portfolios | Public HTTPS URL (`hf.space/*`) |
| **Docker (Local or VPS)** | **$0.00** | 1 minute | Private Servers, Self-Hosting | `http://localhost:8501` or Custom Domain |
| **Render.com** | **$0.00** (Free Web Tier) | 4 minutes | Web App Hosting | Public HTTPS URL (`*.onrender.com`) |

---

## Method 1: Streamlit Community Cloud (Recommended & Easiest)

Streamlit Community Cloud is **100% free**, requires **no credit card**, and connects directly to your GitHub repository to generate a live, shareable public HTTPS link.

### Step 1: Push QuantEdge to GitHub

If you have not already pushed your code to GitHub:

1. Open your terminal in the `project 3` directory:
   ```bash
   cd "/Users/yashi/Desktop/project 3"
   ```

2. Create a new repository on [GitHub.com](https://github.com/new):
   - Repository name: `quantedge`
   - Visibility: **Public** (required for free Streamlit Cloud hosting)
   - Do **not** initialize with README or .gitignore (QuantEdge already has them).

3. Link and push your local commits to GitHub:
   ```bash
   git remote add origin https://github.com/<YOUR_GITHUB_USERNAME>/quantedge.git
   git branch -M main
   git push -u origin main
   ```

### Step 2: Deploy on Streamlit Community Cloud

1. Go to **[share.streamlit.io](https://share.streamlit.io/)** and click **"Continue with GitHub"** to sign in.
2. Click the **"New app"** button (top right).
3. Fill in the deployment form:
   - **Repository**: `<YOUR_GITHUB_USERNAME>/quantedge`
   - **Branch**: `main`
   - **Main file path**: `app/main.py`
   - **App URL** (Optional): Choose a custom subdomain (e.g. `quantedge-workstation.streamlit.app`)
4. Click **"Advanced settings..."**:
   - **Python version**: Select **3.12** or **3.11**
   - **Secrets**: Leave empty (QuantEdge operates with zero external paid API keys).
5. Click **"Deploy!"**.

Streamlit Cloud will automatically clone your repository, install dependencies from [`requirements.txt`](file:///Users/yashi/Desktop/project%203/requirements.txt), and launch the terminal in ~2 minutes.

---

## Method 2: Hugging Face Spaces (Free 16 GB RAM)

Hugging Face Spaces provides a free dedicated CPU environment with 2 vCPUs and 16 GB RAM.

1. Go to **[huggingface.co/spaces](https://huggingface.co/spaces)** and log in.
2. Click **"Create new Space"**:
   - **Space name**: `quantedge`
   - **License**: `MIT`
   - **Space SDK**: Select **Streamlit**
   - **Space hardware**: `Free - 2 vCPU · 16 GB`
3. In the repository settings or `README.md`, set:
   ```yaml
   ---
   title: QuantEdge Workstation
   emoji: ⚡
   colorFrom: green
   colorTo: blue
   sdk: streamlit
   sdk_version: 1.43.2
   app_file: app/main.py
   pinned: false
   ---
   ```
4. Push your code to the Hugging Face Git remote:
   ```bash
   git remote add hf https://huggingface.co/spaces/<YOUR_HF_USERNAME>/quantedge
   git push hf main
   ```

---

## Method 3: Docker Deployment (Containerized)

QuantEdge includes a production-ready [`Dockerfile`](file:///Users/yashi/Desktop/project%203/Dockerfile) and [`docker-compose.yml`](file:///Users/yashi/Desktop/project%203/docker-compose.yml).

### Option A: Using Docker Compose
```bash
# Build and start container in the background
docker-compose up -d

# View real-time logs
docker-compose logs -f
```
The workstation is live at: **`http://localhost:8501`**

### Option B: Using Standard Docker CLI
```bash
# Build the Docker image
docker build -t quantedge:latest .

# Run the container
docker run -d \
  -p 8501:8501 \
  --name quantedge \
  -e MPLCONFIGDIR="/tmp/matplotlib" \
  -e HF_HOME="/tmp/huggingface" \
  quantedge:latest
```

---

## Method 4: Local Server / Background Service

To run QuantEdge continuously on your local Mac or an internal Linux server:

```bash
# Launch via the optimized startup script:
./scripts/run_app.sh
```

To run in the background (detached process):
```bash
nohup ./scripts/run_app.sh > quantedge.log 2>&1 &
```
Check running process:
```bash
ps aux | grep streamlit
```
To stop the background process:
```bash
pkill -f "streamlit run app/main.py"
```

---

## Verification & Post-Deployment Checklist

After deployment launches:
1. Open the public URL in your browser.
2. Verify that **Market Overview** loads and charts render with the dark navy terminal theme.
3. Switch to **Econometrics** and click *"Estimate Conditional Volatility"* to verify that GARCH(1,1) estimates properly.
4. Switch to **Backtesting Studio** and click *"Run Backtest"* to confirm the simulation runs smoothly without errors.
