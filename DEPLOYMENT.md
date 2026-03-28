# Deploying House Price Predictor Pro 🚀

This guide provides a step-by-step walkthrough for deploying your full-stack ML application completely for free using **Render** (a modern cloud provider). 

To keep things simple, we'll deploy the Backend (Flask) and Frontend (HTML/JS) simultaneously as a single Web Service.

---

## Step 1: Prepare Your Project for Production

Before uploading to the internet, we need a WSGI production server (like `gunicorn`) instead of the built-in Flask development server.

1. **Add gunicorn to requirements:**
   Open `backend/requirements.txt` and add this line to the bottom:
   ```text
   gunicorn==21.2.0
   ```
2. **Configure Flask to serve the frontend:**
   By default, Flask only serves API endpoints. We need to tell it to render your `index.html`. 
   Add this route to the bottom of `backend/app.py`:
   ```python
   @app.route("/")
   def serve_frontend():
       return send_from_directory("../frontend", "index.html")
       
   @app.route("/<path:filename>")
   def serve_static(filename):
       return send_from_directory("../frontend", filename)
   ```

3. **Initialize Git:**
   Open your terminal in the project folder (`house-price-predictor-pro`) and run:
   ```bash
   git init
   git add .
   git commit -m "Initial commit for production"
   ```

---

## Step 2: Push to GitHub

Render pulls your code directly from GitHub, so we need to upload it there.

1. Create a free account on [GitHub](https://github.com/).
2. Create a **New Repository** (name it `house-price-predictor-pro`). Leave it public or private.
3. Link your local project to GitHub by running these commands in your terminal (replace `USERNAME` with yours):
   ```bash
   git remote add origin https://github.com/USERNAME/house-price-predictor-pro.git
   git branch -M main
   git push -u origin main
   ```

---

## Step 3: Deploy to Render 

1. Create a free account on [Render](https://render.com/).
2. Click **New** -> **Web Service**.
3. Connect your GitHub account and select your `house-price-predictor-pro` repository.
4. Fill out the configuration form:
   - **Name:** `house-price-predictor`
   - **Environment:** `Python 3`
   - **Region:** Pick whatever is closest to you.
   - **Build Command:** 
     ```bash
     pip install -r backend/requirements.txt
     ```
   - **Start Command:** 
     ```bash
     cd backend && gunicorn app:app --bind 0.0.0.0:$PORT
     ```
5. Choose the **Free Instance Type**.
6. Click **Create Web Service**.

> **Note:** Render will now download your code, install the pandas/scikit-learn libraries, and start your ML server. This usually takes 3 to 5 minutes.

---

## Step 4: Update the Frontend API URL

When you deploy, your API will no longer be at `http://localhost:5000`. It will be at your new Render domain (e.g., `https://house-price-predictor.onrender.com`).

1. Open `frontend/script.js`.
2. Find line 13:
   ```javascript
   const API_BASE = "http://localhost:5000";
   ```
3. Change it to use the current domain dynamically:
   ```javascript
   const API_BASE = window.location.origin;
   ```
4. Push this change to GitHub:
   ```bash
   git commit -am "Update API URL for production"
   git push origin main
   ```

Render will automatically detect the new commit, rebuild your server, and boom! Your ML property valuation tool is live on the internet! 🎉

---

### ⚠️ Important Production Notes
- **Cold Booting:** On Render's free tier, your server will go to "sleep" after 15 minutes of inactivity. When someone visits the link, it might take 30–50 seconds to wake up.
- **Joblib Size:** The `model.pkl` file is roughly 1-2MB. GitHub natively handles files up to 100MB, so you won't need Git LFS.
