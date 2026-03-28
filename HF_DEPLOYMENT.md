# Deploying to Hugging Face Spaces 🤗

Hugging Face Spaces is an incredible platform for hosting Machine Learning apps. Since we built a custom, beautiful frontend using real HTML/CSS instead of basic Gradio/Streamlit, we will use a **Docker Space**. This allows us to run our exact Flask web server and UI on Hugging Face exactly as it looks right now!

## Step 1: Push the Dockerfile to GitHub
I have automatically created a `Dockerfile` in your project folder that is perfectly tuned for Hugging Face.

First, let's push this new configuration up to your GitHub:
1. Open your terminal in the project folder.
2. Run these commands:
   ```bash
   git add Dockerfile
   git commit -m "Add Docker deployment configuration for Hugging Face"
   git push origin main
   ```

## Step 2: Create your Space on Hugging Face
1. Go to [Hugging Face Spaces](https://huggingface.co/spaces) and create a free account if needed.
2. Click the **"Create new Space"** button in the top right.
3. Fill out the form:
   - **Space name:** `house-price-predictor-pro` (or anything you like)
   - **License:** `MIT` (optional)
   - **Select the Space SDK:** Choose **Docker** 🐳
   - **Docker template:** Choose **Blank**
4. Click **Create Space** at the bottom.

## Step 3: Link to GitHub
Hugging Face handles git directly, but there is an even easier way. We'll simply mirror your GitHub code into the Hugging Face Space repository.

1. On your new empty Space page, you will see a command to clone the space (e.g., `git clone https://huggingface.co/spaces/YOUR_USER/house-price-predictor-pro`).
2. Alternatively, you can directly pull your GitHub code into the Space by going to the **Settings** tab of your Space.
3. Scroll down to **GitHub repo syncing** (or "Action Sync"). Follow the simple prompt to connect your Space to `kit2824bcs005-cell/House-price-prediction` on GitHub.
4. Click Sync!

Hugging Face will detect the `Dockerfile` I generated, automatically install the ML libraries, launch your Flask server on Port `7860` (their default internal port), and securely deliver your dark-theme UI to the browser via HTTPS!
