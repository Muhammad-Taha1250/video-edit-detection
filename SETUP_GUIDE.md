# Video Edit Detection — Full Setup Guide

This project now has **two parts**:

1. **Frontend** (React + Vite) — in the root folder
2. **Backend API** (Node.js + Express + MySQL) — in the `server` folder

The frontend talks to the backend over HTTP (`http://localhost:4000`), and the backend stores everything in a real MySQL database.

---

## Prerequisites

- Node.js installed (you already have this ✅)
- MySQL installed and running locally

If you don't have MySQL yet, install it:
- macOS: `brew install mysql` then `brew services start mysql`
- Or download MySQL Community Server from https://dev.mysql.com/downloads/mysql/

---

## Step 1 — Set up the database

Open Terminal and log into MySQL:

```bash
mysql -u root -p
```

Then run the schema file (creates the database, tables, and demo data):

```bash
mysql -u root -p < server/schema.sql
```

This creates a database called `video_edit_detection` with 4 tables (`users`, `videos`, `analysis_results`, `detected_frames`) and pre-loads **6 demo videos** with realistic frame/accuracy data — so the app looks populated immediately.

---

## Step 2 — Configure and start the backend

```bash
cd server
npm install
```

Copy the example env file and put in your real MySQL password:

```bash
cp .env.example .env
```

Open `.env` in VS Code and set `DB_PASSWORD` to your MySQL root password (leave blank if you have no password).

Start the API server:

```bash
npm start
```

You should see:
```
✅ API server running at http://localhost:4000
```

Leave this terminal running.

---

## Step 3 — Start the frontend

Open a **second** Terminal tab/window:

```bash
cd frontend   # or wherever you unzipped this
npm install
npm run dev
```

Open the URL shown (usually `http://localhost:5173`).

---

## What to expect

- **History page** — pulls real rows from MySQL via `GET /api/videos`, shows the 6 seeded demo videos with frame counts, edits detected, and confidence scores
- **Upload page** — uploading a video now sends a `POST /api/videos` request that inserts a row into `videos`, `analysis_results`, and `detected_frames`
- **Analysis page** — fetches the specific video's details via `GET /api/videos/:id`, including its real detected frames from the database
- **Database Schema page** — visual reference of the same 4 tables (no API needed, just documentation)

---

## API Endpoints Reference

| Method | Endpoint | Purpose |
|---|---|---|
| GET | `/api/health` | Check server is alive |
| GET | `/api/videos` | List all videos + their analysis summary |
| GET | `/api/videos/:id` | Get one video's full details + frames |
| POST | `/api/videos` | Save a new video + analysis + frames |
| DELETE | `/api/videos/:id` | Delete a video record |
| GET | `/api/stats` | Get total/suspicious/clean counts |

---

## For your FYP evaluation

This satisfies **Section B, Criterion 2** (Core Features Working / Database Schema / Integrations) two ways at once:
- A real **MySQL database schema** with proper foreign key relationships
- A real **working integration**: frontend ↔ Node.js API ↔ MySQL, not just mock data

Be ready to explain: "When a user uploads a video, the frontend sends a POST request to our Express API, which inserts the video info and analysis results into MySQL. The History and Analysis pages then fetch this data back with GET requests."
