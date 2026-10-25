# Twinergy 2.0 — Frontend Dashboard (Step 11B)

A modern, responsive React web dashboard for **Twinergy 2.0 Digital Twin Energy Analytics**, interfacing directly with the FastAPI REST API.

---

## ⚠️ Important Operating Disclaimer: Historical & Model-Derived Data

> **Notice:** The energy readings, forecasts, anomaly detections, and carbon/cost metrics presented in this dashboard are derived from **historical building energy datasets and trained machine learning models** (LightGBM 24-hour forecasting, residual-based anomaly detectors, and ASHRAE Great Energy Predictor III benchmarks).
>
> This dashboard **does NOT claim live IoT telemetry or automatic physical building control**. Recommendations and what-if simulations are strictly advisory decision-support tools.

---

## Prerequisites

- **Node.js**: `v18.0.0` or higher (tested on `v22+`)
- **npm**: `v9.0.0` or higher
- **Backend API**: Running Twinergy 2.0 FastAPI backend on `http://127.0.0.1:8000` (Python 3.10+ virtual environment)

---

## Environment Setup

1. Copy the example environment configuration:
   ```bash
   cp .env.example .env
   ```
2. Verify the API base URL in `.env`:
   ```bash
   VITE_API_BASE_URL=http://127.0.0.1:8000
   ```

---

## Installation

Install frontend dependencies:
```bash
cd frontend
npm install
```

Dependencies installed:
- `react`, `react-dom` (v19)
- `lucide-react` (iconography)
- `vite` (bundler / dev server)
- `vitest`, `@testing-library/react`, `@testing-library/jest-dom`, `jsdom` (testing)

---

## Running the Application

### 1. Start the FastAPI Backend Server
In the project root directory, activate your virtual environment and start Uvicorn:
```bash
# From workspace root:
major/bin/python -m uvicorn src.api.main:app --reload --host 127.0.0.1 --port 8000
```
Verify the API is running at: [http://127.0.0.1:8000/docs](http://127.0.0.1:8000/docs) or [http://127.0.0.1:8000/health](http://127.0.0.1:8000/health).

### 2. Start the React Frontend Development Server
In a separate terminal, run:
```bash
cd frontend
npm run dev
```
Open [http://localhost:5173](http://localhost:5173) in your browser.

---

## Running Tests

Run the complete frontend Vitest test suite:
```bash
cd frontend
npm test
```
All unit and component tests run in a headless JSDOM environment with mocked API responses.

---

## Production Build

To compile and optimize the frontend for production:
```bash
cd frontend
npm run build
```
To preview the production bundle locally:
```bash
npm run preview
```
Production assets are generated in `frontend/dist/`.

---

## Consumed API Endpoints

The frontend client (`src/services/api.js`) integrates with the following FastAPI endpoints:

| Endpoint | Method | Purpose in Dashboard |
|---|---|---|
| `/health` | `GET` | Health check & API connectivity badge in the top navigation bar |
| `/buildings` | `GET` | Populates the building dropdown and available meters |
| `/buildings/{id}/state` | `GET` | Comprehensive Digital Twin State (baseline energy, weather, status) |
| `/buildings/{id}/forecast` | `GET` | 24-hour ahead hourly energy forecast and peak metrics |
| `/buildings/{id}/anomalies` | `GET` | Residual anomaly flags and auxiliary Isolation Forest signals |
| `/buildings/{id}/impact` | `GET` | Cost (₹) and carbon (kg CO₂e) breakdown and hourly curves |
| `/buildings/{id}/recommendations` | `GET` | Priority-ordered advisory efficiency recommendations |
| `/buildings/{id}/simulate` | `POST` | What-if hypothetical demand adjustments without altering baseline |

---

## Dashboard Architecture & Design

- **Cyber-Industrial Glassmorphic UI**: High-contrast, dark-theme aesthetic using HSL custom properties, responsive flex/grid layouts, and clean visual hierarchy.
- **Pure SVG Responsive Charts**: Custom zero-dependency line charts with dual-series comparison, tooltips, dashed baseline curves, and interactive axes.
- **Request Cancellation**: Uses standard `AbortController` signals to cancel in-flight requests when buildings or meters are switched rapidly.
- **Defensive Error Handling**: Explicit loading spinners, error retry banners, and empty states. No fabricated or assumed default values.
- **Strict Simulation Safety**: Simulation outcomes are kept strictly distinct from the baseline Twin State, ensuring model ground truth is never mutated.
