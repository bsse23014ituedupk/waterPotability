/**
 * AquaGuard — Water Potability Prediction UI
 * Handles form validation, API calls, result display, and tooltips.
 */

'use strict';

// ---------------------------------------------------------------------------
// Configuration
// ---------------------------------------------------------------------------
const API_BASE = window.location.origin;
const PREDICT_ENDPOINT = `${API_BASE}/predict`;
const HEALTH_ENDPOINT  = `${API_BASE}/health`;

// Validation constraints matching Pydantic schema
const FIELD_CONSTRAINTS = {
  ph:              { min: 0.0, max: 14.0, label: 'pH Level' },
  Hardness:        { min: 0.0, max: null, label: 'Hardness' },
  Solids:          { min: 0.0, max: null, label: 'Total Dissolved Solids' },
  Chloramines:     { min: 0.0, max: null, label: 'Chloramines' },
  Sulfate:         { min: 0.0, max: null, label: 'Sulfate' },
  Conductivity:    { min: 0.0, max: null, label: 'Conductivity' },
  Organic_carbon:  { min: 0.0, max: null, label: 'Organic Carbon' },
  Trihalomethanes: { min: 0.0, max: null, label: 'Trihalomethanes' },
  Turbidity:       { min: 0.0, max: null, label: 'Turbidity' },
};

const DEFAULT_VALUES = {
  ph:              7.0,
  Hardness:        196.0,
  Solids:          20791.0,
  Chloramines:     7.3,
  Sulfate:         368.0,
  Conductivity:    564.0,
  Organic_carbon:  10.4,
  Trihalomethanes: 86.0,
  Turbidity:       2.96,
};

const INTERPRETATIONS = {
  potable: `This water sample meets potability criteria based on the ML model's
    analysis. Note: this is a machine learning prediction with ~73% accuracy — 
    always verify with certified laboratory analysis for official assessments.`,
  notPotable: `This water sample shows characteristics associated with non-potable water.
    The model detected parameters outside typical safe ranges. Consult a certified
    water quality laboratory before consuming this water.`,
};

// ---------------------------------------------------------------------------
// DOM References
// ---------------------------------------------------------------------------
const form         = document.getElementById('predictionForm');
const predictBtn   = document.getElementById('predictBtn');
const resetBtn     = document.getElementById('resetBtn');
const btnLoader    = document.getElementById('btnLoader');
const resultCard   = document.getElementById('resultCard');
const errorCard    = document.getElementById('errorCard');
const resultVerdict  = document.getElementById('resultVerdict');
const verdictIcon    = document.getElementById('verdictIcon');
const verdictLabel   = document.getElementById('verdictLabel');
const probBar        = document.getElementById('probBar');
const probValue      = document.getElementById('probValue');
const confidenceBadge  = document.getElementById('confidenceBadge');
const thresholdValue   = document.getElementById('thresholdValue');
const resultInterp     = document.getElementById('resultInterpretation');
const errorBody        = document.getElementById('errorBody');
const statusDot        = document.getElementById('statusDot');
const statusText       = document.getElementById('statusText');
const tooltipPopup     = document.getElementById('tooltipPopup');

// ---------------------------------------------------------------------------
// API Health Check
// ---------------------------------------------------------------------------
async function checkApiHealth() {
  try {
    const res = await fetch(HEALTH_ENDPOINT, { signal: AbortSignal.timeout(5000) });
    if (!res.ok) throw new Error(`HTTP ${res.status}`);
    const data = await res.json();

    statusDot.className = 'status-dot online';
    statusText.textContent = data.model_loaded
      ? `API Online · v${data.model_version}`
      : 'API Online · Model not loaded';
  } catch (err) {
    statusDot.className = 'status-dot offline';
    statusText.textContent = 'API Offline';
  }
}

// ---------------------------------------------------------------------------
// Form Validation
// ---------------------------------------------------------------------------
function validateField(fieldName, value) {
  const { min, max, label } = FIELD_CONSTRAINTS[fieldName];
  const errorEl = document.getElementById(`error-${fieldName}`);
  const inputEl = document.getElementById(fieldName);

  if (value === '' || isNaN(value)) {
    errorEl.textContent = `${label} is required.`;
    inputEl.classList.add('invalid');
    return false;
  }
  if (min !== null && value < min) {
    errorEl.textContent = `Must be ≥ ${min}.`;
    inputEl.classList.add('invalid');
    return false;
  }
  if (max !== null && value > max) {
    errorEl.textContent = `Must be ≤ ${max}.`;
    inputEl.classList.add('invalid');
    return false;
  }

  errorEl.textContent = '';
  inputEl.classList.remove('invalid');
  return true;
}

function validateForm(data) {
  let valid = true;
  for (const [field, value] of Object.entries(data)) {
    if (!validateField(field, value)) valid = false;
  }
  return valid;
}

// ---------------------------------------------------------------------------
// Collect Form Data
// ---------------------------------------------------------------------------
function getFormData() {
  const data = {};
  for (const field of Object.keys(FIELD_CONSTRAINTS)) {
    const el = document.getElementById(field);
    data[field] = el ? parseFloat(el.value) : NaN;
  }
  return data;
}

// ---------------------------------------------------------------------------
// UI State: Loading
// ---------------------------------------------------------------------------
function setLoading(loading) {
  if (loading) {
    predictBtn.disabled = true;
    document.querySelector('.btn-text').style.display = 'none';
    document.querySelector('.btn-icon').style.display = 'none';
    btnLoader.style.display = 'inline-flex';
  } else {
    predictBtn.disabled = false;
    document.querySelector('.btn-text').style.display = '';
    document.querySelector('.btn-icon').style.display = '';
    btnLoader.style.display = 'none';
  }
}

// ---------------------------------------------------------------------------
// Display Prediction Result
// ---------------------------------------------------------------------------
function showResult(data) {
  errorCard.style.display = 'none';
  resultCard.style.display = 'block';

  // Force reflow for animation
  resultCard.style.animation = 'none';
  resultCard.offsetHeight; // trigger reflow
  resultCard.style.animation = '';

  const isPotable = data.potability === 1;
  const prob = data.probability;
  const probPercent = Math.round(prob * 100);

  // Verdict
  verdictIcon.textContent   = isPotable ? '✅' : '❌';
  verdictLabel.textContent  = data.interpretation;
  verdictLabel.className    = `verdict-label ${isPotable ? 'text-potable' : 'text-not-potable'}`;
  resultVerdict.className   = `result-verdict ${isPotable ? 'verdict-potable' : 'verdict-not-potable'}`;

  // Probability bar
  setTimeout(() => {
    probBar.style.width = `${probPercent}%`;
  }, 50);
  probBar.className = `prob-bar${isPotable ? '' : ' danger'}`;
  probValue.textContent = `${probPercent}%`;

  // Confidence badge
  const confClass = {
    HIGH:   'badge-high',
    MEDIUM: 'badge-medium',
    LOW:    'badge-low',
  }[data.confidence] || 'badge-medium';
  confidenceBadge.textContent = data.confidence;
  confidenceBadge.className   = `confidence-badge ${confClass}`;

  // Threshold
  thresholdValue.textContent = data.threshold_used.toFixed(2);

  // Interpretation text
  resultInterp.textContent = isPotable ? INTERPRETATIONS.potable : INTERPRETATIONS.notPotable;
}

// ---------------------------------------------------------------------------
// Display Error
// ---------------------------------------------------------------------------
function showError(message) {
  resultCard.style.display = 'none';
  errorCard.style.display  = 'block';
  errorBody.textContent    = message;
}

// ---------------------------------------------------------------------------
// Form Submit → API Call
// ---------------------------------------------------------------------------
form.addEventListener('submit', async (e) => {
  e.preventDefault();

  const data = getFormData();
  if (!validateForm(data)) return;

  setLoading(true);
  resultCard.style.display = 'none';
  errorCard.style.display  = 'none';

  try {
    const res = await fetch(PREDICT_ENDPOINT, {
      method:  'POST',
      headers: { 'Content-Type': 'application/json' },
      body:    JSON.stringify(data),
      signal:  AbortSignal.timeout(15000),
    });

    if (!res.ok) {
      const errData = await res.json().catch(() => ({ detail: `HTTP ${res.status}` }));
      const msg = errData.detail
        ? (typeof errData.detail === 'string' ? errData.detail : JSON.stringify(errData.detail))
        : `HTTP ${res.status}`;
      showError(msg);
      return;
    }

    const result = await res.json();
    showResult(result);
    resultCard.scrollIntoView({ behavior: 'smooth', block: 'nearest' });

  } catch (err) {
    if (err.name === 'TimeoutError') {
      showError('Request timed out. Ensure the API server is running.');
    } else if (err.name === 'TypeError') {
      showError('Cannot connect to the API. Run: python main.py --mode serve');
    } else {
      showError(err.message || 'An unexpected error occurred.');
    }
  } finally {
    setLoading(false);
  }
});

// ---------------------------------------------------------------------------
// Reset Form
// ---------------------------------------------------------------------------
resetBtn.addEventListener('click', () => {
  for (const [field, value] of Object.entries(DEFAULT_VALUES)) {
    const el = document.getElementById(field);
    if (el) el.value = value;
  }
  for (const field of Object.keys(FIELD_CONSTRAINTS)) {
    const errorEl = document.getElementById(`error-${field}`);
    const inputEl = document.getElementById(field);
    if (errorEl) errorEl.textContent = '';
    if (inputEl) inputEl.classList.remove('invalid');
  }
  resultCard.style.display = 'none';
  errorCard.style.display  = 'none';
});

// ---------------------------------------------------------------------------
// Real-time field validation on blur
// ---------------------------------------------------------------------------
for (const field of Object.keys(FIELD_CONSTRAINTS)) {
  const el = document.getElementById(field);
  if (!el) continue;
  el.addEventListener('blur', () => {
    validateField(field, parseFloat(el.value));
  });
  el.addEventListener('input', () => {
    // Clear error on input start
    const errEl = document.getElementById(`error-${field}`);
    if (errEl && errEl.textContent) {
      validateField(field, parseFloat(el.value));
    }
  });
}

// ---------------------------------------------------------------------------
// Tooltip System
// ---------------------------------------------------------------------------
document.querySelectorAll('.tooltip-trigger').forEach((trigger) => {
  trigger.addEventListener('mouseenter', (e) => {
    const text = e.target.getAttribute('data-tooltip');
    if (!text) return;
    tooltipPopup.textContent = text;
    tooltipPopup.classList.add('visible');
    positionTooltip(e);
  });

  trigger.addEventListener('mousemove', positionTooltip);

  trigger.addEventListener('mouseleave', () => {
    tooltipPopup.classList.remove('visible');
  });
});

function positionTooltip(e) {
  const x = e.clientX + 12;
  const y = e.clientY + 12;
  const vw = window.innerWidth;
  const vh = window.innerHeight;
  const tipW = 270;
  const tipH = tooltipPopup.offsetHeight || 80;

  tooltipPopup.style.left = `${Math.min(x, vw - tipW - 12)}px`;
  tooltipPopup.style.top  = `${Math.min(y, vh - tipH - 12)}px`;
}

// ---------------------------------------------------------------------------
// Initialise
// ---------------------------------------------------------------------------
document.addEventListener('DOMContentLoaded', () => {
  checkApiHealth();
  // Re-check health every 30 seconds
  setInterval(checkApiHealth, 30000);
});
