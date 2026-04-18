const state = {
  home: null,
  hiringReference: null,
  promotionReference: null,
  dataSummary: null,
  riskDashboard: null,
  statisticalAnalysis: null,
  organizationalImpact: null,
};

const pageMeta = {
  home: { title: 'Dashboard', subtitle: 'Overview of all analytical models and key metrics' },
  hiring: { title: 'Hiring Predictor', subtitle: 'Predict candidate hiring probability based on merit and connection factors' },
  promotion: { title: 'Promotion Predictor', subtitle: 'Predict employee promotion probability based on performance and connections' },
  'data-summary': { title: 'Data Summary', subtitle: 'Overview of generated data, metrics, and scenario comparisons' },
  'risk-dashboard': { title: 'Risk Dashboard', subtitle: 'Network nepotism risk analysis and anomaly detection insights' },
  'statistical-analysis': { title: 'Statistical Analysis', subtitle: 'Matched pairs analysis and sensitivity curves' },
  'organizational-impact': { title: 'Organizational Impact', subtitle: 'Cross-scenario comparisons of hiring quality, fairness, and efficiency' },
};

let charts = {};

document.addEventListener('DOMContentLoaded', async () => {
  setupNavigation();
  showLoading(true);
  try {
    await loadInitialData();
    renderHome();
    renderHiring();
    renderPromotion();
    renderDataSummary();
    renderRiskDashboard();
    renderStatisticalAnalysis();
    renderOrganizationalImpact();
  } catch (error) {
    console.error('Initialization error:', error);
    document.querySelector('.views-container').innerHTML = `
      <div class="alert error">
        <svg viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2">
          <circle cx="12" cy="12" r="10"/>
          <line x1="12" y1="8" x2="12" y2="12"/>
          <line x1="12" y1="16" x2="12.01" y2="16"/>
        </svg>
        <span>Failed to initialize the application. Please check that the backend server is running.</span>
      </div>
    `;
  }
  showLoading(false);
});

function showLoading(show) {
  document.getElementById('page-loader').classList.toggle('visible', show);
}

function setupNavigation() {
  document.querySelectorAll('.nav-btn').forEach((button) => {
    button.addEventListener('click', () => {
      document.querySelectorAll('.nav-btn').forEach((item) => item.classList.remove('active'));
      document.querySelectorAll('.page-view').forEach((view) => view.classList.remove('active'));
      button.classList.add('active');
      const viewId = `${button.dataset.view}-view`;
      document.getElementById(viewId).classList.add('active');
    });
  });
}

async function loadInitialData() {
  const [home, hiringReference, promotionReference, dataSummary, riskDashboard, statisticalAnalysis, organizationalImpact] = await Promise.all([
    fetchJson('/api/meta/home'),
    fetchJson('/api/reference/hiring'),
    fetchJson('/api/reference/promotion'),
    fetchJson('/api/data-summary'),
    fetchJson('/api/risk-dashboard'),
    fetchJson('/api/statistical-analysis'),
    fetchJson('/api/organizational-impact'),
  ]);
  state.home = home;
  state.hiringReference = hiringReference;
  state.promotionReference = promotionReference;
  state.dataSummary = dataSummary;
  state.riskDashboard = riskDashboard;
  state.statisticalAnalysis = statisticalAnalysis;
  state.organizationalImpact = organizationalImpact;
}

async function fetchJson(url, options = {}) {
  const response = await fetch(url, options);
  if (!response.ok) {
    throw new Error(`Request failed: ${response.status}`);
  }
  return response.json();
}

function refreshData() {
  showLoading(true);
  loadInitialData().then(() => {
    renderHome();
    renderDataSummary();
    renderRiskDashboard();
    renderStatisticalAnalysis();
    renderOrganizationalImpact();
    showLoading(false);
  }).catch(() => {
    showLoading(false);
  });
}

function escapeHtml(value) {
  return String(value)
    .replace(/&/g, '&amp;')
    .replace(/</g, '&lt;')
    .replace(/>/g, '&gt;')
    .replace(/"/g, '&quot;');
}

function formatNumber(value, format) {
  if (format === ',.0f') {
    return Number(value).toLocaleString(undefined, { maximumFractionDigits: 0 });
  }
  if (format === '.1f') return Number(value).toFixed(1);
  if (format === '.2f') return Number(value).toFixed(2);
  if (format === '.3f') return Number(value).toFixed(3);
  if (format === '.4f') return Number(value).toFixed(4);
  if (format === '.4%') return `${(Number(value) * 100).toFixed(2)}%`;
  if (format === ',.0f' || format === ',') return Number(value).toLocaleString();
  return Number(value).toLocaleString();
}

function renderHome() {
  const container = document.getElementById('home-view');
  const home = state.home;
  const risk = state.riskDashboard || {};
  const summary = state.dataSummary || {};

  container.innerHTML = `
    <div class="page-header">
      <h1 class="page-title">${escapeHtml(home.title)}</h1>
      <p class="page-subtitle">${escapeHtml(home.hero.body)}</p>
    </div>

    <div class="kpi-grid">
      ${renderKpiCard('Total Candidates', summary.overview_metrics?.find(m => m.label === 'Total candidates')?.value || 'N/A', 'blue', 0)}
      ${renderKpiCard('Total Employees', summary.overview_metrics?.find(m => m.label === 'Total employees')?.value || 'N/A', 'green', 1)}
      ${renderKpiCard('Overall Hiring Rate', summary.overview_metrics?.find(m => m.label === 'Overall hiring rate')?.value || 'N/A', 'yellow', 2)}
      ${renderKpiCard('Overall Promotion Rate', summary.overview_metrics?.find(m => m.label === 'Overall promotion rate')?.value || 'N/A', 'green', 3)}
    </div>

    <div class="two-col">
      <div class="chart-card">
        <div class="card-header">
          <div>
            <h3 class="card-title">Scenario Overview</h3>
            <p class="card-subtitle">Hiring and promotion rates by scenario</p>
          </div>
        </div>
        <div class="chart-container">
          <canvas id="home-scenario-chart"></canvas>
        </div>
      </div>
      <div class="chart-card">
        <div class="card-header">
          <div>
            <h3 class="card-title">Anomaly Rates</h3>
            <p class="card-subtitle">Suspicious hiring and promotion rates</p>
          </div>
        </div>
        <div class="chart-container">
          <canvas id="home-anomaly-chart"></canvas>
        </div>
      </div>
    </div>

    <div class="card">
      <div class="card-header">
        <div>
          <h3 class="card-title">Analytical Models</h3>
          <p class="card-subtitle">Four machine learning models for nepotism risk analytics</p>
        </div>
      </div>
      <div class="model-grid">
        ${home.pipeline.map((item, i) => renderModelCard(item, i + 1)).join('')}
      </div>
    </div>

    <div class="two-col">
      <div class="card">
        <div class="card-header">
          <div>
            <h3 class="card-title">Network Risk Summary</h3>
            <p class="card-subtitle">Manager and department risk assessment</p>
          </div>
        </div>
        <div class="metric-row">
          ${(risk.network_metrics || []).slice(0, 3).map(m => renderMetricBox(m.label, m.value)).join('')}
        </div>
      </div>
      <div class="card">
        <div class="card-header">
          <div>
            <h3 class="card-title">Anomaly Detection</h3>
            <p class="card-subtitle">Suspicious hires and promotions flagged</p>
          </div>
        </div>
        <div class="metric-row">
          ${(risk.anomaly_metrics || []).slice(0, 3).map(m => renderMetricBox(m.label, m.value)).join('')}
        </div>
      </div>
    </div>
  `;

  initializeHomeCharts();
}

function renderKpiCard(label, value, color, index) {
  const icons = [
    '<svg viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2"><path d="M17 21v-2a4 4 0 0 0-4-4H5a4 4 0 0 0-4 4v2"/><circle cx="9" cy="7" r="4"/></svg>',
    '<svg viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2"><rect x="2" y="7" width="20" height="14" rx="2"/><path d="M16 21V5a2 2 0 0 0-2-2h-4a2 2 0 0 0-2 2v16"/></svg>',
    '<svg viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2"><path d="M16 21v-2a4 4 0 0 0-4-4H5a4 4 0 0 0-4 4v2"/><circle cx="8.5" cy="7" r="4"/><line x1="20" y1="8" x2="20" y2="14"/><line x1="23" y1="11" x2="17" y2="11"/></svg>',
    '<svg viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2"><line x1="12" y1="19" x2="12" y2="5"/><polyline points="5 12 12 5 19 12"/></svg>',
  ];
  return `
    <div class="kpi-card">
      <div class="kpi-icon ${color}">${icons[index % icons.length]}</div>
      <div class="kpi-label">${escapeHtml(label)}</div>
      <div class="kpi-value">${escapeHtml(value)}</div>
    </div>
  `;
}

function renderModelCard(item, modelNum) {
  const colors = ['blue', 'green', 'yellow', 'red'];
  return `
    <div class="model-card ${colors[modelNum - 1]}">
      <div class="model-number">Model ${modelNum}</div>
      <h4 class="model-title">${escapeHtml(item.title)}</h4>
      <p class="model-desc">${escapeHtml(item.description)}</p>
    </div>
  `;
}

function renderMetricBox(label, value) {
  return `
    <div class="metric-box">
      <div class="metric-label">${escapeHtml(label)}</div>
      <div class="metric-value">${escapeHtml(value)}</div>
    </div>
  `;
}

function initializeHomeCharts() {
  if (charts.homeScenario) charts.homeScenario.destroy();
  if (charts.homeAnomaly) charts.homeAnomaly.destroy();

  const summary = state.dataSummary?.scenario_summary_raw || [];
  const risk = state.riskDashboard || {};

  const scenarioLabels = summary.map(r => r.Scenario || r.scenario);
  const hiringRates = summary.map(r => {
    const key = Object.keys(r).find(k => k.toLowerCase().includes('hiring'));
    return key ? parseFloat(r[key]) : 0;
  });
  const promotionRates = summary.map(r => {
    const key = Object.keys(r).find(k => k.toLowerCase().includes('promotion'));
    return key ? parseFloat(r[key]) : 0;
  });

  const ctxScenario = document.getElementById('home-scenario-chart');
  if (ctxScenario && scenarioLabels.length) {
    charts.homeScenario = new Chart(ctxScenario, {
      type: 'bar',
      data: {
        labels: scenarioLabels,
        datasets: [
          {
            label: 'Hiring Rate (%)',
            data: hiringRates,
            backgroundColor: 'rgba(59, 130, 246, 0.7)',
            borderColor: '#3b82f6',
            borderWidth: 2,
            borderRadius: 6,
          },
          {
            label: 'Promotion Rate (%)',
            data: promotionRates,
            backgroundColor: 'rgba(16, 185, 129, 0.7)',
            borderColor: '#10b981',
            borderWidth: 2,
            borderRadius: 6,
          },
        ],
      },
      options: {
        responsive: true,
        maintainAspectRatio: false,
        plugins: {
          legend: { position: 'bottom', labels: { usePointStyle: true, padding: 20 } },
        },
        scales: {
          y: { beginAtZero: true, grid: { color: 'rgba(0,0,0,0.05)' } },
          x: { grid: { display: false } },
        },
      },
    });
  }

  const hiringAnomalies = (risk.hiring_anomaly_series || []).map(r => (parseFloat(r.value) * 100).toFixed(2));
  const promoAnomalies = (risk.promotion_anomaly_series || []).map(r => (parseFloat(r.value) * 100).toFixed(2));
  const anomalyLabels = (risk.hiring_anomaly_series || []).map(r => r.scenario);

  const ctxAnomaly = document.getElementById('home-anomaly-chart');
  if (ctxAnomaly && anomalyLabels.length) {
    charts.homeAnomaly = new Chart(ctxAnomaly, {
      type: 'line',
      data: {
        labels: anomalyLabels,
        datasets: [
          {
            label: 'Hiring Anomaly Rate (%)',
            data: hiringAnomalies,
            borderColor: '#f59e0b',
            backgroundColor: 'rgba(245, 158, 11, 0.1)',
            fill: true,
            tension: 0.4,
            pointRadius: 6,
            pointBackgroundColor: '#f59e0b',
          },
          {
            label: 'Promotion Anomaly Rate (%)',
            data: promoAnomalies,
            borderColor: '#ef4444',
            backgroundColor: 'rgba(239, 68, 68, 0.1)',
            fill: true,
            tension: 0.4,
            pointRadius: 6,
            pointBackgroundColor: '#ef4444',
          },
        ],
      },
      options: {
        responsive: true,
        maintainAspectRatio: false,
        plugins: {
          legend: { position: 'bottom', labels: { usePointStyle: true, padding: 20 } },
        },
        scales: {
          y: { beginAtZero: true, grid: { color: 'rgba(0,0,0,0.05)' } },
          x: { grid: { display: false } },
        },
      },
    });
  }
}

function renderHiring() {
  const container = document.getElementById('hiring-view');
  const ref = state.hiringReference;
  const numeric = ref.reference_data.numeric_ranges;
  const educationLevels = ref.reference_data.education_levels;
  const channels = ref.reference_data.discretionary_channels;

  container.innerHTML = `
    <div class="predictor-layout">
      <div class="form-panel">
        <h3 class="form-title">Candidate Profile</h3>
        <p class="form-subtitle">Enter candidate information to predict hiring probability</p>
        <form id="hiring-form">
          <div class="form-grid">
            <div class="form-field">
              <label class="form-label">Model Version</label>
              <select class="form-select" name="model_name">
                ${ref.model_choices.map(o => `<option value="${o.name}" ${o.name === ref.default_model_name ? 'selected' : ''}>${o.label}</option>`).join('')}
              </select>
            </div>
            <div class="form-field">
              <label class="form-label">Education Level</label>
              <select class="form-select" name="education_level">
                ${educationLevels.map(l => `<option value="${l}">${l}</option>`).join('')}
              </select>
            </div>
            ${renderSliderField('hiring-hs', 'high_school_gpa', 'High School GPA', numeric.high_school_gpa.median.toFixed(1), 0.1, 50, 100)}
            ${renderSliderField('hiring-college', 'college_gpa', 'College GPA', numeric.college_gpa.median.toFixed(1), 0.1, 50, 100)}
            ${renderSliderField('hiring-test', 'test_score', 'Test Score', numeric.test_score.median.toFixed(1), 0.1, 30, 100)}
            ${renderSliderField('hiring-interview', 'interview_score', 'Interview Score', numeric.interview_score.median.toFixed(1), 0.1, 30, 100)}
            ${renderSliderField('hiring-connection', 'connection_strength', 'Connection Strength', numeric.connection_strength.median.toFixed(2), 0.01, 0, 1)}
            ${renderNumberField('hiring-exp', 'years_experience', 'Years of Experience', numeric.years_experience.median, 0.5, 0)}
            <div class="form-field full">
              <label class="form-label">Recruitment Channel</label>
              <select class="form-select" name="discretionary_channel">
                ${channels.map(c => `<option value="${c}">${c}</option>`).join('')}
              </select>
            </div>
            <div class="form-field full">
              <label class="form-label">Connection Type</label>
              <div class="checkbox-grid">
                ${renderCheckbox('referral_flag', 'Employee Referral')}
                ${renderCheckbox('family_link_flag', 'Family Connection')}
                ${renderCheckbox('close_family_relation_flag', 'Close Family')}
                ${renderCheckbox('same_high_school_flag', 'Same High School')}
                ${renderCheckbox('same_city_flag', 'Same City')}
                ${renderCheckbox('same_college_flag', 'Same College')}
                ${renderCheckbox('same_last_name_flag', 'Same Last Name')}
              </div>
            </div>
          </div>
          <button type="submit" class="submit-btn">
            <svg width="18" height="18" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2">
              <path d="M22 11.08V12a10 10 0 1 1-5.93-9.14"/>
              <polyline points="22 4 12 14.01 9 11.01"/>
            </svg>
            Predict Hiring Probability
          </button>
        </form>
      </div>
      <div class="result-panel" id="hiring-result">
        ${renderResultPlaceholder('hiring')}
      </div>
    </div>
  `;

  document.getElementById('hiring-form').addEventListener('submit', (e) => {
    e.preventDefault();
    submitPrediction(e.currentTarget, '/api/predict/hiring', 'hiring-result', 'hiring');
  });

  document.querySelectorAll('.form-slider').forEach(slider => {
    slider.addEventListener('input', () => {
      const valueDisplay = document.getElementById(`${slider.id}-value`);
      if (valueDisplay) valueDisplay.textContent = slider.value;
    });
  });
}

function renderResultPlaceholder(type) {
  return `
    <div class="result-placeholder">
      <svg viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="1.5">
        <path d="M9 17v-2m3 2v-4m3 4v-6m2 10H7a2 2 0 01-2-2V5a2 2 0 012-2h5.586a1 1 0 01.707.293l5.414 5.414a1 1 0 01.293.707V19a2 2 0 01-2 2z"/>
      </svg>
      <p>Submit a candidate profile to see the prediction result, driver analysis, and model explanation</p>
    </div>
  `;
}

function renderNumberField(id, name, label, value, step, min, max = '') {
  const maxAttr = max !== '' ? `max="${max}"` : '';
  return `
    <div class="form-field">
      <label class="form-label" for="${id}">${label}</label>
      <input class="form-input" id="${id}" name="${name}" type="number" step="${step}" min="${min}" ${maxAttr} value="${value}">
    </div>
  `;
}

function renderSliderField(id, name, label, value, step, min, max) {
  return `
    <div class="form-field">
      <label class="form-label" for="${id}">${label}: <span class="slider-value" id="${id}-value">${value}</span></label>
      <input class="form-slider" id="${id}" name="${name}" type="range" step="${step}" min="${min}" max="${max}" value="${value}">
    </div>
  `;
}

function renderCheckbox(name, label) {
  return `<label class="checkbox-item"><input type="checkbox" name="${name}"><span>${label}</span></label>`;
}

function renderPromotion() {
  const container = document.getElementById('promotion-view');
  const ref = state.promotionReference;
  const numeric = ref.reference_data.numeric_ranges;
  const channels = ref.reference_data.discretionary_channels;
  const roleLevels = ref.reference_data.role_levels;
  const roleLabels = ref.role_labels || {};

  container.innerHTML = `
    <div class="predictor-layout">
      <div class="form-panel">
        <h3 class="form-title">Employee Profile</h3>
        <p class="form-subtitle">Enter employee information to predict promotion probability</p>
        <form id="promotion-form">
          <div class="form-grid">
            <div class="form-field">
              <label class="form-label">Model Version</label>
              <select class="form-select" name="model_name">
                ${ref.model_choices.map(o => `<option value="${o.name}" ${o.name === ref.default_model_name ? 'selected' : ''}>${o.label}</option>`).join('')}
              </select>
            </div>
            <div class="form-field">
              <label class="form-label">Current Role</label>
              <select class="form-select" name="role_level">
                ${roleLevels.map(l => `<option value="${l}">${roleLabels[l] || `Role ${l}`}</option>`).join('')}
              </select>
            </div>
            ${renderSliderField('promotion-performance', 'performance_score', 'Performance Score', numeric.performance_score.median.toFixed(1), 0.1, 35, 100)}
            ${renderSliderField('promotion-merit', 'merit_score', 'Merit Score', numeric.merit_score.median.toFixed(1), 0.1, 30, 100)}
            ${renderSliderField('promotion-connection', 'connection_strength', 'Connection Strength', numeric.connection_strength.median.toFixed(2), 0.01, 0, 1)}
            ${renderNumberField('promotion-tenure', 'tenure_months', 'Tenure (months)', Math.round(numeric.tenure_months.median), 1, 0)}
            ${renderNumberField('promotion-salary', 'salary', 'Salary', Math.round(numeric.salary.median), 100, 0)}
            ${renderNumberField('promotion-exp', 'years_experience', 'Years of Experience', numeric.years_experience.median, 0.5, 0)}
            <div class="form-field full">
              <label class="form-label">Recruitment Channel</label>
              <select class="form-select" name="discretionary_channel">
                ${channels.map(c => `<option value="${c}">${c}</option>`).join('')}
              </select>
            </div>
            <div class="form-field full">
              <label class="form-label">Connection Type</label>
              <div class="checkbox-grid">
                ${renderCheckbox('family_link_flag', 'Family Connection')}
                ${renderCheckbox('close_family_relation_flag', 'Close Family')}
                ${renderCheckbox('same_high_school_flag', 'Same High School')}
                ${renderCheckbox('same_city_flag', 'Same City')}
                ${renderCheckbox('same_college_flag', 'Same College')}
                ${renderCheckbox('same_last_name_flag', 'Same Last Name')}
              </div>
            </div>
          </div>
          <button type="submit" class="submit-btn">
            <svg width="18" height="18" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2">
              <path d="M22 11.08V12a10 10 0 1 1-5.93-9.14"/>
              <polyline points="22 4 12 14.01 9 11.01"/>
            </svg>
            Predict Promotion Probability
          </button>
        </form>
      </div>
      <div class="result-panel" id="promotion-result">
        ${renderResultPlaceholder('promotion')}
      </div>
    </div>
  `;

  document.getElementById('promotion-form').addEventListener('submit', (e) => {
    e.preventDefault();
    submitPrediction(e.currentTarget, '/api/predict/promotion', 'promotion-result', 'promotion');
  });

  document.querySelectorAll('#promotion-form .form-slider').forEach(slider => {
    slider.addEventListener('input', () => {
      const valueDisplay = document.getElementById(`${slider.id}-value`);
      if (valueDisplay) valueDisplay.textContent = slider.value;
    });
  });
}

async function submitPrediction(form, endpoint, resultId, task) {
  const resultContainer = document.getElementById(resultId);
  resultContainer.innerHTML = `
    <div class="result-placeholder">
      <div class="loading-spinner" style="width:32px;height:32px;border-width:2px;"></div>
      <p>Running prediction...</p>
    </div>
  `;

  try {
    const payload = formToJson(form);
    if ('role_level' in payload) {
      payload.role_level = Number(payload.role_level);
    }
    const result = await fetchJson(endpoint, {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify(payload),
    });
    resultContainer.innerHTML = renderPredictionResult(result, task);
    drawLogisticCurve(`${task}-curve-svg`, result.math.curve_points, result.math.marker_points);
  } catch (error) {
    resultContainer.innerHTML = `
      <div class="alert error">
        <svg viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2">
          <circle cx="12" cy="12" r="10"/>
          <line x1="15" y1="9" x2="9" y2="15"/>
          <line x1="9" y1="9" x2="15" y2="15"/>
        </svg>
        <span>Prediction failed: ${error.message}</span>
      </div>
    `;
  }
}

function formToJson(form) {
  const payload = {};
  for (const field of form.elements) {
    if (!field.name) continue;
    if (field.type === 'checkbox') {
      payload[field.name] = field.checked;
    } else if (field.type === 'number') {
      payload[field.name] = Number(field.value);
    } else {
      payload[field.name] = field.value;
    }
  }
  return payload;
}

function renderPredictionResult(result, task) {
  const model = result.model;
  const math = result.math;

  return `
    <div class="metric-row">
      <div class="metric-box highlight">
        <div class="metric-label">Predicted Probability</div>
        <div class="metric-value large">${result.probability_label}</div>
      </div>
      <div class="metric-box">
        <div class="metric-label">Likelihood Band</div>
        <div class="metric-value">${result.likelihood_band}</div>
      </div>
      <div class="metric-box">
        <div class="metric-label">Training Data</div>
        <div class="metric-value">${Number(model.training_rows).toLocaleString()}</div>
      </div>
    </div>

    ${result.merit_breakdown ? `
    <div class="info-box">
      <h4>Merit Score Breakdown</h4>
      <p>Derived merit score: <strong>${result.merit_breakdown.merit_score.toFixed(1)}</strong></p>
    </div>
    ` : ''}

    <div class="info-box">
      <h4>Primary Drivers</h4>
      <ul>
        ${result.drivers.map(d => `<li>${escapeHtml(d)}</li>`).join('')}
      </ul>
    </div>

    <div class="info-box">
      <h4>Model Information</h4>
      <p><strong>${model.description}</strong></p>
      <p style="margin-top:8px;">Brier Score: ${Number(model.brier_score).toFixed(4)} | ROC AUC: ${Number(model.roc_auc).toFixed(4)}</p>
    </div>

    <div class="info-box">
      <h4>Model Equation</h4>
      <div class="code-block">
        <pre>${escapeHtml(math.equation_lines.join('\n'))}</pre>
      </div>
    </div>

    <div class="card">
      <h3 class="card-title">Coefficient Contributions</h3>
      <div class="table-shell compact">
        <div class="table-scroll">
          <table class="data-table data-table-compact">
            <thead>
              <tr>
                <th>Feature</th>
                <th>Value</th>
                <th>Coefficient</th>
                <th>Contribution</th>
              </tr>
            </thead>
            <tbody>
              ${math.contributions.slice(0, 12).map(row => `
                <tr>
                  <td>${escapeHtml(row.feature_label)}</td>
                  <td>${Number(row.x_value).toFixed(4)}</td>
                  <td>${Number(row.coefficient).toFixed(4)}</td>
                  <td class="${Number(row.contribution) >= 0 ? 'positive' : 'negative'}">
                    ${Number(row.contribution) >= 0 ? '+' : ''}${Number(row.contribution).toFixed(4)}
                  </td>
                </tr>
              `).join('')}
            </tbody>
          </table>
        </div>
      </div>
    </div>

    <div class="curve-container">
      <h4>Logistic Curve Position</h4>
      <svg id="${task}-curve-svg" viewBox="0 0 700 280" preserveAspectRatio="xMidYMid meet"></svg>
    </div>
  `;
}

function drawLogisticCurve(svgId, curvePoints, markerPoints) {
  const svg = document.getElementById(svgId);
  if (!svg || !curvePoints?.length || !markerPoints?.length) return;

  const width = 700;
  const height = 280;
  const margin = { top: 18, right: 22, bottom: 34, left: 42 };
  const xMin = Math.min(...curvePoints.map(p => p.linear_score));
  const xMax = Math.max(...curvePoints.map(p => p.linear_score));
  const scaleX = v => margin.left + ((v - xMin) / (xMax - xMin || 1)) * (width - margin.left - margin.right);
  const scaleY = v => height - margin.bottom - v * (height - margin.top - margin.bottom);

  const points = curvePoints.map(p => `${scaleX(p.linear_score)},${scaleY(p.probability)}`).join(' ');
  const marker = markerPoints[0];
  const markerX = scaleX(marker.linear_score);
  const markerY = scaleY(marker.probability);

  svg.innerHTML = `
    <line x1="${margin.left}" y1="${height - margin.bottom}" x2="${width - margin.right}" y2="${height - margin.bottom}" stroke="#94a3b8" />
    <line x1="${margin.left}" y1="${margin.top}" x2="${margin.left}" y2="${height - margin.bottom}" stroke="#94a3b8" />
    <polyline fill="none" stroke="#3b82f6" stroke-width="3" points="${points}" />
    <circle cx="${markerX}" cy="${markerY}" r="7" fill="#d4a017" stroke="#b45309" stroke-width="2" />
    <text x="${markerX + 10}" y="${markerY - 10}" fill="#1e293b" font-size="12" font-weight="600">Current input</text>
    <text x="${width / 2}" y="${height - 8}" fill="#64748b" font-size="12" text-anchor="middle">Linear score (z)</text>
    <text x="16" y="${height / 2}" fill="#64748b" font-size="12" text-anchor="middle" transform="rotate(-90 16 ${height / 2})">Probability</text>
  `;
}

function renderDataSummary() {
  const container = document.getElementById('data-summary-view');
  const summary = state.dataSummary;
  const baselineOptions = ['Merit-based', 'Moderate favoritism', 'High nepotism risk'];

  container.innerHTML = `
    <div class="page-header">
      <h1 class="page-title">Data Summary</h1>
      <p class="page-subtitle">Generated and processed data overview, including scenario metrics and comparisons.</p>
    </div>

    ${summary.warnings?.length ? `
    <div class="alert warning">
      <svg viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2">
        <path d="M10.29 3.86L1.82 18a2 2 0 0 0 1.71 3h16.94a2 2 0 0 0 1.71-3L13.71 3.86a2 2 0 0 0-3.42 0z"/>
        <line x1="12" y1="9" x2="12" y2="13"/>
        <line x1="12" y1="17" x2="12.01" y2="17"/>
      </svg>
      <span>${summary.warnings.map(escapeHtml).join('<br>')}</span>
    </div>
    ` : ''}

    <div class="kpi-grid">
      ${(summary.overview_metrics || []).map((m, i) => {
        const colors = ['blue', 'green', 'yellow', 'red'];
        return renderKpiCard(m.label, m.value, colors[i % colors.length], i);
      }).join('')}
    </div>

    <div class="table-heavy-grid">
      <div class="card">
        <div class="card-header">
          <div>
            <h3 class="card-title">Dataset Overview</h3>
            <p class="card-subtitle">Available data sources and their dimensions</p>
          </div>
        </div>
        <div class="table-shell">
          <div class="table-scroll">
            <table class="data-table">
              <thead>
                <tr>
                  <th>Dataset</th>
                  <th>Rows</th>
                  <th>Columns</th>
                </tr>
              </thead>
              <tbody>
                ${(summary.source_rows || []).map(r => `
                  <tr>
                    <td><strong>${escapeHtml(r.dataset)}</strong></td>
                    <td>${Number(r.rows).toLocaleString()}</td>
                    <td>${Number(r.columns).toLocaleString()}</td>
                  </tr>
                `).join('')}
              </tbody>
            </table>
          </div>
        </div>
      </div>
      <div class="card">
        <div class="card-header">
          <div>
            <h3 class="card-title">Scenario Summary</h3>
            <p class="card-subtitle">Key metrics comparison across scenarios</p>
          </div>
        </div>
        ${renderTableFromObjects(summary.scenario_summary_table)}
      </div>
    </div>

    <div class="two-col">
      <div class="chart-card">
        <div class="card-header">
          <div>
            <h3 class="card-title">Hiring Metrics</h3>
            <p class="card-subtitle">Candidate pool analysis</p>
          </div>
        </div>
        <div class="chart-container">
          <canvas id="data-hiring-chart"></canvas>
        </div>
      </div>
      <div class="chart-card">
        <div class="card-header">
          <div>
            <h3 class="card-title">Employee Metrics</h3>
            <p class="card-subtitle">Performance and connection analysis</p>
          </div>
        </div>
        <div class="chart-container">
          <canvas id="data-employee-chart"></canvas>
        </div>
      </div>
    </div>

    <div class="card">
      <div class="card-header">
        <div>
          <h3 class="card-title">Scenario Comparison</h3>
          <p class="card-subtitle">Visual comparison of key metrics</p>
        </div>
        <select class="form-select" id="baseline-select">
          ${baselineOptions.map(o => `<option value="${o}">${o}</option>`).join('')}
        </select>
      </div>
      <div id="comparison-grid"></div>
      <div id="difference-table" style="margin-top:24px;"></div>
    </div>
  `;

  initializeDataCharts();

  const baselineSelect = document.getElementById('baseline-select');
  const differenceTable = document.getElementById('difference-table');
  const renderDifferenceTable = () => {
    const rows = summary.difference_tables?.[baselineSelect.value] || [];
    differenceTable.innerHTML = rows.length ? renderTableFromObjects(rows) : '<p>No difference data available.</p>';
  };
  baselineSelect.addEventListener('change', renderDifferenceTable);
  renderDifferenceTable();

  renderComparisons();
}

function renderTableFromObjects(rows) {
  if (!rows?.length) return '<p class="muted">No data available.</p>';
  const columns = Object.keys(rows[0]);
  const compact = columns.length >= 6;
  return `
    <div class="table-shell${compact ? ' compact' : ''}">
      <div class="table-scroll">
        <table class="data-table${compact ? ' data-table-compact' : ''}">
          <thead>
            <tr>${columns.map(c => `<th>${escapeHtml(c)}</th>`).join('')}</tr>
          </thead>
          <tbody>
            ${rows.map(row => `
              <tr>${columns.map(c => `<td>${escapeHtml(row[c] ?? 'N/A')}</td>`).join('')}</tr>
            `).join('')}
          </tbody>
        </table>
      </div>
    </div>
  `;
}

function renderComparisons() {
  const summary = state.dataSummary;
  const grid = document.getElementById('comparison-grid');
  if (!grid) return;

  const rows = summary.scenario_summary_raw || [];
  if (!rows.length) return;

  const scenarios = rows.map(r => r.scenario);
  const scenarioColors = ['var(--success)', 'var(--warning)', 'var(--danger)'];

  const legacyMetrics = [
    { key: 'hiring_rate', label: 'Hiring Rate', format: '.1f', suffix: '%', icon: '👥' },
    { key: 'promotion_rate', label: 'Promotion Rate', format: '.1f', suffix: '%', icon: '📈' },
    { key: 'avg_candidate_merit', label: 'Avg Merit Score', format: '.2f', icon: '⭐' },
    { key: 'avg_candidate_connection', label: 'Avg Connection', format: '.3f', icon: '🔗' },
    { key: 'avg_performance', label: 'Avg Performance', format: '.2f', icon: '🎯' },
    { key: 'avg_salary', label: 'Avg Salary', format: ',.0f', prefix: '$', icon: '💰' },
  ];
  const metrics = [
    { key: 'hiring_rate_pct', label: 'Hiring Rate', format: '.1f', suffix: '%', icon: 'H' },
    { key: 'promotion_rate_pct', label: 'Promotion Rate', format: '.1f', suffix: '%', icon: 'P' },
    { key: 'avg_candidate_merit_score', label: 'Avg Merit Score', format: '.2f', icon: 'M' },
    { key: 'avg_candidate_connection_strength', label: 'Avg Connection', format: '.3f', icon: 'C' },
    { key: 'avg_performance_score', label: 'Avg Performance', format: '.2f', icon: 'F' },
    { key: 'avg_salary', label: 'Avg Salary', format: ',.0f', prefix: '$', icon: 'S' },
  ];

  grid.innerHTML = `
    <div class="comparison-header">
      <div class="comparison-label"></div>
      ${scenarios.map((s, i) => `
        <div class="comparison-scenario-header" style="border-bottom: 3px solid ${scenarioColors[i]};">
          <span class="scenario-dot" style="background: ${scenarioColors[i]};"></span>
          ${escapeHtml(s)}
        </div>
      `).join('')}
    </div>
    ${metrics.map(m => {
      const values = rows.map(r => Number(r[m.key] ?? 0));
      const max = Math.max(...values);
      return `
        <div class="comparison-row">
          <div class="comparison-label">
            <span class="metric-icon">${m.icon}</span>
            ${m.label}
          </div>
          ${values.map((v, i) => `
            <div class="comparison-value ${v === max ? 'highlight' : ''}">
              ${m.prefix || ''}${formatNumber(v, m.format)}${m.suffix || ''}
            </div>
          `).join('')}
        </div>
      `;
    }).join('')}
  `;
}

function initializeDataCharts() {
  if (charts.dataHiring) charts.dataHiring.destroy();
  if (charts.dataEmployee) charts.dataEmployee.destroy();

  const summary = state.dataSummary?.scenario_summary_raw || [];
  if (!summary.length) return;

  const labels = summary.map(r => r.scenario);
  const hiringRates = summary.map(r => Number(r.hiring_rate_pct || 0));
  const promoRates = summary.map(r => Number(r.promotion_rate_pct || 0));

  const ctxHiring = document.getElementById('data-hiring-chart');
  if (ctxHiring) {
    charts.dataHiring = new Chart(ctxHiring, {
      type: 'bar',
      data: {
        labels,
        datasets: [
          { label: 'Hiring Rate (%)', data: hiringRates, backgroundColor: 'rgba(59, 130, 246, 0.7)', borderColor: '#3b82f6', borderWidth: 2, borderRadius: 6 },
          { label: 'Promotion Rate (%)', data: promoRates, backgroundColor: 'rgba(16, 185, 129, 0.7)', borderColor: '#10b981', borderWidth: 2, borderRadius: 6 },
        ],
      },
      options: {
        responsive: true,
        maintainAspectRatio: false,
        plugins: { legend: { position: 'bottom' } },
        scales: { y: { beginAtZero: true, grid: { color: 'rgba(0,0,0,0.05)' } }, x: { grid: { display: false } } },
      },
    });
  }

  const performance = summary.map(r => Number(r.avg_performance_score || 0));
  const salary = summary.map(r => Number(r.avg_salary || 0) / 1000);

  const ctxEmployee = document.getElementById('data-employee-chart');
  if (ctxEmployee) {
    charts.dataEmployee = new Chart(ctxEmployee, {
      type: 'radar',
      data: {
        labels: labels.map(l => l.split(' ')[0]),
        datasets: [
          { label: 'Performance Score', data: performance, borderColor: '#8b5cf6', backgroundColor: 'rgba(139, 92, 246, 0.2)' },
          { label: 'Salary (K)', data: salary, borderColor: '#ec4899', backgroundColor: 'rgba(236, 72, 153, 0.2)' },
        ],
      },
      options: {
        responsive: true,
        maintainAspectRatio: false,
        plugins: { legend: { position: 'bottom' } },
        scales: { r: { beginAtZero: true, grid: { color: 'rgba(0,0,0,0.1)' } } },
      },
    });
  }
}

function renderRiskDashboard() {
  const container = document.getElementById('risk-dashboard-view');
  const risk = state.riskDashboard;

  container.innerHTML = `
    <div class="page-header">
      <h1 class="page-title">Risk Dashboard</h1>
      <p class="page-subtitle">Network nepotism risk analysis and anomaly detection insights.</p>
    </div>

    ${risk.warnings?.length ? `
    <div class="alert warning">
      <svg viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2">
        <path d="M10.29 3.86L1.82 18a2 2 0 0 0 1.71 3h16.94a2 2 0 0 0 1.71-3L13.71 3.86a2 2 0 0 0-3.42 0z"/>
        <line x1="12" y1="9" x2="12" y2="13"/>
        <line x1="12" y1="17" x2="12.01" y2="17"/>
      </svg>
      <span>${risk.warnings.map(escapeHtml).join('<br>')}</span>
    </div>
    ` : ''}

    <div class="card">
      <div class="card-header">
        <div>
          <span class="badge red">Model 3</span>
          <h3 class="card-title">Network Nepotism Risk</h3>
        </div>
      </div>
      <p class="card-subtitle">Structural risk based on family concentration and connected-employee behavior.</p>
      
      <div class="kpi-grid">
        ${(risk.network_metrics || []).map((m, i) => {
          const colors = ['blue', 'yellow', 'red'];
          return renderKpiCard(m.label, m.value, colors[i % colors.length], i);
        }).join('')}
      </div>

      <div class="two-col">
        <div class="chart-container">
          <canvas id="risk-manager-chart"></canvas>
        </div>
        <div class="chart-container">
          <canvas id="risk-department-chart"></canvas>
        </div>
      </div>
    </div>

    <div class="table-heavy-grid">
      <div class="card">
        <h3 class="card-title">Top Risky Managers</h3>
        ${renderTableFromObjects(risk.top_managers)}
      </div>
      <div class="card">
        <h3 class="card-title">Top Risky Departments</h3>
        ${renderTableFromObjects(risk.top_departments)}
      </div>
    </div>

    <div class="card">
      <div class="card-header">
        <div>
          <span class="badge yellow">Model 4</span>
          <h3 class="card-title">Anomaly Detection</h3>
        </div>
      </div>
      <p class="card-subtitle">Suspicious hires and promotions based on merit mismatch and connection intensity.</p>

      <div class="kpi-grid">
        ${(risk.anomaly_metrics || []).map((m, i) => {
          const colors = ['red', 'yellow', 'blue', 'green'];
          return renderKpiCard(m.label, m.value, colors[i % colors.length], i + 2);
        }).join('')}
      </div>

      <div class="chart-container">
        <canvas id="risk-anomaly-chart"></canvas>
      </div>
    </div>

    <div class="table-heavy-grid">
      <div class="card">
        <h3 class="card-title" style="text-align:center;">Top Suspicious Hires</h3>
        ${renderTableFromObjects(risk.top_hiring)}
      </div>
      <div class="card">
        <h3 class="card-title" style="text-align:center;">Top Suspicious Promotions</h3>
        ${renderTableFromObjects(risk.top_promotion)}
      </div>
    </div>
  `;

  initializeRiskCharts();
}

function initializeRiskCharts() {
  if (charts.riskManager) charts.riskManager.destroy();
  if (charts.riskDepartment) charts.riskDepartment.destroy();
  if (charts.riskAnomaly) charts.riskAnomaly.destroy();

  const risk = state.riskDashboard;
  const managerData = risk.manager_risk_series || [];
  const deptData = risk.department_risk_series || [];
  const hiringAnomaly = risk.hiring_anomaly_series || [];
  const promoAnomaly = risk.promotion_anomaly_series || [];

  const ctxManager = document.getElementById('risk-manager-chart');
  if (ctxManager && managerData.length) {
    charts.riskManager = new Chart(ctxManager, {
      type: 'bar',
      data: {
        labels: managerData.map(r => r.scenario),
        datasets: [{
          label: 'Avg Manager Risk Score',
          data: managerData.map(r => parseFloat(r.value)),
          backgroundColor: ['rgba(16, 185, 129, 0.7)', 'rgba(245, 158, 11, 0.7)', 'rgba(239, 68, 68, 0.7)'],
          borderColor: ['#10b981', '#f59e0b', '#ef4444'],
          borderWidth: 2,
          borderRadius: 8,
        }],
      },
      options: {
        responsive: true,
        maintainAspectRatio: false,
        plugins: { legend: { display: false } },
        scales: { y: { beginAtZero: true, grid: { color: 'rgba(0,0,0,0.05)' } }, x: { grid: { display: false } } },
      },
    });
  }

  const ctxDept = document.getElementById('risk-department-chart');
  if (ctxDept && deptData.length) {
    charts.riskDepartment = new Chart(ctxDept, {
      type: 'bar',
      data: {
        labels: deptData.map(r => r.scenario),
        datasets: [{
          label: 'Avg Department Risk Score',
          data: deptData.map(r => parseFloat(r.value)),
          backgroundColor: ['rgba(16, 185, 129, 0.7)', 'rgba(245, 158, 11, 0.7)', 'rgba(239, 68, 68, 0.7)'],
          borderColor: ['#10b981', '#f59e0b', '#ef4444'],
          borderWidth: 2,
          borderRadius: 8,
        }],
      },
      options: {
        responsive: true,
        maintainAspectRatio: false,
        plugins: { legend: { display: false } },
        scales: { y: { beginAtZero: true, grid: { color: 'rgba(0,0,0,0.05)' } }, x: { grid: { display: false } } },
      },
    });
  }

  const ctxAnomaly = document.getElementById('risk-anomaly-chart');
  if (ctxAnomaly && hiringAnomaly.length) {
    charts.riskAnomaly = new Chart(ctxAnomaly, {
      type: 'bar',
      data: {
        labels: hiringAnomaly.map(r => r.scenario),
        datasets: [
          {
            label: 'Hiring Anomaly Rate',
            data: hiringAnomaly.map(r => (parseFloat(r.value) * 100).toFixed(2)),
            backgroundColor: 'rgba(245, 158, 11, 0.7)',
            borderColor: '#f59e0b',
            borderWidth: 2,
            borderRadius: 6,
          },
          {
            label: 'Promotion Anomaly Rate',
            data: promoAnomaly.map(r => (parseFloat(r.value) * 100).toFixed(2)),
            backgroundColor: 'rgba(239, 68, 68, 0.7)',
            borderColor: '#ef4444',
            borderWidth: 2,
            borderRadius: 6,
          },
        ],
      },
      options: {
        responsive: true,
        maintainAspectRatio: false,
        plugins: { legend: { position: 'bottom' } },
        scales: { y: { beginAtZero: true, grid: { color: 'rgba(0,0,0,0.05)' } }, x: { grid: { display: false } } },
      },
    });
  }
}

function renderStatisticalAnalysis() {
  const container = document.getElementById('statistical-analysis-view');
  const analysis = state.statisticalAnalysis;
  const candidatePairs = analysis.candidate_pairs?.pairs || [];
  const employeePairs = analysis.employee_pairs?.pairs || [];

  container.innerHTML = `
    <div class="page-header">
      <h1 class="page-title">Statistical Analysis</h1>
      <p class="page-subtitle">${escapeHtml(analysis.hero?.body || '')}</p>
    </div>

    <div class="two-col">
      <div class="card">
        <h3 class="card-title">Hiring Model Statistics</h3>
        <div class="metric-row" style="grid-template-columns:repeat(2,1fr);">
          <div class="metric-box">
            <div class="metric-label">Model Type</div>
            <div class="metric-value" style="font-size:1rem;">${escapeHtml(analysis.models?.hiring?.description?.split('(')[0] || 'Logistic Regression')}</div>
          </div>
          <div class="metric-box">
            <div class="metric-label">ROC AUC</div>
            <div class="metric-value">${Number(analysis.models?.hiring?.roc_auc || 0).toFixed(4)}</div>
          </div>
          <div class="metric-box">
            <div class="metric-label">Brier Score</div>
            <div class="metric-value">${Number(analysis.models?.hiring?.brier_score || 0).toFixed(4)}</div>
          </div>
          <div class="metric-box">
            <div class="metric-label">Training Rows</div>
            <div class="metric-value">${Number(analysis.models?.hiring?.training_rows || 0).toLocaleString()}</div>
          </div>
        </div>
      </div>
      <div class="card">
        <h3 class="card-title">Promotion Model Statistics</h3>
        <div class="metric-row" style="grid-template-columns:repeat(2,1fr);">
          <div class="metric-box">
            <div class="metric-label">Model Type</div>
            <div class="metric-value" style="font-size:1rem;">${escapeHtml(analysis.models?.promotion?.description?.split('(')[0] || 'Logistic Regression')}</div>
          </div>
          <div class="metric-box">
            <div class="metric-label">ROC AUC</div>
            <div class="metric-value">${Number(analysis.models?.promotion?.roc_auc || 0).toFixed(4)}</div>
          </div>
          <div class="metric-box">
            <div class="metric-label">Brier Score</div>
            <div class="metric-value">${Number(analysis.models?.promotion?.brier_score || 0).toFixed(4)}</div>
          </div>
          <div class="metric-box">
            <div class="metric-label">Training Rows</div>
            <div class="metric-value">${Number(analysis.models?.promotion?.training_rows || 0).toLocaleString()}</div>
          </div>
        </div>
      </div>
    </div>

    <div class="card" style="margin-bottom:2rem;">
      <div class="card-header">
        <div>
          <h3 class="card-title">Sensitivity Analysis</h3>
          <p class="card-subtitle">See how each factor impacts the prediction outcome</p>
        </div>
      </div>
      <div class="sensitivity-charts">
        <div class="sens-panel">
          <div class="sens-panel-header blue">
            <span class="sens-icon">⭐</span>
            <div>
              <h4>Hiring — Merit Impact</h4>
              <p>How merit score affects hiring chances</p>
            </div>
          </div>
          <div class="sens-chart-container">
            <canvas id="sens-hiring-merit"></canvas>
          </div>
        </div>
        <div class="sens-panel">
          <div class="sens-panel-header orange">
            <span class="sens-icon">🔗</span>
            <div>
              <h4>Hiring — Connection Impact</h4>
              <p>How connection strength affects hiring chances</p>
            </div>
          </div>
          <div class="sens-chart-container">
            <canvas id="sens-hiring-connection"></canvas>
          </div>
        </div>
        <div class="sens-panel">
          <div class="sens-panel-header green">
            <span class="sens-icon">🎯</span>
            <div>
              <h4>Promotion — Performance Impact</h4>
              <p>How performance score affects promotion chances</p>
            </div>
          </div>
          <div class="sens-chart-container">
            <canvas id="sens-promotion-performance"></canvas>
          </div>
        </div>
        <div class="sens-panel">
          <div class="sens-panel-header red">
            <span class="sens-icon">🔗</span>
            <div>
              <h4>Promotion — Connection Impact</h4>
              <p>How connection strength affects promotion chances</p>
            </div>
          </div>
          <div class="sens-chart-container">
            <canvas id="sens-promotion-connection"></canvas>
          </div>
        </div>
      </div>
    </div>

    <div class="two-col">
      <div class="card">
        <h3 class="card-title">Candidate Matched Pairs</h3>
        <p class="card-subtitle">Profiles with similar hiring probability but different merit/connection profiles</p>
        ${renderPairSection('candidate', candidatePairs, analysis.candidate_pairs?.note)}
      </div>
      <div class="card">
        <h3 class="card-title">Employee Matched Pairs</h3>
        <p class="card-subtitle">Profiles with similar promotion probability but different performance/connection profiles</p>
        ${renderPairSection('employee', employeePairs, analysis.employee_pairs?.note)}
      </div>
    </div>
  `;

  initializeSensitivityCharts();
  initializePairSelectors('candidate', candidatePairs);
  initializePairSelectors('employee', employeePairs);
}

function renderPairSection(prefix, pairs, note) {
  if (!pairs.length) {
    return `<div class="alert warning"><span>${escapeHtml(note || 'No matched pairs available.')}</span></div>`;
  }

  const typeLabel = prefix === 'candidate' ? 'Hiring' : 'Promotion';
  return `
    <p class="muted" style="font-size:0.85rem;margin-bottom:1rem;">Found ${pairs.length} matched pair${pairs.length > 1 ? 's' : ''} - profiles with similar ${typeLabel.toLowerCase()} probability but different merit/performance vs connection profiles.</p>
    ${pairs.length > 1 ? `
    <div class="pair-selector">
      <select id="${prefix}-pair-select" class="form-select">
        ${pairs.map((p, i) => `<option value="${i}">Pair ${i + 1}: ${p.scenario_merit || p.scenario_connection || 'Unknown'}</option>`).join('')}
      </select>
    </div>
    ` : ''}
    <div id="${prefix}-pair-detail"></div>
  `;
}

function initializePairSelectors(prefix, pairs) {
  if (!pairs.length) return;

  const select = document.getElementById(`${prefix}-pair-select`);
  const detail = document.getElementById(`${prefix}-pair-detail`);

  const renderPair = (index) => {
    const pair = pairs[index] || pairs[0];
    detail.innerHTML = renderPairDetail(pair, prefix);
  };

  if (select) {
    select.addEventListener('change', () => renderPair(Number(select.value)));
  }
  renderPair(0);
}

function renderPairDetail(pair, prefix) {
  const metricLabel = prefix === 'candidate' ? 'Merit Score' : 'Performance Score';
  const probLabel = prefix === 'candidate' ? 'Hire Probability' : 'Promo Probability';

  return `
    <div class="pair-comparison">
      <div class="pair-card merit">
        <div class="pair-card-header">
          <span class="pair-badge">MERIT</span>
          <h4>${prefix === 'candidate' ? 'High Merit Candidate' : 'High Performance Employee'}</h4>
        </div>
        <div class="pair-card-body">
          <div class="pair-stat">
            <span class="pair-stat-label">Name</span>
            <span class="pair-stat-value">${escapeHtml(pair.full_name_merit || 'N/A')}</span>
          </div>
          <div class="pair-stat">
            <span class="pair-stat-label">${metricLabel}</span>
            <span class="pair-stat-value highlight">${Number(pair.merit_value_merit || 0).toFixed(1)}</span>
          </div>
          <div class="pair-stat">
            <span class="pair-stat-label">Connection Index</span>
            <span class="pair-stat-value">${Number(pair.connection_index_merit || 0).toFixed(3)}</span>
          </div>
          <div class="pair-stat">
            <span class="pair-stat-label">${probLabel}</span>
            <span class="pair-stat-value">${(Number(pair.probability_merit || 0) * 100).toFixed(1)}%</span>
          </div>
        </div>
      </div>
      <div class="pair-vs">VS</div>
      <div class="pair-card connection">
        <div class="pair-card-header">
          <span class="pair-badge">CONNECTION</span>
          <h4>High Connection Profile</h4>
        </div>
        <div class="pair-card-body">
          <div class="pair-stat">
            <span class="pair-stat-label">Name</span>
            <span class="pair-stat-value">${escapeHtml(pair.full_name_connection || 'N/A')}</span>
          </div>
          <div class="pair-stat">
            <span class="pair-stat-label">${metricLabel}</span>
            <span class="pair-stat-value">${Number(pair.merit_value_connection || 0).toFixed(1)}</span>
          </div>
          <div class="pair-stat">
            <span class="pair-stat-label">Connection Index</span>
            <span class="pair-stat-value highlight">${Number(pair.connection_index_connection || 0).toFixed(3)}</span>
          </div>
          <div class="pair-stat">
            <span class="pair-stat-label">${probLabel}</span>
            <span class="pair-stat-value">${(Number(pair.probability_connection || 0) * 100).toFixed(1)}%</span>
          </div>
        </div>
      </div>
    </div>
    <p class="pair-insight">Both profiles receive <strong>similar predictions</strong> despite different merit/performance and connection profiles - demonstrating the substitution effect in the model.</p>
  `;
}

function initializeSensitivityCharts() {
  const analysis = state.statisticalAnalysis;
  const sens = analysis.sensitivity || {};

  Object.keys(charts).forEach(key => {
    if (key.startsWith('sens-')) {
      charts[key].destroy();
      delete charts[key];
    }
  });

  const chartConfigs = [
    { id: 'sens-hiring-merit', data: sens.hiring_merit, xLabel: 'Merit Score', color: '#3b82f6' },
    { id: 'sens-hiring-connection', data: sens.hiring_connection, xLabel: 'Connection Strength', color: '#f59e0b' },
    { id: 'sens-promotion-performance', data: sens.promotion_performance, xLabel: 'Performance Score', color: '#10b981' },
    { id: 'sens-promotion-connection', data: sens.promotion_connection, xLabel: 'Connection Strength', color: '#ef4444' },
  ];

  chartConfigs.forEach(config => {
    const canvas = document.getElementById(config.id);
    if (!canvas || !config.data?.curve?.length) return;

    const points = config.data.curve;
    charts[`sens-${config.id}`] = new Chart(canvas, {
      type: 'line',
      data: {
        labels: points.map(p => Number(p.x_value).toFixed(1)),
        datasets: [{
          label: 'Predicted Probability',
          data: points.map(p => (Number(p.predicted_probability) * 100).toFixed(1)),
          borderColor: config.color,
          backgroundColor: `${config.color}20`,
          fill: true,
          tension: 0.4,
          pointRadius: 4,
          pointBackgroundColor: config.color,
        }],
      },
      options: {
        responsive: true,
        maintainAspectRatio: false,
        plugins: { legend: { display: false } },
        scales: {
          x: { title: { display: true, text: config.xLabel }, grid: { display: false } },
          y: { title: { display: true, text: 'Probability (%)' }, beginAtZero: true, grid: { color: 'rgba(0,0,0,0.05)' } },
        },
      },
    });
  });
}

function renderOrganizationalImpact() {
  const container = document.getElementById('organizational-impact-view');
  const impact = state.organizationalImpact;

  container.innerHTML = `
    <div class="page-header">
      <h1 class="page-title">Organizational Impact</h1>
      <p class="page-subtitle">${escapeHtml(impact.hero?.body || '')}</p>
    </div>

    ${impact.warnings?.length ? `
    <div class="alert warning" style="margin-bottom:1.5rem;">
      <svg viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2">
        <path d="M10.29 3.86L1.82 18a2 2 0 0 0 1.71 3h16.94a2 2 0 0 0 1.71-3L13.71 3.86a2 2 0 0 0-3.42 0z"/>
        <line x1="12" y1="9" x2="12" y2="13"/>
        <line x1="12" y1="17" x2="12.01" y2="17"/>
      </svg>
      <span>${impact.warnings.map(escapeHtml).join('<br>')}</span>
    </div>
    ` : ''}

    <div class="card" style="margin-bottom:2rem;">
      <div class="card-header">
        <div>
          <h3 class="card-title">Executive Overview</h3>
          <p class="card-subtitle">Key metrics comparison across scenarios</p>
        </div>
      </div>
      <div class="chart-container">
        <canvas id="impact-executive-chart"></canvas>
      </div>
    </div>

    <div class="table-heavy-grid" style="margin-bottom:2rem;">
      <div class="card">
        <h3 class="card-title">Hiring Quality</h3>
        <p class="card-subtitle">Merit-based hiring effectiveness by scenario</p>
        <div class="chart-container short">
          <canvas id="impact-hiring-chart"></canvas>
        </div>
        <div style="margin-top:1rem;">
          ${renderTableFromObjects(impact.hiring_quality_table)}
        </div>
      </div>
      <div class="card">
        <h3 class="card-title">Promotion Fairness</h3>
        <p class="card-subtitle">Performance-aligned promotion effectiveness</p>
        <div class="chart-container short">
          <canvas id="impact-promotion-chart"></canvas>
        </div>
        <div style="margin-top:1rem;">
          ${renderTableFromObjects(impact.promotion_fairness_table)}
        </div>
      </div>
    </div>

    <div class="card" style="margin-bottom:2rem;">
      <h3 class="card-title">Model Coefficients</h3>
      <p class="card-subtitle">Statistical significance of model features</p>
      <div class="table-heavy-grid" style="margin-top:1rem;">
        ${renderCoefficientPanel('Hiring Model Coefficients', impact.coefficients?.hiring)}
        ${renderCoefficientPanel('Promotion Model Coefficients', impact.coefficients?.promotion)}
      </div>
    </div>
  `;

  initializeImpactCharts();
}

function renderCoefficientPanel(title, block) {
  return `
    <div class="info-panel">
      <h4>${escapeHtml(title)}</h4>
      ${block?.note ? `<p class="muted" style="margin-bottom:1rem;">${escapeHtml(block.note)}</p>` : ''}
      ${renderTableFromObjects(block?.table)}
    </div>
  `;
}

function initializeImpactCharts() {
  if (charts.impactExecutive) charts.impactExecutive.destroy();
  if (charts.impactHiring) charts.impactHiring.destroy();
  if (charts.impactPromotion) charts.impactPromotion.destroy();

  const impact = state.organizationalImpact;
  const executive = impact.executive_table || [];
  if (!executive.length) return;

  const proxyKey = Object.keys(executive[0] || {}).find(k => k.toLowerCase().includes('proxy')) || 'proxy_per_100_employees';

  const ctxExec = document.getElementById('impact-executive-chart');
  if (ctxExec) {
    charts.impactExecutive = new Chart(ctxExec, {
      type: 'bar',
      data: {
        labels: executive.map(r => r.Scenario || r.scenario),
        datasets: [
          {
            label: impact.proxy_metric_label || 'Output per 100 employees',
            data: executive.map(r => Number(r[proxyKey] || 0).toFixed(2)),
            backgroundColor: ['rgba(16, 185, 129, 0.7)', 'rgba(245, 158, 11, 0.7)', 'rgba(239, 68, 68, 0.7)'],
            borderColor: ['#10b981', '#f59e0b', '#ef4444'],
            borderWidth: 2,
            borderRadius: 8,
          },
        ],
      },
      options: {
        responsive: true,
        maintainAspectRatio: false,
        plugins: { legend: { display: false } },
        scales: { y: { beginAtZero: true, grid: { color: 'rgba(0,0,0,0.05)' } }, x: { grid: { display: false } } },
      },
    });
  }

  const hiringQuality = impact.hiring_quality_series || {};
  const ctxHiring = document.getElementById('impact-hiring-chart');
  if (ctxHiring && hiringQuality.avg_hired_merit?.length) {
    charts.impactHiring = new Chart(ctxHiring, {
      type: 'doughnut',
      data: {
        labels: hiringQuality.avg_hired_merit.map(r => r.scenario),
        datasets: [{
          data: hiringQuality.avg_hired_merit.map(r => Number(r.value).toFixed(2)),
          backgroundColor: ['rgba(16, 185, 129, 0.7)', 'rgba(245, 158, 11, 0.7)', 'rgba(239, 68, 68, 0.7)'],
          borderColor: ['#10b981', '#f59e0b', '#ef4444'],
          borderWidth: 2,
        }],
      },
      options: {
        responsive: true,
        maintainAspectRatio: false,
        plugins: { legend: { position: 'bottom' } },
      },
    });
  }

  const promoFairness = impact.promotion_fairness_series || {};
  const ctxPromo = document.getElementById('impact-promotion-chart');
  if (ctxPromo && promoFairness.promotion_quality_ratio?.length) {
    charts.impactPromotion = new Chart(ctxPromo, {
      type: 'polarArea',
      data: {
        labels: promoFairness.promotion_quality_ratio.map(r => r.scenario),
        datasets: [{
          data: promoFairness.promotion_quality_ratio.map(r => Number(r.value).toFixed(3)),
          backgroundColor: ['rgba(16, 185, 129, 0.5)', 'rgba(245, 158, 11, 0.5)', 'rgba(239, 68, 68, 0.5)'],
          borderColor: ['#10b981', '#f59e0b', '#ef4444'],
          borderWidth: 2,
        }],
      },
      options: {
        responsive: true,
        maintainAspectRatio: false,
        plugins: { legend: { position: 'bottom' } },
      },
    });
  }
}
