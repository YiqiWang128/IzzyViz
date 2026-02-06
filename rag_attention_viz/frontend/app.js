// RAG + Attention Visualization Frontend
// API base URL
const API_BASE = window.location.origin;

// State
let questions = [];
let currentResults = null;

// DOM Elements
const questionSelect = document.getElementById('questionSelect');
const customQuestion = document.getElementById('customQuestion');
const kRetrieveSlider = document.getElementById('kRetrieve');
const kContextSlider = document.getElementById('kContext');
const layerSlider = document.getElementById('layer');
const headSlider = document.getElementById('head');
const queryBtn = document.getElementById('queryBtn');
const loadingIndicator = document.getElementById('loadingIndicator');
const resultsSection = document.getElementById('resultsSection');

// Value display elements
const kRetrieveValue = document.getElementById('kRetrieveValue');
const kContextValue = document.getElementById('kContextValue');
const layerValue = document.getElementById('layerValue');
const headValue = document.getElementById('headValue');

// Initialize
document.addEventListener('DOMContentLoaded', () => {
    console.log('Initializing RAG Attention Visualization App...');
    loadQuestions();
    setupEventListeners();
});

// Load questions from API
async function loadQuestions() {
    try {
        const response = await fetch(`${API_BASE}/api/questions`);
        const data = await response.json();
        questions = data.questions;

        // Populate select dropdown
        questionSelect.innerHTML = '<option value="">-- Choose a question --</option>';
        questions.forEach(q => {
            const option = document.createElement('option');
            option.value = q.question;
            option.textContent = `${q.id}. ${q.question}`;
            questionSelect.appendChild(option);
        });

        console.log(`Loaded ${questions.length} questions`);
    } catch (error) {
        console.error('Error loading questions:', error);
        questionSelect.innerHTML = '<option value="">加载失败 - Error loading questions</option>';
    }
}

// Setup event listeners
function setupEventListeners() {
    // Slider updates
    kRetrieveSlider.addEventListener('input', (e) => {
        kRetrieveValue.textContent = e.target.value;
        // Ensure k_context <= k_retrieve
        if (parseInt(kContextSlider.value) > parseInt(e.target.value)) {
            kContextSlider.value = e.target.value;
            kContextValue.textContent = e.target.value;
        }
    });

    kContextSlider.addEventListener('input', (e) => {
        kContextValue.textContent = e.target.value;
    });

    layerSlider.addEventListener('input', (e) => {
        const val = parseInt(e.target.value);
        layerValue.textContent = val === -1 ? '-1 (Last)' : val;
    });

    headSlider.addEventListener('input', (e) => {
        headValue.textContent = e.target.value;
    });

    // Question select
    questionSelect.addEventListener('change', (e) => {
        if (e.target.value) {
            customQuestion.value = '';
        }
    });

    // Custom question
    customQuestion.addEventListener('input', (e) => {
        if (e.target.value) {
            questionSelect.value = '';
        }
    });

    // Query button
    queryBtn.addEventListener('click', handleQuery);

    // Enter key in custom question
    customQuestion.addEventListener('keypress', (e) => {
        if (e.key === 'Enter') {
            handleQuery();
        }
    });
}

// Handle query submission
async function handleQuery() {
    // Get question
    const question = customQuestion.value.trim() || questionSelect.value;

    if (!question) {
        alert('请选择或输入一个问题！\nPlease select or enter a question!');
        return;
    }

    // Get parameters
    const params = {
        question: question,
        k_retrieve: parseInt(kRetrieveSlider.value),
        k_context: parseInt(kContextSlider.value),
        layer: parseInt(layerSlider.value),
        head: parseInt(headSlider.value),
        max_tokens: 48
    };

    console.log('Processing query with params:', params);

    // Show loading
    queryBtn.disabled = true;
    queryBtn.textContent = '⏳ Processing...';
    loadingIndicator.classList.remove('hidden');
    resultsSection.classList.add('hidden');

    try {
        // Call API
        const response = await fetch(`${API_BASE}/api/query`, {
            method: 'POST',
            headers: {
                'Content-Type': 'application/json'
            },
            body: JSON.stringify(params)
        });

        if (!response.ok) {
            const error = await response.json();
            throw new Error(error.detail || 'Query failed');
        }

        const results = await response.json();
        currentResults = results;

        console.log('Query results:', results);

        // Display results
        displayResults(results);

    } catch (error) {
        console.error('Error processing query:', error);
        alert(`Error: ${error.message}\n\nPlease check the console for more details.`);
    } finally {
        // Hide loading
        queryBtn.disabled = false;
        queryBtn.textContent = '🚀 Process Query';
        loadingIndicator.classList.add('hidden');
    }
}

// Display results
function displayResults(results) {
    // Show results section
    resultsSection.classList.remove('hidden');

    // Scroll to results
    resultsSection.scrollIntoView({ behavior: 'smooth' });

    // Display query
    document.getElementById('queryText').textContent = results.query;

    // Display answers
    document.getElementById('answerNoRAG').innerHTML = `
        <p>${results.answer_no_rag || 'No answer generated'}</p>
        <small style="color: var(--text-secondary); margin-top: 10px; display: block;">
            Tokens: ${results.tokens_no_rag_length || 'N/A'}
        </small>
    `;

    document.getElementById('answerWithRAG').innerHTML = `
        <p>${results.answer_with_rag || 'No answer generated'}</p>
        <small style="color: var(--text-secondary); margin-top: 10px; display: block;">
            Tokens: ${results.tokens_rag_length || 'N/A'}
        </small>
    `;

    // Display retrieved documents
    displayRetrievedDocs(results.retrieved_docs, results.parameters.k_context);

    // Display visualizations
    displayVisualizations(results.visualization_paths);
}

// Display retrieved documents
function displayRetrievedDocs(docs, k_context) {
    const docsList = document.getElementById('retrievedDocsList');

    if (!docs || docs.length === 0) {
        docsList.innerHTML = '<p>No documents retrieved</p>';
        return;
    }

    docsList.innerHTML = docs.map((doc, index) => {
        const isUsedInContext = index < k_context;
        return `
            <div class="doc-item" style="${isUsedInContext ? 'border-left-color: var(--success-color); font-weight: 500;' : ''}">
                <span class="doc-rank">#${index + 1}</span>
                ${isUsedInContext ? '<span style="color: var(--success-color);">✓ Used in context</span><br>' : ''}
                <span>${doc}</span>
            </div>
        `;
    }).join('');
}

// Display visualizations
function displayVisualizations(paths) {
    console.log('Visualization paths:', paths);

    if (!paths) {
        console.warn('No visualization paths provided');
        return;
    }

    // Display prefill stage visualizations (original)
    if (paths.comparison_path) {
        displayVisualization('comparisonViz', paths.comparison_path);
    }

    if (paths.no_rag_path) {
        displayVisualization('noRAGViz', paths.no_rag_path);
    }

    if (paths.rag_path) {
        displayVisualization('withRAGViz', paths.rag_path);
    }

    // Display decode stage visualizations (NEW!)
    if (paths.decode_no_rag_path) {
        displayVisualization('decodeNoRAGViz', paths.decode_no_rag_path);
    }

    if (paths.decode_rag_path) {
        displayVisualization('decodeWithRAGViz', paths.decode_rag_path);
    }
}

// Display single visualization
function displayVisualization(elementId, path) {
    const container = document.getElementById(elementId);

    if (!container) {
        console.error(`Container ${elementId} not found`);
        return;
    }

    // Extract filename from path
    const filename = path.split('/').pop();

    // Create embed element for PDF
    container.innerHTML = `
        <embed src="/api/visualization/${filename}" type="application/pdf" width="100%" height="500px">
        <p style="margin-top: 10px; text-align: center;">
            <a href="/api/visualization/${filename}" target="_blank" style="color: var(--primary-color);">
                📄 Open in new tab
            </a>
        </p>
    `;

    console.log(`Displayed visualization: ${filename} in ${elementId}`);
}

// Utility: Format timestamp
function formatTimestamp(date) {
    return date.toLocaleString('zh-CN');
}

// Health check on load
async function checkHealth() {
    try {
        const response = await fetch(`${API_BASE}/api/health`);
        const health = await response.json();
        console.log('Server health:', health);

        if (!health.rag_system_loaded || !health.visualizer_loaded) {
            console.warn('Server is still initializing...');
        }
    } catch (error) {
        console.error('Health check failed:', error);
    }
}

// Run health check
checkHealth();

console.log('App initialized successfully!');
