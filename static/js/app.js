// Custom JavaScript for Portfolio Rebalancer

// Update momentum weight display
document.addEventListener('DOMContentLoaded', function() {
    const momentumWeightSlider = document.getElementById('momentum_weight');
    const momentumWeightValue = document.getElementById('momentum_weight_value');
    
    if (momentumWeightSlider && momentumWeightValue) {
        momentumWeightSlider.addEventListener('input', function() {
            momentumWeightValue.textContent = this.value;
        });
    }
});

// Handle period mode switching
document.addEventListener('DOMContentLoaded', function() {
    const periodModeRadios = document.querySelectorAll('input[name="period_mode"]');
    const periodMonthsInput = document.getElementById('period_months_input');
    const analysisPeriodInput = document.getElementById('analysis_period');
    
    periodModeRadios.forEach(radio => {
        radio.addEventListener('change', function() {
            if (this.value === 'months') {
                periodMonthsInput.style.display = 'block';
            } else {
                periodMonthsInput.style.display = 'none';
            }
            updateAnalysisForm();
        });
    });
    
    if (periodMonthsInput) {
        periodMonthsInput.addEventListener('input', updateAnalysisForm);
    }
    
    // Update analysis form inputs
    function updateAnalysisForm() {
        const periodMode = document.querySelector('input[name="period_mode"]:checked')?.value || 'months';
        let period = '';
        if (periodMode === 'months') {
            period = periodMonthsInput?.value || '12';
        } else {
            period = periodMode.toUpperCase();
        }
        
        if (analysisPeriodInput) {
            analysisPeriodInput.value = period;
        }
        
        const rfInput = document.getElementById('analysis_rf');
        const benchInput = document.getElementById('analysis_bench');
        const momentumInput = document.getElementById('analysis_momentum_weight');
        
        if (rfInput) rfInput.value = (document.getElementById('rf')?.value || 0) / 100;
        if (benchInput) benchInput.value = document.getElementById('bench')?.value || 'SPY';
        if (momentumInput) momentumInput.value = document.getElementById('momentum_weight')?.value || '0.2';
    }
    
    // Update evaluation form inputs
    function updateEvaluationForm() {
        const rcOverInput = document.getElementById('evaluation_rc_over_thresh');
        const eThreshInput = document.getElementById('evaluation_e_thresh');
        
        if (rcOverInput) rcOverInput.value = document.getElementById('rc_over_thresh')?.value || '1.5';
        if (eThreshInput) eThreshInput.value = document.getElementById('e_thresh')?.value || '0.5';
    }
    
    // Update forms when sidebar inputs change
    ['rf', 'bench', 'momentum_weight', 'rc_over_thresh', 'e_thresh'].forEach(id => {
        const input = document.getElementById(id);
        if (input) {
            input.addEventListener('input', function() {
                updateAnalysisForm();
                updateEvaluationForm();
            });
        }
    });
    
    // Initial update
    updateAnalysisForm();
    updateEvaluationForm();
});

// Handle Chart.js initialization after HTMX swaps content
document.body.addEventListener('htmx:afterSwap', function(event) {
    // Check if evaluation results were swapped
    if (event.detail.target.id === 'evaluation_results') {
        // Call the chart initialization function if it exists
        if (typeof initQuadrantChart === 'function') {
            setTimeout(initQuadrantChart, 50);
        }
    }
});
