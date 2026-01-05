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
    // Initialize tooltips after HTMX swaps content
    setTimeout(initTooltips, 100);
});

// Global tooltip state
let currentTooltip = {
    element: null,
    trigger: null,
    timeout: null
};

// Initialize tooltips by attaching event listeners to each trigger
function initTooltips() {
    const tooltipTriggers = document.querySelectorAll('.tooltip-trigger');
    
    tooltipTriggers.forEach((trigger, index) => {
        // Skip if already initialized (check for data attribute)
        if (trigger.dataset.tooltipInitialized === 'true') {
            return;
        }
        
        const tooltipText = trigger.getAttribute('data-tooltip');
        if (!tooltipText) {
            return;
        }
        
        // Mark as initialized
        trigger.dataset.tooltipInitialized = 'true';
        
        trigger.addEventListener('mouseenter', function(e) {
            // Clear any existing tooltip
            hideTooltip();
            
            // Create tooltip element
            const tooltipElement = document.createElement('div');
            tooltipElement.className = 'tooltip-content';
            tooltipElement.textContent = tooltipText;
            tooltipElement.style.display = 'block';
            tooltipElement.style.visibility = 'hidden'; // Initially hidden to measure
            document.body.appendChild(tooltipElement);
            
            // Store reference
            currentTooltip.element = tooltipElement;
            currentTooltip.trigger = trigger;
            
            // Force reflow to get accurate dimensions
            const tooltipWidth = tooltipElement.offsetWidth;
            const tooltipHeight = tooltipElement.offsetHeight;
            
            // Make visible for positioning
            tooltipElement.style.visibility = 'visible';
            
            // Position tooltip
            positionTooltip(trigger, tooltipElement);
            
            // Show tooltip with slight delay for smooth animation
            currentTooltip.timeout = setTimeout(() => {
                if (tooltipElement && tooltipElement.parentNode) {
                    tooltipElement.classList.add('show');
                    // Force opacity with inline style to override CSS
                    tooltipElement.style.setProperty('opacity', '1', 'important');
                }
            }, 50);
        });
        
        trigger.addEventListener('mouseleave', function() {
            hideTooltip();
        });
        
        trigger.addEventListener('mousemove', function() {
            if (currentTooltip.element && currentTooltip.trigger === trigger) {
                positionTooltip(trigger, currentTooltip.element);
            }
        });
    });
}

function hideTooltip() {
    if (currentTooltip.timeout) {
        clearTimeout(currentTooltip.timeout);
        currentTooltip.timeout = null;
    }
    
    if (currentTooltip.element) {
        currentTooltip.element.classList.remove('show');
        setTimeout(() => {
            if (currentTooltip.element && currentTooltip.element.parentNode) {
                currentTooltip.element.remove();
            }
            currentTooltip.element = null;
            currentTooltip.trigger = null;
        }, 200);
    }
}

function positionTooltip(trigger, tooltip) {
    const triggerRect = trigger.getBoundingClientRect();
    const tooltipWidth = tooltip.offsetWidth;
    const tooltipHeight = tooltip.offsetHeight;
    const viewportWidth = window.innerWidth;
    const viewportHeight = window.innerHeight;
    const spacing = 8;
    const scrollX = window.pageXOffset || document.documentElement.scrollLeft;
    const scrollY = window.pageYOffset || document.documentElement.scrollTop;
    
    // Default: show tooltip above
    let top = triggerRect.top + scrollY - tooltipHeight - spacing;
    let left = triggerRect.left + scrollX + (triggerRect.width / 2) - (tooltipWidth / 2);
    let position = 'tooltip-bottom';
    
    // Check if tooltip goes off screen vertically
    if (triggerRect.top - tooltipHeight - spacing < 0) {
        // Show below instead
        top = triggerRect.bottom + scrollY + spacing;
        position = 'tooltip-top';
    }
    
    // Check horizontal boundaries
    if (left < scrollX + spacing) {
        left = scrollX + spacing;
    } else if (left + tooltipWidth > scrollX + viewportWidth - spacing) {
        left = scrollX + viewportWidth - tooltipWidth - spacing;
    }
    
    // If still doesn't fit vertically, try left/right
    const tooltipBottom = top + tooltipHeight - scrollY;
    if (tooltipBottom > viewportHeight || top - scrollY < 0) {
        // Try right side
        left = triggerRect.right + scrollX + spacing;
        top = triggerRect.top + scrollY + (triggerRect.height / 2) - (tooltipHeight / 2);
        position = 'tooltip-left';
        
        if (left + tooltipWidth > scrollX + viewportWidth) {
            // Try left side
            left = triggerRect.left + scrollX - tooltipWidth - spacing;
            position = 'tooltip-right';
        }
        
        // Ensure tooltip stays within viewport vertically
        if (top - scrollY < spacing) {
            top = scrollY + spacing;
        } else if (top + tooltipHeight - scrollY > viewportHeight - spacing) {
            top = scrollY + viewportHeight - tooltipHeight - spacing;
        }
    }
    
    tooltip.className = 'tooltip-content ' + position;
    tooltip.style.position = 'absolute';
    tooltip.style.top = top + 'px';
    tooltip.style.left = left + 'px';
    tooltip.style.zIndex = '10000';
}

// Initialize tooltips on page load
document.addEventListener('DOMContentLoaded', function() {
    initTooltips();
    
    // Watch for new tooltip triggers added dynamically (e.g., via HTMX)
    const observer = new MutationObserver(function(mutations) {
        let shouldInit = false;
        mutations.forEach(function(mutation) {
            if (mutation.addedNodes.length > 0) {
                mutation.addedNodes.forEach(function(node) {
                    if (node.nodeType === 1) { // Element node
                        if (node.classList && node.classList.contains('tooltip-trigger')) {
                            shouldInit = true;
                        } else if (node.querySelector && node.querySelector('.tooltip-trigger')) {
                            shouldInit = true;
                        }
                    }
                });
            }
        });
        if (shouldInit) {
            console.log('[Tooltips] New tooltip triggers detected, reinitializing...');
            setTimeout(initTooltips, 50);
        }
    });
    
    // Start observing
    observer.observe(document.body, {
        childList: true,
        subtree: true
    });
});
