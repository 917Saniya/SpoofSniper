(function(){
  const $ = (sel) => document.querySelector(sel);

  // Enhanced reveal on scroll with staggered animations
  function bindReveal(){
    const obs = new IntersectionObserver((entries)=>{
      entries.forEach((e, index) => { 
        if(e.isIntersecting){ 
          // Add staggered delay for multiple elements
          setTimeout(() => {
            e.target.classList.add('show');
          }, index * 100);
          obs.unobserve(e.target);
        } 
      });
    }, { threshold: 0.1, rootMargin: '0px 0px -50px 0px' });
    document.querySelectorAll('.reveal').forEach(el=>obs.observe(el));
  }

  // Analysis form
  function bindAnalysisForm(){
    const form = $('#analysisForm');
    if(!form) return;

    form.addEventListener('submit', async (e) => {
      e.preventDefault();
      const text = $('#textInput').value.trim();
      const username = $('#usernameInput').value.trim();
      const followers = $('#followersInput').value ? Number($('#followersInput').value) : undefined;
      if(!text){
        $('#resultBox').textContent = 'Please provide text.';
        return;
      }
      $('#resultBox').textContent = 'Analyzing...';
      $('#explainBox').textContent = '';
      try{
        const result = await localWorkflow.run({text, username, followers});
        $('#resultBox').textContent = `${result.label} (overall risk: ${result.overallRiskPct}%)`;
        $('#explainBox').textContent = result.explanations.join(' • ');
        // Also write a mini breakdown
        const breakdown = `Text risk: ${result.textRiskPct}%, Account risk: ${result.accountRiskPct}%`;
        $('#explainBox').textContent += ` • ${breakdown}`;
      }catch(err){
        $('#resultBox').textContent = 'Error: ' + err.message;
      }
    });
  }

  // Settings risk chart (text/account/overall)
  function bindSettings(){
    const generateBtn = $('#generateReport');
    if(!generateBtn) return;

    const ctx = $('#metricsBar');
    let chart;

    generateBtn.addEventListener('click', async ()=>{
      const text = $('#textInputSettings')?.value?.trim() || 'Sample giveaway! Click to verify now';
      const username = $('#usernameSettings')?.value?.trim() || 'official_support_123';
      const followers = $('#followersSettings')?.value ? Number($('#followersSettings').value) : 120;

      $('#reportResult').textContent = 'Computing risk...';
      try{
        const result = await localWorkflow.run({text, username, followers});
        $('#reportResult').textContent = JSON.stringify(result, null, 2);
        const labels = ['Text Risk','Account Risk','Overall Risk'];
        const values = [result.textRiskPct, result.accountRiskPct, result.overallRiskPct];
        if(chart) chart.destroy();
        chart = new Chart(ctx, {
          type: 'bar',
          data: { labels, datasets: [{ label: 'Risk (%)', data: values, backgroundColor: ['#0d6efd','#fd7e14','#6f42c1'] }]},
          options: { scales: { y: { beginAtZero: true, max: 100 } } }
        });
      }catch(err){
        $('#reportResult').textContent = 'Error: ' + err.message;
      }
    });
  }

  document.addEventListener('DOMContentLoaded', ()=>{
    bindReveal();
    bindAnalysisForm();
    bindSettings();
  });
})();
