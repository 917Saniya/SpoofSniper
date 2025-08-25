// Handle profile submission
document.getElementById('profileForm')?.addEventListener('submit', function(e) {
  e.preventDefault();
  alert("Profile submitted successfully!");
  window.location.href = "result.html";
});

// Simulate result display
if (window.location.pathname.includes("result.html")) {
  document.getElementById('resultBox').innerHTML = `
    <h3>Profile: John Doe</h3>
    <p><strong>Risk Score:</strong> 78%</p>
    <p><strong>Status:</strong> ⚠️ Suspicious</p>
  `;
}

// Simulate admin dashboard
if (window.location.pathname.includes("admin.html")) {
  const tableBody = document.querySelector('#flaggedTable tbody');
  const sampleData = [
    { name: "Pratik Gode", email: "pratikg@example.com", score: "85%", status: "Flagged" },
    { name: "Saniya Pathan", email: "saniyap@example.com", score: "92%", status: "Flagged" }
  ];
  sampleData.forEach(profile => {
    const row = `<tr>
      <td>${profile.name}</td>
      <td>${profile.email}</td>
      <td>${profile.score}</td>
      <td>${profile.status}</td>
    </tr>`;
    tableBody.innerHTML += row;
  });
}
