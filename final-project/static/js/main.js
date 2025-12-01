const form = document.getElementById("credit-form");
const resultDiv = document.getElementById("result");
const probaBar = document.getElementById("proba-bar");

form.addEventListener("submit", async (e) => {
    e.preventDefault();

    const formData = new FormData(form);
    const data = Object.fromEntries(formData.entries());

    ["person_age","person_income","person_emp_length","loan_amnt",
     "loan_int_rate","loan_percent_income","cb_person_cred_hist_length"]
        .forEach(key => data[key] = parseFloat(data[key]));

    const response = await fetch("/predict", {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify(data)
    });

    const result = await response.json();

    const colors = { "Низкий": "green", "Средний": "orange", "Высокий": "red" };

    resultDiv.style.color = colors[result.risk_level];
    resultDiv.textContent = `Prediction: ${result.prediction}, Probability: ${result.probability_default.toFixed(2)}, Risk: ${result.risk_level}`;

    probaBar.style.width = (result.probability_default*100) + "%";
    probaBar.textContent = (result.probability_default*100).toFixed(1) + "%";
    probaBar.style.backgroundColor = colors[result.risk_level];
});
